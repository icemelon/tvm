import collections
import numpy as np

import tvm._ffi
from tvm import te
from tvm import runtime
from tvm.ir import container
from tvm.tir import expr
from .. import _ffi_api

@tvm._ffi.register_object("autotvm.Workload")
class Workload(runtime.Object):
    def __init__(self, task_name, args):
        sargs, vmap = self._serialize_args(args)
        self.__init_handle_by_constructor__(
            _ffi_api.CreateWorkload, task_name, args, vmap)
        self._hash_key = (task_name,) + sargs

    @property
    def cargs(self):
        cargs = []
        for arg in self.args:
            if isinstance(arg, runtime.container.String):
                cargs.append(str(arg))
            else:
                cargs.append(arg)
        return cargs

    @property
    def has_wildcard(self):
        return self.wildcard_map.empty()

    @property
    def hash_key(self):
        if not hasattr(self, "_hash_key"):
            sargs, _ = self._serialize_args(self.args)
            self._hash_key = (self.task_name,) + sargs
        return self._hash_key

    @property
    def serialized_args(self):
        sargs, vmap = self._serialize_args(self.args)
        if vmap:
            wildcard = ["__wildcard"]
            for v in vmap:
                if isinstance(v, expr.SizeVar):
                    wildcard.append((vmap[v], v.name, "size_var", v.dtype))
                else:
                    wildcard.append((vmap[v], v.name, "var", v.dtype))
            sargs += (tuple(wildcard),)
        return sargs

    @property
    def serialized_workload(self):
        return (self.task_name,) + self.serialized_args

    def __hash__(self):
        return hash(self.hash_key)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return str(self.hash_key)

    def _serialize_args(self, args):
        vmap = collections.OrderedDict()

        def _encode(x):
            if isinstance(x, te.Tensor):
                shape = _encode(list(x.shape))
                return ('TENSOR', shape, x.dtype)
            if isinstance(x, (tuple, list, container.Array)):
                return tuple([_encode(a) for a in x])
            if isinstance(x, (str, int, float, np.int, np.float)):
                return x
            if isinstance(x, expr.Var):
                if x in vmap:
                    return vmap[x]
                name = "_w%d" % len(vmap)
                vmap[x] = name
                return name
            if isinstance(x, (expr.StringImm, expr.IntImm, expr.FloatImm)):
                return x.value
            if isinstance(x, runtime.container.String):
                return str(x)
            if x is None:
                return None
            raise RuntimeError('Do not support type "%s" in argument. Consider to use'
                               'primitive types or tvm.tir.Var only' % type(x))
        ret = []
        for t in args:
            ret.append(_encode(t))
        return tuple(ret), vmap

    @staticmethod
    def from_serialized(serialized_workload):
        wkl = list(serialized_workload)
        vmap = {}
        if isinstance(wkl[-1], tuple) and wkl[-1][0] == "__wildcard":
            wildcard = wkl.pop()
            for i in range(1, len(wildcard)):
                wc_name, vname, var_type, dtype = wildcard[i]
                assert var_type in ["var", "size_var"]
                if var_type == "size_var":
                    v = expr.SizeVar(vname, dtype)
                else:
                    v = expr.Var(vname, dtype)
                vmap[wc_name] = v
        task_name = wkl.pop(0)
        args = []
        for t in wkl:
            if isinstance(t, tuple) and t[0] == 'TENSOR':
                shape = []
                for dim in t[1]:
                    if isinstance(dim, str):
                        assert dim in vmap
                        shape.append(vmap[dim])
                    else:
                        shape.append(dim)
                args.append(te.placeholder(shape=shape, dtype=t[2]))
            else:
                args.append(t)
        return Workload(task_name, args)
