# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""
A pass for analyzing device attribute of each IR node.
"""
from typing import Optional
from collections import defaultdict

from ..expr_functor import ExprVisitor
from ..function import Function
from .. import op
from ... import register_func
from ..._ffi.runtime_ctypes import TVMContext


def is_primitive(call):
    """Check if a call node is a primitive call.

    Parameters
    ----------
    call : tvm.relay.Call
        The call node to be checked.

    Returns
    -------
    ret : Boolean
        True if the call is primitive. Otherwise, false.
    """
    return hasattr(call, 'op') and hasattr(call.op, 'attrs') and \
           hasattr(call.op.attrs, 'Primitive') and int(call.op.attrs.Primitive) == 1


class DeviceDomain:
    """A class to represent the device of a domain, i.e. a segment of relay
    program.

    Parameters
    ----------
    ctx : Optional[tvm.runtime.TVMContext]
        The device to be assigned to the current domain. It is optional.
    """
    def __init__(self, ctx: Optional[TVMContext]):
        self.domain = ctx

    def join(self, other: 'DeviceDomain') -> 'DeviceDomain':
        """Merge the device of two domains.

        Parameters
        ----------
        other : DeviceDomain
            The other domain to be merged.

        Returns
        -------
        ret : DeviceDomain
            The merged domain. An error will be raised if two domain has
            conflict, i.e. they have different context.
        """
        if self.domain is None and other.domain is None:
            return self
        elif self.domain is None:
            return other
        elif other.domain is None:
            return self
        elif (self.domain.device_type == other.domain.device_type and
              self.domain.device_id == other.domain.device_id):
            return self
        else:
            raise Exception("all expressions must have a singular device")

    def __hash__(self):
        if self.domain is None:
            return id(self)
        else:
            return hash((self.domain.device_type, self.domain.device_id))

    def __eq__(self, other):
        if self.domain is None and other.domain is None:
            return id(self) == id(other)
        else:
            return self.domain == other.domain


def bottom():
    """Create an empty domain. This would usually happen when we enter a new
    scope, i.e. Function.
    """
    return DeviceDomain(None)


def device_type(ctx):
    """Create a domain with the given device context.

    Parameters
    ----------
    ctx : tvm.runtime.TVMContext
        The device context used to construct a domain.

    Returns
    -------
    ret : DeviceDomain
        The constructed domain.
    """
    return DeviceDomain(ctx)


class ContextAnalysis(ExprVisitor):
    """Compute on which device each sub-expression will execute. A union find
    algorithm is used to assign and merge the context domains.

    Parameters
    ----------
    fallback_device : tvm.rutnime.TVMContext
        The default device that could be attached to an expression.
    """
    def __init__(self, fallback_device):
        super().__init__()
        self.expr_to_device = defaultdict(bottom)
        self.device_uf = {}
        self.fallback_device = fallback_device

    def lookup(self, device):
        """Find the root domain of a given device domain.

        Parameters
        ----------
        device : DeviceDomain
            The domain that is used to query the root domain.

        Returns
        -------
        ret : DeviceDomain
            The root domain.
        """
        while device in self.device_uf:
            device = self.device_uf[device]
        return device

    def unify(self, lhs, rhs):
        """Unify the device context of two domains.

        Parameters
        ----------
        lhs : DeviceDomain
            The lhs domain to unify.

        rhs : DeviceDomain
            The rhs domain to unify.

        Returns
        -------
        ret : DeviceDomain
            The unified domain.
        """
        lhs = self.lookup(lhs)
        rhs = self.lookup(rhs)
        unified_device = lhs.join(rhs)
        if not lhs == unified_device:
            self.device_uf[lhs] = unified_device
        if not rhs == unified_device:
            self.device_uf[rhs] = unified_device
        return unified_device

    def unify_expr(self, lhs, rhs):
        """Compute the device type of both expressions and unify them.

        Parameters
        ----------
        lhs : tvm.relay.Expr
            The lhs expression to unify.

        rhs : tvm.relay.Expr
            The rhs expression to unify.

        Returns
        -------
        ret : DeviceDomain
            The unified domain.
        """
        return self.unify(self.device_for(lhs), self.device_for(rhs))

    def device_for(self, expr):
        """Find the domain that contains the given expr.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression used to lookup a domain.

        Returns
        -------
        ret : DeviceDomain
            The domain that contains the expression.
        """
        return self.lookup(self.expr_to_device[expr])

    def device_copy(self, inp, output, src_dev_type, dst_dev_type):
        """Unify the device context for device copy node. Device copy node is
        the only node that carries information in the input program. The device
        attribute of other nodes are propagated from it.

        Parameters
        ----------
        inp : tvm.relay.Expr
            The input expression to the device copy node. The device type of
            the input should be the same as the source device type of the
            copy node.

        output : tvm.relay.Expr
            The output expression of the device copy node. The device type of
            the output should be the same as the destination device type of the
            copy node.

        src_dev_type : int
            The source device type of the copy node.

        dst_dev_type : int
            The destination device type of the copy node.
        """
        src_dev_type = device_type(TVMContext(src_dev_type, 0))
        self.unify(self.device_for(inp), src_dev_type)
        dst_dev_type = device_type(TVMContext(dst_dev_type, 0))
        self.unify(self.device_for(output), dst_dev_type)

    def unify_call(self, call_op, inputs, outputs):
        """Unify the domain of inputs and outputs of a relay Call.

        Parameters
        ----------
        op : tvm.relay.Expr
            The op of a call node.

        inputs : List[tvm.relay.Expr]
            The inputs of the call.

        outputs : List[tvm.relay.Expr]
            The outputs of the call.

        Returns
        -------
            The unified domain.

        Note
        ----
        For most call nodes, the op, inputs, and outputs should all be in the
        same domain, i.e. have the same context. However, device_copy call node
        needs to be handled different as it copies data from one device to
        another.
        """
        device = bottom()
        for arg in inputs:
            device = self.unify(device, self.device_for(arg))

        device = self.unify(device, self.device_for(call_op))

        for out in outputs:
            device = self.unify(device, self.device_for(out))

        return device

    def visit_call(self, call):
        if call.op == op.op.get("device_copy"):
            (input_tensor,) = call.args
            # Device copy op only has one input which is now annotated with the
            # same device to the source device type of the device copy op.
            # The call itself has the same device type to the destination.
            self.device_copy(input_tensor, call,
                             call.attrs.src_dev_type,
                             call.attrs.dst_dev_type)
        elif call.op == op.op.get("memory.alloc_storage"):
            call_dev = device_type(TVMContext(call.attrs.device_type,
                                              call.attrs.device_id))
            self.unify(self.device_for(call), call_dev)
            # The arguments should be one the same device as the call.
            size = call.args[0]
            alignment = call.args[1]
            self.unify(self.device_for(size), call_dev)
            self.unify(self.device_for(alignment), call_dev)
        elif call.op == op.op.get("memory.alloc_tensor"):
            storage = call.args[0]
            shape = call.args[1]
            self.unify(self.device_for(storage), self.device_for(call))
            self.unify(self.device_for(shape), self.device_for(call))
        elif call.op == op.op.get("memory.invoke_tvm_op"):
            if call.args[0].body.op == op.op.get("device_copy"):
                input_tensor = call.args[1][0]
                output_tensor = call.args[2][0]
                self.device_copy(input_tensor, output_tensor,
                                 call.attrs.src_dev_type,
                                 call.attrs.dst_dev_type)
            else:
                self.unify_call(call.args[0], call.args[1].fields,
                                call.args[2].fields)
                super().visit_call(call)
        elif isinstance(call.op, Function):
            device = bottom()
            for arg in call.args:
                self.visit(arg)
                device = self.unify(device, self.device_for(arg))

            for param in call.op.params:
                self.visit(param)
                device = self.unify(device, self.device_for(param))

            out_device = self.device_for(call.op)
            self.unify(self.device_for(call), out_device)
            super().visit_call(call)
        else:
            self.unify_call(call.op, call.args, [call])
            super().visit_call(call)

    def visit_let(self, let):
        self.unify(self.device_for(let.var), self.device_for(let.value))
        self.unify_expr(let, let.body)
        super().visit_let(let)

    def visit_function(self, f):
        self.unify(self.device_for(f), self.device_for(f.body))
        super().visit_function(f)

    def visit_tuple(self, tup):
        # TODO(@zhiics) How to handle tuple with different device context for
        # different fields.
        self.unify(self.device_for(tup), self.device_for(tup[0]))
        super().visit_tuple(tup)

    def visit_tuple_getitem(self, t):
        self.unify(self.device_for(t), self.device_for(t.tuple[t.index]))
        super().visit_tuple_getitem(t)

    def visit_var(self, var):
        self.device_for(var)

    def visit_constant(self, const):
        self.device_for(const)

    def results(self):
        """Return the analysis result.

        Returns
        -------
        ret : Dict[tvm.relay.Expr, DeviceDomain]
            The dictionary mapping each expression to a device context.
        """
        results = {}
        for exp in self.expr_to_device:
            device = self.lookup(self.expr_to_device[exp])
            if device.domain is None:
                results[exp] = self.fallback_device
            else:
                results[exp] = device.domain

        return results


def mk_analysis_annotator(results):
    """Pretty print the annotated relay program with device info"""
    def _annotator(exp):
        if exp in results:
            return f"<{results[exp]}>"
        else:
            return ""

    return _annotator


def context_analysis(expr, fallback_device):
    """Perform device context analysis on a given relay program. This requires
    that the program has already been annotated and rewritten by replacing on
    device annotations with device copy nodes.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expression for analysis

    fallback_device : tvm.runtime.TVMContext
        The default device context

    Returns
    -------
    ret : Dict[tvm.relay.Expr, [int]]
        The mapping of each expression to the device context that is
        represented in a list form as TVMContext is not a runtime object.
    """
    ca = ContextAnalysis(fallback_device)
    ca.visit(expr)
    ret = defaultdict(list)
    for key, val in ca.results().items():
        ret[key] = [val.device_type, val.device_id]
    return ret


register_func("relay.analysis.ContextAnalysis", context_analysis)
