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
A pass for manifesting explicit memory allocations.
"""
import numpy as np
import logging

from tvm.ir.transform import PassContext
from tvm import nd, container, tir
from ..expr_functor import ExprMutator, ExprVisitor
from ..function import Function
from ..scope_builder import ScopeBuilder
from . import transform
from .. import op
from ... import DataType, register_func
from .. import ty, expr
from ..backend import compile_engine
from ..op.memory import flatten_tuple_type, from_tuple_type, to_tuple_type
from ..analysis.context_analysis import ContextAnalysis, mk_analysis_annotator, is_primitive
from ...import cpu


logging.basicConfig(level=logging.DEBUG)

class IsReshapePass(ExprVisitor):
    def __init__(self):
        super(ExprVisitor, self).__init__()
        self.is_reshape = True

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)
        if call.op != op.get("reshape") and call.op != op.get("_contrib_reverse_reshape"):
            self.is_reshape = False


def is_reshape(func):
    is_reshape_pass = IsReshapePass()
    is_reshape_pass.visit(func)
    return is_reshape_pass.is_reshape


class ManifestAllocPass(ExprMutator):
    """A pass for explicitly manifesting all memory allocations in Relay."""

    def __init__(self, target_host, context_analysis):
        self.invoke_tvm = op.memory.invoke_tvm_op
        self.alloc_storage = op.memory.alloc_storage
        self.alloc_tensor = op.memory.alloc_tensor
        self.shape_func = op.memory.shape_func
        self.reshape_tensor = op.memory.reshape_tensor
        self.scopes = [ScopeBuilder()]
        self.target_host = target_host
        self.default_context = cpu(0)
        self.compute_dtype = "int64"
        self.context_analysis = context_analysis
        super().__init__()

    def get_context(self, expr):
        # TODO(@zhiics) Need to add all newly created epxressions to the map.
        # assert expr in self.context_analysis, expr.astext(False)
        if expr in self.context_analysis:
            return self.context_analysis[expr]
        else:
            return self.default_context

    def device_copy(self, scope, inp, src_ctx, dst_ctx, idx):
        # TODO(@zhiics) move divice_copy to memory namespace
        copy = self.visit(op.tensor.device_copy(inp, src_ctx, dst_ctx))
        copy_out = scope.let("copy_out_{0}".format(idx), copy)
        self.context_analysis[copy] = dst_ctx
        self.context_analysis[copy_out] = dst_ctx
        return copy_out

    def current_scope(self):
        return self.scopes[-1]

    def shape_of(self, e):
        return op.shape_of(e, self.compute_dtype)

    def visit_tuple(self, tup):
        scope = self.current_scope()
        new_fields = []
        for field in tup.fields:
            field = self.visit(field)
            if isinstance(field, expr.Constant):
                field = scope.let('const', field)
            new_fields.append(field)
        return expr.Tuple(new_fields)

    def compute_alignment(self, dtype):
        dtype = DataType(dtype)
        align = (dtype.bits // 8) * dtype.lanes
        # MAGIC CONSTANT FROM device_api.h
        if align < 64:
            align = 64

        return expr.const(align, dtype="int64")

    def compute_storage_in_relay(self, shape, dtype):
        dtype = DataType(dtype)
        els = op.prod(shape)
        num = expr.const(dtype.bits * dtype.lanes, self.compute_dtype)
        num = num + expr.const(7, self.compute_dtype)
        div = expr.const(8, self.compute_dtype)
        return els * (num / div)

    def compute_storage(self, tensor_type):
        dtype = DataType(tensor_type.dtype)
        shape = [int(sh) for sh in tensor_type.shape]
        size = 1
        for sh in shape:
            size *= sh
        size *= (dtype.bits * dtype.lanes + 7) // 8
        return expr.const(size, dtype=self.compute_dtype)

    def make_static_allocation(self, scope, tensor_type, ctx, name_hint):
        """Allocate a tensor with a statically known shape."""
        shape = [int(sh) for sh in tensor_type.shape]
        if len(shape) == 0:
            shape = expr.const(np.array([]).astype(
                self.compute_dtype), dtype=self.compute_dtype)
        else:
            shape = expr.const(np.array(shape), dtype=self.compute_dtype)
        size = self.compute_storage(tensor_type)
        alignment = self.compute_alignment(tensor_type.dtype)
        dtype = tensor_type.dtype

        sto = scope.let("storage_{0}".format(name_hint), self.alloc_storage(
            size, alignment, ctx, dtype))
        # TODO(@jroesch): There is a bug with typing based on the constant shape.
        tensor = self.alloc_tensor(sto, shape, dtype, tensor_type.shape)
        return scope.let("tensor_{0}".format(name_hint), tensor)

    def visit_let(self, let):
        scope = ScopeBuilder()

        self.scopes.append(scope)
        while isinstance(let, expr.Let):
            new_val = self.visit(let.value)
            scope.let(let.var, new_val)
            let = let.body

        new_body = self.visit(let)
        scope.ret(new_body)
        self.scopes.pop()

        return scope.get()

    def emit_shape_func(self, scope, func, new_args):
        shape_func_ins = []
        engine = compile_engine.get()
        cfunc = engine.lower_shape_func(func, self.target_host)
        input_states = cfunc.shape_func_param_states

        is_inputs = []
        input_pos = 0
        cpu_ctx = nd.cpu(0)
        for i, (arg, state) in enumerate(zip(new_args, input_states)):
            state = int(state)
            # Pass Shapes
            if state == 2:
                for j, subexp in enumerate(from_tuple_type(arg.type_annotation, arg)):
                    ctx = self.get_context(subexp)
                    if ctx.device_type != cpu_ctx.device_type:
                        subexp = self.device_copy(scope, subexp, ctx, cpu_ctx, j)
                    let_in_arg = scope.let("in_arg_{0}".format(input_pos + j), subexp)
                    sh_of = self.visit(self.shape_of(let_in_arg))
                    shape_func_ins.append(
                        scope.let("in_shape_{0}".format(input_pos + j), sh_of))
                    input_pos += 1
                is_inputs.append(0)
            # Pass Inputs
            elif state == 1:
                new_arg = self.visit(arg)
                ctx = self.get_context(arg)
                if ctx.device_type != cpu_ctx.device_type:
                    new_arg = self.device_copy(scope, new_arg, ctx, cpu_ctx, i)
                shape_func_ins.append(
                    scope.let("in_shape_{0}".format(input_pos), new_arg))
                input_pos += 1
                is_inputs.append(1)
            else:
                # TODO(@jroesch): handle 3rd case
                raise Exception("unsupported shape function input state")

        out_shapes = []
        for i, out in enumerate(cfunc.outputs):
            tt = ty.TensorType(out.shape, out.dtype)
            alloc = self.make_static_allocation(scope, tt, cpu_ctx, i)
            self.context_analysis[alloc] = cpu_ctx
            alloc = scope.let("shape_func_out_{0}".format(i), alloc)
            self.context_analysis[alloc] = cpu_ctx
            out_shapes.append(alloc)

        shape_call = self.shape_func(
            func,
            expr.Tuple(shape_func_ins),
            expr.Tuple(out_shapes), is_inputs)

        self.context_analysis[shape_call] = cpu_ctx
        scope.let("shape_func", shape_call)

        copy_out_shapes = []
        for i, alloc in enumerate(out_shapes):
            if ctx.device_type != cpu_ctx.device_type:
                copy = self.device_copy(scope, alloc, cpu_ctx, ctx, i)
                copy_out_shapes.append(copy)

        return copy_out_shapes if copy_out_shapes else out_shapes

    def dynamic_invoke(self, scope, func, ins, new_args, out_types, ret_type):
        """Generate the code for invoking a TVM op with a dynamic shape."""
        out_shapes = self.emit_shape_func(scope, func, new_args)
        storages = []
        for i, (out_shape, out_type) in enumerate(zip(out_shapes, out_types)):
            size = self.compute_storage_in_relay(
                out_shape, out_type.dtype)
            alignment = self.compute_alignment(out_type.dtype)
            # Let CPU compute the shape func.
            context = nd.cpu(0)
            sto = scope.let("storage_{i}".format(i=i), self.alloc_storage(
                size, alignment, context, out_type.dtype))
            storages.append(sto)

        outs = []
        sh_ty_storage = zip(out_shapes, out_types, storages)
        for i, (out_shape, out_type, storage) in enumerate(sh_ty_storage):
            alloc = self.alloc_tensor(
                storage,
                out_shape,
                out_type.dtype,
                out_type.shape)
            alloc = scope.let("out_{i}".format(i=i), alloc)
            outs.append(alloc)

        tuple_outs = expr.Tuple(outs)
        invoke = self.invoke_tvm(func, ins, tuple_outs)
        scope.let("", invoke)
        return to_tuple_type(ret_type, tuple_outs.fields)

    def emit_reshape_tensor(self, scope, func, new_args, ret_type):
        if self.is_dynamic(ret_type):
            out_shapes = self.emit_shape_func(scope, func, new_args)
            shape_expr = out_shapes[0]
        else:
            # constant output shape
            shape = [int(dim) for dim in ret_type.shape]
            shape_expr = expr.const(np.array(shape), dtype=self.compute_dtype)
        return self.reshape_tensor(new_args[0], shape_expr, ret_type.shape)

    def is_dynamic(self, ret_type):
        is_dynamic = ty.type_has_any(ret_type)
        # TODO(@jroesch): restore this code, more complex then it seems
        # for arg in call.args:
        #     is_dynamic = is_dynamic or arg.checked_type.is_dynamic()
        return is_dynamic

    def visit_call(self, call):
        if is_primitive(call):
            # Because we are in ANF we do not need to visit the arguments.
            scope = self.current_scope()
            new_args = [self.visit(arg) for arg in call.args]
            ins = expr.Tuple(new_args)
            ret_type = call.checked_type
            out_types = flatten_tuple_type(ret_type)

            if is_reshape(call.op):
                # Handle op with only reshape
                return self.emit_reshape_tensor(scope, call.op, new_args, ret_type)
            if self.is_dynamic(ret_type):
                # Handle dynamic case.
                return self.dynamic_invoke(scope, call.op, ins, new_args, out_types, ret_type)
            else:
                # Handle static case.
                outs = []
                for i, out_ty in enumerate(out_types):
                    ctx = self.get_context(call)
                    out = self.make_static_allocation(scope, out_ty, ctx, i)
                    outs.append(out)

                output = expr.Tuple(outs)
                invoke = self.invoke_tvm(call.op, ins, output)
                scope.let("", invoke)
                return to_tuple_type(ret_type, output.fields)
        else:
            return super().visit_call(call)


@transform.function_pass(opt_level=0)
class ManifestAlloc:
    """The explicit pass wrapper around ManifestAlloc."""
    def __init__(self, target_host, targets):
        self.target_host = target_host
        self.targets = targets

    def transform_function(self, func, mod, _):
        # TODO(@jroesch): Is there a way to do one shot initilization?
        # can we have def pass_init?
        mod.import_from_std("core.rly")

        assert isinstance(self.targets, (dict, container.Map))
        if len(self.targets) > 1:
            pass_ctx = PassContext.current()
            fallback_ctx = nd.context(pass_ctx.fallback_device)
            ca = ContextAnalysis(fallback_ctx)
        else:
            dev, _ = self.targets.items()[0]
            ca = ContextAnalysis(nd.context(dev.value))

        # We use logger here to help debug.
        logging.debug("-----BEFORE ANALYSIS-----")
        logging.debug(func.astext(False))
        ca.visit(func)
        logging.debug("-----AFTER ANALYSIS-----")
        logging.debug(func.astext(show_meta_data=False,
                                  annotate=mk_analysis_annotator(ca.results())))
        ea = ManifestAllocPass(self.target_host, ca.results())
        func = ea.visit(func)
        return func


register_func("relay.transform.ManifestAlloc", ManifestAlloc)
