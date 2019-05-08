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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs

import json
import tvm
from .. import ir_pass
from .. import expr as _expr
from .. import op as _op
from .. import ty as _ty
from .. import scope_builder as _scope_builder
from .. import module as _module
from ... import nd as _nd

from .common import StrAttrsDict
from .nnvm_common import _rename, _binop_scalar, _rbinop_scalar, _reduce
from .nnvm_common import _arg_reduce, _init_op, _softmax_op, _cast
from .nnvm_common import _clip, _transpose, _upsampling
from .nnvm_common import _elemwise_sum, _reshape
from .nnvm_common import _warn_not_used

__all__ = ['from_mxnet']

def _mx_fully_connected(inputs, attrs, mod):
    import mxnet as mx
    units = attrs.get_int("num_hidden")
    use_bias = not attrs.get_bool("no_bias", False)
    try:
        _ = mx.sym.FullyConnected(mx.sym.var("x"), num_hidden=1, flatten=True)
        has_flatten = True
    except mx.base.MXNetError:
        # no flatten attribute in old mxnet
        has_flatten = False
    use_flatten = attrs.get_bool("flatten", True)
    if has_flatten and use_flatten:
        inputs[0] = _op.nn.batch_flatten(inputs[0])
    res = _op.nn.dense(inputs[0], inputs[1], units=units)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=-1)
    return res


def _get_channel_axis(layout, op_name):
    if layout == "NCHW":
        return 1
    if layout == "NHWC":
        return 3
    raise tvm.error.OpAttributeInvalid(
        'Value {} in attribute "layout" of operator {} is not valid.'.format(layout, op_name))


def _mx_activations(inputs, attrs, mod):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "sigmoid":
        return _op.sigmoid(inputs[0])
    if act_type == "tanh":
        return _op.tanh(inputs[0])
    if act_type == "relu":
        return _op.nn.relu(inputs[0])
    if act_type == "softrelu":
        def _stable_softrelu(x):
            # log(1 + exp(-abs(x))) + relu(x)
            one = _expr.const(1, dtype="float32")
            exp_neg_abs_x = _op.exp(_op.negative(_op.abs(x)))
            return _op.add(_op.log(_op.add(one, exp_neg_abs_x)),
                           _op.nn.relu(x))
        return _stable_softrelu(inputs[0])
    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported for frontend MXNet.'.format(act_type))


def _mx_compare(new_op, wrapper):
    def impl(inputs, attrs, mod):
        dtype = ir_pass.infer_type(inputs[0], mod=mod).checked_type.dtype
        return wrapper(new_op, mod)(inputs, attrs).astype(dtype)
    return impl


def _mx_conv2d(inputs, attrs, mod):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid(
            'Non-2D kernels are not supported for operator Conv2D.')
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_conv2d_transpose(inputs, attrs, mod):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "target_shape" is not supported for operator Conv2D-transpose.')
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid(
            'Non-2D kernels are not supported for operator Conv2D-transpose.')
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d_transpose")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["output_padding"] = attrs.get_int_tuple("adj", (0, 0))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d_transpose(inputs[0], inputs[1], **new_attrs)

    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_pooling(inputs, attrs, mod):
    global_pool = attrs.get_bool("global_pool", False)
    pool_type = attrs.get_str("pool_type")

    def _pool2d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 2:
            raise tvm.error.OpAttributeInvalid(
                'Only 2D kernels are supported for operator Pool2D.')
        new_attrs = {}
        new_attrs["pool_size"] = kernel_size
        new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = (attrs.get_str("pooling_convention", "valid") == "full")
        if is_avg:
            new_attrs["count_include_pad"] = attrs.get_bool("count_include_pad", True)
        return new_op(inputs[0], **new_attrs)

    if pool_type == "max":
        if global_pool:
            return _op.nn.global_max_pool2d(inputs[0])
        return _pool2d(_op.nn.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            return _op.nn.global_avg_pool2d(inputs[0])
        return _pool2d(_op.nn.avg_pool2d, True)
    raise tvm.error.OpNotImplemented(
        'Operator {} Pooling is not supported for frontend MXNet.'.format(pool_type.capitalize()))


def _mx_adaptive_pooling(inputs, attrs, mod):
    output_size = attrs.get_int_tuple("output_size", [])
    if output_size != (1,):
        raise RuntimeError("AdaptiveAvgPooling with output_size other than 1 is not supported yet.")
    return _op.nn.global_avg_pool2d(inputs[0])


def _mx_dropout(inputs, attrs, mod):
    rate = attrs.get_float("p", 0.5)
    return _op.nn.dropout(inputs[0], rate=rate)


def _mx_BlockGrad(inputs, attrs, mod): #pylint: disable=
    return inputs


def _mx_batch_norm(inputs, attrs, mod):
    if attrs.get_bool("output_mean_var", False):
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "output_mean_var" is not supported for operator Batch Norm.')
    if attrs.get_bool("use_global_stats", False):
        _warn_not_used("use_global_stats", "batch_norm")
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", 1)
    new_attrs["epsilon"] = attrs.get_float("eps", 0.001)
    new_attrs["center"] = True
    new_attrs["scale"] = not attrs.get_bool("fix_gamma", False)
    return _op.nn.batch_norm(*inputs, **new_attrs)


def _mx_slice(inputs, attrs, mod):
    new_attrs = {}
    begin = attrs.get_int_tuple('begin', None)
    end = attrs.get_int_tuple('end', None)
    stride = attrs.get_int_tuple('step', None)
    if begin is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "begin" not found in operator Slice.')
    if end is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "end" not found in operator Slice.')
    if None in begin:
        raise tvm.error.OpAttributeInvalid(
            'Value None in attribute "begin" of operator Slice is not valid.')
    if None in end:
        raise tvm.error.OpAttributeInvalid(
            'Value None in attribute "end" of operator Slice is not valid.')
    new_attrs = {'begin': begin, 'end': end}
    if stride is not None:
        new_attrs['strides'] = stride
    return _op.strided_slice(inputs[0], **new_attrs)


def _mx_slice_like(inputs, attrs, mod):
    assert len(inputs) == 2
    new_attrs = {}
    new_attrs["axes"] = attrs.get_int_tuple("axes", None)
    return _op.slice_like(*inputs, **new_attrs)


def _mx_slice_axis(inputs, attrs, mod):
    assert len(inputs) == 1
    shape = ir_pass.infer_type(inputs[0]).checked_type.shape
    axis = attrs.get_int("axis")
    ax_beg = attrs.get_int("begin")
    ax_end = attrs.get_str("end")
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape)
    if ax_end == "None":
        ax_end = int(shape[axis])
    else:
        ax_end = int(ax_end)
    if ax_beg < 0:
        ax_beg += int(shape[axis])
    if ax_end < 0:
        ax_end += int(shape[axis])
    assert 0 <= ax_beg < int(shape[axis])
    assert ax_beg < ax_end <= int(shape[axis])
    begin = []
    end = []
    for i, dim in enumerate(shape):
        if i != axis:
            begin.append(0)
            end.append(dim)
        else:
            begin.append(ax_beg)
            end.append(ax_end)
    return _op.strided_slice(inputs[0], begin, end)


def _mx_split(inputs, attrs, mod):
    axis = attrs.get_int("axis", 1)
    new_attrs = {}
    new_attrs["indices_or_sections"] = attrs.get_int("num_outputs")
    new_attrs["axis"] = axis
    res = _op.split(inputs[0], **new_attrs)
    if attrs.get_bool("squeeze_axis", False):
        return tuple([_op.squeeze(x, axis=[axis]) for x in res])
    return res


def _mx_softmax_activation(inputs, attrs, mod):
    mode = attrs.get_str("mode", "instance")
    axis = 0 if mode == "instance" else 1
    return _op.nn.softmax(inputs[0], axis=axis)


def _mx_softmax_output(inputs, attrs, mod):
    if attrs.get_bool("multi_output", False):
        return _op.nn.softmax(inputs[0], axis=1)
    return _op.nn.softmax(inputs[0])


def _mx_concat(inputs, attrs, mod):
    axis = attrs.get_int("dim", 1)
    return _op.concatenate(tuple(inputs), axis=axis)


def _mx_stack(inputs, attrs, mod):
    axis = attrs.get_int("axis", 0)
    return _op.stack(tuple(inputs), axis=axis)


def _mx_expand_dims(inputs, attrs, mod):
    axis = attrs.get_int("axis")
    return _op.expand_dims(inputs[0], axis=axis)


def _mx_leaky_relu(inputs, attrs, mod):
    act_type = attrs.get_str("act_type")
    if act_type == "leaky":
        return _op.nn.leaky_relu(inputs[0], alpha=attrs.get_float("slope", 0.25))
    if act_type == "prelu":
        assert len(inputs) == 2
        return _op.nn.prelu(*inputs)
    if act_type == "elu":
        # -slope * relu(1-exp(x)) + relu(x)
        slope = attrs.get_float("slope", 0.25)
        one = _expr.const(1, dtype="float32")
        x = inputs[0]
        mslope = _op.nn.relu(_op.subtract(one, _op.exp(x)))
        mslope = _op.multiply(mslope, _expr.const(-slope, dtype="float32"))
        return _op.add(mslope, _op.nn.relu(x))
    if act_type == "rrelu":
        # NOTE this is only converted for inference.
        lower_bound = attrs.get_float("lower_bound")
        upper_bound = attrs.get_float("upper_bound")
        alpha = (lower_bound + upper_bound) / 2.0
        return _op.nn.leaky_relu(inputs[0], alpha=alpha)
    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported for frontend MXNet.'.format(act_type))


def _mx_make_power(power):
    def _impl(inputs, attrs, mod):  # Note: no attrs
        assert len(inputs) == 1
        dtype = ir_pass.infer_type(inputs[0], mod=mod).checked_type.dtype
        scalar = _expr.const(power, dtype=dtype)
        # Note: int maps to "int32", float maps to "float32"
        return _op.power(inputs[0], scalar)
    return _impl


def _mx_make_exponent(base):
    # exp(b, x) = e^b * e^x
    def _impl(inputs, attrs, mod):  # Note: no attrs
        assert len(inputs) == 1
        scalar = _op.exp(_expr.const(base, dtype="float32"))
        return _op.multiply(inputs[0], scalar)
    return _impl


def _mx_make_logarithm(base):
    # log(b, x) = log(x) / log(b)
    def _impl(inputs, attrs, mod):  # Note: no attrs
        assert len(inputs) == 1
        scalar = _op.log(_expr.const(base, dtype="float32"))
        return _op.divide(inputs[0], scalar)
    return _impl


def _mx_expm1():
    # exp_minus_1 x = exp(x) - 1
    def _impl(inputs, attrs, mod):  # Note: no attrs
        assert len(inputs) == 1
        one = _expr.const(1, dtype="float32")
        return _op.log(_op.subtract(inputs[0], one))
    return _impl


def _mx_log1p():
    # 1_plus_log x = log(x + 1)
    def _impl(inputs, attrs, mod):  # Note: no attrs
        assert len(inputs) == 1
        one = _expr.const(1, dtype="float32")
        return _op.log(_op.add(inputs[0], one))
    return _impl


def _mx_lrn(inputs, attrs, mod):
    new_attrs = {}
    new_attrs["alpha"] = attrs.get_float("alpha", 0.0001)
    new_attrs["beta"] = attrs.get_float("beta", 0.75)
    new_attrs["bias"] = attrs.get_float("knorm", 2)
    # NCHW format and normalization along channel axis
    new_attrs["axis"] = 1
    new_attrs["size"] = attrs.get_int("nsize")
    assert len(inputs) == 1
    return _op.nn.lrn(inputs[0], **new_attrs)


def _mx_multibox_prior(inputs, attrs, mod):
    new_attrs = {}
    new_attrs["sizes"] = attrs.get_float_tuple("sizes", (1.0, ))
    new_attrs["steps"] = attrs.get_float_tuple("steps", (-1.0, -1.0))
    new_attrs["offsets"] = attrs.get_float_tuple("offsets", (0.5, 0.5))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (1.0, ))
    new_attrs["clip"] = attrs.get_bool("clip", False)
    return _op.vision.multibox_prior(inputs[0], **new_attrs)


def _mx_multibox_detection(inputs, attrs, mod):
    new_attrs0 = {}
    new_attrs0["clip"] = attrs.get_bool("clip", True)
    new_attrs0["threshold"] = attrs.get_float("threshold", 0.01)
    new_attrs0["variances"] = attrs.get_float_tuple("variances", (0.1, 0.1,
                                                                  0.2, 0.2))

    new_attrs1 = {}
    new_attrs1["return_indices"] = False
    new_attrs1["iou_threshold"] = attrs.get_float("nms_threshold", 0.5)
    new_attrs1["force_suppress"] = attrs.get_bool("force_suppress", False)
    new_attrs1["top_k"] = attrs.get_int("nms_topk", -1)

    ret = _op.vision.multibox_transform_loc(inputs[0], inputs[1],
                                            inputs[2], **new_attrs0)
    return _op.vision.non_max_suppression(ret[0], ret[1], **new_attrs1)


def _mx_batch_dot(inputs, attrs, mod):
    assert len(inputs) == 2
    a, b = inputs
    transpose_a = attrs.get_bool("transpose_a", False)
    transpose_b = attrs.get_bool("transpose_b", False)
    if transpose_a is True:
        msg = 'Value {} in attribute "transpose_a" of operator batch_dot ' \
              'is not valid.'
        raise tvm.error.OpAttributeInvalid(msg.format(transpose_a))
    if transpose_b is False:
        b = _op.transpose(b, axes=[0, 2, 1])
    return _op.nn.batch_matmul(a, b)


def _mx_arange(inputs, attrs, mod):
    assert len(inputs) == 0
    if attrs.get_int("repeat", 1) != 1:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "repeat" is not supported in operator arange.')
    dtype = attrs.get_str("dtype", "float32")
    new_attrs = {}
    new_attrs["start"] = _op.const(attrs.get_float("start", 0), dtype)
    new_attrs["stop"] = _op.const(attrs.get_float("stop"), dtype)
    new_attrs["step"] = _op.const(attrs.get_float("step", 1), dtype)
    new_attrs["dtype"] = dtype

    return _op.arange(**new_attrs)


def _mx_contrib_sarange(inputs, attrs, mod):
    dtype = "float32"
    stop = _op.take(inputs[0], _op.const(0, 'int32'))
    return _op.arange(stop, dtype=dtype)


def _mx_repeat(inputs, attrs, mod):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["repeats"] = attrs.get_int("repeats")
    new_attrs["axis"] = attrs.get_int("axis", 0)
    return _op.repeat(inputs[0], **new_attrs)


def _mx_tile(inputs, attrs, mod):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["reps"] = attrs.get_int_tuple("reps")
    return _op.tile(inputs[0], **new_attrs)


def _mx_take(inputs, attrs, mod):
    assert len(inputs) == 2
    mode = attrs.get_str("mode", "clip")
    if mode == "raise":
        raise RuntimeError("take doesn't support raise mode")
    axis = attrs.get_int("axis", 0)
    return _op.take(inputs[0], inputs[1].astype("int32"), axis, mode)


def _mx_reverse(inputs, attrs, mod):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis")
    return _op.reverse(inputs[0], **new_attrs)


def _mx_roi_align(inputs, attrs, mod):
    new_attrs = {}
    new_attrs["pooled_size"] = attrs.get_int_tuple("pooled_size")
    new_attrs["spatial_scale"] = attrs.get_float("spatial_scale")
    new_attrs["sample_ratio"] = attrs.get_int("sample_ratio", -1)
    new_attrs["layout"] = "NCHW"
    return _op.vision.roi_align(inputs[0], inputs[1], **new_attrs)


def _mx_resize(inputs, attrs, mod):
    scale_height = attrs.get_float("scale_height", None)
    scale_width = attrs.get_float("scale_width", None)
    height = attrs.get_int("height", 1)
    width = attrs.get_int("width", 1)
    shape = ir_pass.infer_type(inputs[0]).checked_type.shape
    if scale_height is not None:
        height = (scale_height * shape[2]).astype("int32")
    if scale_width is not None:
        width = (scale_width * shape[3]).astype("int32")
    size = (height, width)
    return _op.image.resize(inputs[0], size, align_corners=True)


def _mx_roi_pooling(inputs, attrs, mod):
    new_attrs = {}
    new_attrs["pooled_size"] = attrs.get_int_tuple("pooled_size")
    new_attrs["spatial_scale"] = attrs.get_float("spatial_scale")
    new_attrs["layout"] = "NCHW"
    return _op.vision.roi_pool(inputs[0], inputs[1], **new_attrs)


def _mx_proposal(inputs, attrs, mod):
    new_attrs = {}
    new_attrs["scales"] = attrs.get_float_tuple("scales", (4.0, 8.0, 16.0, 32.0))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (0.5, 1.0, 2.0))
    new_attrs["feature_stride"] = attrs.get_int("feature_stride", 16)
    new_attrs["threshold"] = attrs.get_float("threshold", 0.7)
    new_attrs["rpn_pre_nms_top_n"] = attrs.get_int("rpn_pre_nms_top_n", 6000)
    new_attrs["rpn_post_nms_top_n"] = attrs.get_int("rpn_post_nms_top_n", 300)
    new_attrs["rpn_min_size"] = attrs.get_int("rpn_min_size", 16)
    new_attrs["iou_loss"] = attrs.get_bool("iou_loss", False)
    assert not attrs.get_bool("output_score", False), "proposal doesn't support output score"
    return _op.vision.proposal(inputs[0], inputs[1], inputs[2], **new_attrs)


def _mx_box_nms(inputs, attrs, mod):
    force_suppress = attrs.get_bool("force_suppress", False)
    iou_thresh = attrs.get_float('overlap_thresh', 0.5)
    top_k = attrs.get_int('topk', -1)
    valid_thresh = attrs.get_float('valid_thresh', 0)
    coord_start = attrs.get_int('coord_start', 2)
    score_index = attrs.get_int('score_index', 1)
    id_index = attrs.get_int('id_index', -1)
    in_format = attrs.get_str('in_format', 'corner')
    out_format = attrs.get_str('out_format', 'corner')
    if in_format != 'corner':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "in_format" must equal "corner" for operator box_nms.')
    if out_format != 'corner':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "out_format" must equal "corner" for operator box_nms.')

    ret = _op.vision.get_valid_counts(inputs[0], score_threshold=valid_thresh)
    nms_out = _op.vision.non_max_suppression(ret[1],
                                             ret[0],
                                             iou_threshold=iou_thresh,
                                             force_suppress=force_suppress,
                                             top_k=top_k,
                                             coord_start=coord_start,
                                             score_index=score_index,
                                             id_index=id_index,
                                             return_indices=False,
                                             invalid_to_bottom=True)
    return nms_out


def _mx_l2_normalize(inputs, attrs, mod):
    new_attrs = {}
    mode = attrs.get_str('mode', 'instance')
    if mode != 'channel':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "mode" must equal "channel" for operator l2_normalize.')
    new_attrs['eps'] = attrs.get_float('eps', 1e-10)
    new_attrs['axis'] = [1]
    return _op.nn.l2_normalize(inputs[0], **new_attrs)


def _mx_shape_array(inputs, attrs, mod):
    assert len(inputs) == 1
    if attrs.get_int("lhs_begin", None) is not None:
        raise RuntimeError("shape_array doesn't support lhs_begin")
    if attrs.get_int("lhs_end", None) is not None:
        raise RuntimeError("shape_array doesn't support lhs_end")
    if attrs.get_int("rhs_begin", None) is not None:
        raise RuntimeError("shape_array doesn't support rhs_begin")
    if attrs.get_int("rhs_end", None) is not None:
        raise RuntimeError("shape_array doesn't support rhs_end")
    return _op.shape_of(inputs[0], dtype='int64')


def _mx_full(inputs, attrs, mod):
    assert len(inputs) == 0
    val = attrs.get_float("value")
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_str("dtype", "float32")
    return _op.full(_expr.const(val, dtype), shape, dtype)


def _mx_squeeze(inputs, attrs, mod):
    assert len(inputs) == 1
    axis = attrs.get_int_tuple("axis", None)
    return _op.squeeze(inputs[0], axis)


def _mx_broadcast_axis(inputs, attrs, mod):
    assert len(inputs) == 1
    axis = attrs.get_int_tuple("axis", [])
    size = attrs.get_int_tuple("size", [])
    assert len(axis) == len(size)
    if len(axis) == 0:
        return inputs[0]
    src_shape = ir_pass.infer_type(inputs[0])._checked_type_.shape
    tgt_shape = []
    for i, dim in enumerate(src_shape):
        if i not in axis:
            tgt_shape.append(dim)
        else:
            assert int(dim) == 1
            idx = axis.index(i)
            tgt_shape.append(size[idx])
    return _op.broadcast_to(inputs[0], tgt_shape)


def _mx_embedding(inputs, attrs, mod):
    assert len(inputs) == 2
    indices, weight = inputs
    return _op.take(weight, indices.astype('int32'), axis=0)


def _mx_smooth_l1(inputs, attrs, mod):
    scalar = attrs.get_float("scalar", 1.0)
    scalar_sq = scalar * scalar
    mask = _op.less(inputs[0], _expr.const(1.0 / scalar_sq, dtype='float32'))
    return _op.where(mask,
                     _expr.const(scalar_sq / 2.0, dtype='float32') * inputs[0] * inputs[0],
                     _op.abs(inputs[0]) - _expr.const(0.5 / scalar_sq))


def _mx_deformable_convolution(inputs, attrs, mod):
    new_attrs = {}
    assert attrs.get_bool("no_bias")
    new_attrs["kernel_size"] = attrs.get_int_tuple("kernel")
    new_attrs["strides"] = attrs.get_int_tuple("stride")
    new_attrs["padding"] = attrs.get_int_tuple("pad")
    new_attrs["dilation"] = attrs.get_int_tuple("dilate")
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["deformable_groups"] = attrs.get_int("num_deformable_group", 1)
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    assert attrs.get_str("layout", "NCHW") == "NCHW", "Deformable conv2d only supports NCHW layout"
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.deformable_conv2d(inputs[0], inputs[1], inputs[2], **new_attrs)
    if use_bias:
        assert len(inputs) == 4
        res = _op.nn.bias_add(res, inputs[3])
    return res


def _mx_argsort(inputs, attrs, mod):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["is_ascend"] = attrs.get_bool("is_ascend", True)
    new_attrs["dtype"] = attrs.get_str("dtype", "float32")
    return _op.vision.argsort(inputs[0], **new_attrs)


def _mx_contrib_div_sqrt_dim(inputs, attrs, mod):
    assert len(inputs) == 1
    ndim = len(ir_pass.infer_type(inputs[0])._checked_type_.shape)
    dim = _op.take(_op.shape_of(inputs[0]), _expr.const(ndim-1, dtype="int32"))
    sqrt_dim = _op.sqrt(dim.astype('float32'))
    out = inputs[0] / sqrt_dim
    return out


def _mx_cond(inputs, attrs, subgraphs, mod):
    assert len(subgraphs) == 3
    cond_input_locs = json.loads(attrs.get_str("cond_input_locs"))
    then_input_locs = json.loads(attrs.get_str("then_input_locs"))
    else_input_locs = json.loads(attrs.get_str("else_input_locs"))
    num_outputs = attrs.get_int("num_outputs")

    input_args = []
    for i, arg in enumerate(inputs):
        var = _expr.var("arg%s" % i, ir_pass.infer_type(arg, mod=mod).checked_type)
        input_args.append(var)
    cond_args = [input_args[i] for i in cond_input_locs]
    then_args = [input_args[i] for i in then_input_locs]
    else_args = [input_args[i] for i in else_input_locs]

    cond_arg_shapes = [arg.type_annotation.shape for arg in cond_args]
    cond_arg_dtype_info = [arg.type_annotation.dtype for arg in cond_args]
    cond_func = _from_mxnet_impl(mod, subgraphs[0], cond_arg_shapes, cond_arg_dtype_info)
    cond = _expr.Call(cond_func, cond_args).astype("bool")

    sb = _scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        then_arg_shapes = [arg.type_annotation.shape for arg in then_args]
        then_arg_dtype_info = [arg.type_annotation.dtype for arg in then_args]
        then_func = _from_mxnet_impl(mod, subgraphs[1], then_arg_shapes, then_arg_dtype_info)
        sb.ret(_expr.Call(then_func, then_args))
    with sb.else_scope():
        else_arg_shapes = [arg.type_annotation.shape for arg in else_args]
        else_arg_dtype_info = [arg.type_annotation.dtype for arg in else_args]
        else_func = _from_mxnet_impl(mod, subgraphs[2], else_arg_shapes, else_arg_dtype_info)
        sb.ret(_expr.Call(else_func, else_args))
    func = _expr.Function(input_args, sb.get())
    ret = _expr.Call(func, inputs)
    if num_outputs > 1:
        ret = _expr.TupleWrapper(ret, num_outputs)
    return ret


def _mx_foreach(inputs, attrs, subgraphs, mod):
    from tvm.relay.prelude import Prelude
    p = Prelude(mod)
    nil = p.nil
    cons = p.cons
    l = p.l

    assert len(subgraphs) == 1
    in_data_locs = json.loads(attrs.get_str('in_data_locs'))
    in_state_locs = json.loads(attrs.get_str('in_state_locs'))
    remain_locs = json.loads(attrs.get_str('remain_locs'))
    num_data = len(in_data_locs)
    num_states = len(in_state_locs)
    num_outputs = len(subgraphs[0]["heads"]) - num_states

    input_args = []
    for i, arg in enumerate(inputs):
        var = _expr.var("arg%s" % i, ir_pass.infer_type(arg, mod=mod).checked_type)
        input_args.append(var)

    data = input_args[:num_data]
    prev_states = input_args[num_data:num_data+num_states]
    params = input_args[num_data+num_states:]
    num_iter = _expr.var("num_iter", dtype='int32', shape=())
    loop_iter = _expr.var("i", dtype='int32', shape=())
    all_outs = _expr.var("all_outs")

    loop = _expr.GlobalVar("foreach")
    body_sb = _scope_builder.ScopeBuilder()
    with body_sb.if_scope(_op.equal(loop_iter, num_iter)):
        body_sb.ret(_expr.Tuple([all_outs] + prev_states))
    with body_sb.else_scope():
        body_args = [None] * len(inputs)
        for k, v in enumerate(in_data_locs):
            assert body_args[v] is None
            body_args[v] = _op.take(data[k], loop_iter, 0)
        for k, v in enumerate(in_state_locs):
            assert body_args[v] is None
            body_args[v] = prev_states[k]
        for k, v in enumerate(remain_locs):
            assert body_args[v] is None
            body_args[v] = params[k]
        body_arg_shapes = []
        body_arg_dtype_info = []
        for arg in body_args:
            ty = ir_pass.infer_type(arg).checked_type
            body_arg_shapes.append(ty.shape)
            body_arg_dtype_info.append(ty.dtype)
        loop_body = _from_mxnet_impl(mod, subgraphs[0], body_arg_shapes, body_arg_dtype_info)
        loop_body_ret = _expr.Call(loop_body, body_args)

        if num_outputs == 1:
            out = _expr.TupleGetItem(loop_body_ret, 0)
        else:
            out = _expr.Tuple([_expr.TupleGetItem(loop_body_ret, i) for i in range(num_outputs)])
        states = [_expr.TupleGetItem(loop_body_ret, num_outputs+i) for i in range(num_states)]
        new_all_outs = cons(out, all_outs)
        recur_ret = _expr.Call(loop, data + states + params + [num_iter, loop_iter + _expr.const(1), new_all_outs])
        body_sb.ret(recur_ret)

    body = body_sb.get()
    foreach_args = input_args + [num_iter, loop_iter, all_outs]
    func = _expr.Function(foreach_args, body)
    mod[loop] = func

    data0_shape = _op.shape_of(inputs[0])
    num_iter = _op.take(data0_shape, _expr.const(0))
    ret = _expr.Call(loop, inputs + [num_iter, _expr.const(0), nil()])
    # Currently return the all_outs in reverse order because foldl and rev fail
    # to compile in the vm
    # all_outs = p.rev(_expr.TupleGetItem(ret, 0))
    # states = _expr.TupleGetItem(ret, 1)
    # return _expr.TupleWrapper(_expr.Tuple([all_outs, states]), 2)
    return _expr.TupleWrapper(ret, num_states+1)


def _mx_while_loop(inputs, attrs, subgraphs, mod):
    from tvm.relay.prelude import Prelude
    p = Prelude(mod)
    nil = p.nil
    cons = p.cons
    l = p.l

    assert len(subgraphs) == 2
    input_args = []
    for i, arg in enumerate(inputs):
        var = _expr.var("arg%s" % i, ir_pass.infer_type(arg, mod=mod).checked_type)
        input_args.append(var)

    cond_input_locs = attrs.get_int_tuple("cond_input_locs")
    func_input_locs = attrs.get_int_tuple("func_input_locs")
    # indices of state vars in the func_input_locs
    func_var_locs = attrs.get_int_tuple("func_var_locs")
    num_out_data = attrs.get_int("num_out_data")
    num_outputs = attrs.get_int("num_outputs")

    if num_out_data == 0:
        all_outs = None
    else:
        all_outs = _expr.var("all_outs")
    while_loop = _expr.GlobalVar("while_loop")
    prev_states = [input_args[func_input_locs[j]] for j in func_var_locs]

    cond_args = [input_args[j] for j in cond_input_locs]
    cond_arg_shapes = [arg.type_annotation.shape for arg in cond_args]
    cond_arg_dtype_info = [arg.type_annotation.dtype for arg in cond_args]
    cond_body = _from_mxnet_impl(mod, subgraphs[0], cond_arg_shapes, cond_arg_dtype_info)
    cond_ret = _expr.Call(cond_body, cond_args)
    cond = _op.take(cond_ret, _expr.const(0)).astype("bool")

    sb = _scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        func_args = [input_args[j] for j in func_input_locs]
        func_arg_shapes = [arg.type_annotation.shape for arg in func_args]
        func_arg_dtype_info = [arg.type_annotation.dtype for arg in func_args]
        func = _from_mxnet_impl(mod, subgraphs[1], func_arg_shapes, func_arg_dtype_info)
        func_ret = _expr.Call(func, func_args)
        if num_out_data == 0:
            out = None
        elif num_out_data == 1:
            out = _expr.TupleGetItem(func_ret, 0)
        else:
            out = _expr.Tuple([_expr.TupleGetItem(func_ret, j) for j in range(num_out_data)])
        new_all_outs = None if out is None else cons(out, all_outs)
        states = [_expr.TupleGetItem(func_ret, j) for j in range(num_out_data, num_outputs)]
        recur_args = input_args[:]
        for i, func_idx in enumerate(func_var_locs):
            recur_args[func_input_locs[func_idx]] = states[i]
        if num_out_data == 0:
            recur_ret = _expr.Call(while_loop, recur_args)
        else:
            recur_ret = _expr.Call(while_loop, recur_args + [new_all_outs])
        sb.ret(recur_ret)
    with sb.else_scope():
        if num_out_data == 0:
            sb.ret(_expr.Tuple(prev_states))
        else:
            sb.ret(_expr.Tuple([all_outs] + prev_states))

    body = sb.get()
    if num_out_data == 0:
        while_args = input_args
    else:
        while_args = input_args + [all_outs]
    func = _expr.Function(while_args, body)
    mod[while_loop] = func
    if num_out_data == 0:
        ret = _expr.Call(while_loop, inputs)
    else:
        ret = _expr.Call(while_loop, inputs + [nil()])
    return _expr.TupleWrapper(ret, num_outputs)


def _mx_layer_norm(inputs, attrs, mod):
    assert len(inputs) == 3
    if attrs.get_bool("output_mean_var", False):
        raise RuntimeError("layer_norm does not support output_mean_var")
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["epsilon"] = attrs.get_float("eps", 1e-5)
    return _op.nn.layer_norm(*inputs, **new_attrs)


def _mx_sequence_mask(inputs, attrs, mod):
    return inputs[0]


def _mx_topk(inputs, attrs, mod):
    assert len(inputs) == 1
    axis = attrs.get_int("axis", -1)
    ret_type = attrs.get_str("ret_typ", "indices")
    k = attrs.get_int("k", 1)
    is_ascend = attrs.get_bool("is_ascend", False)
    dtype = attrs.get_str("dtype", "float32")

    if ret_type == "mask":
        raise RuntimeError("topk does not support ret_typ as mask")

    argsort_attrs = {}
    argsort_attrs["axis"] = axis
    argsort_attrs["is_ascend"] = is_ascend
    argsort_attrs["dtype"] = dtype if ret_type == "indices" else "int32"

    indices = _op.vision.argsort(inputs[0], **argsort_attrs)
    if k >= 1:
        in_shape = ir_pass.infer_type(inputs[0]).checked_type.shape
        begin = [0] * len(in_shape)
        end = []
        for dim in in_shape:
            end.append(int(dim))
        end[axis] = k
        indices = _op.strided_slice(indices, begin, end)
    if ret_type == "indices":
        return indices
    # return indices
    # print(ir_pass.infer_type(indices))
    # print(axis)
    # values = _op.take(inputs[0], indices, axis=axis)
    # if ret_type == "value":
    #     return values
    # ret_type == "both"
    return _expr.TupleWrapper(_expr.Tuple([indices.astype(dtype), indices.astype(dtype)]), 2)


# Note: due to attribute conversion constraint
# ops in the identity set must be attribute free
_identity_list = [
    "log",
    "exp",
    "sqrt",
    "floor",
    "ceil",
    "sigmoid",
    "tanh",
    "negative",
    "reshape_like",
    "zeros_like",
    "ones_like",
    "where",
    "gather_nd",
]

_convert_map = {
    "_copy"                  : _rename(_op.copy),
    "relu"                   : _rename(_op.nn.relu),
    "broadcast_add"          : _rename(_op.add),
    "broadcast_sub"          : _rename(_op.subtract),
    "broadcast_mul"          : _rename(_op.multiply),
    "broadcast_div"          : _rename(_op.divide),
    "broadcast_mod"          : _rename(_op.mod),
    "broadcast_maximum"      : _rename(_op.maximum),
    "broadcast_minimum"      : _rename(_op.minimum),
    "broadcast_equal"        : _mx_compare(_op.equal, _rename),
    "broadcast_not_equal"    : _mx_compare(_op.not_equal, _rename),
    "broadcast_greater"      : _mx_compare(_op.greater, _rename),
    "broadcast_greater_equal": _mx_compare(_op.greater_equal, _rename),
    "broadcast_lesser"       : _mx_compare(_op.less, _rename),
    "broadcast_lesser_equal" : _mx_compare(_op.less_equal, _rename),
    "_lesser"                : _mx_compare(_op.less, _rename),
    "elemwise_add"           : _rename(_op.add),
    "elemwise_sub"           : _rename(_op.subtract),
    "elemwise_mul"           : _rename(_op.multiply),
    "elemwise_div"           : _rename(_op.divide),
    "_maximum"               : _rename(_op.maximum),
    "_minimum"               : _rename(_op.minimum),
    "flatten"                : _rename(_op.nn.batch_flatten),
    "Flatten"                : _rename(_op.nn.batch_flatten),
    # scalar power
    "square"                 : _mx_make_power(2),
    "rsqrt"                  : _mx_make_power(-1/2),
    "cbrt"                   : _mx_make_power(1/3),
    "rcbrt"                  : _mx_make_power(-1/3),
    "__pow_scalar__"         : _binop_scalar(_op.power),
    "_power_scalar"          : _binop_scalar(_op.power),
    "__rsub_scalar__"        : _rbinop_scalar(_op.subtract),
    "_rminus_scalar"         : _rbinop_scalar(_op.subtract),
    "__rdiv_scalar__"        : _rbinop_scalar(_op.divide),
    "_rdiv_scalar"           : _rbinop_scalar(_op.divide),
    "__rpow_scalar__"        : _rbinop_scalar(_op.power),
    # scalar op
    "__add_scalar__"         : _binop_scalar(_op.add),
    "_plus_scalar"           : _binop_scalar(_op.add),
    "__sub_scalar__"         : _binop_scalar(_op.subtract),
    "_minus_scalar"          : _binop_scalar(_op.subtract),
    "__mul_scalar__"         : _binop_scalar(_op.multiply),
    "_mul_scalar"            : _binop_scalar(_op.multiply),
    "__div_scalar__"         : _binop_scalar(_op.divide),
    "_div_scalar"            : _binop_scalar(_op.divide),
    "log2"                   : _mx_make_logarithm(2),
    "log10"                  : _mx_make_logarithm(10),
    "log1p"                  : _mx_log1p,
    "expm1"                  : _mx_expm1,
    "_equal_scalar"          : _mx_compare(_op.equal, _binop_scalar),
    "_not_equal_scalar"      : _mx_compare(_op.not_equal, _binop_scalar),
    "_greater_scalar"        : _mx_compare(_op.greater, _binop_scalar),
    "_greater_equal_scalar"  : _mx_compare(_op.greater_equal, _binop_scalar),
    "_lesser_scalar"         : _mx_compare(_op.less, _binop_scalar),
    "_lesser_equal_scalar"   : _mx_compare(_op.less_equal, _binop_scalar),
    "_maximum_scalar"        : _binop_scalar(_op.maximum),
    "_minimum_scalar"        : _binop_scalar(_op.minimum),
    # reduction ops
    "mean"          : _reduce(_op.mean),
    "max"           : _reduce(_op.max),
    "min"           : _reduce(_op.min),
    "sum"           : _reduce(_op.sum),
    "max_axis"      : _reduce(_op.max),
    "min_axis"      : _reduce(_op.min),
    "sum_axis"      : _reduce(_op.sum),
    "argmax"        : _arg_reduce(_op.argmax),
    "argmin"        : _arg_reduce(_op.argmin),
    # init ops
    "_ones"         : _init_op(_op.ones),
    "_zeros"        : _init_op(_op.zeros),
    # softmax
    "softmax"       : _softmax_op(_op.nn.softmax),
    "log_softmax"   : _softmax_op(_op.nn.log_softmax),
    "Softmax"       : _softmax_op(_op.nn.softmax),
    # per op specialization
    "Reshape"       : _reshape,
    "reshape"       : _reshape,
    "Cast"          : _cast,
    "clip"          : _clip,
    "transpose"     : _transpose,
    "UpSampling"    : _upsampling,
    "add_n"         : _elemwise_sum,
    # MXNet specific implementations
    "FullyConnected": _mx_fully_connected,
    "Activation"    : _mx_activations,
    "Convolution"   : _mx_conv2d,
    "Convolution_v1": _mx_conv2d,
    "Deconvolution" : _mx_conv2d_transpose,
    "Pooling"       : _mx_pooling,
    "Pooling_v1"    : _mx_pooling,
    "Dropout"       : _mx_dropout,
    "BatchNorm"     : _mx_batch_norm,
    "BatchNorm_v1"  : _mx_batch_norm,
    "LayerNorm"     : _mx_layer_norm,
    "LRN"           : _mx_lrn,
    "L2Normalization"  : _mx_l2_normalize,
    "slice"         : _mx_slice,
    "slice_like"    : _mx_slice_like,
    "slice_axis"    : _mx_slice_axis,
    "SliceChannel"  : _mx_split,
    "split"         : _mx_split,
    "expand_dims"   : _mx_expand_dims,
    "Concat"        : _mx_concat,
    "concat"        : _mx_concat,
    "stack"         : _mx_stack,
    "batch_dot"     : _mx_batch_dot,
    "LeakyReLU"     : _mx_leaky_relu,
    "_arange"       : _mx_arange,
    "_full"         : _mx_full,
    "repeat"        : _mx_repeat,
    "tile"          : _mx_tile,
    "take"          : _mx_take,
    "reverse"       : _mx_reverse,
    "squeeze"       : _mx_squeeze,
    "broadcast_axis": _mx_broadcast_axis,
    "BlockGrad"     : _mx_BlockGrad,
    "shape_array"   : _mx_shape_array,
    "Embedding"     : _mx_embedding,
    "argsort"       : _mx_argsort,
    "SoftmaxOutput" : _mx_softmax_output,
    "SoftmaxActivation" : _mx_softmax_activation,
    "smooth_l1"     : _mx_smooth_l1,
    "SequenceMask"  : _mx_sequence_mask,
    "topk"          : _mx_topk,
    # vision
    "_contrib_BilinearResize2D" : _mx_resize,
    "_contrib_MultiBoxPrior" : _mx_multibox_prior,
    "_contrib_MultiBoxDetection" : _mx_multibox_detection,
    "_contrib_ROIAlign" : _mx_roi_align,
    "ROIPooling"        : _mx_roi_pooling,
    "_contrib_Proposal" : _mx_proposal,
    "_contrib_MultiProposal" : _mx_proposal,
    "_contrib_box_nms" : _mx_box_nms,
    "_contrib_DeformableConvolution" : _mx_deformable_convolution,
    "_contrib_AdaptiveAvgPooling2D" : _mx_adaptive_pooling,
    # control flow
    "_cond"       : _mx_cond,
    "_foreach"    : _mx_foreach,
    "_while_loop" : _mx_while_loop,
    # junru branch op
    "_contrib_div_sqrt_dim": _mx_contrib_div_sqrt_dim,
    "_contrib_sarange"     : _mx_contrib_sarange,
    # List of missing operators that are present in NNVMv1
    # TODO(tvm-tvm): support all operators.
    #
    # "broadcast_to",
    # "Crop"          : _crop_like,
}

# set identity list
_convert_map.update({k : _rename(k) for k in _identity_list})


def _from_mxnet_impl(mod, symbol, shape_dict, dtype_info):
    """Convert mxnet symbol to compatible relay Function.

    Reconstruct a relay Function by traversing the mxnet symbol.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet.
        The op_name and attrs inside are not always compatible.

    shape_dict : dict
        Known parameter shapes

    dtype_info : dict or str.
        Known parameter dtypes

    Returns:
    -------
    func : tvm.relay.Function
        Converted relay Function
    """
    assert symbol is not None
    if isinstance(symbol, dict):
        jgraph = symbol
    else:
        jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    free_vars = []

    for nid, node in enumerate(jnodes):
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = StrAttrsDict(node.get("attrs", {}))
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            if isinstance(shape_dict, dict):
                shape = shape_dict[node_name] if node_name in shape_dict else None
            elif isinstance(shape_dict, list):
                shape = shape_dict.pop(0)
            else:
                raise ValueError("Unknown type of shape_dict: %s" + type(shape_dict))
            if isinstance(dtype_info, dict):
                dtype = dtype_info[node_name] if node_name in dtype_info else "float32"
            elif isinstance(dtype_info, list):
                dtype = dtype_info.pop(0)
            else:
                dtype = dtype_info
            var = _expr.var(node_name, shape=shape, dtype=dtype)
            node_map[nid] = [var]
            free_vars.append(var)
        elif op_name in _convert_map:
            if op_name in ['_cond', '_foreach', '_while_loop']:
                subgraphs = node['subgraphs']
                res = _convert_map[op_name](children, attrs, subgraphs, mod)
            else:
                res = _convert_map[op_name](children, attrs, mod)
            if isinstance(res, (_expr.TupleWrapper, tuple, list)):
                pass
            elif isinstance(res, _expr.Expr):
                res = [res]
            else:
                print("res is {}:{}".format(res, type(res)))
                raise RuntimeError("unexpected type %s" % type(res))
            node_map[nid] = res
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend MXNet.'.format(op_name))

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    if isinstance(outputs, _expr.Function):
        return outputs
    func = _expr.Function(free_vars, outputs)
    return func

def _update_shape_dtype(shape, dtype, params):
    """Update shape dtype given params information"""
    shape = {} if shape is None else shape
    if not params:
        return shape, dtype
    shape = shape.copy()
    shape.update({k : v.shape for k, v in params.items()})
    if isinstance(dtype, str):
        for k, v in params.items():
            if v.dtype != dtype:
                raise ValueError(
                    "%s: dtype not expected %s vs %s" % (k, dtype, v.dtype))
    else:
        dtype = dtype.copy()
        dtype.update({k : str(v.dtype) for k, v in params.items()})
    return shape, dtype


def from_mxnet(symbol,
               shape=None,
               dtype="float32",
               arg_params=None,
               aux_params=None,
               input_symbols=None):
    """Convert from MXNet"s model into compatible relay Function.

    Parameters
    ----------
    symbol : mxnet.Symbol or mxnet.gluon.HybridBlock
        MXNet symbol.

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    arg_params : dict of str to mx.NDArray
        The argument parameters in mxnet

    aux_params : dict of str to mx.NDArray
        The auxiliary parameters in mxnet

    Returns
    -------
    sym : tvm.relay.Function
        Compatible relay Function

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import mxnet as mx
    except ImportError as e:
        raise ImportError("{}. MXNet is required to parse symbols.".format(e))
    mod = _module.Module()

    if isinstance(symbol, mx.sym.Symbol):
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = _nd.array(v.asnumpy())
        for k, v in aux_params.items():
            params[k] = _nd.array(v.asnumpy())
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        sym = _from_mxnet_impl(mod, symbol, shape, dtype)
    elif isinstance(symbol, mx.gluon.HybridBlock):
        if arg_params is not None or aux_params is not None:
            raise ValueError("arg_params and aux_params ae not used when importing HybridBlock")
        params = {}
        for k, v in symbol.collect_params().items():
            params[k] = _nd.array(v.data().asnumpy())
        if input_symbols is not None:
            inputs = input_symbols
        else:
            inputs = []
            inputs.append(mx.sym.Variable("data"))
        sym = symbol(*inputs)
        if isinstance(sym, (list, tuple)):
            sym = mx.sym.Group(sym)
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        sym = _from_mxnet_impl(mod, sym, shape, dtype)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    mod[mod.entry_func] = sym
    return mod, params
