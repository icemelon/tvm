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
"""Definition of x86 operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from __future__ import absolute_import

import logging

import topi
from topi.util import get_const_int, get_const_float, get_const_tuple, get_float_tuple
from .generic import *
from .. import op as _op
from ....schedule import SpecializedCondition

logger = logging.getLogger('strategy')

@schedule_injective.register("cpu")
def schedule_injective_cpu(attrs, outs, target):
    """schedule injective ops for x86"""
    with target:
        return topi.x86.schedule_injective(outs)

@schedule_reduce.register("cpu")
def schedule_reduce_cpu(attrs, outs, target):
    """schedule reduction ops for x86"""
    with target:
        return topi.x86.schedule_reduce(outs)

@schedule_concatenate.register("cpu")
def schedule_concatenate_cpu(attrs, outs, target):
    """schedule concatenate op for x86"""
    with target:
        return topi.x86.schedule_concatenate(outs)

@schedule_pool.register("cpu")
def schedule_pool_cpu(attrs, outs, target):
    """schedule pooling ops for x86"""
    with target:
        return topi.x86.schedule_pool(outs, attrs.layout)

@schedule_adaptive_pool.register("cpu")
def schedule_adaptive_pool_cpu(attrs, outs, target):
    """schedule adaptive pooling ops for x86"""
    with target:
        return topi.x86.schedule_adaptive_pool(outs)

@schedule_softmax.register("cpu")
def schedule_softmax_cpu(attrs, outs, target):
    """schedule softmax for x86"""
    with target:
        return topi.x86.schedule_softmax(outs)

@conv2d_strategy.register("cpu")
def conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d x86 strategy"""
    strategy = _op.OpStrategy()

    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    dtype = inputs[0].dtype

    assert layout in ["NCHW", "NHWC"]
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            if dtype == "int8":
                raise RuntimeError("Yet to be added")
            else:
                strategy.add_implement(wrap_compute_conv2d_NCHWc(topi.x86.conv2d_NCHWc),
                                       wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc))
        elif layout == "NHWC":
            logger.warning("For x86 target, NCHW layout is recommended for conv2d.")
            strategy.add_implement(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.x86.schedule_conv2d_nhwc))
    else:
        if layout == "NCHW" and get_conv2d_out_depth(inputs[1], kernel_layout) == groups:
            strategy.add_implement(
                wrap_compute_depthwise_conv2d_NCHWc(topi.x86.depthwise_conv2d_NCHWc),
                wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_NCHWc))
        elif layout == "NHWC" and kernel_layout == "HWOI" \
                and get_conv2d_out_depth(inputs[1], kernel_layout) == groups:
            logger.warning("For x86 target, NCHW layout is recommended for depthwise conv2d.")
            strategy.add_implement(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc))
        elif layout in ['NCHW', 'NCHW4c']:
            logger.warning("Group conv2d is not optimized for x86 target.")
            strategy.add_implement(
                wrap_compute_conv2d(topi.nn.group_conv2d_nchw),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw))
        else:
            raise RuntimeError("Unsupported group number %d" % groups)

    return strategy

@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_conv2d_NCHWc(topi.x86.conv2d_NCHWc),
                           wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc))
    return strategy

@depthwise_conv2d_NCHWc_strategy.register("cpu")
def depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_compute_depthwise_conv2d_NCHWc(topi.x86.depthwise_conv2d.depthwise_conv2d_NCHWc),
        wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_NCHWc))
    return strategy

@conv2d_transpose_strategy.register("cpu")
def conv2d_transpose_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_transpose x86 strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_comptue_conv2d_transpose(topi.x86.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.x86.schedule_conv2d_transpose_nchw))
    return strategy

@dense_strategy.register("cpu")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""
    strategy = _op.OpStrategy()
    m, k = inputs[0].shape
    strategy.add_implement(wrap_compute_dense(topi.x86.dense_nopack),
                           wrap_topi_schedule(topi.x86.schedule_dense_nopack),
                           10)
    if "cblas" in target.libs:
        strategy.add_implement(wrap_compute_dense(topi.x86.dense_cblas),
                               wrap_topi_schedule(topi.x86.schedule_dense_cblas),
                               5)
    with SpecializedCondition(k > 16):
        strategy.add_implement(wrap_compute_dense(topi.x86.dense_pack),
                               wrap_topi_schedule(topi.x86.schedule_dense_pack))
    return strategy

@batch_matmul_strategy.register("cpu")
def batch_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """batch_matmul x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_batch_matmul(topi.x86.batch_matmul),
                           wrap_topi_schedule(topi.x86.schedule_batch_matmul),
                           10)
    if "cblas" in target.libs:
        strategy.add_implement(wrap_compute_batch_matmul(topi.x86.batch_matmul_cblas),
                               wrap_topi_schedule(topi.x86.schedule_batch_matmul_cblas),
                               5)
    return strategy

@schedule_sparse_dense.register("cpu")
def schedule_sparse_dense_cpu(attrs, outs, target):
    """schedule sparse_dense for x86"""
    with target:
        return topi.x86.schedule_sparse_dense(outs)

@roi_align_strategy.register("cpu")
def roi_align_strategy_cpu(attrs, inputs, out_type, target):
    """roi_align x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_roi_align(topi.x86.roi_align_nchw),
                           wrap_topi_schedule(topi.generic.schedule_roi_align))
    return strategy

@bitserial_conv2d_strategy.register("cpu")
def bitserial_conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implement(
            wrap_compute_bitserial_conv2d(topi.x86.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.x86.schedule_bitserial_conv2d_nchw))
    elif layout == "NHWC":
        strategy.add_implement(
            wrap_compute_bitserial_conv2d(topi.x86.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.x86.schedule_bitserial_conv2d_nhwc))
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy

@bitserial_dense_strategy.register("cpu")
def bitserial_dense_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_dense x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_compute_bitserial_dense(topi.x86.bitserial_dense),
        wrap_topi_schedule(topi.x86.schedule_bitserial_dense))
    return strategy
