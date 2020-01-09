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
"""Definition of ARM CPU operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from __future__ import absolute_import

import topi
from .generic import *
from .. import op as _op

@schedule_injective.register("arm_cpu")
def schedule_injective_arm_cpu(_, outs, target):
    """schedule injective ops for arm cpu"""
    with target:
        return topi.arm_cpu.schedule_injective(outs)

@schedule_concatenate.register("arm_cpu")
def schedule_concatenate_arm_cpu(_, outs, target):
    """schedule concatenate for arm cpu"""
    with target:
        return topi.arm_cpu.schedule_concatenate(outs)

@conv2d_transpose_strategy.register("arm_cpu")
def conv2d_transpose_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d_transpose arm cpu strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_comptue_conv2d_transpose(topi.arm_cpu.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_transpose_nchw))
    return strategy

@bitserial_conv2d_strategy.register("arm_cpu")
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
            wrap_compute_bitserial_conv2d(topi.arm_cpu.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.arm_cpu.schedule_bitserial_conv2d_nhwc))
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy

@bitserial_dense_strategy.register("arm_cpu")
def schedule_bitserial_dense_arm_cpu(attrs, inputs, out_type, target):
    """bitserial_dense arm cpu strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_compute_bitserial_dense(topi.arm_cpu.bitserial_dense),
        wrap_topi_schedule(topi.arm_cpu.schedule_bitserial_dense))
    return strategy
