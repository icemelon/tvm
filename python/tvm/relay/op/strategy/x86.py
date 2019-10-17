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
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from .generic import *
from .. import op as _op
from ....schedule import SpecializedCondition

@schedule_injective.register("cpu")
def schedule_injective(attrs, outs, target):
    with target:
        return topi.x86.schedule_injective(outs)

@schedule_reduce.register("cpu")
def schedule_reduce(attrs, outs, target):
    with target:
        return topi.x86.schedule_reduce(outs)

@schedule_concatenate.register("cpu")
def schedule_concatenate(attrs, outs, target):
    with target:
        return topi.x86.schedule_concatenate(outs)

@schedule_pool.register("cpu")
def schedule_pool(attrs, outs, target):
    with target:
        return topi.x86.schedule_pool(outs, attrs.layout)

@schedule_adaptive_pool.register("cpu")
def schedule_adaptive_pool(attrs, outs, target):
    with target:
        return topi.x86.schedule_adaptive_pool(outs)

@schedule_softmax.register("cpu")
def schedule_softmax(attrs, outs, target):
    with target:
        return topi.x86.schedule_softmax(outs)

@conv2d_strategy.register("cpu")
def conv2d_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dtype = out_type.dtype
    assert layout in ["NCHW", "NHWC", "NCHWc"]
    if layout == "NCHW":
        strategy.add_implement(wrap_compute_conv2d(topi.x86.conv2d_nchw),
                               wrap_topi_schedule(topi.x86.schedule_conv2d_nchw))
    elif layout == "NHWC":
        strategy.add_implement(wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                               wrap_topi_schedule(topi.x86.schedule_conv2d_nhwc))
    return strategy

@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy(attrs, inputs, out_type, target):
    print('inside x86 conv2d_NCHWc_strategy')
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_conv2d_NCHWc(topi.x86.conv2d_NCHWc),
                           wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc))
    return strategy

@dense_strategy.register("cpu")
def dense_strategy(attrs, inputs, out_type, target):
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
def batch_matmul_strategy(attrs, inputs, out_type, target):
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
def schedule_sparse_dense(attrs, outs, target):
    with target:
        return topi.x86.schedule_sparse_dense(outs)

@roi_align_strategy.register("cpu")
def roi_align_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_roi_align(topi.x86.roi_align_nchw),
                           wrap_topi_schedule(topi.generic.schedule_roi_align))
    return strategy
