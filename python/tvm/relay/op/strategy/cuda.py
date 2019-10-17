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
"""Definition of CUDA/GPU operator strategy."""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from .generic import *
from .. import op as _op
from ....schedule import SpecializedCondition

@schedule_injective.register(["cuda", "gpu"])
def schedule_injective(attrs, outs, target):
    with target:
        return topi.cuda.schedule_injective(outs)

@schedule_reduce.register(["cuda", "gpu"])
def schedule_reduce(attrs, outs, target):
    with target:
        return topi.cuda.schedule_reduce(outs)

@schedule_concatenate.register(["cuda", "gpu"])
def schedule_concatenate(attrs, outs, target):
    with target:
        return topi.cuda.schedule_injective(outs)

@schedule_pool.register(["cuda", "gpu"])
def schedule_pool(attrs, outs, target):
    with target:
        return topi.cuda.schedule_pool(outs, attrs.layout)

@schedule_pool_grad.register(["cuda", "gpu"])
def schedule_pool_grad(attrs, outs, target):
    with target:
        return topi.cuda.schedule_pool_grad(outs)

@schedule_adaptive_pool.register(["cuda", "gpu"])
def schedule_adaptive_pool(attrs, outs, target):
    with target:
        return topi.cuda.schedule_adaptive_pool(outs)

@schedule_softmax.register(["cuda", "gpu"])
def schedule_softmax(attrs, outs, target):
    with target:
        return topi.cuda.schedule_softmax(outs)

@schedule_lrn.register(["cuda", "gpu"])
def schedule_lrn(attrs, outs, target):
    with target:
        return topi.cuda.schedule_lrn(outs)

@schedule_l2_normalize.register(["cuda", "gpu"])
def schedule_l2_normalize(attrs, outs, target):
    with target:
        return topi.cuda.schedule_l2_normalize(outs)

@deformable_conv2d_strategy.register(["cuda", "gpu"])
def deformable_conv2d_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_deformable_conv2d(topi.cuda.deformable_conv2d_nchw),
                           wrap_topi_schedule(topi.cuda.schedule_deformable_conv2d_nchw))
    return strategy

@conv3d_strategy.register(["cuda", "gpu"])
def conv3d_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    assert layout in ["NCDHW", "NDHWC"], "Not support this layout {} yet".format(layout)
    if layout == "NCDHW":
        strategy.add_implement(wrap_compute_conv3d(topi.cuda.conv3d_ncdhw),
                               wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw),
                               10)
    else: # layout == "NDHWC":
        strategy.add_implement(wrap_compute_conv3d(topi.cuda.conv3d_ndhwc),
                               wrap_topi_schedule(topi.cuda.schedule_conv3d_ndhwc),
                               10)
    if target.target_name == "cuda" and "cudnn" in target.libs:
        strategy.add_implement(wrap_compute_conv3d(topi.cuda.conv3d_cudnn),
                               wrap_topi_schedule(topi.cuda.schedule_conv3d_cudnn),
                               15)
    return strategy

@conv1d_transpose_strategy.register(["cuda", "gpu"])
def conv1d_transpose_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCW", "conv1d_transpose ncw only supported"
    assert dilation == (1,), "conv1d_transpose dilation is not supported"
    assert groups == 1, "conv1d_transpose groups == 1 only supported"
    strategy.add_implement(wrap_compute_conv1d_transpose(topi.cuda.conv1d_transpose_ncw),
                           wrap_topi_schedule(topi.cuda.schedule_conv1d_transpose_ncw))
    return strategy

@dense_strategy.register(["cuda", "gpu"])
def dense_strategy(attrs, inputs, out_type, target):
    # Todo(@icemelon9): update dense strategy
    strategy = _op.OpStrategy()
    if out_type.dtype == "int8":
        strategy.add_implement(wrap_compute_dense(topi.cuda.dense_int8),
                               wrap_topi_schedule(topi.cuda.schedule_dense_int8))
    else:
        strategy.add_implement(wrap_compute_dense(topi.nn.dense),
                               wrap_topi_schedule(topi.cuda.schedule_dense_small_batch))
        b = inputs[0].shape[0]
        with SpecializedCondition(b >= 32):
            strategy.add_implement(wrap_compute_dense(topi.nn.dense),
                                   wrap_topi_schedule(topi.cuda.schedule_dense_large_batch))
    if target.target_name == "cuda" and "cublas" in target.libs:
        strategy.add_implement(wrap_compute_dense(topi.cuda.dense_cblas),
                               wrap_topi_schedule(topi.generic.schedule_extern), 5)
    return strategy

@batch_matmul_strategy.register(["cuda", "gpu"])
def batch_matmul_strategy(attrs, inputs, out_type, target):
    strategy =_op.OpStrategy()
    strategy.add_implement(wrap_compute_batch_matmul(topi.nn.batch_matmul),
                           wrap_topi_schedule(topi.cuda.schedule_batch_matmul),
                           10)
    if target.target_name == "cuda" and "cublas" in target.libs:
        strategy.add_implement(wrap_compute_batch_matmul(topi.cuda.batch_matmul_cublas),
                               wrap_topi_schedule(topi.generic.schedule_extern),
                               15)
    return strategy

@argsort_strategy.register(["cuda", "gpu"])
def argsort_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_argsort(topi.cuda.argsort_gpu),
                           wrap_topi_schedule(topi.cuda.schedule_argsort))
    return strategy

@topk_strategy.register(["cuda", "gpu"])
def topk_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_topk(topi.cuda.topk_gpu),
                           wrap_topi_schedule(topi.cuda.schedule_topk))
    return strategy

@schedule_multibox_prior.register(["cuda", "gpu"])
def schedule_multibox_prior(attrs, outs, target):
    with target:
        return topi.cuda.schedule_multibox_prior(outs)

@schedule_multibox_transform_loc.register(["cuda", "gpu"])
def schedule_multibox_transform_loc(attrs, outs, target):
    with target:
        return topi.cuda.schedule_multibox_transform_loc(outs)

@get_valid_counts_strategy.register(["cuda", "gpu"])
def get_valid_counts_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_get_valid_counts(topi.cuda.get_valid_counts),
                           wrap_topi_schedule(topi.cuda.schedule_get_valid_counts))
    return strategy

@nms_strategy.register(["cuda", "gpu"])
def nms_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_nms(topi.cuda.non_max_suppression),
                           wrap_topi_schedule(topi.cuda.schedule_nms))
    return strategy

@roi_align_strategy.register(["cuda", "gpu"])
def roi_align_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_roi_align(topi.vision.rcnn.roi_align_nchw),
                           wrap_topi_schedule(topi.cuda.schedule_roi_align))
    return strategy

@schedule_roi_pool.register(["cuda", "gpu"])
def schedule_roi_pool(attrs, outs, target):
    with target:
        return topi.cuda.schedule_roi_pool(outs)

@proposal_strategy.register(["cuda", "gpu"])
def proposal_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_proposal(topi.cuda.proposal),
                           wrap_topi_schedule(topi.cuda.schedule_proposal))
    return strategy
