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
"""Definition of ROCm operator strategy."""
# pylint: disable=invalid-name,unused-argument,unused-wildcard-import,wildcard-import
from __future__ import absolute_import

import topi
from .generic import *

@schedule_lrn.register("rocm")
def schedule_lrn_rocm(attrs, outs, target):
    """schedule LRN for rocm"""
    with target:
        return topi.rocm.schedule_lrn(outs)

@schedule_l2_normalize.register("rocm")
def schedule_l2_normalize_rocm(attrs, outs, target):
    """schedule L2 normalize for rocm"""
    with target:
        return topi.rocm.schedule_l2_normalize(outs)

@dense_strategy.register(["rocm"])
def dense_strategy_rocm(attrs, inputs, out_type, target):
    """Dense strategy for ROCM"""
    strategy = _op.OpStrategy()
    assert len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2, "Only support 2-dim dense"

    strategy.add_implement(wrap_compute_dense(topi.rocm.dense),
                           wrap_topi_schedule(topi.rocm.schedule_dense))
    if target.target_name == "rocm" and "rocblas" in target.libs:
        assert out_dtype == inputs[0].dtype, "Mixed precision not supported."
        strategy.add_implement(
            wrap_compute_dense(topi.rocm.dense_cblas),
            wrap_topi_schedule(topi.rocm.dense_cblas), 5)
    return strategy
