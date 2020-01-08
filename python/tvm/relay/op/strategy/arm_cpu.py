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

@bitserial_dense_strategy.register("arm_cpu")
def schedule_bitserial_dense_arm_cpu(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(
        wrap_compute_bitserial_dense(topi.arm_cpu.bitserial_dense_default),
        wrap_topi_schedule(topi.arm_cpu.schedule_bitserial_dense))
    return strategy
