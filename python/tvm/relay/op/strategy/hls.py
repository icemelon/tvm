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
"""Definition of HLS operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from __future__ import absolute_import

import topi
from .generic import *

@schedule_injective.register("hls")
def schedule_injective_hls(attrs, outs, target):
    """schedule injective ops for hls"""
    with target:
        return topi.hls.schedule_injective(outs)

@schedule_reduce.register("hls")
def schedule_reduce_hls(attrs, outs, target):
    """schedule reduction ops for hls"""
    with target:
        return topi.hls.schedule_reduce(outs)

@schedule_concatenate.register("hls")
def schedule_concatenate_hls(attrs, outs, target):
    """schedule concatenate for hls"""
    with target:
        return topi.hls.schedule_injective(outs)

@schedule_pool.register("hls")
def schedule_pool_hls(attrs, outs, target):
    """schedule pooling ops for hls"""
    with target:
        return topi.hls.schedule_pool(outs, attrs.layout)

@schedule_adaptive_pool.register("hls")
def schedule_adaptive_pool_hls(attrs, outs, target):
    """schedule adaptive pooling ops for hls"""
    with target:
        return topi.hls.schedule_adaptive_pool(outs)

@schedule_softmax.register("hls")
def schedule_softmax_hls(attrs, outs, target):
    """schedule softmax for hls"""
    with target:
        return topi.hls.schedule_softmax(outs)
