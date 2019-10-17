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
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from .generic import *
from .. import op as _op

@schedule_lrn.register("rocm")
def schedule_lrn(attrs, outs, target):
    with target:
        return topi.rocm.schedule_lrn(outs)

@schedule_l2_normalize.register("rocm")
def schedule_l2_normalize(attrs, outs, target):
    with target:
        return topi.rocm.schedule_l2_normalize(outs)
