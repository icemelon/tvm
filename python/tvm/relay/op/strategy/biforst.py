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
"""Definition of biforst operator strategy."""
# pylint: disable=invalid-name,unused-argument

from __future__ import absolute_import

import topi
from .generic import dense_strategy
from .. import op as _op


@dense_strategy.register(["biforst"])
def dense_strategy_biforst(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implement(wrap_compute_dense(topi.biforst.dense_default),
                           wrap_topi_schedule(topi.biforst.schedule_dense))
    return strategy
