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
"""Unit tests for context analysis pass"""
import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay.analysis import context_analysis

def check_context_analysis(func, check_fn):
    mod = tvm.IRModule()
    mod['main'] = func
    ex = relay.create_executor('vm', mod)
    args = []
    for param in func.params:
        param = param.type_annotation
        shape = [int(sh) for sh in param.shape]
        data = np.random.rand(*shape).astype(param.dtype)
        args.append(tvm.nd.array(data))
    result = ex.evaluate(mod['main'])(*args)
    py_res = check_fn(*[arg.asnumpy() for arg in args])
    np.testing.assert_allclose(result.asnumpy(), py_res)
