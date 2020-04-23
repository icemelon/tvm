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
from tvm import te
from tvm import relay as rly
from tvm.ir import structural_equal
from tvm.relay.transform import SimplifyExpr
from tvm.relay.testing import run_opt_pass, run_infer_type

def test_simplify_reshape():
    l = te.var('l')
    def before():
        x = rly.var("x", shape=(l, 128), dtype="float32")
        y = rly.reshape(x, newshape=(0, 8, -1)) # (l, 8, 16)
        y = rly.reshape(y, newshape=(-1, 0, 4, 4)) # (l, 8, 4, 4)
        y = rly.reshape(y, newshape=(-1, 1, 8, 4, 4)) # (l, 1, 8, 4, 4)
        return rly.Function([x], y)

    def expected():
        x = rly.var("x", shape=(l, 128), dtype="float32")
        z = rly.reshape(x, newshape=(-1, 1, 8, 4, 4))
        return rly.Function([x], z)

    f = before()
    ff = run_opt_pass(f, SimplifyExpr())
    after = run_infer_type(expected())
    assert structural_equal(ff, after)

if __name__ == "__main__":
    test_simplify_reshape()
