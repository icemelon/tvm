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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
from tvm import autotvm
from tvm.contrib import rocblas
import topi
from ..nn.dense import dense
from .. import generic

#@autotvm.register_topi_compute(dense, "rocm", "direct")
@autotvm.register_topi_compute2('dense.rocm')
def dense_default(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for rocm backend.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert bias is None, "bias is not supported"
    return dense(data, weight, bias, out_dtype)


#@autotvm.register_topi_schedule(generic.schedule_dense, "rocm", "direct")
@autotvm.register_topi_schedule2('dense.rocm')
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    return topi.cuda.schedule_dense(cfg, outs)


@autotvm.register_topi_compute2('dense_cblas.rocm')
def dense_cblas(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for rocm backend with cblas.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert bias is None, "bias is not supported"
    return rocblas.matmul(data, weight, False, True)


@autotvm.register_topi_schedule2('dense_cblas.rocm')
def schedule_dense_cblas(_, outs):
    """Schedule for dense operator with rocm cblas"""
    return generic.schedule_extern(outs)
