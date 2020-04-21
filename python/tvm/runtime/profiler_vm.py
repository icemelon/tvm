# License .to the Apache Software Foundation (ASF) under one
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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
The Relay Virtual Machine profiler.

Provides extra APIs for profiling vm execution.
"""
import tvm
from tvm.runtime import Object
from tvm.runtime import _ffi_api
from . import vm

def enabled():
    """Whether vm profiler is enabled."""
    return hasattr(_ffi_api, "_VirtualMachineDebug")

@tvm._ffi.register_object("vm_profiler.ProfileRecord")
class ProfileRecord(Object):
    """Profile record"""


@tvm._ffi.register_object("vm_profiler.KernelRecord")
class KernelRecord(ProfileRecord):
    """Kernel profile record"""


@tvm._ffi.register_object("vm_profiler.AllocStorageRecord")
class AllocStorageRecord(ProfileRecord):
    """Allocate storage profile record"""


class VirtualMachineProfiler(vm.VirtualMachine):
    """Relay profile VM runtime."""
    def __init__(self, mod):
        super(VirtualMachineProfiler, self).__init__(mod)
        m = mod.module if isinstance(mod, vm.Executable) else mod
        self.mod = _ffi_api._VirtualMachineDebug(m)
        self._init = self.mod["init"]
        self._invoke = self.mod["invoke"]
        self._get_profile_result = self.mod["get_profile_result"]
        self._set_input = self.mod["set_input"]
        self._reset = self.mod["reset"]

    def get_profile_result(self, sort_by_time=True):
        """Get the statistics of executed ops.

        Parameter
        ---------
        sort_by_time: Optional[Boolean]
           Set to indicate the returned results are sorted by execution time in
           the descending order. It is printed in the random order if this
           field is not set.

        Returns
        -------
            The execution statistics in string.
        """
        res = self._get_profile_result()
        header = ["Kernel Name", "Time(us)", "Time(%)", "Shape", "Inputs", "Outputs"]
        sep    = ["-----------", "--------", "-------", "-----", "------", "-------"]

        kernel_time = 0
        num_allocs = 0
        alloc_time = 0
        data = []
        for rec in res:
            if isinstance(rec, KernelRecord):
                kernel_name = rec.kernel_name
                time_us = round(rec.duration * 1e6, 3)
                inputs = rec.num_inputs
                outputs = rec.num_outputs
                shape = str(rec.output_shapes[0])
                data.append([kernel_name, time_us, shape, inputs, outputs])
                kernel_time += time_us
            elif isinstance(rec, AllocStorageRecord):
                num_allocs += 1
                alloc_time += rec.duration * 1e6

        total_time = kernel_time + alloc_time
        for row in data:
            time_percent = round((row[1] / kernel_time) * 100, 3)
            row.insert(2, time_percent)

        if sort_by_time:
            data = sorted(data, key=lambda x: x[1], reverse=True)

        fmt = ""
        for i in range(len(header)):
            max_len = len(header[i])
            for j in range(len(data)):
                item_len = len(str(data[j][i]))
                max_len = max(item_len, max_len)
            fmt += f"{{:<{max_len + 2}}}"

        log = [f"Total time: {total_time:.3f} us",
               f"Kernel time: {kernel_time:.3f} us",
               f"Allocation time: {alloc_time:.3f} us",
               f"Number of allocation: {num_allocs}",
               "",
               fmt.format(*header),
               fmt.format(*sep)]
        for row in data:
            log.append(fmt.format(*row))
        return "\n".join(log)

    def reset(self):
        """Reset the profile result."""
        self._reset()
