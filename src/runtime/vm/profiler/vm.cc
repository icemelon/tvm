/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/vm/profiler/vm.cc
 * \brief The Relay debug virtual machine.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "vm.h"

namespace tvm {
namespace runtime {
namespace vm {

TVM_REGISTER_NODE_TYPE(ProfileRecordObj);
TVM_REGISTER_NODE_TYPE(KernelRecordObj);
TVM_REGISTER_NODE_TYPE(AllocStorageRecordObj);

PackedFunc VirtualMachineDebug::GetFunction(
    const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_profile_result") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = profile_records_;
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      profile_records_ = Array<ProfileRecord>();
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineDebug::LoadExecutable(const Executable* exec) {
  VirtualMachine::LoadExecutable(exec);
  CHECK(exec_);
  for (auto kv : exec_->primitive_map) {
    packed_index_map_[kv.second] = kv.first;
  }
}

void VirtualMachineDebug::InvokePacked(Index packed_index,
                                       const PackedFunc& func, Index arg_count,
                                       Index output_size,
                                       const std::vector<ObjectRef>& args) {
  CHECK(exec_);
  CHECK(!ctxs_.empty()) << "Context has not been initialized yet.";
  // TODO(@zhiics) Need to record the device type of each packed func so that
  // we can correctly sync.
  Index fallback_device_type = static_cast<Index>(ctxs_[0].device_type);
  auto ctx = this->GetContext(fallback_device_type);

  auto op_begin = std::chrono::high_resolution_clock::now();
  VirtualMachine::InvokePacked(packed_index, func, arg_count, output_size, args);
  TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
  auto op_end = std::chrono::high_resolution_clock::now();
  double op_duration =
      std::chrono::duration_cast<std::chrono::duration<double> >(op_end -
                                                                 op_begin)
          .count();

  auto n = make_object<KernelRecordObj>();
  n->opcode = static_cast<int32_t>(Opcode::InvokePacked);
  n->duration = op_duration;
  n->kernel_name = packed_index_map_[packed_index];
  n->num_inputs = arg_count;
  n->num_outputs = 0;

  auto add_out_shape = [&n](NDArray arr) {
    ++n->num_outputs;
    Array<Integer> shape;
    for (auto dim : arr.Shape()) {
      shape.push_back(dim);
    }
    n->output_shapes.push_back(shape);
  };

  for (int i = arg_count - output_size; i < arg_count; ++i) {
    if (const auto* adt = args[i].as<ADTObj>()) {
      for (size_t j = 0; j < adt->size; ++j) {
        auto obj = (*adt)[j];
        add_out_shape(Downcast<NDArray>(obj));
      }
    } else {
      add_out_shape(Downcast<NDArray>(args[i]));
    }
  }
  profile_records_.push_back(KernelRecord(n));
}

void VirtualMachineDebug::AllocateStorage(const Instruction& instr) {
  auto begin = std::chrono::high_resolution_clock::now();
  VirtualMachine::AllocateStorage(instr);
  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration_cast<std::chrono::duration<double> >(end - begin).count();

  auto n = make_object<AllocStorageRecordObj>();
  n->opcode = static_cast<int32_t>(Opcode::AllocStorage);
  n->duration = duration;
  n->nbytes = LoadScalarInt(instr.alloc_storage.allocation_size);
  profile_records_.push_back(AllocStorageRecord(n));
}

runtime::Module CreateVirtualMachineDebug(const Executable* exec) {
  auto vm = make_object<VirtualMachineDebug>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachineDebug")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec) << "Virtual machine has not been defined yet."
              << "\n";
  *rv = CreateVirtualMachineDebug(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
