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
 * \file src/runtime/vm/profiler/vm.h
 * \brief The Relay debug virtual machine.
 */

#ifndef TVM_RUNTIME_VM_PROFILER_VM_H_
#define TVM_RUNTIME_VM_PROFILER_VM_H_

#include <tvm/node/reflection.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/vm.h>
#include <tvm/runtime/object.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

class ProfileRecordObj : public Object {
 public:
  int32_t opcode;
  double duration;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("opcode", &opcode);
    v->Visit("duration", &duration);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const uint32_t _type_child_slots = 2;
  static constexpr const char* _type_key = "vm_profiler.ProfileRecord";
  TVM_DECLARE_BASE_OBJECT_INFO(ProfileRecordObj, Object);
};

class ProfileRecord : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ProfileRecord, ObjectRef, ProfileRecordObj);
};

class KernelRecordObj : public ProfileRecordObj {
 public:
  std::string kernel_name;
  int32_t num_inputs;
  int32_t num_outputs;
  Array<Array<Integer>> output_shapes;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kernel_name", &kernel_name);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("num_outputs", &num_outputs);
    v->Visit("output_shapes", &output_shapes);
    ProfileRecordObj::VisitAttrs(v);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "vm_profiler.KernelRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelRecordObj, ProfileRecordObj);
};

class KernelRecord : public ProfileRecord {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(KernelRecord, ProfileRecord, KernelRecordObj);
};

class AllocStorageRecordObj : public ProfileRecordObj {
 public:
  int64_t nbytes;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("nbytes", &nbytes);
    ProfileRecordObj::VisitAttrs(v);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "vm_profiler.AllocStorageRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocStorageRecordObj, ProfileRecordObj);
};

class AllocStorageRecord : public ProfileRecord {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AllocStorageRecord, ProfileRecord, AllocStorageRecordObj);
};

class VirtualMachineDebug : public VirtualMachine {
 public:
  VirtualMachineDebug() : VirtualMachine() {}

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final;

  void LoadExecutable(const Executable* exec) final;

  ~VirtualMachineDebug() {}

 private:
  void InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                    Index output_size, const std::vector<ObjectRef>& args) final;

  void AllocateStorage(const Instruction& instr) final;

  std::unordered_map<Index, std::string> packed_index_map_;
  Array<ProfileRecord> profile_records_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_PROFILER_VM_H_
