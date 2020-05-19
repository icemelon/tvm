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
 * \file workload.h
 * \brief AutoTVM workload.
 */
#ifndef TVM_AUTOTVM_WORKLOAD_H_
#define TVM_AUTOTVM_WORKLOAD_H_

#include <tvm/tir/expr.h>
#include <tvm/runtime/container.h>

namespace tvm {
namespace autotvm {

using namespace tvm::tir;

class WorkloadObj : public Object {
 public:
  runtime::String task_name;
  Array<ObjectRef> args;
  Map<Var, runtime::String> wildcard_map;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("task_name", &task_name);
    v->Visit("args", &args);
    v->Visit("wildcard_map", &wildcard_map);
  }

  static constexpr const char* _type_key = "autotvm.Workload";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadObj, Object);
};

class Workload : public ObjectRef {
 public:
  Workload(runtime::String task_name, Array<ObjectRef> args,
           Map<Var, runtime::String> wildcard_map);

  TVM_DEFINE_OBJECT_REF_METHODS(Workload, ObjectRef, WorkloadObj);
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_WORKLOAD_H_
