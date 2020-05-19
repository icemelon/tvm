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
 * \file workload.cc
 */
#include <tvm/runtime/registry.h>
#include "workload.h"

namespace tvm {
namespace autotvm {

TVM_REGISTER_NODE_TYPE(WorkloadObj);

Workload::Workload(runtime::String task_name, Array<ObjectRef> args,
                   Map<Var, runtime::String> wildcard_map) {
  auto n = make_object<WorkloadObj>();
  n->task_name = std::move(task_name);
  n->args = std::move(args);
  n->wildcard_map = std::move(wildcard_map);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("autotvm.CreateWorkload")
.set_body_typed([](runtime::String task_name, Array<ObjectRef> args,
                   Map<Var, runtime::String> wildcard_map) {
  return Workload(task_name, args, wildcard_map);
});

}  // namespace autotvm
}  // namespace tvm
