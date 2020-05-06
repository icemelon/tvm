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
 * \file specialized_condition.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/specialized_condition.h>
#include <stack>

namespace tvm {
namespace te {

using namespace tir;

TVM_REGISTER_NODE_TYPE(SpecializedConditionNode);

SpecializedCondition::SpecializedCondition(Array<PrimExpr> conditions) {
  ObjectPtr<SpecializedConditionNode> n = make_object<SpecializedConditionNode>();
  n->clauses = std::move(conditions);
  data_ = std::move(n);
}

/*! \brief Entry to hold the SpecializedCondition context stack. */
struct TVMSpecializationThreadLocalEntry {
  /*! \brief The current specialized condition */
  std::stack<SpecializedCondition> condition_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMSpecializationThreadLocalEntry> TVMSpecializationThreadLocalStore;

void SpecializedCondition::EnterWithScope() {
  TVMSpecializationThreadLocalEntry *entry = TVMSpecializationThreadLocalStore::Get();
  entry->condition_stack.push(*this);
}

void SpecializedCondition::ExitWithScope() {
  TVMSpecializationThreadLocalEntry *entry = TVMSpecializationThreadLocalStore::Get();
  CHECK(!entry->condition_stack.empty());
  CHECK(entry->condition_stack.top().same_as(*this));
  entry->condition_stack.pop();
}

SpecializedCondition SpecializedCondition::Current() {
  TVMSpecializationThreadLocalEntry *entry = TVMSpecializationThreadLocalStore::Get();
  SpecializedCondition cond;
  if (entry->condition_stack.size() > 0) {
    cond = entry->condition_stack.top();
  }
  return cond;
}

class SpecializedCondition::Internal {
 public:
  static void EnterScope(SpecializedCondition cond) {
    cond.EnterWithScope();
  }

  static void ExitScope(SpecializedCondition cond) {
    cond.ExitWithScope();
  }
};

// Printer
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SpecializedConditionNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SpecializedConditionNode*>(node.get());
    p->stream << "specialized_condition(";
    p->Print(op->clauses);
    p->stream << ')';
});

TVM_REGISTER_GLOBAL("te.CreateSpecializedCondition")
.set_body_typed([](Array<PrimExpr> condition) {
    return SpecializedCondition(condition);
});

TVM_REGISTER_GLOBAL("te.GetCurrentSpecialization")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = SpecializedCondition::Current();
});

TVM_REGISTER_GLOBAL("te.EnterSpecializationScope")
.set_body_typed(SpecializedCondition::Internal::EnterScope);

TVM_REGISTER_GLOBAL("te.ExitSpecializationScope")
.set_body_typed(SpecializedCondition::Internal::ExitScope);

}  // namespace te
}  // namespace tvm
