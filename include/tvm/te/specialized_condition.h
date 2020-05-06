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
 * \file tvm/te/specialized_condition.h
 * \brief Define specialized condition.
 */
#ifndef TVM_TE_SPECIALIZED_CONDITION_H_
#define TVM_TE_SPECIALIZED_CONDITION_H_

#include <tvm/tir/expr.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/support/with.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace te {

/*! \brief Container for specialization conditions. */
class SpecializedConditionNode : public Object {
 public:
  /*!
   * \brief List of conditions in conjunctive joint form (CNF).
   *   Each condition should be a simple expression, e.g., n > 16, m % 8 == 0, etc.,
   *   where n, m are tvm::Var that represents a dimension in the tensor shape.
   */
  Array<PrimExpr> clauses;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("clauses", &clauses);
  }

  static constexpr const char* _type_key = "SpecializedCondition";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpecializedConditionNode, Object);
};

/*!
 * \brief Specialized condition to enable op specialization
 */
class SpecializedCondition : public ObjectRef {
 public:
  /*!
   * \brief construct from conditions
   * \param conditions The clauses in the specialized condition.
   */
  TVM_DLL SpecializedCondition(Array<PrimExpr> conditions);  // NOLINT(*)

  /*!
   * \brief Get the current specialized condition.
   * \return the current specialized condition.
   */
  TVM_DLL static SpecializedCondition Current();

  TVM_DEFINE_OBJECT_REF_METHODS(SpecializedCondition, ObjectRef, SpecializedConditionNode);
  class Internal;

 private:
  // enable with syntax.
  friend class Internal;
  friend class With<SpecializedCondition>;
  /*! \brief Push a new specialized condition onto the thread local stack. */
  TVM_DLL void EnterWithScope();
  /*! \brief Pop a specialized condition off the thread local context stack. */
  TVM_DLL void ExitWithScope();
};

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_SPECIALIZED_CONDITION_H_
