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
 * \file simplify_inference.cc
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class ExprSimplifier : public ExprMutator {
 public:
  ExprSimplifier()
      : reshape_op_(Op::Get("reshape")) {}

  Expr VisitExpr_(const CallNode* n) {
    static auto make_reshape = tvm::runtime::Registry::Get("relay.op._make.reshape");
    CHECK(make_reshape) << "relay.op._make.reshape is not registered.";

    auto new_n = ExprMutator::VisitExpr_(n);
    auto new_call = new_n.as<CallNode>();
    if (new_call == nullptr) {
      return new_n;
    }
    if (new_call->op == reshape_op_) {
      if (auto arg = new_call->args[0].as<CallNode>()) {
        if (arg->op == reshape_op_) {
          // Merge two reshape
          int num_sym_axes = 0;
          Array<Integer> newshape;
          auto out_type = Downcast<TensorType>(n->checked_type());
          for (size_t i = 0; i < out_type->shape.size(); ++i) {
            if (auto val = out_type->shape[i].as<IntImmNode>()) {
              newshape.push_back(val->value);
            } else {
              if (++num_sym_axes > 1) {
                break;
              }
              newshape.push_back(-1);
            }
          }
          if (num_sym_axes <= 1) {
            return (*make_reshape)(arg->args[0], newshape);
          }
        }
      }
    }
    return new_n;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.
  const Op& reshape_op_;
};

Expr SimplifyExpr(const Expr& e) {
  return ExprSimplifier().Mutate(e);
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
          return Downcast<Function>(SimplifyExpr(f));
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr")
.set_body_typed(SimplifyExpr);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

