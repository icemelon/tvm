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
 * \file bind_specialized_condition.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/specialized_condition.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <unordered_map>
#include <unordered_set>
//#include "ir_util.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

using namespace te;

struct ModBind {
  Var origin_var;
  Var bind_var;
  PrimExpr divisor;
  PrimExpr remainder;
};

class ModSubstitue : public StmtExprMutator {
 public:
  explicit ModSubstitue(const std::unordered_map<const VarNode*, PrimExpr>& vmap)
      : vmap_(vmap) {
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = vmap_.find(var.get());
    if (it != vmap_.end()) {
      used_vars_.insert(var.get());
      return it->second;
    }
    return std::move(var);
  }

  std::unordered_set<const VarNode*> used_vars() const {
    return used_vars_;
  }

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& vmap_;
  std::unordered_set<const VarNode*> used_vars_;
};

Stmt BindSpecializedCondition(Stmt stmt) {
  SpecializedCondition scond = SpecializedCondition::Current();
  if (!scond.defined()) {
    return stmt;
  }
  std::unordered_map<const VarNode*, ModBind> bind_map;
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  arith::PVar<Var> x;
  arith::PVar<IntImm> c1, c2;
  auto pattern = (floormod(x, c1) == c2);
  for (auto clause : scond->clauses) {
    if (pattern.Match(clause)) {
      int64_t divisor = c1.Eval()->value;
      int64_t remainder = c2.Eval()->value;
      if (divisor > 0 && remainder >= 0) {
        ModBind bind;
        bind.origin_var = x.Eval();
        bind.divisor = IntImm(bind.origin_var->dtype, divisor);
        bind.remainder = IntImm(bind.origin_var->dtype, remainder);
        if (bind.origin_var.as<SizeVarNode>()) {
          bind.bind_var = SizeVar("k");
        } else {
          bind.bind_var = Var("k");
        }
        bind_map.emplace(bind.origin_var.get(), bind);
        vmap.emplace(bind.origin_var.get(), bind.divisor * bind.bind_var + bind.remainder);
      }
    }
  }
  ModSubstitue sub(vmap);
  Stmt ret = sub(stmt);

  std::vector<Stmt> seq_init;
  std::vector<Stmt> seq_check;
  const Stmt nop = tir::EvaluateNode::make(0);
  for (auto var : sub.used_vars()) {
    auto bind = bind_map.at(var);
    seq_init.push_back(LetStmtNode::make(
        bind.bind_var, floordiv(bind.origin_var - bind.remainder, bind.divisor), nop));
    std::ostringstream os;
    os << "Specialized condition \"" << bind.origin_var << " % " << bind.divisor
       << " == " << bind.remainder << "\" is not satisfied";
    seq_check.push_back(AssertStmtNode::make(
        floormod(bind.origin_var, bind.divisor) == bind.remainder,
        StringImmNode::make(os.str()), nop));
  }

  for (auto it : seq_init) {
    const auto* let = it.as<tir::LetStmtNode>();
    CHECK(let);
    auto n = make_object<tir::LetStmtNode>(*let);
    n->body = ret;
    ret = tir::Stmt(n);
  }
  for (auto it : seq_check) {
    const auto* assert = it.as<tir::AssertStmtNode>();
    CHECK(assert);
    auto n = make_object<tir::AssertStmtNode>(*assert);
    n->body = ret;
    ret = tir::Stmt(n);
  }
  //LOG(INFO) << ret;
  
  return ret;
}

namespace transform {

Pass BindSpecializedCondition() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BindSpecializedCondition(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BindSpecializedcondition", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BindSpecializedCondition")
.set_body_typed(BindSpecializedCondition);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
