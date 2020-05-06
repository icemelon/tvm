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
 * \file tvm/arith/symbolic_bound.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr_functor.h>
#include <algorithm>
#include "int_operator.h"
#include "pattern_match.h"
#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

TVM_REGISTER_NODE_TYPE(SymbolicBoundNode);

const PrimExpr SymbolicBound::kUnknown = Var("unknown");

SymbolicBound::SymbolicBound(PrimExpr lower_bound, PrimExpr upper_bound) {
  auto node = make_object<SymbolicBoundNode>();
  node->lower_bound = lower_bound;
  node->upper_bound = upper_bound;
  data_ = std::move(node);
}

//TVM_REGISTER_GLOBAL("arith.SymbolicBound")
//.set_body_typed(MakeConstIntBound);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SymbolicBoundNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const SymbolicBoundNode*>(node.get());
    p->stream << "SymbolicBound[";
    p->stream << op->lower_bound;
    p->stream << ',';
    p->stream << op->upper_bound;
    p->stream << ']';
});

class RelaxRewriteSimplifier : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;

  explicit RelaxRewriteSimplifier(Analyzer* parent)
      : IRMutatorWithAnalyzer(parent) {}

  PrimExpr VisitExpr_(const MulNode* op) override {
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    PVar<PrimExpr> x;
    PVar<IntImm> c1;
    if ((floordiv(x, c1) * c1).Match(ret)) {
      if (analyzer_->CanProveGreaterEqual(x.Eval(), 0) && c1.Eval()->value >= 0) {
        return x.Eval();
      }
    }
    return ret;
  }
};

// internal entry for const int bound
struct SymbolicBoundAnalyzer::Entry {
  PrimExpr lower_bound;
  PrimExpr upper_bound;

  bool is_const() const {
    if (const IntImmNode* lb = lower_bound.as<IntImmNode>()) {
      if (const IntImmNode* ub = upper_bound.as<IntImmNode>()) {
        return lb->value == ub->value;
      }
    }
    return false;
  }

  bool operator==(const Entry& other) const {
    return (StructuralEqual()(lower_bound, other.lower_bound) &&
            StructuralEqual()(upper_bound, other.upper_bound));
  }
};

class SymbolicBoundAnalyzer::Impl :
    public ExprFunctor<SymbolicBoundAnalyzer::Entry(const PrimExpr&)> {
 public:
  /*! \brief additional bound info about expr \in bound */
//  struct BoundInfo {
//    /*! \brief The expr */
//    PrimExpr expr;
//    /*! \brief The additional bound */
//    Entry bound;BoundInfo
//
//    BoundInfo() {}
//    BoundInfo(PrimExpr expr, Entry bound)
//        : expr(expr), bound(bound) {
//    }
//  };
  Impl(Analyzer* parent) : analyzer_(parent) {}

  void Bind(const Var& var, const Range& range, bool override) {
    Entry a = VisitExpr(range->min);
    Entry b = VisitExpr(range->extent);
    Entry ret;
    ret.lower_bound = a.lower_bound;
    ret.upper_bound = analyzer_->Simplify(a.upper_bound + b.upper_bound - 1);
    Update(var, ret, override);
  }

  void Update(const Var& var,
              const Entry& info,
              bool override) {
    if (!override) {
      auto it = var_map_.find(var);
      if (it != var_map_.end()) {
        CHECK(it->second == info)
          << "Trying to update var \'" << var << "\'"
          << " with a different const bound: "
          << "original=" << SymbolicBound(it->second.lower_bound, it->second.upper_bound)
          << ", new=" << SymbolicBound(info.lower_bound, info.upper_bound);
      }
    }
    var_map_[var] = info;
  }

  void Update(const Var& var,
              const SymbolicBound& info,
              bool override) {
    Update(var, MakeBound(info->lower_bound, info->upper_bound), override);
  }

  Entry MakeUnknownBound() const {
    return MakeBound(SymbolicBound::kUnknown, SymbolicBound::kUnknown);
  }

  // Override visitor behaviors
  Entry VisitExprDefault_(const Object* op) final {
    return MakeUnknownBound();
  }

  Entry VisitExpr(const PrimExpr& expr) final {
    Entry res = ExprFunctor::VisitExpr(expr);
    return res;
  }

//  Entry VisitExpr_(const RampNode* op) final {
//    // op = {base + i * stride | 0 <= i < lanes}
//    // Entry(op) = Union(Entry(base + i * stride) | 0 <= i < lanes)
//    // Note that `base + i * stride` is linear w.r.t. `i`
//    // Entry(op) = Union(Entry(base + i * stride) | i = 0, i = lanes-1)
//    Entry a = VisitExpr(op->base);
//    Entry b = VisitExpr(op->base + (op->lanes - 1) * op->stride);
//    return Union(a, b);
//  }

//  Entry VisitExpr_(const CastNode* op) final {
//    Entry a = VisitExpr(op->value);
//    Entry b = Everything(op->dtype);
//    return Intersect(a, b);
//  }

  Entry VisitExpr_(const IntImmNode* op) final {
    IntImm i = GetRef<IntImm>(op);
    return MakeBound(i, i);
  }

  Entry VisitExpr_(const AddNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    if (a.lower_bound.same_as(SymbolicBound::kUnknown) || b.lower_bound.same_as(SymbolicBound::kUnknown)) {
      ret.lower_bound = SymbolicBound::kUnknown;
    } else {
      ret.lower_bound = analyzer_->Simplify(a.lower_bound + b.lower_bound);
    }
    if (a.upper_bound.same_as(SymbolicBound::kUnknown) || b.upper_bound.same_as(SymbolicBound::kUnknown)) {
      ret.upper_bound = SymbolicBound::kUnknown;
    } else {
      ret.upper_bound = analyzer_->Simplify(a.upper_bound + b.upper_bound);
    }
    return ret;
  }

  Entry VisitExpr_(const SubNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    if (a.lower_bound.same_as(SymbolicBound::kUnknown) || b.upper_bound.same_as(SymbolicBound::kUnknown)) {
      ret.lower_bound = SymbolicBound::kUnknown;
    } else {
      ret.lower_bound = analyzer_->Simplify(a.lower_bound - b.upper_bound);
    }
    if (a.upper_bound.same_as(SymbolicBound::kUnknown) || b.lower_bound.same_as(SymbolicBound::kUnknown)) {
      ret.upper_bound = SymbolicBound::kUnknown;
    } else {
      ret.upper_bound = analyzer_->Simplify(a.upper_bound - b.lower_bound);
    }
    return ret;
  }

  Entry VisitExpr_(const MulNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret = MakeUnknownBound();
    if (a.is_const()) {
      auto a_val = Downcast<IntImm>(a.lower_bound);
      if (a_val->value > 0) {
        ret.lower_bound = analyzer_->Simplify(a_val * b.lower_bound);
        ret.upper_bound = analyzer_->Simplify(a_val * b.upper_bound);
      } else if (a_val->value < 0) {
        ret.lower_bound = analyzer_->Simplify(a_val * b.upper_bound);
        ret.upper_bound = analyzer_->Simplify(a_val * b.lower_bound);
      } else {
        ret.lower_bound = a_val;  // 0
        ret.upper_bound = a_val;  // 0
      }
    } else if (b.is_const()) {
      auto b_val = Downcast<IntImm>(b.lower_bound);
      if (b_val->value > 0) {
        ret.lower_bound = analyzer_->Simplify(a.lower_bound * b_val);
        ret.upper_bound = analyzer_->Simplify(a.upper_bound * b_val);
      } else if (b_val->value < 0) {
        ret.lower_bound = analyzer_->Simplify(a.upper_bound * b_val);
        ret.upper_bound = analyzer_->Simplify(a.lower_bound * b_val);
      } else {
        ret.lower_bound = b_val;  // 0
        ret.upper_bound = b_val;  // 0
      }
    } else if (analyzer_->CanProveGreaterEqual(a.lower_bound, 0) &&
               analyzer_->CanProveGreaterEqual(b.lower_bound, 0)) {
      ret.lower_bound = analyzer_->Simplify(a.lower_bound * b.lower_bound);
      if (a.upper_bound.same_as(SymbolicBound::kUnknown) || b.lower_bound.same_as(SymbolicBound::kUnknown)) {
        ret.upper_bound = SymbolicBound::kUnknown;
      } else {
        ret.upper_bound = analyzer_->Simplify(a.upper_bound * b.upper_bound);
      }
    }
    ret.upper_bound = analyzer_->Simplify(RelaxRewriteSimplifier(analyzer_)(ret.upper_bound));
    return ret;
  }

  Entry VisitExpr_(const DivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      auto b_val = Downcast<IntImm>(b.lower_bound);
      CHECK_NE(b_val->value, 0);
      Entry ret;
      if (b_val->value > 0) {
        if (a.lower_bound.same_as(SymbolicBound::kUnknown)) {
          ret.lower_bound = SymbolicBound::kUnknown;
        } else {
          ret.lower_bound = analyzer_->Simplify(div(a.lower_bound, b_val));
        }
        if (a.upper_bound.same_as(SymbolicBound::kUnknown)) {
          ret.upper_bound = SymbolicBound::kUnknown;
        } else {
          ret.upper_bound = analyzer_->Simplify(div(a.upper_bound, b_val));
        }
      } else {
        if (a.upper_bound.same_as(SymbolicBound::kUnknown)) {
          ret.lower_bound = SymbolicBound::kUnknown;
        } else {
          ret.lower_bound = analyzer_->Simplify(div(a.upper_bound, b_val));
        }
        if (a.lower_bound.same_as(SymbolicBound::kUnknown)) {
          ret.upper_bound = SymbolicBound::kUnknown;
        } else {
          ret.upper_bound = analyzer_->Simplify(div(a.lower_bound, b_val));
        }
      }
      return ret;
    }
    return MakeUnknownBound();
  }

  Entry VisitExpr_(const FloorDivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      auto b_val = Downcast<IntImm>(b.lower_bound);
      CHECK_NE(b_val->value, 0);
      Entry ret;
      if (b_val->value > 0) {
        if (a.lower_bound.same_as(SymbolicBound::kUnknown)) {
          ret.lower_bound = SymbolicBound::kUnknown;
        } else {
          ret.lower_bound = analyzer_->Simplify(floordiv(a.lower_bound, b_val));
        }
        if (a.upper_bound.same_as(SymbolicBound::kUnknown)) {
          ret.upper_bound = SymbolicBound::kUnknown;
        } else {
          ret.upper_bound = analyzer_->Simplify(floordiv(a.upper_bound, b_val));
        }
      } else {
        if (a.upper_bound.same_as(SymbolicBound::kUnknown)) {
          ret.lower_bound = SymbolicBound::kUnknown;
        } else {
          ret.lower_bound = analyzer_->Simplify(floordiv(a.upper_bound, b_val));
        }
        if (a.lower_bound.same_as(SymbolicBound::kUnknown)) {
          ret.upper_bound = SymbolicBound::kUnknown;
        } else {
          ret.upper_bound = analyzer_->Simplify(floordiv(a.lower_bound, b_val));
        }
      }
      return ret;
    }
    return MakeUnknownBound();
  }

  Entry VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return MakeBound(v, v);
    }
  }

  Entry VisitExpr_(const SizeVarNode* op) final {
    SizeVar v = GetRef<SizeVar>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return MakeBound(v, v);
    }
  }

//  Entry VisitExpr_(const ModNode* op) final {
//    Entry a = VisitExpr(op->a);
//    Entry b = VisitExpr(op->b);
//    if (b.min_value > 0) {
//      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
//      if (a.min_value >= 0) {
//        // 0 <= [a_min, a_max] < b_min
//        if (a.max_value < b.min_value) return a;
//        // other case, we can get close to 0
//        return MakeBound(0,
//                         std::min(a.max_value, b_max_cap));
//      } else {
//        return MakeBound(std::max(a.min_value, -b_max_cap),
//                         std::min(std::max(a.max_value, (int64_t)0), b_max_cap));
//      }
//    } else {
//      CHECK(!b.is_const(0)) << "mod by zero";
//      // mod by negative value is rare,
//      // and we just use the simpliest rule.
//      return Everything(op->dtype);
//    }
//  }
//
//  Entry VisitExpr_(const FloorModNode* op) final {
//    Entry a = VisitExpr(op->a);
//    Entry b = VisitExpr(op->b);
//    if (analyzer_->CanProveGreaterEqual(b.lower_bound, 0)) {
//      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
//      if (a.min_value >= 0) {
//        // 0 <= [a_min, a_max] < b_min
//        if (a.max_value < b.min_value) return a;
//        // other case, we can get close to 0
//        return MakeBound(0, std::min(a.max_value, b_max_cap));
//      } else {
//        return MakeBound(0, b_max_cap);
//      }
//    } else {
//      CHECK(!b.is_const(0)) << "floormod by zero";
//      // mod by negative value is rare,
//      // and we just use the simpliest rule.
//      return Everything(op->dtype);
//    }
//  }
//
//  Entry VisitExpr_(const MinNode* op) final {
//    Entry a = VisitExpr(op->a);
//    Entry b = VisitExpr(op->b);
//    Entry ret;
//    ret.min_value = std::min(a.min_value, b.min_value);
//    ret.max_value = std::min(a.max_value, b.max_value);
//    return ret;
//  }
//
//  Entry VisitExpr_(const MaxNode* op) final {
//    Entry a = VisitExpr(op->a);
//    Entry b = VisitExpr(op->b);
//    Entry ret;
//    ret.min_value = std::max(a.min_value, b.min_value);
//    ret.max_value = std::max(a.max_value, b.max_value);
//    return ret;
//  }
//
//  Entry VisitExpr_(const SelectNode* op) final {
//    Entry a = VisitExpr(op->true_value);
//    Entry b = VisitExpr(op->false_value);
//    return Union(a, b);
//  }
//
//  Entry VisitExpr_(const CallNode* op) final {
//    // only special handle >> and & which can be
//    // used for index calculation.
//    if (op->is_intrinsic(CallNode::shift_right)) {
//      return VisitRightShift(op);
//    } else if (op->is_intrinsic(CallNode::bitwise_and)) {
//      return VisitBitwiseAnd(op);
//    } else {
//      return Everything(op->dtype);
//    }
//  }
//
//  Entry VisitRightShift(const CallNode* op) {
//    Entry a = VisitExpr(op->args[0]);
//    Entry b = VisitExpr(op->args[1]);
//    return BinaryOpBoundry(a, b, InfAwareRightShift);
//  }
//
//  Entry VisitBitwiseAnd(const CallNode* op) {
//    Entry a = VisitExpr(op->args[0]);
//    Entry b = VisitExpr(op->args[1]);
//    // handle positive index case.
//    if (a.min_value >= 0 && b.min_value >= 0) {
//      return MakeBound(0, std::min(a.max_value, b.max_value));
//    } else {
//      if (b.min_value >= 0) {
//        return MakeBound(0, b.max_value);
//      }
//      if (a.min_value >= 0) {
//        return MakeBound(0, a.max_value);
//      }
//      return Everything(op->dtype);
//    }
//  }
//
//  std::function<void()> EnterConstraint(const PrimExpr& constraint) {
//    std::vector<BoundInfo> info = DetectBoundInfo(constraint);
//    if (info.size() == 0) return nullptr;
//    size_t old_size = additional_info_.size();
//    additional_info_.insert(additional_info_.end(), info.begin(), info.end());
//    size_t new_size = old_size + info.size();
//    auto frecover = [old_size, new_size, this]() {
//        CHECK_EQ(additional_info_.size(), new_size);
//        additional_info_.resize(old_size);
//    };
//    return frecover;
//  }

 private:
  friend class SymbolicBoundAnalyzer;

  Analyzer* analyzer_;
  // internal variable map
  std::unordered_map<Var, Entry, ObjectHash, ObjectEqual> var_map_;
  // look up table for memorization
  std::unordered_map<const PrimExprNode*, SymbolicBound>* bound_{nullptr};
  // constants: the limit value means umlimited
  // NOTE: kNegInf/kPosInf are used to represent infinity.
//  using SymbolicBound::kPosInf;
//  using SymbolicBound::kNegInf;
//  static const constexpr int64_t kNegInf = ConstIntBound::kNegInf;
//  static const constexpr int64_t kPosInf = ConstIntBound::kPosInf;
//  static_assert(-kNegInf == kPosInf, "invariant of inf");
  // internal helper functions
  /*!
   * \brief Make a new bound entry.
   */
  static Entry MakeBound(PrimExpr lower_bound, PrimExpr upper_bound) {
    Entry e;
    e.lower_bound = lower_bound;
    e.upper_bound = upper_bound;
    return e;
  }

#if 0
  /*!
   * \brief Get boundary of binary op who are monotonic wrt to one argument.
   * \param param a The entry of the left operand.
   * \param param a The entry of the right operand.
   * \param op The operator.
   * \tparam F the operator function type.
   * \return The result.
   */
  template<typename F>
  static Entry BinaryOpBoundry(Entry a, Entry b, const F& op) {
    Entry ret;
    // The boundary point must be shihft of the original boundary.
    int64_t v1 = op(a.min_value, b.min_value);
    int64_t v2 = op(a.max_value, b.max_value);
    int64_t v3 = op(a.min_value, b.max_value);
    int64_t v4 = op(a.max_value, b.min_value);
    ret.min_value = std::min(std::min(std::min(v1, v2), v3), v4);
    ret.max_value = std::max(std::max(std::max(v1, v2), v3), v4);
    return ret;
  }
  /*!
   * \brief Compute x + y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static PrimExpr InfAwareAdd(PrimExpr x, PrimExpr y) {
    if (x == SymbolicBound::kPosInf) {
      CHECK(y != SymbolicBound::kNegInf);
      return SymbolicBound::kPosInf;
    }
    if (x == kNegInf) {
      CHECK(y != kPosInf);
      return kNegInf;
    }
    if (y == kPosInf || y == kNegInf) return y;
    if (WillOverflow<AddNode>(x, y, kNegInf, kPosInf)) {
      if (x > 0) return kPosInf;
      return kNegInf;
    }
    return x + y;
  }
  /*!
   * \brief Compute x * y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareMul(int64_t x, int64_t y) {
    if (!WillOverflow<MulNode>(x, y, kNegInf, kPosInf)) return x * y;
    if ((x > 0 && y > 0) || (x < 0 && y < 0)) return kPosInf;
    return kNegInf;
  }
  /*!
   * \brief Compute x / y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareDiv(int64_t x, int64_t y) {
    CHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return x / y;
  }
  /*!
   * \brief Compute floodiv(x, y), aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareFloorDiv(int64_t x, int64_t y) {
    CHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return floordiv(x, y);
  }
  /*!
   * \brief Compute x / y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareRightShift(int64_t x, int64_t y) {
    if (x == kPosInf || x == kNegInf) return x;
    return x >> y;
  }
  /*!
   * \brief Create union of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Union(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::min(a.min_value, b.min_value);
    ret.max_value = std::max(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief Create intersect of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::max(a.min_value, b.min_value);
    ret.max_value = std::min(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief return everything dtype can represent.
   * \param dtype The data type.
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Everything(DataType dtype) {
    if (!dtype.is_int() && !dtype.is_uint()) {
      return MakeBound(kNegInf, kPosInf);
    }
    Entry ret;
    int64_t vbits = dtype.bits() - static_cast<int>(dtype.is_int());
    if (dtype.is_uint()) {
      ret.min_value = 0;
    } else {
      if (vbits >= 63) {
        ret.min_value = kNegInf;
      } else {
        ret.min_value = -(static_cast<int64_t>(1) << vbits);
      }
    }
    if (vbits >= 63) {
      ret.max_value = kPosInf;
    } else {
      ret.max_value = (static_cast<int64_t>(1) << vbits) - 1;
    }
    return ret;
  }

  /*!
   * \brief Detect additional constant bound from cond, if any
   * \param cond The constraint condition.
   * \return List of detected bounds.
   */
  static std::vector<BoundpInfo> DetectBoundInfo(const PrimExpr& cond) {
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    // NOTE: canonical form always use <= or <
    if ((c <= x).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(c.Eval()->value, kPosInf))};
    }
    if ((c < x).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(c.Eval()->value + 1, kPosInf))};
    }
    if ((x <= c).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(kNegInf, c.Eval()->value))};
    }
    if ((x < c).Match(cond)) {
      return {BoundInfo(x.Eval(), MakeBound(kNegInf, c.Eval()->value - 1))};
    }
    if ((x && y).Match(cond)) {
      auto ret1 = DetectBoundInfo(x.Eval());
      auto ret2 = DetectBoundInfo(y.Eval());
      ret1.insert(ret1.end(), ret2.begin(), ret2.end());
      return ret1;
    }
    return {};
  }
#endif
};

SymbolicBound SymbolicBoundAnalyzer::operator()(const PrimExpr& expr) {
  Entry ret = impl_->VisitExpr(expr);
  return SymbolicBound(ret.lower_bound, ret.upper_bound);
}

void SymbolicBoundAnalyzer::Update(const Var& var,
                                   const SymbolicBound& info,
                                   bool override) {
  impl_->Update(var, info, override);
}

void SymbolicBoundAnalyzer::Bind(const Var& var, const Range& range, bool override) {
  impl_->Bind(var, range, override);
}

SymbolicBoundAnalyzer::SymbolicBoundAnalyzer(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

SymbolicBoundAnalyzer::~SymbolicBoundAnalyzer() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm

