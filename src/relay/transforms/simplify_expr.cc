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
 * \file src/relay/transforms/simplify_expr.cc
 * \brief A pass for simplifying the Relay expression.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/support/logging.h>

#include "../op/tensor/transform.h"

namespace tvm {
namespace relay {

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op.
 */
class SimplifyReshape {
 public:
  SimplifyReshape() {
    x_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto reshape1 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    auto reshape2 = IsOp("reshape") || IsOp("contrib_reverse_reshape");
    pattern_ = reshape1({reshape2({x_})});
  }

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto x = node_map[x_][0];
    bool const_shape = true;
    Array<Integer> newshape;
    for (auto dim : Downcast<TensorType>(pre->checked_type())->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_shape = false;
        break;
      }
      newshape.push_back(Downcast<Integer>(dim));
    }
    if (const_shape) {
      return MakeReshape(x, newshape);
    }
    return post;
  }

  DFPattern pattern() const { return pattern_; }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Pattern for consecutive reshape or reverse_reshape ops */
  DFPattern pattern_;
};

class SimplifyCast {
 public:
  SimplifyCast() {
    x_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto cast1_ = IsOp("cast")({x_});
    auto cast2_ = IsOp("cast")({cast1_});
    pattern_ = cast2_;
  }

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto x = node_map[x_][0];
    return MakeCast(x, Downcast<TensorType>(pre->checked_type())->dtype);
  }

  DFPattern pattern() const { return pattern_; }

 private:
  DFPattern x_;
  DFPattern cast1_;
  DFPattern cast2_;
  DFPattern pattern_;
};

class SameCast {
 public:
  SameCast() {
    x_fp32_ = WildcardPattern();
    Map<String, ObjectRef> attrs;
    attrs.Set("dtype", String("float32"));
    auto cast_fp32 = IsOp("cast")({x_fp32_.HasDtype("float32")}).HasAttr(attrs);

    x_fp16_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto fp16 = DataTypePattern(x_fp16_, DataType::Float(16));
    Map<String, ObjectRef> fp16_attrs;
    fp16_attrs.Set("dtype", String("float16"));
    auto call_cast_fp16 = CallPattern(ExprPattern(cast_op), {fp16}, Attrs{}, {});
    auto cast_fp16 = AttrPattern(call_cast_fp16, DictAttrs(fp16_attrs));

    pattern_ = AltPattern(cast_fp32, cast_fp16);
  }

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    if (node_map.count(x_fp32_)) {
      return node_map[x_fp32_][0];
    }
    if (node_map.count(x_fp16_)) {
      return node_map[x_fp16_][0];
    }
    return post;
  }

  DFPattern pattern() const { return pattern_; }

 private:
  DFPattern x_fp32_, x_fp16_;
  DFPattern pattern_;
};


/*!
 * \brief ExprSimplifier simplifies the Relay expression.
 */
class ExprSimplifier {
 public:
  explicit ExprSimplifier(IRModule mod) : mod_(mod) {
    auto reshape_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = simplify_reshape_.callback(pre, post, node_map);
    };
    auto cast_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = simplify_cast_.callback(pre, post, node_map);
    };
    auto same_cast_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = same_cast_.callback(pre, post, node_map);
    };
    callbacks_.push_back(
        DFPatternCallback(simplify_reshape_.pattern(), PackedFunc(reshape_func), true));
    callbacks_.push_back(
        DFPatternCallback(simplify_cast_.pattern(), PackedFunc(cast_func), true));
    callbacks_.push_back(
        DFPatternCallback(same_cast_.pattern(), PackedFunc(same_cast_func), true));
  }

  Expr Simplify(const Expr& expr) { return RewritePatterns(callbacks_, expr, mod_); }

 private:
  IRModule mod_;
  /*! \brief Simplify reshape pattern */
  SimplifyReshape simplify_reshape_;
  SimplifyCast simplify_cast_;
  SameCast same_cast_;
  /*! \brief Callbacks for expr simplification */
  Array<DFPatternCallback> callbacks_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  return ExprSimplifier(mod).Simplify(expr);
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SimplifyExpr(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
