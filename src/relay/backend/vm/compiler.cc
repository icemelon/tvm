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
 *  Copyright (c) 2019 by Contributors
 * \file src/relay/backend/vm/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/logging.h>
#include <tvm/operation.h>
#include <tvm/relay/pass.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../../../runtime/vm/naive_allocator.h"
#include "../../backend/compile_engine.h"

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;

// (@jroesch): VM passes, eventually declare as passes.
bool IsClosure(const Function& func);
Module LambdaLift(const Module& module);
Module InlinePrimitives(const Module& module);

bool IsConstantShape(const Type& ty) {
  if (auto tty = ty.as<TensorTypeNode>()) {
    for (auto sh : tty->shape) {
      if (sh.as<IntImm>() == nullptr) {
        return false;
      }
    }
    return true;
  }
  if (auto tty = ty.as<TupleTypeNode>()) {
    for (auto field : tty->fields) {
      if (!IsConstantShape(field)) {
        return false;
      }
    }
    return true;
  }
  return true;
}

template <typename T, typename U>
using NodeMap = std::unordered_map<T, U, NodeHash, NodeEqual>;
using TagMap = NodeMap<tvm::relay::Constructor, Index>;
using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;
using GlobalMap = NodeMap<GlobalVar, Index>;
using ConstMap = NodeMap<Constant, Index>;
using ConstTensorShapeMap = NodeMap<TensorType, std::pair<Index, NDArray>>;

struct VMCompilerContext {
  // The module context for the compilation
  Module module;
  // Error reporter
  ErrorReporter err_reporter;
  // Map from a unique integer to ADT constructor tag
  TagNameMap tag_index_map;
  // Map from ADT constructor tag to a unique integer
  TagMap tag_map;
  // Map from global var to a unique integer
  GlobalMap global_map;
  // Map from Const object to its index in const pool
  ConstMap const_map;
  // Map from Const tensor shape to its index in const pool
  ConstTensorShapeMap const_tensor_shape_map;
  // List of lowered functions
  std::vector<LoweredFunc> lowered_funcs;
};

// Compute the constant pool, i.e a mapping from Constant node to constant index.
struct ConstantPool : ExprVisitor {
  std::set<GlobalVar> visited;
  Module module;
  ConstMap const_map;
  ConstTensorShapeMap const_tensor_shape_map;

  size_t index;

  explicit ConstantPool(const Module& mod) : module(mod), const_map(), index(0) {}

  void VisitExpr_(const GlobalVarNode* var_node) {
    auto gvar = GetRef<GlobalVar>(var_node);
    if (visited.find(gvar) == visited.end()) {
      visited.insert(gvar);
      this->VisitExpr(this->module->Lookup(gvar));
    }
  }

  void VisitExpr_(const ConstantNode* const_node) {
    auto konst = GetRef<Constant>(const_node);
    auto it = this->const_map.find(konst);
    if (it == this->const_map.end()) {
      this->const_map.insert({konst, index++});
    }
  }
};

std::tuple<ConstMap, ConstTensorShapeMap> LayoutConstantPool(const Module& module) {
  auto cp = ConstantPool(module);
  for (auto& func : module->functions) {
    cp.VisitExpr(func.first);
  }
  return std::make_tuple(cp.const_map, cp.const_tensor_shape_map);
}

void InstructionPrint(std::ostream& os, const Instruction& instr);

struct VMCompiler : ExprFunctor<void(const Expr& expr)> {
  /*! \brief Store the expression a variable points to. */
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> expr_map;

  std::vector<Instruction> instructions;

  // var -> register num
  std::unordered_map<Var, RegName, NodeHash, NodeEqual> var_register_map;

  size_t last_register;

  // Total number of virtual registers allocated
  size_t registers_num;
  CompileEngine engine;

  /*! \brief The functions that have been lowered. */
  std::unordered_map<LoweredFunc, size_t, NodeHash, NodeEqual> seen_funcs;

  /*! \brief Global shared meta data */
  VMCompilerContext* context;

  VMCompiler(VMCompilerContext* context)
      : instructions(),
        var_register_map(),
        last_register(0),
        registers_num(0),
        engine(CompileEngine::Global()),
        context(context)
        {}

  size_t NewRegister() { return registers_num++; }

  inline void Emit(const Instruction& instr) {
    DLOG(INFO) << "VMCompiler::Emit: instr=" << instr;
    CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
    switch (instr.op) {
      case Opcode::AllocDatatype:
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::GetField:
      case Opcode::LoadConst:
      case Opcode::Select:
      case Opcode::Invoke:
      case Opcode::AllocClosure:
      case Opcode::Move:
      case Opcode::InvokeClosure:
        last_register = instr.dst;
        break;
      case Opcode::InvokePacked:
        last_register = instr.packed_args[instr.arity - 1];
        break;
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
        break;
    }
    instructions.push_back(instr);
  }

  void VisitExpr_(const ConstantNode* const_node) {
    auto rconst = GetRef<Constant>(const_node);
    auto it = this->context->const_map.find(rconst);
    CHECK(it != this->context->const_map.end());
    Emit(Instruction::LoadConst(it->second, NewRegister()));
  }

  void VisitExpr_(const VarNode* var_node) {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map.find(var);
    CHECK(reg_it != this->var_register_map.end());
    last_register = reg_it->second;
  }

  void VisitExpr_(const TupleNode* tuple_node) {
    auto tuple = GetRef<Tuple>(tuple_node);
    std::vector<Index> fields_registers;

    for (auto& field : tuple->fields) {
      this->VisitExpr(field);
      fields_registers.push_back(last_register);
    }

    // TODO(@jroesch): use correct tag
    Emit(Instruction::AllocDatatype(
      0,
      tuple->fields.size(),
      fields_registers,
      NewRegister()));
  }

  void VisitExpr_(const MatchNode* match_node) {
    auto match = GetRef<Match>(match_node);
    LOG(FATAL) << "translation of match nodes to the VM is"
               << "currently unsupported" << std::endl;
  }

  void VisitExpr_(const LetNode* let_node) {
    DLOG(INFO) << let_node->value;
    this->VisitExpr(let_node->value);
    DLOG(INFO) << this->last_register;
    var_register_map.insert({let_node->var, this->last_register});
    this->VisitExpr(let_node->body);
  }

  void VisitExpr_(const TupleGetItemNode* get_node) {
    auto get = GetRef<TupleGetItem>(get_node);
    this->VisitExpr(get->tuple);
    auto tuple_register = last_register;
    Emit(Instruction::GetField(tuple_register, get->index, NewRegister()));
  }

  void VisitExpr_(const GlobalVarNode* gvar) {
    LOG(FATAL) << "Global variables should only appear in the call position";
  }

  void VisitExpr_(const IfNode* if_node) {
    this->VisitExpr(if_node->cond);

    size_t cond_register = last_register;

    auto after_cond = this->instructions.size();

    this->Emit(Instruction::If(cond_register, 0, 0));
    this->VisitExpr(if_node->true_branch);

    size_t true_register = last_register;

    Emit(Instruction::Goto(0));

    // Finally store how many instructions there are in the
    // true branch.
    auto after_true = this->instructions.size();

    this->VisitExpr(if_node->false_branch);

    size_t false_register = last_register;

    // Compute the total number of instructions
    // after generating false.
    auto after_false = this->instructions.size();

    // Now we will compute the jump targets in order
    // to properly patch the instruction with the
    // the requiste targets.

    // After we emit the true body, and false body,
    // we patch up the if instruction, and goto.
    auto true_offset = 1;
    auto false_offset = after_true - after_cond;
    this->instructions[after_cond].true_offset = true_offset;
    this->instructions[after_cond].false_offset = false_offset;

    // Patch the Goto.
    this->instructions[after_true - 1].pc_offset = (after_false - after_true) + 1;

    Emit(Instruction::Select(cond_register, true_register, false_register, NewRegister()));
  }

  size_t EmitShapeFunc(const Type& ret_type, const Function& func, std::vector<Index>* args_registers) {
    auto call_node = func->body.as<CallNode>();
    auto op = Downcast<Op>(call_node->op);
    auto args = call_node->args;

    static auto fshape_func = Op::GetAttr<FShapeFunc>("FShapeFunc");
    CHECK_GT(fshape_func.count(op), 0) << "internal error, cannot find ShapeFunc for " << op->name;

    // Prepare input and output shapes for shape func
    Array<tvm::Tensor> shape_func_in_tensors;
    Array<Shape> shape_func_out_shapes;
    std::vector<DataType> out_types;
    for (auto arg : args) {
      auto ty = arg->checked_type().as<TensorTypeNode>();
      shape_func_in_tensors.push_back(tvm::placeholder(ty->shape, ty->dtype));
    }
    if (const auto* tuple_type = ret_type.as<TupleTypeNode>()) {
      for (auto field : tuple_type->fields) {
        const TensorTypeNode* tty = field.as<TensorTypeNode>();
        CHECK(tty);
        int64_t ndim = tty->shape.size();
        shape_func_out_shapes.push_back({Integer(ndim)});
        out_types.push_back(tty->dtype);
      }
    } else {
      auto tty = ret_type.as<TensorTypeNode>();
      CHECK(tty);
      int64_t ndim = tty->shape.size();
      shape_func_out_shapes.push_back({Integer(ndim)});
      out_types.push_back(tty->dtype);
    }

    // Lower the shape func
    auto shape_func_out_tensors = fshape_func[op](call_node->attrs, shape_func_in_tensors, shape_func_out_shapes);
    auto shape_func = LowerShapeFunc(shape_func_in_tensors, shape_func_out_tensors);
    int func_idx = -1;
    if (seen_funcs.count(shape_func) > 0) {
      func_idx = seen_funcs[shape_func];
    } else {
      func_idx = this->context->lowered_funcs.size();
      this->context->lowered_funcs.push_back(shape_func);
      seen_funcs[shape_func] = func_idx;
    }

    // Emit instructions
    std::vector<Index> shape_func_args(*args_registers);
    for (auto tensor : shape_func_out_tensors) {
      std::vector<uint64_t> shape;
      for (auto dim : tensor->shape) {
        shape.push_back(Downcast<Integer>(dim)->value);
      }
      Emit(Instruction::AllocTensor(shape, Type2TVMType(tensor->dtype), NewRegister()));
      shape_func_args.push_back(last_register);
    }
    size_t num_inputs = shape_func_in_tensors.size();
    size_t num_outputs = shape_func_out_tensors.size();
    size_t arity = shape_func_args.size();
    Emit(Instruction::InvokePacked(func_idx, arity, num_outputs, shape_func_args));
    for (size_t i = 0; i < num_outputs; ++i) {
      Emit(Instruction::AllocTensorReg(shape_func_args[num_inputs + i], Type2TVMType(out_types[i]), NewRegister()));
      args_registers->push_back(last_register);
    }
    return num_outputs;
  }

  size_t AllocReturnTensors(const Type& ret_type, const Function& func, std::vector<Index>* args_registers) {
    if (IsConstantShape(ret_type)) {
      size_t ret_num = 0;
      auto alloc_tensor = [&](const TensorTypeNode* ttype) {
        const TensorType& tensor_type = GetRef<TensorType>(ttype);
        std::vector<uint64_t> shape;
        for (auto dim : tensor_type->shape) {
          shape.push_back(Downcast<tvm::Integer>(dim)->value);
        }
        Emit(Instruction::AllocTensor(shape, Type2TVMType(tensor_type->dtype), NewRegister()));
        args_registers->push_back(last_register);
        ++ret_num;
      };
      if (const TensorTypeNode* ttype = ret_type.as<TensorTypeNode>()) {
        alloc_tensor(ttype);
      } else if (const TupleTypeNode* ttype = ret_type.as<TupleTypeNode>()) {
        for (auto field : ttype->fields) {
          alloc_tensor(field.as<TensorTypeNode>());
        }
      }
      return ret_num;
    }
    return EmitShapeFunc(ret_type, func, args_registers);
  }

  void EmitInvokePrimitive(const Function& func, std::vector<Index> args_registers,
                           const Type& ret_type) {
    std::vector<Index> unpacked_arg_regs;

    // Arity calculation must flatten tuples.
    size_t arity = 0;
    CHECK_EQ(func->params.size(), args_registers.size());
    for (size_t i = 0; i < func->params.size(); i++) {
      auto ty = func->params[i]->checked_type();
      if (ty.as<TensorTypeNode>()) {
        unpacked_arg_regs.push_back(args_registers[i]);
        arity += 1;
      } else if (auto tuple_ty = ty.as<TupleTypeNode>()) {
        for (size_t f = 0; f < tuple_ty->fields.size(); f++) {
          const auto& field = tuple_ty->fields[f];
          CHECK(field.as<TensorTypeNode>())
            << "only supports non-nested tuples currently "
            << "found " << field;
          auto dst =  NewRegister();
          Emit(Instruction::GetField(args_registers[i], f, dst));
          unpacked_arg_regs.push_back(dst);
        }
        arity += tuple_ty->fields.size();
      } else {
        LOG(FATAL) << "unsupported parameter type " << ty;
      }
    }

    size_t return_val_count = AllocReturnTensors(ret_type, func, &args_registers);
    arity += return_val_count;

    // Next generate the invoke instruction.
    CHECK(func->IsPrimitive());
    auto target = Target::Create("llvm");
    // TODO(icemelon9): Fix me
    auto key = CCacheKeyNode::make(func, {}, target);
    auto cfunc = engine->Lower(key);
    // TODO(jroesch): support lowered funcs for multiple targets
    CHECK_EQ(cfunc->funcs.size(), 1);
    auto op_index = -1;
    if (seen_funcs.find(cfunc->funcs[0]) == seen_funcs.end()) {
      op_index = this->context->lowered_funcs.size();
      this->context->lowered_funcs.push_back(cfunc->funcs[0]);
      seen_funcs[cfunc->funcs[0]] = op_index;
    } else {
      op_index = seen_funcs[cfunc->funcs[0]];
    }

    Emit(Instruction::InvokePacked(op_index, arity, return_val_count, unpacked_arg_regs));

    if (return_val_count > 1) {
      // return value is a tuple, we need to create a tuple
      std::vector<Index> fields_registers;
      for (size_t i = arity - return_val_count; i < arity; ++i) {
        fields_registers.push_back(unpacked_arg_regs[i]);
      }
      Emit(Instruction::AllocDatatype(0, return_val_count, fields_registers, NewRegister()));
    }
  }

  void VisitExpr_(const CallNode* call_node) {
    std::vector<Index> args_registers;

    for (auto arg : call_node->args) {
      CHECK(arg.as<VarNode>()) << "found: " << AsText(arg, false) << std::endl << arg;
      this->VisitExpr(arg);
      args_registers.push_back(last_register);
    }

    Expr op = call_node->op;

    if (auto func_node = op.as<FunctionNode>()) {
      CHECK(func_node->IsPrimitive());
      EmitInvokePrimitive(GetRef<Function>(func_node), args_registers, call_node->checked_type());
    } else if (auto global_node = op.as<GlobalVarNode>()) {
      auto global = GetRef<GlobalVar>(global_node);
      auto it = this->context->global_map.find(global);
      CHECK(it != this->context->global_map.end());
      DLOG(INFO) << "VisitExpr_: generating invoke for " << global->name_hint
                      << " with func_index=" << it->second;

      auto func = this->context->module->Lookup(global);
      if (IsClosure(func)) {
        auto arity = func->params.size();
        std::vector<Index> free_var_registers;
        for (size_t i = 0; i < arity; ++i) {
          free_var_registers.push_back(var_register_map.at(func->params[i]));
        }
        Emit(Instruction::AllocClosure(it->second, arity, free_var_registers, NewRegister()));
      } else {
        Emit(Instruction::Invoke(it->second, args_registers, NewRegister()));
      }
    } else if (auto constructor_node = op.as<ConstructorNode>()) {
      auto constructor = GetRef<Constructor>(constructor_node);
      auto tag = GetConstructorTag(constructor);
      Emit(Instruction::AllocDatatype(tag, call_node->args.size(), args_registers, NewRegister()));
    } else if (auto var_node = op.as<VarNode>()) {
      VisitExpr(GetRef<Var>(var_node));
      Emit(Instruction::InvokeClosure(last_register, args_registers, NewRegister()));
    } else {
      LOG(FATAL) << "unsupported case in vm compiler: " << op;
    }
  }

  size_t GetConstructorTag(tvm::relay::Constructor constructor) {
    auto it = this->context->tag_map.find(constructor);
    if (it != this->context->tag_map.end()) {
      return it->second;
    } else {
      auto tag = this->context->tag_map.size();
      this->context->tag_map[constructor] = tag;
      this->context->tag_index_map[tag] = constructor;
      return tag;
    }
  }

  void VisitExpr_(const FunctionNode* func_node) {
    if (!func_node->IsPrimitive()) {
      LOG(FATAL) << "local functions should have been removed by lambda lifting:" << std::endl
                 << "Program: " << AsText(GetRef<Function>(func_node), false) << std::endl
                 << "AST: " << GetRef<Function>(func_node);
    }
  }

  void CompileClosure(const Function& func) {
    // We first layout the function arguments.
    auto inner_func = Downcast<Function>(func->body);

    size_t i = 0;
    for (auto param : inner_func->params) {
      auto arg_register = NewRegister();
      CHECK_EQ(i, arg_register);
      var_register_map.insert({param, arg_register});
      i++;
    }

    // We then assign register num to the free variables
    for (auto param : func->params) {
      auto arg_register = NewRegister();
      CHECK_EQ(i, arg_register);
      var_register_map.insert({param, arg_register});
      i++;
    }

    // We will now process the body like normal.
    this->VisitExpr(inner_func->body);
  }

  void Compile(const Function& func) {
    // We need to generate code specially for lifted closures.
    if (IsClosure(func)) {
      CompileClosure(func);
      return;
    }

    for (size_t i = 0; i < func->params.size(); ++i) {
      auto arg_register = NewRegister();
      CHECK_EQ(arg_register, i);
      var_register_map.insert({func->params[i], arg_register});
    }

    this->VisitExpr(func->body);
  }
};

void PopulatePackedFuncMap(const std::vector<LoweredFunc>& lowered_funcs,
                           std::vector<PackedFunc>* packed_funcs) {
  runtime::Module mod;
  if (lowered_funcs.size() > 0) {
    // TODO(@jroesch): we need to read target from build config
    // TODO(@icemelon): we need to always use llvm target for shape func
    Target target = Target::Create("llvm");
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      mod = (*f)(tvm::Array<LoweredFunc>(lowered_funcs.begin(), lowered_funcs.end()), target);
    } else {
      LOG(FATAL) << "relay.backend.build is not registered";
    }
    CHECK(mod.operator->());
    for (auto lfunc : lowered_funcs) {
      packed_funcs->push_back(mod.GetFunction(lfunc->name));
    }
  }
}

VMFunction CompileFunc(VMCompilerContext* context, const GlobalVar& var, const Function& func) {
  DLOG(INFO) << "CompileFunc: " << std::endl << AsText(func, false) << std::endl;
  size_t params = func->params.size();
  VMCompiler compiler(context);
  compiler.Compile(func);
  // return the last evaluated expression
  compiler.instructions.push_back(Instruction::Ret(compiler.last_register));

  // Would like to refactor this so we only check if closure once.
  if (IsClosure(func)) {
    auto inner_params = Downcast<Function>(func->body)->params.size();
    return VMFunction(var->name_hint, params + inner_params, compiler.instructions,
                      compiler.registers_num);
  } else {
    return VMFunction(var->name_hint, params, compiler.instructions, compiler.registers_num);
  }
}

Module OptimizeModule(const Module& mod) {
  ToANormalForm(mod->entry_func, mod);
  InlinePrimitives(mod);
  LambdaLift(mod);
  return InlinePrimitives(mod);
}

void PopulateGlobalMap(GlobalMap* global_map, const Module& mod) {
  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    global_map->insert({gvar, global_index++});
  }
}

VirtualMachine CompileModule(const Module& mod_ref) {
  Module mod = mod_ref;

  // Run some optimizations first, this code should
  // be moved to pass manager.
  mod = OptimizeModule(mod);

  VirtualMachine vm;

  VMCompilerContext context;
  context.module = mod;

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap(&context.global_map, mod);

  // Next we populate constant map.
  auto constant_analysis_result = LayoutConstantPool(mod);
  context.const_map = std::get<0>(constant_analysis_result);
  context.const_tensor_shape_map = std::get<1>(constant_analysis_result);

  // Next we get ready by allocating space for
  // the global state.
  vm.functions.resize(mod->functions.size());
  vm.constants.resize(context.const_map.size() + context.const_tensor_shape_map.size());

  for (auto pair : context.const_map) {
    vm.constants[pair.second] = Object::Tensor(pair.first->data);
  }

  for (auto pair : context.const_tensor_shape_map) {
    vm.constants[pair.second.first] = Object::Tensor(pair.second.second);
  }

  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    auto func = named_func.second;
    auto vm_func = CompileFunc(&context, gvar, func);

    size_t func_index = context.global_map.at(gvar);
    CHECK(func_index < vm.functions.size());
    vm.functions[func_index] = vm_func;
  }

#ifdef USE_RELAY_DEBUG
  for (auto vm_func : vm.functions) {
    std::cout << "Function: " << vm_func.name << std::endl
              << vm_func << "-------------" << std::endl;
  }
#endif  // USE_RELAY_DEBUG

  PopulatePackedFuncMap(context.lowered_funcs, &vm.packed_funcs);

  for (auto gv : context.global_map) {
    vm.global_map_.insert({gv.first->name_hint, gv.second});
  }

  return vm;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
