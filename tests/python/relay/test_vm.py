import os

import mxnet as mx
from mxnet import gluon

import tvm
import numpy as np
from tvm import relay
from tvm.relay.vm import _eval_vm, eta_expand
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude

def veval(f, *args, ctx=tvm.cpu()):
    if isinstance(f, relay.Expr):
        ex = relay.create_executor('vm', mod=relay.Module(), ctx=ctx)
        if len(args) == 0:
            return ex.evaluate(f)
        else:
            return ex.evaluate(f)(*args)
    else:
        assert isinstance(f, relay.Module), "expected expression or module"
        mod = f
        ex = relay.create_executor('vm', mod=mod, ctx=ctx)
        if len(args) == 0:
            return ex.evaluate(mod[mod.entry_func])
        else:
            return ex.evaluate(mod[mod.entry_func])(*args)

def test_shape_of():
    a = tvm.var('a')
    b = tvm.var('b')
    x = relay.var("x", shape=(a, b), dtype='float32')
    s = relay.op.shape_of(x)
    t = relay.take(s, relay.const(1))
    sb = ScopeBuilder()
    seq_length = sb.let('seq_length', t)
    with sb.if_scope(relay.equal(seq_length, relay.const(20, dtype='int32'))):
        sb.ret(seq_length)
    with sb.else_scope():
        sb.ret(seq_length)
    func = relay.Function([x], sb.get())

    x_data = np.random.rand(20,10).astype('float32')
    res = eval_vm(func, tvm.cpu(), x_data)
    print("res is {}".format(res))

def test_split():
    x = relay.var('x', shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    z = relay.concatenate([relay.TupleGetItem(y, 0)], axis=0)
    f = relay.Function([x], z)

    x_data = np.random.rand(12,).astype('float32')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), np.split(x_data, 3, axis=0)[0])

def test_id():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + x_data)

def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])

def test_cond():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = veval(f, x_data, x_data)
    np.testing.assert_allclose(res.asnumpy(), True)

    # diff
    res = veval(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), False)


def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = veval(f, x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = veval(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data)

def test_simple_call():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    mod[mod.entry_func] = relay.Function([iarg], sum_up(iarg))
    result = veval(mod, i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_count_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    mod[mod.entry_func] = relay.Function([iarg], sum_up(iarg))
    result = veval(mod, i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_sum_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    loop_bound = 0
    i_data = np.array(loop_bound, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    aarg = relay.var('accum', shape=[], dtype='int32')
    mod[mod.entry_func] = relay.Function([iarg, aarg], sum_up(iarg, aarg))
    result = veval(mod, i_data, accum_data)
    tvm.testing.assert_allclose(result.asnumpy(), sum(range(1, loop_bound + 1)))

def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = veval(f, (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_tuple_second():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = veval(f, (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), j_data)

# FIX ME
# def test_list_constructor():
#     def to_list(o):
#         if isinstance(o, tvm.relay.backend.interpreter.TensorValue):
#             return [o.data.asnumpy().tolist()]
#         if isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
#             result = []
#             for f in o.fields:
#                 result.extend(to_list(f))
#             return result

#     mod = relay.Module()
#     p = Prelude(mod)

#     nil = p.nil
#     cons = p.cons
#     l = p.l

#     one2 = cons(relay.const(1), nil())
#     one3 = cons(relay.const(2), one2)
#     one4 = cons(relay.const(3), one3)
#     f = relay.Function([], one4)

#     mod[mod.entry_func] = f

#     result = veval(mod)
#     obj = to_list(result)
#     tvm.testing.assert_allclose(obj, np.array([3,2,1]))

def test_let_tensor():
    sb = relay.ScopeBuilder()
    shape = (1,)
    x = relay.var('x', shape=shape, dtype='float32')
    x1 = relay.var('x1', shape=shape, dtype='float32')

    x1 = sb.let(x1, x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.random.rand(*shape).astype('float32')
    result = veval(f, x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def test_let_scalar():
    sb = relay.ScopeBuilder()

    x = relay.var('x', 'float32')
    x1 = sb.let('x1', x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.array(np.random.rand()).astype('float32')
    result = veval(f, x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def import_mxnet_model(fname, num_states):
    ctx = mx.context.cpu()
    data_names = ['data0']
    for i in range(num_states):
        data_names.append('data%s' % (i+1))

    model_data_dir = os.path.dirname(os.path.realpath(__file__))
    net = gluon.nn.SymbolBlock.imports("%s/model_zoo_data/%s-symbol.json.data" % (model_data_dir, fname), data_names,
                                       "%s/model_zoo_data/%s-0001.params.data" % (model_data_dir, fname), ctx=ctx)
    net.hybridize()
    return net

def test_rnn(cell_type):
    input_size = 128
    hidden_size = 128
    batch, seq_len = 1, tvm.var('seq_len')
    data_shape= (seq_len, batch, input_size)
    state_shape = (batch, hidden_size)
    num_states = 2 if cell_type == 'lstm' else 1
    mx_net = import_mxnet_model("%s_i128_h128" % cell_type, num_states)

    shapes = {'data': data_shape}
    mx_input_syms = []
    mx_input_syms.append(mx.sym.Variable("data"))
    for i in range(num_states):
        shapes['state%s' % i] = state_shape
        mx_input_syms.append(mx.sym.Variable("state%s" % i))

    mod = relay.module.Module()
    relay_net, params = relay.frontend.from_mxnet(mx_net, shape=shapes, input_symbols=mx_input_syms, module=mod)
    params = params.items()

    loop = None
    for v, func in mod.functions.items():
        if v.name_hint == 'foreach':
            loop = v
            print(relay.ir_pass.infer_type(func, mod=mod))
            # print("func params: {}".format(func.params))

    inputs = [relay.var('data')]
    for i in range(num_states):
        inputs.append(relay.var('state%s' % i))
    for name, _ in params:
        inputs.append(relay.var(name))
    mod[mod.entry_func] = relay.Function(inputs, relay.Call(relay_net, inputs))
    print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))

    l = 5
    data_v = np.random.rand(l, batch, 128).astype('float32')
    states_v = [np.random.rand(*state_shape).astype('float32') for _ in range(num_states)]
    params_v = [e[1].asnumpy() for e in params]
    print('eval vm')
    result = _eval_vm(mod, tvm.cpu(), data_v, *states_v, *params_v)
    print("Relay result is {}".format(result))

    mx_inputs = [mx.nd.array(x) for x in [data_v, *states_v]]
    mx_outputs = mx_net(*mx_inputs)
    print("======== MXNet result ==========")
    for o in mx_outputs:
        print("MXNet output : {}".format(o.asnumpy()))

def test_while():
    n = 5
    class Scan(gluon.HybridBlock):
        def hybrid_forward(self, F, data):
            def sum(state, i):
                s = state + F.take(data, i)
                return [s], [s, i + 1]
            def sum_cond(state, i):
                return i < n
            out, state = F.contrib.while_loop(sum_cond, sum,
                                              [F.zeros((1)), F.zeros((1))], max_iterations=5)
            return out, state

    data = mx.nd.arange(n)
    scan_layer = Scan()
    scan_layer.hybridize()
    scan_layer(data)
    mx_sym = scan_layer._cached_graph[1]
    mod = relay.module.Module()
    sym, _ = relay.frontend.from_mxnet(mx_sym, shape={'data': (5,)}, module=mod)
    print(sym)

    for v, func in mod.functions.items():
        if v.name_hint == 'while_loop':
            print(relay.ir_pass.infer_type(func, mod=mod))

    inputs = [relay.var('data')]
    mod[mod.entry_func] = relay.Function(inputs, relay.Call(sym, inputs))
    print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))

    data_np = np.arange(n).astype('float32')

    print('eval interpreter')
    intrp = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
    op_res = intrp.evaluate(mod.entry_func)(data_np)
    print("Interpreter result is {}".format(op_res))

    print('eval vm')
    result = _eval_vm(mod, tvm.cpu(), data_np)
    print("Relay result is {}".format(result))

    mx_inputs = [mx.nd.array(data_np)]
    mx_outputs = scan_layer(*mx_inputs)
    print("======== MXNet result ==========")
    for o in mx_outputs:
        print("MXNet output : {}".format(o))


def test_closure():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))
    res = veval(main)
    tvm.testing.assert_allclose(res.asnumpy(), 3.0)

if __name__ == "__main__":
    # test_id()
    # test_op()
    # test_cond()
    # test_simple_if()
    # test_simple_call()
    # test_count_loop()
    # test_sum_loop()
    # test_tuple_fst()
    # test_tuple_second()
    # test_let_scalar()
    # test_let_tensor()
    # test_list_constructor()
    # test_closure()
    # test_rnn('lstm')
    test_while()
