import os
import numpy as np
import mxnet as mx
from mxnet import gluon
import tvm
from tvm import relay
from tvm.relay.vm import _eval_vm, eta_expand

def import_mxnet_model(fname, num_states):
    ctx = mx.context.cpu()
    data_names = ['data0']
    for i in range(num_states):
        data_names.append('data%s' % (i+1))

    model_data_dir = os.path.dirname(os.path.realpath(__file__))
    net = gluon.nn.SymbolBlock.imports("%s/model_zoo_data/%s-symbol.json" % (model_data_dir, fname), data_names,
                                       "%s/model_zoo_data/%s-0001.params" % (model_data_dir, fname), ctx=ctx)
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

    # mod = relay.module.Module()
    mod, params = relay.frontend.from_mxnet(mx_net, shape=shapes, input_symbols=mx_input_syms)
    # print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))
    # loop = None
    # for v, func in mod.functions.items():
    #     if v.name_hint == 'foreach':
    #         loop = v
    #         print(relay.ir_pass.infer_type(func, mod=mod))

    l = 5
    data_v = np.random.rand(l, batch, 128).astype('float32')
    states_v = {}
    for i in range(num_states):
        states_v['state%s' % i] = np.random.rand(*state_shape).astype('float32')
    print('eval relay')
    intrp = relay.create_executor('debug', mod=mod, ctx=tvm.cpu(), target='llvm')
    result = intrp.evaluate(mod[mod.entry_func])(data=data_v, **states_v, **params)
    print("Relay result is {}".format(result))

    mx_inputs = [mx.nd.array(data_v)]
    for n in states_v:
        mx_inputs.append(mx.nd.array(states_v[n]))
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
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape={'data': (5,)})
    # print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))
    # for v, func in mod.functions.items():
    #     if v.name_hint == 'while_loop':
    #         print(relay.ir_pass.infer_type(func, mod=mod))

    data_np = np.arange(n).astype('float32')
    print('eval interpreter')
    intrp = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
    op_res = intrp.evaluate(mod.entry_func)(data_np)
    print("Interpreter result is {}".format(op_res))

    # print('eval vm')
    # result = _eval_vm(mod, tvm.cpu(), data_np)
    # print("Relay result is {}".format(result))

    mx_inputs = [mx.nd.array(data_np)]
    mx_outputs = scan_layer(*mx_inputs)
    print("======== MXNet result ==========")
    for o in mx_outputs:
        print("MXNet output : {}".format(o))


def test_cond():
    x = mx.sym.var('x')
    y = mx.sym.var('y')
    z = mx.sym.var('z')
    def then_func():
        return x + z
    def else_func():
        return mx.sym.square(y)
    out = mx.sym.contrib.cond(mx.sym.min(x) < mx.sym.min(y), then_func, else_func)

    relay.frontend.from_mxnet(out)

if __name__ == "__main__":
    test_rnn('lstm')
    # test_while()
    # test_cond()
