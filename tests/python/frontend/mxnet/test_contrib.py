import numpy as np
import mxnet as mx
import tvm
from tvm import relay
from tvm.contrib import graph_runtime

def test_slice_axis():
    x_np = np.arange(20).reshape(4, 5).astype('float32')

    x = mx.sym.Variable(name="data")
    y = mx.sym.broadcast_axis(x, axis=(0, 2), size=(2, 3))

    yy, _ = relay.frontend.from_mxnet(y, shape={"data": (1, 2, 1)})
    yy = relay.ir_pass.infer_type(yy)
    print(yy.astext())

def test_embedding():
    data_np = np.asarray([[0, 2], [1, 3]]).astype('float32')
    weight_np = np.arange(20).reshape(4, 5).astype('float32')

    data = mx.nd.array(data_np)
    weight = mx.nd.array(weight_np)
    out_mx = mx.nd.Embedding(data, weight, input_dim=4, output_dim=5)
    print(out_mx.shape)
    print(out_mx.asnumpy())
    
    data = mx.sym.Variable(name="data")
    weight = mx.sym.Variable(name="weight")
    z = mx.sym.Embedding(data, weight, input_dim=4, output_dim=5)

    zz, _ = relay.frontend.from_mxnet(z, shape={"data": (2, 2), "weight": (4, 5)})
    print(relay.ir_pass.infer_type(zz).astext(False))
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(zz, 'llvm')
    m = graph_runtime.create(graph, lib, tvm.cpu())
    m.set_input("data", tvm.nd.array(data_np))
    m.set_input("weight", tvm.nd.array(weight_np))
    m.run()
    out_tvm = m.get_output(0, tvm.nd.empty(out_mx.shape, 'float32'))
    print(out_tvm.asnumpy())

def test_div_sqrt_dim():
    x = mx.sym.Variable(name="data")
    y = mx.sym.contrib.div_sqrt_dim(x)

    n, m = tvm.var("n"), tvm.var("m")
    yy, _ = relay.frontend.from_mxnet(y, shape={"data": (n, m)})
    yy = relay.ir_pass.infer_type(yy)
    print(yy.astext(False))

def test_var():
    shape = (3, 4, 5)
    x = relay.var("x", shape=shape, dtype="float32")
    y = relay.variance(x)
    f = relay.Function([x], y)
    print(f.astext())

    x_data = np.random.uniform(size=shape).astype('float32')
    ref_res = np.var(x_data)
    
    ctx = tvm.cpu()
    target = 'llvm'
    intrp = relay.create_executor("graph", ctx=ctx, target=target)
    mod = intrp.evaluate(f)
    op_res = mod(x_data)
    tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_layer_norm():
    dim = 16
    shape = (1, 4, dim)
    data_np = np.random.uniform(size=shape).astype("float32")
    gamma_np = np.random.uniform(size=(dim,)).astype("float32")
    beta_np = np.random.uniform(size=(dim,)).astype("float32")

    data = mx.nd.array(data_np)
    gamma = mx.nd.array(gamma_np)
    beta = mx.nd.array(beta_np)
    out_mx = mx.nd.LayerNorm(data, gamma, beta)
    # print(out_mx.shape)
    # print(out_mx.asnumpy())
    
    data = mx.sym.Variable(name="data")
    gamma = mx.sym.Variable(name="gamma")
    beta = mx.sym.Variable(name="beta")
    mx_sym = mx.sym.LayerNorm(data, gamma, beta)

    zz, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"data": shape, "gamma": (dim,), "beta": (dim,)})
    print(relay.ir_pass.infer_type(zz).astext(False))
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(zz, 'llvm')
    m = graph_runtime.create(graph, lib, tvm.cpu())
    m.set_input("data", tvm.nd.array(data_np))
    m.set_input("gamma", tvm.nd.array(gamma_np))
    m.set_input("beta", tvm.nd.array(beta_np))
    m.run()
    out_tvm = m.get_output(0, tvm.nd.empty(out_mx.shape, 'float32'))
    # print(out_tvm.asnumpy())
    tvm.testing.assert_allclose(out_tvm.asnumpy(), out_mx.asnumpy(), rtol=1e-3)

if __name__ == '__main__':
    test_slice_axis()
    test_embedding()
    test_div_sqrt_dim()
    test_var()
    test_layer_norm()
