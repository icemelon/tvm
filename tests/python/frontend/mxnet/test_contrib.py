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

if __name__ == '__main__':
    #test_slice_axis()
    #test_embedding()
    test_div_sqrt_dim()
