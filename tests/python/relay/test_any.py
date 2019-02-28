import tvm
from tvm import relay
from tvm.relay import Kind
import numpy as np

def test_dyn_arange():
    m, n, k = relay.TypeVar('m', Kind.ShapeVar), relay.TypeVar('n', Kind.ShapeVar), relay.TypeVar('k', Kind.ShapeVar)
    # m, n, k = tvm.var('m'), tvm.var('n'), tvm.var('k')
    x = relay.var('x', shape=(m.var, n.var, k.var), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, 'int32'))
    y2 = relay.op.arange(y1)
    print(y2)
    ex = relay.create_executor()
    f = relay.Function([x], y2, type_params=[m, n, k])
    data = np.random.rand(10, 5, 3).astype('float32')
    result = ex.evaluate(f)(data)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_dyn_arange()
