"""Chapter 9: reverse-mode autodiff from scratch, manual batched backprop, and initialisation.

Part A: a scalar autograd engine (about 60 lines) checked against finite differences, then used to train a tiny MLP.
Part B: batched manual backprop for an MLP with ReLU + softmax cross-entropy, checked against PyTorch autograd.
Part C: why initialisation scale matters: activations and gradients through 50 layers.
"""
import math
import numpy as np
import torch

# =============================================================================================
# Part A. Scalar reverse-mode autodiff.
# =============================================================================================
class Value:
    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, backward

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw():                                   # d(out)/d(self) = d(out)/d(o) = 1
            self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw; return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        def bw(): self.grad += (self.data > 0) * out.grad
        out._backward = bw; return out

    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,))
        def bw(): self.grad += e * out.grad
        out._backward = bw; return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def bw(): self.grad += out.grad / self.data
        out._backward = bw; return out

    __radd__ = __add__; __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o if isinstance(o, Value) else Value(-o))
    def __rsub__(self, o): return (-self) + o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else Value(1.0 / o))

    def backward(self):
        order, seen = [], set()
        def topo(v):                                # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents: topo(p)
                order.append(v)
        topo(self)
        self.grad = 1.0                             # d(loss)/d(loss)
        for v in reversed(order): v._backward()     # children before parents: the reverse sweep

def f(a, b, c):
    return (a * b + c).tanh() * (a - c).relu() + (b ** 2) / (1.5 + a)

a, b, c = Value(0.8), Value(-1.3), Value(0.4)
out = f(a, b, c); out.backward()
def f_num(x):                                       # same function on floats
    a, b, c = x
    return math.tanh(a * b + c) * max(0.0, a - c) + b ** 2 / (1.5 + a)
x0 = np.array([0.8, -1.3, 0.4]); num = []
for i in range(3):
    e = np.zeros(3); e[i] = 1e-6
    num.append((f_num(x0 + e) - f_num(x0 - e)) / 2e-6)
print("A. autograd:", [round(v.grad, 6) for v in (a, b, c)], " finite differences:", [round(float(v), 6) for v in num])

# a tiny MLP built from Value objects, trained on two noisy spirals
import random; random.seed(0)
class Neuron:
    def __init__(self, n_in, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]; self.b = Value(0.0); self.act = act
    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return s.tanh() if self.act else s
    def params(self): return self.w + [self.b]
class MLP:
    def __init__(self, sizes):
        self.layers = [[Neuron(sizes[i], act=i < len(sizes) - 2) for _ in range(sizes[i + 1])] for i in range(len(sizes) - 1)]
    def __call__(self, x):
        for layer in self.layers: x = [n(x) for n in layer]
        return x[0]
    def params(self): return [p for layer in self.layers for n in layer for p in n.params()]

pts, labels = [], []
for i in range(30):
    t = i / 30 * 3.0
    for sign in (1, -1):
        pts.append([sign * t * math.cos(t) / 3 + random.gauss(0, 0.05), sign * t * math.sin(t) / 3 + random.gauss(0, 0.05)])
        labels.append(sign)
net = MLP([2, 8, 8, 1])
for step in range(60):
    loss = sum(((net(x) - y) ** 2 for x, y in zip(pts, labels)), Value(0.0)) / len(pts)
    for p in net.params(): p.grad = 0.0
    loss.backward()
    for p in net.params(): p.data -= 0.3 * p.grad
acc = sum((net(x).data > 0) == (y > 0) for x, y in zip(pts, labels)) / len(pts)
print(f"   scalar-autograd MLP (2-8-8-1, {len(net.params())} params): final loss {loss.data:.3f}, train accuracy {acc:.2f}")

# =============================================================================================
# Part B. Manual batched backprop vs torch autograd (ReLU MLP, softmax cross-entropy).
# =============================================================================================
rng = np.random.default_rng(0)
B, d_in, d_h, K = 16, 10, 12, 4
X = rng.normal(size=(B, d_in)); y = rng.integers(0, K, B)
W1, b1 = rng.normal(size=(d_h, d_in)) * 0.5, rng.normal(size=d_h) * 0.1
W2, b2 = rng.normal(size=(K, d_h)) * 0.5, rng.normal(size=K) * 0.1

# forward  (shapes: X (B,d_in), Z1 (B,d_h), A1 (B,d_h), Z2 (B,K))
Z1 = X @ W1.T + b1; A1 = np.maximum(Z1, 0); Z2 = A1 @ W2.T + b2
P = np.exp(Z2 - Z2.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
loss = -np.log(P[np.arange(B), y]).mean()
# backward: delta = dL/dZ at each layer
D2 = P.copy(); D2[np.arange(B), y] -= 1; D2 /= B           # softmax-CE: (p - onehot)/B       (B,K)
gW2 = D2.T @ A1; gb2 = D2.sum(0)                            # dL/dW2 = Delta^T A_prev          (K,d_h)
D1 = (D2 @ W2) * (Z1 > 0)                                   # delta^(l) = (W^T delta^(l+1)) * relu'(z)  (B,d_h)
gW1 = D1.T @ X; gb1 = D1.sum(0)

tW1, tb1, tW2, tb2 = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
tl = torch.nn.functional.cross_entropy(torch.relu(torch.tensor(X) @ tW1.T + tb1) @ tW2.T + tb2, torch.tensor(y))
tl.backward()
errs = [np.abs(g - t.grad.numpy()).max() for g, t in ((gW1, tW1), (gb1, tb1), (gW2, tW2), (gb2, tb2))]
print(f"B. manual backprop vs torch autograd: loss {loss:.6f} vs {tl.item():.6f}; max gradient difference = {max(errs):.1e}")

# =============================================================================================
# Part C. Initialisation: forward activation scale and backward gradient scale through 50 ReLU layers.
# =============================================================================================
def depth_experiment(scale_fn, depth=50, width=256):
    r = np.random.default_rng(1)
    Ws = [r.normal(size=(width, width)) * scale_fn(width) for _ in range(depth)]
    h = r.normal(size=(64, width)); masks = []
    for W in Ws:
        z = h @ W.T; masks.append(z > 0); h = np.maximum(z, 0)
    act_rms = np.sqrt((h ** 2).mean())
    g = np.ones((64, width))                               # a gradient arriving at the top
    for W, m in zip(reversed(Ws), reversed(masks)):
        g = (g * m) @ W
    return act_rms, np.sqrt((g ** 2).mean())
print("C. RMS activation at layer 50 and RMS gradient at layer 1 (width 256, ReLU):")
for name, fn in (("N(0,1)          ", lambda n: 1.0), ("N(0,1/n) (Xavier-like)", lambda n: np.sqrt(1.0 / n)), ("N(0,2/n) (He)   ", lambda n: np.sqrt(2.0 / n))):
    a_rms, g_rms = depth_experiment(fn)
    print(f"   {name:24s} activation RMS = {a_rms:10.3e}   gradient RMS = {g_rms:10.3e}")
