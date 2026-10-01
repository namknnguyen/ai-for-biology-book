"""Chapter 11: RNN gradient decay, SSM recurrence == convolution, parallel scan, and long-context cost."""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Vanishing / exploding gradients through time in a tanh RNN: ||d h_T / d h_0|| versus T.
#    The Jacobian is the product of diag(tanh'(a_t)) W_h; its norm scales like (spectral radius * gain)^T.
# ---------------------------------------------------------------------------------------------
d = 64
def jacobian_norm(rho, T, seed=0):
    r = np.random.default_rng(seed)
    W = r.normal(size=(d, d)) / np.sqrt(d)                       # spectral radius ~ 1 before rescaling
    W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h = r.normal(size=d) * 0.5; J = np.eye(d)
    for _ in range(T):
        a = W @ h + r.normal(size=d) * 0.3                        # input drive keeps the state active
        h = np.tanh(a); J = (np.diag(1 - h ** 2) @ W) @ J         # dh_t/dh_{t-1} = diag(tanh') W
    return np.linalg.norm(J, 2)
print("RNN gradient path ||dh_T/dh_0|| (spectral radius rho of W):")
for rho in (0.8, 1.0, 1.5, 3.0):
    print(f"  rho = {rho:3.1f}: " + "  ".join(f"T={T}: {jacobian_norm(rho, T):9.2e}" for T in (10, 50, 200)))

# ---------------------------------------------------------------------------------------------
# 2. A linear state-space model has two equivalent computations: a recurrence and a convolution.
#    h_t = Abar h_{t-1} + Bbar x_t, y_t = C h_t   <=>   y = K * x with K_j = C Abar^j Bbar.
# ---------------------------------------------------------------------------------------------
N, L, dt = 16, 512, 0.05
A = -np.exp(rng.normal(size=N)) - 0.1                              # stable diagonal continuous-time A (negative reals)
B = rng.normal(size=N); C = rng.normal(size=N)
Abar = np.exp(dt * A)                                              # zero-order-hold discretisation (diagonal case)
Bbar = (Abar - 1.0) / A * B                                        # (dt A)^{-1} (exp(dt A) - I) dt B
x = rng.normal(size=L)

h = np.zeros(N); y_rec = np.empty(L)
for t in range(L):                                                  # recurrent mode: O(1) memory per step
    h = Abar * h + Bbar * x[t]; y_rec[t] = C @ h
K = np.array([(C * Abar ** j * Bbar).sum() for j in range(L)])     # convolution kernel
n_fft = 2 * L
y_conv = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(K, n_fft), n_fft)[:L]   # parallel mode: O(L log L)
print(f"\nSSM: max |recurrent - convolutional| = {np.abs(y_rec - y_conv).max():.1e}   "
      f"(kernel decays: K[0]={K[0]:.3f}, K[100]={K[100]:.1e}, K[500]={K[500]:.1e})")

# ---------------------------------------------------------------------------------------------
# 3. Selective (input-dependent) recurrences h_t = a_t h_{t-1} + b_t can still be parallelised by an associative scan.
#    Compose affine maps: (a2, b2) o (a1, b1) = (a2 a1, a2 b1 + b2).
# ---------------------------------------------------------------------------------------------
def sequential(a, b):
    h, out = 0.0, []
    for at, bt in zip(a, b):
        h = at * h + bt; out.append(h)
    return np.array(out)

def parallel_scan(a, b):
    a, b = a.copy(), b.copy(); n = len(a); step = 1
    while step < n:                                                 # Hillis-Steele: log2(n) rounds of vectorised work
        a_prev = np.concatenate([np.ones(step), a[:-step]]); b_prev = np.concatenate([np.zeros(step), b[:-step]])
        b = a * b_prev + b; a = a * a_prev; step *= 2
    return b
a = rng.uniform(0.5, 1.0, 1000); b = rng.normal(size=1000)         # input-dependent decay a_t and drive b_t
print(f"parallel scan vs sequential recurrence: max difference = {np.abs(sequential(a, b) - parallel_scan(a, b)).max():.1e}  "
      f"({int(np.ceil(np.log2(1000)))} rounds instead of 1000 sequential steps)")

# ---------------------------------------------------------------------------------------------
# 4. What does context length cost? (per layer, per sequence; width d = 4096; fp16 attention score matrix)
# ---------------------------------------------------------------------------------------------
d_model = 4096
print("\ncost per layer of mixing information across L positions (d = 4096; illustrative orders of magnitude):")
print(f"  {'L':>10s} {'attention FLOPs 4 L^2 d':>26s} {'attn score memory (fp16)':>26s} {'long-conv FLOPs ~ 10 d L log2 L':>34s} {'SSM scan FLOPs ~ 10 d N L':>28s}")
for Lc in (8_192, 131_072, 1_000_000):
    attn = 4 * Lc ** 2 * d_model; mem = Lc ** 2 * 2 / 1e9
    conv = 10 * d_model * Lc * np.log2(Lc); ssm = 10 * d_model * 16 * Lc
    print(f"  {Lc:10,d} {attn:26.2e} {mem:23.1f} GB {conv:34.2e} {ssm:28.2e}")
