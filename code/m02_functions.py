"""Chapter M2: exponentials, logarithms, power laws, Hill curves, and numerically safe log-sum-exp."""
import math

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Exponential growth: PCR doubles the template each cycle, N_c = N_0 * (1 + E)^c with efficiency E <= 1.
#    Taking logs turns the curve into a straight line whose slope gives the efficiency.
# ---------------------------------------------------------------------------------------------
N0, E_true, cycles = 100.0, 0.92, np.arange(0, 31)
N = N0 * (1 + E_true) ** cycles
noisy = N * np.exp(rng.normal(0, 0.03, size=N.size))
slope, intercept = np.polyfit(cycles, np.log10(noisy), 1)
print(f"PCR: true efficiency {E_true:.2f}; estimated from the log-line slope {10 ** slope - 1:.3f}; "
      f"cycles for a 1e6-fold increase: {math.log(1e6) / math.log(1 + E_true):.1f}")
print(f"doubling time with 100% efficiency: 1 cycle;  with {E_true:.0%}: {math.log(2) / math.log(1 + E_true):.2f} cycles")

# half-life: exp(-k t); k = ln2 / t_half
t_half = 4.0
k = math.log(2) / t_half
print(f"mRNA with half-life {t_half} h: decay rate k = {k:.4f} /h; fraction left after 12 h = {math.exp(-k * 12):.3f} (= 1/8)")

# ---------------------------------------------------------------------------------------------
# 2. Power laws: Kleiber's law, metabolic rate ~ mass^(3/4). On log-log axes a power law is a straight line.
# ---------------------------------------------------------------------------------------------
mass = np.logspace(-3, 6, 60)                       # kg, from a shrew to a whale
rate = 3.4 * mass ** 0.75 * np.exp(rng.normal(0, 0.15, size=mass.size))
b, a = np.polyfit(np.log(mass), np.log(rate), 1)
print(f"\npower law: fitted exponent {b:.3f} (true 0.75), prefactor {math.exp(a):.2f} (true 3.40)")
print("a 10,000-fold increase in mass multiplies the rate by", f"{10 ** (4 * 0.75):.0f}", "(not 10,000)")

# ---------------------------------------------------------------------------------------------
# 3. The Hill function: fraction of a promoter bound/active as a function of transcription-factor level x.
#    f(x) = x^n / (K^n + x^n): K is the half-saturation level, n the steepness (cooperativity).
# ---------------------------------------------------------------------------------------------
def hill(x, K, n):
    return x ** n / (K ** n + x ** n)

x = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
print("\nHill function, K = 1:    x =", x)
for n in (1, 2, 4):
    print(f"  n = {n}:  f(x) =", np.round(hill(x, 1.0, n), 3), f"  x for 10%->90%: {(81) ** (1 / n):.2f}-fold range")

# ---------------------------------------------------------------------------------------------
# 4. Logs for expression data: fold change is multiplicative, log fold change is additive and symmetric.
# ---------------------------------------------------------------------------------------------
print("\nlog2 fold change: 2x up =", math.log2(2), "| 2x down =", math.log2(0.5), "| 8x up =", math.log2(8))
print("raw ratios 2 and 0.5 average to", (2 + 0.5) / 2, "(wrongly >1); logs average to", (math.log2(2) + math.log2(0.5)) / 2)
counts = np.array([0, 1, 4, 20, 150, 1200])
print("log1p(counts) = ", np.round(np.log1p(counts), 2), "  (log(0) is undefined, so log1p is used for counts)")

# ---------------------------------------------------------------------------------------------
# 5. A sum of exponentials overflows, but log-sum-exp is computed safely (this is the softmax denominator).
# ---------------------------------------------------------------------------------------------
z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore"):
    naive = np.log(np.sum(np.exp(z)))
safe = z.max() + np.log(np.sum(np.exp(z - z.max())))
print(f"\nlog(sum(exp(z))) for z = [1000, 1001, 1002]: naive = {naive}, shifted = {safe:.6f}")
p = np.exp(z - safe)
print("softmax probabilities:", np.round(p, 4), "sum =", p.sum().round(6))

# a long product of probabilities underflows; the sum of logs does not
probs = rng.uniform(0.2, 0.9, size=2000)
print(f"product of 2000 probabilities = {np.prod(probs)}, but sum of logs = {np.sum(np.log(probs)):.1f}")

# ---------------------------------------------------------------------------------------------
# 6. Periodic functions: a cell-cycle gene oscillating with period 24 h; recover the period from the data
#    by fitting a*cos(wt) + b*sin(wt) at the right frequency.
# ---------------------------------------------------------------------------------------------
t = np.arange(0, 72, 2.0)
y = 5 + 2.0 * np.cos(2 * math.pi * t / 24 - 1.0) + rng.normal(0, 0.3, size=t.size)
best = None
for period in np.arange(18, 31, 0.5):
    w = 2 * math.pi / period
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(np.sum((y - X @ coef) ** 2))
    if best is None or sse < best[0]:
        best = (sse, period, coef)
sse, period, coef = best
amp = math.hypot(coef[1], coef[2])
print(f"\ncircadian fit: period {period} h (true 24), amplitude {amp:.2f} (true 2.00), baseline {coef[0]:.2f} (true 5.00)")
