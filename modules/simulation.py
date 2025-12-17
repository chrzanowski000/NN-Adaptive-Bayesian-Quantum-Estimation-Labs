import numpy as np

def measure(omega, T2, t, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p0 = np.exp(-t / T2) * np.cos(omega * t / 2)**2 \
         + (1 - np.exp(-t / T2)) / 2

    return 0 if rng.random() < p0 else 1


# example usage
omega = 0.7
T2 = 10.0
t = 3.2
N = 100_000
samples = [measure(omega, T2, t) for _ in range(N)]
print("freq(0) =", samples.count(0) / N)

