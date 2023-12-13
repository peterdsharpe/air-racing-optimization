import numpy as np

from scipy import fft, ndimage

N = 1000

T = 10

t = np.linspace(0, T, N)

f = 3 + (
        np.sin(0.5 * (2 * np.pi) * t) +
        1 * np.sin(2 * (2 * np.pi) * t) +
        0.0 * np.random.randn(N)
)

f /= np.std(f)

fft_vals = fft.dct(
    f,
    type=1,
    norm="forward",
    orthogonalize=False
)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(
    3, 1
)

plt.sca(ax[0])
plt.plot(
    t,
    f,
    color="C0")
plt.xlabel("Time [s]")
plt.ylabel("Signal")

plt.sca(ax[1])
plt.loglog(
    np.arange(N) / (2 * T),
    np.abs(fft_vals),
    linewidth=0.5,
    color="C1")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.sca(ax[2])

from numba import njit


@njit
def reconstructor(t):
    t_norm = t / T

    val = 0

    for i in range(N):
        if i == 0:
            alpha = 1
        elif i == N - 1:
            alpha = 1
        else:
            alpha = 2

        val += alpha * fft_vals[i] * np.cos(np.pi * i * t_norm)

    return val


@njit
def reconstructors(ts):
    return np.array([
        reconstructor(t)
        for t in ts
    ])


# f_reconstructed = np.vectorize(lambda ti: fft_vals[0] + 2 * np.dot(
#     fft_vals[1:],
#     np.cos(np.pi * np.arange(1, N) * (ti / T))
# ))(t)
f_reconstructed = reconstructors(t)

print(np.mean(np.abs(f - f_reconstructed)))
print(np.mean(np.abs(fft.idct(fft_vals, norm="forward", type=1) - f)))

plt.plot(
    t,
    f_reconstructed,
    color="C0"
)
plt.xlabel("Time [s]")
plt.ylabel("Reconstructed\nSignal")

p.show_plot(
    "",
)
