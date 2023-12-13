import numpy as np

np.random.seed(0)
from scipy import fft, ndimage

E = 80
N = 50

e_max = 100
n_max = 100

east_edges = np.linspace(0, e_max, E)
north_edges = np.linspace(0, n_max, N)

east, north = np.meshgrid(
    east_edges,
    north_edges,
    indexing="ij"
)

f = 5 + (
        2 * np.sin(1 / 40 * (2 * np.pi) * east) +
        1 * np.sin(1 / 30 * (2 * np.pi) * north) +
        0.1 * np.random.randn(E, N)
)

fft_vals = fft.dctn(
    f,
    type=1,
    norm="forward",
    orthogonalize=False
)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(
    2, 2
)

ax = ax.flatten()

plt.sca(ax[0])
plt.title("f")
plt.imshow(
    f,
    cmap="turbo",
    extent=[
        east_edges[0],
        east_edges[-1],
        north_edges[0],
        north_edges[-1],
    ],
    origin="lower",
    zorder=4,
    vmin=f.min(),
    vmax=f.max(),
)
plt.colorbar()

plt.sca(ax[1])
plt.title("Spectrum")
plt.imshow(
    np.log10(np.abs(fft_vals) + 1e-100),
    cmap="turbo",
    origin="lower",
    vmin=-6,
    vmax=1,
    zorder=4,
)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")

plt.sca(ax[2])
plt.title("Reconstructed f (2D)")

from numba import njit


# @njit
# def reconstructor(e, n):
#     e_norm = e / e_max
#     n_norm = n / n_max
#
#     val = 0
#
#     for k in range(E):
#         for l in range(N):
#
#             if k == 0:
#                 alpha_E = 1
#             elif k == E - 1:
#                 alpha_E = 1
#             else:
#                 alpha_E = 2
#
#             if l == 0:
#                 alpha_N = 1
#             elif l == N - 1:
#                 alpha_N = 1
#             else:
#                 alpha_N = 2
#
#             val += alpha_E * alpha_N * fft_vals[k, l] * (
#                     np.cos(np.pi * k * e_norm) *
#                     np.cos(np.pi * l * n_norm)
#             )
#
#     return val


@njit
def reconstructors(es, ns):
    output = np.zeros(len(es))
    for i in range(len(es)):

        e_norm = es[i] / e_max
        n_norm = ns[i] / n_max

        output[i] = 0

        for k in range(E):
            for l in range(N):

                if k == 0 or k == E - 1:
                    alpha_E = 1
                else:
                    alpha_E = 2

                if l == 0 or l == N - 1:
                    alpha_N = 1
                else:
                    alpha_N = 2

                output[i] += alpha_E * alpha_N * fft_vals[k, l] * (
                        np.cos(np.pi * k * e_norm) *
                        np.cos(np.pi * l * n_norm)
                )

    return output


f_reconstructed = reconstructors(
    east.reshape(-1),
    north.reshape(-1),
).reshape(east.shape)

plt.imshow(
    f_reconstructed,
    cmap="turbo",
    extent=[
        east_edges[0],
        east_edges[-1],
        north_edges[0],
        north_edges[-1],
    ],
    origin="lower",
    zorder=4,
    vmin=f.min(),
    vmax=f.max(),
)
plt.colorbar()

plt.sca(ax[3])
plt.title("Reconstructed f (ND)")


def manual_inverse_continuous_cosine_transform(
        query_points,
        fft_image,
):
    """
            A manual implementation of the inverse N-dimensional continuous cosine transform using only np.cos().

            Args:
                query_points: Query points, a 2D array of shape (M, N), where M is the number of points and N is the number of dimensions.
                    * Should be normalized to the range [0, 1] in each dimension.

                fft_image: The DCT coefficients of the image, which has N dimensions, each of arbitrary length.

            Returns: Values of the original function at the query points, a 1D array of shape (M,)

            """
    assert len(query_points.shape) == 2
    N = query_points.shape[1]
    assert len(fft_image.shape) == N

    outputs = np.zeros(query_points.shape[0])

    normalized_frequency_edges = [
        np.arange(fft_image.shape[i])
        for i in range(N)
    ]

    for i, query_point in enumerate(query_points):
        output_components = np.ones(fft_image.shape)

        for d in range(N):
            edge_shape = [1] * N
            edge_shape[d] = len(normalized_frequency_edges[d])

            output_components *= np.reshape(
                np.cos(
                    np.pi * normalized_frequency_edges[d] * query_point[d]
                ),
                edge_shape
            )

            output_components *= np.reshape(
                np.where(
                    np.logical_or(
                        normalized_frequency_edges[d] == 0,
                        normalized_frequency_edges[d] == fft_image.shape[d] - 1
                    ),
                    1,
                    2
                ),
                edge_shape
            )

        output = np.sum(output_components * fft_image)

        outputs[i] = output

    return outputs


f_reconstructed_n = manual_inverse_continuous_cosine_transform(
    query_points=np.stack(
        [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6]),
        ],
        axis=0
    ),
    fft_image=fft_vals
)
f_reconstructed_n = manual_inverse_continuous_cosine_transform(
    query_points=np.stack(
        [
            east.reshape(-1) / e_max,
            north.reshape(-1) / n_max,
        ],
        axis=1
    ),
    fft_image=fft_vals
).reshape(east.shape)

plt.imshow(
    f_reconstructed_n,
    cmap="turbo",
    extent=[
        east_edges[0],
        east_edges[-1],
        north_edges[0],
        north_edges[-1],
    ],
    origin="lower",
    zorder=4,
    vmin=f.min(),
    vmax=f.max(),
)
plt.colorbar()

p.show_plot(
    "",
)

print(np.mean(np.abs(fft.idctn(fft_vals, norm="forward", type=1) - f)))
print(np.mean(np.abs(f - f_reconstructed)))
print(np.mean(np.abs(f - f_reconstructed_n)))
