import aerosandbox.numpy as np
from scipy import fft


def dctn(
        x
):
    return fft.dctn(
        x,
        type=1,
        norm="forward",
        orthogonalize=False
    )


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
    M = query_points.shape[0]
    N = query_points.shape[1]
    assert len(fft_image.shape) == N

    outputs = np.zeros(query_points.shape[0])

    normalized_frequency_edges = [
        np.arange(fft_image.shape[i])
        for i in range(N)
    ]

    for i in range(M):
        query_point = query_points[i, :]
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


if __name__ == '__main__':

    x_max = 5
    y_max = 6

    x = np.linspace(0, x_max, 300)
    y = np.linspace(0, y_max, 100)

    X, Y = np.meshgrid(
        x, y,
        indexing="ij"
    )

    f = 5 + (
            2 * np.sin(1 / 40 * (2 * np.pi) * X) +
            1 * np.sin(1 / 30 * (2 * np.pi) * Y) +
            0.1 * np.random.randn(*X.shape)
    )

    fft_vals = dctn(f)

    f_reconstructed = manual_inverse_continuous_cosine_transform(
        query_points=np.stack(
            [
                X.reshape(-1) / x_max,
                Y.reshape(-1) / y_max,
            ],
            axis=1
        ),
        fft_image=fft_vals
    ).reshape(X.shape)

    assert np.allclose(f, f_reconstructed)
