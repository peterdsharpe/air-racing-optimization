import numpy as np
from scipy import fft
from aerosandbox.tools.code_benchmarking import Timer


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
        include_gradient=False,
):
    """
            A manual implementation of the inverse N-dimensional continuous cosine transform using only np.cos().

            Args:
                query_points: Query points, a 2D array of shape (M, N), where M is the number of points and N is the number of dimensions.
                    * Should be normalized to the range [0, 1] in each dimension.

                    * As an example, for a single query point on a 2D DCT, the shape would be (1, 2).

                fft_image: The DCT coefficients of the image, which has N dimensions, each of arbitrary length.

            Returns: Values of the original function at the query points, a 1D array of shape (M,)

            """
    ### Use the fft_image to determine the number of dimensions
    N = len(fft_image.shape)

    ### Handle the query_points input
    query_points = np.array(query_points)  # Convert to a numpy array
    if len(query_points.shape) == 0:
        if N == 1:
            query_points = np.array([[query_points]])
        else:
            raise ValueError("If query_points is a scalar, it must be a 1D DCT (i.e., `fft_image` must be 1D).")
    elif len(query_points.shape) == 1:
        if query_points.shape[0] == N:
            query_points = np.array([query_points])
        else:
            raise ValueError("If query_points is a vector, it must have the same dimensionality as the DCT.")
    elif len(query_points.shape) == 2:
        if query_points.shape[1] != N:
            raise ValueError("If query_points is a 2D array, it must have the same dimensionality as the DCT.")
    else:
        raise ValueError(
            "`query_points` should be a 2D array of size (..., N), where N is the dimensionality of the DCT (as defined by `fft_image`).")

    # At this point, query_points has shape (M, N)
    M = query_points.shape[0]
    N = query_points.shape[1]

    normalized_frequency_edges = [
        np.arange(fft_image.shape[i])
        for i in range(N)
    ]

    ### Shape of intermediate arrays: (<N Axis Dimensions>, <M>). So the len(shape) of each array is N+1.
    output_components_shape = (*fft_image.shape, M)
    output_components = np.ones(output_components_shape)

    query_points_shape = ([1] * N) + [M]

    for d in range(N):
        edge_shape = [1] * (N + 1)
        edge_shape[d] = len(normalized_frequency_edges[d])

        with Timer("cos"):
            output_components *= np.cos(
                np.reshape(
                    np.pi * normalized_frequency_edges[d],
                    edge_shape
                ) *
                np.reshape(
                    query_points[:, d],
                    query_points_shape
                )
            )

        # Multiply every value, except those on the first and last rows, by 2.
        # Part of undoing the DCT-Type-1 normalization.
        with Timer("2x"):
            output_components[*[
                slice(1, -1) if i == d else slice(None)
                for i in range(N)
            ]] *= 2

    with Timer("sum"):
        outputs = np.sum(
            output_components * np.reshape(
                fft_image,
                (*fft_image.shape, 1)
            ),
            axis=tuple(range(N))
        )

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

    with Timer("Overall"):
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

    f_reconstructed_scipy = fft.idctn(
        fft_vals,
        type=1,
        norm="forward",
        orthogonalize=False
    )

    assert np.allclose(f, f_reconstructed)
