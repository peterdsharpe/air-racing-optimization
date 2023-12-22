import numpy as np
from scipy import fft
from aerosandbox.tools.code_benchmarking import Timer

from jax import config

config.update("jax_enable_x64", True)


def dctn(
        x: np.ndarray
) -> np.ndarray:
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
        use_einsum_if_possible=True,
        jax=False,
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
    if jax:
        import jax.numpy as np
    else:
        import numpy as np
        query_points = np.array(query_points)

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

    if include_gradient:
        ### Complex-step method
        eps = 1j * 1e-150

        complex_steps = []
        for d in range(N):
            eps_vector = np.zeros((N)) + 0j
            eps_vector[d] = eps
            eps_vector = np.reshape(eps_vector, (1, N))

            complex_steps.append(
                manual_inverse_continuous_cosine_transform(
                    query_points + eps_vector,
                    fft_image=fft_image,
                    include_gradient=False,
                    use_einsum_if_possible=use_einsum_if_possible,
                )
            )

        value = np.real(complex_steps[0])

        grad = np.stack([
            np.imag(c) / np.abs(eps)
            for c in complex_steps
        ], axis=1)

        return value, grad

    ### True shape of `output_components` is: (<N Axis Dimensions>, <M>). So the len(shape) of each array is N+1.
    output_components_shape = (*fft_image.shape, M)
    output_components = np.reshape(
        np.array([1.]),
        [1] * (N + 1)
    )

    query_points_shape = ([1] * N) + [M]

    for d in range(N):
        edge_shape = [1] * (N + 1)
        edge_shape[d] = fft_image.shape[d]

        # As part of undoing the DCT-Type-1 normalization,
        # We multiply every value, except those on the first and last rows, by 2.
        # This is more-quickly implemented by multiplying the first and last rows by 0.5 (for each dim), and then
        # multiplying the whole thing by 2 ** N at the end.
        multiplier = np.ones(edge_shape)
        if jax:
            multiplier = multiplier.at[  # For JAX
                *[
                    [0, -1] if i == d else slice(None)
                    for i in range(N)
                ]
            ].set(0.5)
        else:
            multiplier[*[
                [0, -1] if i == d else slice(None)
                for i in range(N)
            ]] = 0.5

        output_components = (
                multiplier
                * np.cos(
            np.reshape(
                np.pi * np.arange(fft_image.shape[d]),  # The normalized frequency
                edge_shape
            )
            * np.reshape(
                query_points[:, d],
                query_points_shape
            ))
                * output_components
        )

    if N <= 25 and use_einsum_if_possible:
        input_subscript = ''.join([chr(97 + i) for i in range(N)]) + 'z'  # 'abc...z' for N dimensions
        output_subscript = 'z'  # Sum over all but the last axis
        einsum_subscript = f'{input_subscript},{input_subscript[:-1]}->{output_subscript}'

        return 2 ** N * np.einsum(
            einsum_subscript,  # Analogous to 'ijk,ij->k',
            output_components,
            fft_image,
            optimize="greedy"
        )
    else:
        return 2 ** N * np.sum(
            output_components * np.reshape(
                fft_image,
                (*fft_image.shape, 1)
            ),
            axis=tuple(range(N))
        )


def cas_micct(
        query_points,
        fft_image,
):
    import numpy as np
    import casadi as cas
    # Dynamically construct a callback:
    class Function(cas.Callback):
        def __init__(self):
            cas.Callback.__init__(self)
            self.construct(
                "MICCTfunction",
                # {
                #     "enable_fd": True,
                # }
            )

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, *args):
            return cas.Sparsity.dense(np.prod(query_points.shape), 1)

        def get_sparsity_out(self, *args):
            return cas.Sparsity.dense(query_points.shape[0], 1)

        def eval(self, arg):
            qp_input = np.reshape(
                np.array(arg[0]),
                query_points.shape,
                order="F"
            )

            return [manual_inverse_continuous_cosine_transform(
                query_points=qp_input,
                fft_image=fft_image,
            )]

        def has_jacobian(self, *args):
            return True

        def get_jac_sparsity(self, *args):
            return cas.repmat(cas.Sparsity_diag(query_points.shape[0]), 1, query_points.shape[1])

        def get_jacobian(self, name, inames, onames, opts):

            class JacFun(cas.Callback):
                def __init__(self):
                    cas.Callback.__init__(self)
                    self.construct(name,
                                   {
                                       "enable_fd": True,
                                   }
                                   )

                def get_n_in(self):
                    return 2

                def get_n_out(self):
                    return 1

                def get_sparsity_in(self, n_in):
                    if n_in == 0:
                        return cas.Sparsity.dense(np.prod(query_points.shape), 1)
                    if n_in == 1:
                        return cas.Sparsity.dense(query_points.shape[0], 1)

                def get_sparsity_out(self, n_out):
                    return cas.repmat(cas.Sparsity_diag(query_points.shape[0]), 1, query_points.shape[1])

                def eval(self, arg):
                    qp_input = np.reshape(
                        np.array(arg[0]),
                        query_points.shape,
                        order="F"
                    )

                    jac = manual_inverse_continuous_cosine_transform(
                        query_points=qp_input,
                        fft_image=fft_image,
                        include_gradient=True,
                    )[1]

                    from scipy import sparse

                    casjac = sparse.hstack(
                        [
                            sparse.diags(jac[:, i], format="csc")
                            for i in range(jac.shape[1])
                        ]
                    )

                    return [casjac]

            self.jac_callback = JacFun()
            return self.jac_callback

    function_instance = Function()

    res = function_instance(
        cas.reshape(query_points, -1, 1)
        # cas.reshape(query_points, -1, 1)
    )
    res._function_instance = function_instance

    return res


if __name__ == '__main__':

    ### Checks for correctness
    x_max = 5
    y_max = 6

    x = np.linspace(0, x_max, 50)
    y = np.linspace(0, y_max, 150)

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

    query_points = np.stack(
        [
            X.reshape(-1) / x_max,
            Y.reshape(-1) / y_max,
        ],
        axis=1
    )

    with Timer("Correctness Check"):
        f_reconstructed = manual_inverse_continuous_cosine_transform(
            query_points=query_points,
            fft_image=fft_vals
        ).reshape(X.shape)

    f_reconstructed_scipy = fft.idctn(
        fft_vals,
        type=1,
        norm="forward",
        orthogonalize=False
    )

    assert np.allclose(f, f_reconstructed)

    ### Benchmarking

    x = np.linspace(0, x_max, 500)
    y = np.linspace(0, y_max, 1000)

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

    M = 100

    query_points = np.stack(
        [
            np.linspace(0, 1, M),
            np.linspace(0, 1, M),
        ],
        axis=1
    )

    with Timer("Benchmarking [NP]"):
        f_reconstructed = manual_inverse_continuous_cosine_transform(
            query_points=query_points,
            fft_image=fft_vals,
            jax=False
        )
    with Timer("Benchmarking [NP+grad]"):
        f_reconstructed, grad = manual_inverse_continuous_cosine_transform(
            query_points=query_points,
            fft_image=fft_vals,
            include_gradient=True,
            jax=False
        )


    # Check gradient correctness

    def func(q):
        return manual_inverse_continuous_cosine_transform(
            query_points=np.reshape(q, (1, -1)),
            fft_image=fft_vals,
            jax=True,
        ).reshape(())


    import jax

    func = jax.vmap(jax.value_and_grad(func))
    jfunc = jax.jit(func)

    points = np.random.uniform(0, 1, (100, 2))

    value_manual, grad_manual = manual_inverse_continuous_cosine_transform(
        query_points=points,
        fft_image=fft_vals,
        include_gradient=True,
        use_einsum_if_possible=False
    )
    value_jax, grad_jax = func(
        points
    )

    if not np.allclose(grad_manual, grad_jax):
        print(grad_manual)
        print(grad_jax)
        print(f"atol: {np.max(np.abs(grad_manual - grad_jax))}")
        print(f"rtol: {np.max(np.abs(np.log(grad_manual / grad_jax)))}")

    import casadi as cas

    cas_input = cas.MX(points)
    cas_sym_input = cas.MX.sym("p", *points.shape)
    o1 = cas_micct(cas_input, fft_vals)
    o2 = cas.evalf(o1)

    o1s = cas_micct(cas_sym_input, fft_vals)
    with Timer("jo2s"):
        jo2s = cas.evalf(cas.graph_substitute(jo1s, [cas_sym_input], [cas_input]))
