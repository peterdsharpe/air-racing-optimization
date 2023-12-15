from load_raw_data import terrain_data
from dct_algorithms import dctn, manual_inverse_continuous_cosine_transform
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fft_image = dctn(
    terrain_data["elev"],
)

# fig, ax = plt.subplots()
# plt.imshow(np.log10(np.abs(fft_image)), cmap='turbo')
# plt.colorbar()
# plt.contour(
#     ndimage.gaussian_filter(np.log10(np.abs(fft_image)), sigma=10),
#     levels=np.arange(-1, 10, 1),
#     colors='k',
#     linewidths=0.5
# )
# plt.title("Magnitude Spectrum of the DCT")
# plt.show()

for res in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    # for res in [128]:

    low_res_fft_image = np.copy(fft_image)
    low_res_fft_image[:, 3 * res:] = 0
    low_res_fft_image[res:, :] = 0

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(
            1 + 15,
            1 + 10
        )
    )

    plt.sca(ax[0])

    # reconstruct_res = (100, 300)
    #
    # reconstructed_elev = manual_inverse_continuous_cosine_transform(
    #     query_points=np.stack(
    #         [array.reshape(-1) for array in
    #          np.meshgrid(
    #              np.linspace(0, 1, reconstruct_res[0]),
    #              np.linspace(0, 1, reconstruct_res[1]),
    #              indexing="ij"
    #          )
    #          ],
    #         axis=1
    #     ),
    #     fft_image=fft_image[:res, :3 * res]
    # ).reshape(reconstruct_res)

    reconstructed_elev = fft.idctn(
        low_res_fft_image,
        type=1,
        norm="forward",
        orthogonalize=False
    )

    plt.imshow(
        reconstructed_elev,
        cmap='terrain',
        origin="lower",
        extent=[
            terrain_data["lon_edges"].min(),
            terrain_data["lon_edges"].max(),
            terrain_data["lat_edges"].min(),
            terrain_data["lat_edges"].max()
        ],
        vmin=terrain_data["elev"].min(),
        vmax=terrain_data["elev"].max(),
        zorder=4
    )
    p.equal()
    # plt.colorbar(label='Elevation [m]', orientation="horizontal")
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.suptitle(f"Resolution = {res} $\\times$ {3 * res}")
    plt.suptitle('Contour Plot of Elevation around Riffe Lake')

    plt.sca(ax[1])

    plt.imshow(
        np.log10(np.abs(low_res_fft_image[:256, :3 * 256]) + 1e-100),
        cmap='turbo',
        vmin=-3,
        vmax=11,
    )

    p.show_plot(
        rotate_axis_labels=False,
    )
