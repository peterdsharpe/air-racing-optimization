from terrain_model.load_raw_data import terrain_data, north_east_to_normalized_coordinates, lat_lon_to_north_east
from terrain_model.dct_algorithms import dctn, manual_inverse_continuous_cosine_transform, cas_micct
from scipy import fft
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fft_image = dctn(
    terrain_data["elev"],
)

_default_resolution = (100, 300)


def get_elevation_fourier_north_east(
        query_points_north: np.ndarray,
        query_points_east: np.ndarray,
        resolution=_default_resolution,
):
    query_points_north = np.reshape(np.array(query_points_north), -1)
    query_points_east = np.reshape(np.array(query_points_east), -1)

    query_points_normalized = np.stack(
        north_east_to_normalized_coordinates(
            query_points_north,
            query_points_east,
        ),
        axis=1
    )

    if np.is_casadi_type(query_points_normalized):
        return cas_micct(
            query_points=query_points_normalized,
            fft_image=fft_image[:resolution[0], :resolution[1]]
        )
    else:
        return manual_inverse_continuous_cosine_transform(
            query_points=query_points_normalized,
            fft_image=fft_image[:resolution[0], :resolution[1]]
        )


def get_elevation_fourier_lat_lon(
        query_points_lat: np.ndarray,
        query_points_lon: np.ndarray,
        resolution=_default_resolution,
):
    query_points_lat = np.reshape(np.array(query_points_lat), -1)
    query_points_lon = np.reshape(np.array(query_points_lon), -1)

    query_points_north, query_points_east = lat_lon_to_north_east(
        query_points_lat,
        query_points_lon,
    )

    return get_elevation_fourier_north_east(
        query_points_north,
        query_points_east,
        resolution=resolution,
    )
