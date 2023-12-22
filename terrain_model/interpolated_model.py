from terrain_model.load_raw_data import terrain_data, north_east_to_normalized_coordinates, lat_lon_to_north_east
from terrain_model.dct_algorithms import dctn, manual_inverse_continuous_cosine_transform
from scipy import fft, ndimage, interpolate
import numpy as np
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

_default_resolution = (100, 300)


def get_elevation_interpolated_north_east(
        query_points_north: np.ndarray,
        query_points_east: np.ndarray,
        resolution=_default_resolution,
):
    query_points_north = np.reshape(np.array(query_points_north), -1)
    query_points_east = np.reshape(np.array(query_points_east), -1)

    north_zoom_factor = resolution[0] / terrain_data["elev"].shape[0]
    east_zoom_factor = resolution[1] / terrain_data["elev"].shape[1]

    elev_resampled = ndimage.zoom(
        terrain_data["elev"],
        zoom=(north_zoom_factor, east_zoom_factor),
        order=3,
    )

    north_edges = np.linspace(
        terrain_data["north_edges"].min(),
        terrain_data["north_edges"].max(),
        elev_resampled.shape[0]
    )

    east_edges = np.linspace(
        terrain_data["east_edges"].min(),
        terrain_data["east_edges"].max(),
        elev_resampled.shape[1]
    )

    return asb.InterpolatedModel(
        x_data_coordinates={
            "north": north_edges,
            "east" : east_edges,
        },
        y_data_structured=elev_resampled
    )(
        {
            "north": query_points_north,
            "east" : query_points_east,
        }
    )


def get_elevation_interpolated_lat_lon(
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

    return get_elevation_interpolated_north_east(
        query_points_north,
        query_points_east,
        resolution=resolution,
    )
