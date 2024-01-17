
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from terrain_model.load_raw_data import terrain_data, north_east_to_normalized_coordinates, lat_lon_to_north_east
from scipy import fft, ndimage, interpolate
import numpy as np
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import interpolate
from typing import Tuple
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pickle
import casadi as cas

_default_resolution = (100, 300)


def precompute_and_save_model(
        resolution: Tuple[float, float] = _default_resolution,
        terrain_data=terrain_data
):

    name = f"{resolution[0]}_{resolution[1]}"
    file = Path(__file__).parent / "precomputed_models" / f"{name}.pkl"
    if not file.exists():
        print(f"Pre-computing interpolant: '{name}'...")
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
        interpolant = cas.interpolant(
            "interpolant",
            "bspline",
            [north_edges, east_edges],
            elev_resampled.ravel(order="F"),
        )

        with open(file, "wb+") as f:
            pickle.dump(
                interpolant,
                f
            )


def get_elevation_interpolated_north_east(
        query_points_north: np.ndarray,
        query_points_east: np.ndarray,
        resolution=_default_resolution,
):
    query_points_north = np.reshape(np.array(query_points_north), -1)
    query_points_east = np.reshape(np.array(query_points_east), -1)

    name = f"{resolution[0]}_{resolution[1]}"
    file = Path(__file__).parent / "precomputed_models" / f"{name}.pkl"
    if not file.exists():
        precompute_and_save_model(resolution=resolution)

    with open(file, "rb") as f:
        interpolant = pickle.load(f)

    return interpolant(np.stack([
        query_points_north,
        query_points_east
    ], axis=1).T).T


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


if __name__ == '__main__':
    from aerosandbox.tools.code_benchmarking import Timer

    for res in [
        (100, 300),
        (150, 450),
        (200, 600),
        (300, 900),
        (600, 1800),
        (1200, 3600),
        (1800, 4800),
        (2400, 7200),
        (3600, 10800),
    ]:
        with Timer(str(res)):
            precompute_and_save_model(resolution=res)
