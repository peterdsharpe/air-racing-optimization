from terrain_model.load_raw_data import lat_lon_to_north_east, terrain_data
import numpy as np

lat_start = 46 + 32 / 60 + 53.84 / 3600
lon_start = -(122 + 28 / 60 + 8.98 / 3600)
north_start, east_start = lat_lon_to_north_east(
    46 + 32 / 60 + 53.84 / 3600,
    -(122 + 28 / 60 + 8.98 / 3600)
)

north_end, east_end = lat_lon_to_north_east(
    lat=46 + 16 / 60 + 37.56 / 3600,
    lon=-(121 + 34 / 60 + 38.70 / 3600),
)

north_start_index = np.argmin(np.abs(terrain_data["north_edges"] - north_start))
east_start_index = np.argmin(np.abs(terrain_data["east_edges"] - east_start))
north_end_index = np.argmin(np.abs(terrain_data["north_edges"] - north_end))
east_end_index = np.argmin(np.abs(terrain_data["east_edges"] - east_end))

i_lims = np.sort(np.array([
    north_start_index,
    north_end_index,
]))
j_lims = np.sort(np.array([
    east_start_index,
    east_end_index,
]))

padding_distance = 3000  # meters, around the border
dx_north = np.mean(np.diff(terrain_data["north_edges"]))
dx_east = np.mean(np.diff(terrain_data["east_edges"]))

i_lims += np.round(np.array([-padding_distance, padding_distance]) / dx_north).astype(int)
j_lims += np.round(np.array([-padding_distance, padding_distance]) / dx_east).astype(int)

i_lims = np.clip(i_lims, 0, terrain_data["elev"].shape[0] - 1)
j_lims = np.clip(j_lims, 0, terrain_data["elev"].shape[1] - 1)

terrain_data_zoomed = {
    "elev"             : terrain_data["elev"][i_lims[0]:i_lims[1], j_lims[0]:j_lims[1]],
    "north_edges"      : terrain_data["north_edges"][i_lims[0]:i_lims[1]],
    "east_edges"       : terrain_data["east_edges"][j_lims[0]:j_lims[1]],
    "dx_north": dx_north,
    "dx_east": dx_east,
    "north_start_index": np.argmin(np.abs(terrain_data["north_edges"][i_lims[0]:i_lims[1]] - north_start)),
    "east_start_index" : np.argmin(np.abs(terrain_data["east_edges"][j_lims[0]:j_lims[1]] - east_start)),
    "north_end_index"  : np.argmin(np.abs(terrain_data["north_edges"][i_lims[0]:i_lims[1]] - north_end)),
    "east_end_index"   : np.argmin(np.abs(terrain_data["east_edges"][j_lims[0]:j_lims[1]] - east_end)),
}
