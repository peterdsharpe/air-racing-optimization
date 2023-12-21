import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from terrain_model.load_raw_data import lat_lon_to_north_east, terrain_data
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from airplane import airplane  # See cessna152.py for details.
import skimage as ski
from scipy import ndimage

lat_i = 46 + 32 / 60 + 53.84 / 3600
lon_i = -(122 + 28 / 60 + 8.98 / 3600)
north_i, east_i = lat_lon_to_north_east(lat_i, lon_i)

north_f, east_f = lat_lon_to_north_east(
    lat=46 + 16 / 60 + 37.56 / 3600,
    lon=-(121 + 34 / 60 + 38.70 / 3600),
)

i_lims = np.sort(np.array([
    np.argmin(np.abs(terrain_data["north_edges"] - north_i)),
    np.argmin(np.abs(terrain_data["north_edges"] - north_f)),
]))
j_lims = np.sort(np.array([
    np.argmin(np.abs(terrain_data["east_edges"] - east_i)),
    np.argmin(np.abs(terrain_data["east_edges"] - east_f)),
]))
i_lims += np.array([-100, 100])
j_lims += np.array([-100, 100])
i_lims = np.clip(i_lims, 0, terrain_data["elev"].shape[0] - 1)
j_lims = np.clip(j_lims, 0, terrain_data["elev"].shape[1] - 1)

terrain_data_zoomed = terrain_data["elev"][i_lims[0]:i_lims[1], j_lims[0]:j_lims[1]]
north_edges_zoomed = terrain_data["north_edges"][i_lims[0]:i_lims[1]]
east_edges_zoomed = terrain_data["east_edges"][j_lims[0]:j_lims[1]]

costs = (
        terrain_data_zoomed
        - ndimage.gaussian_filter(
    terrain_data_zoomed,
    500
)
)
costs = np.exp((costs - costs.mean()) / costs.std())

g = ski.graph.MCP_Geometric(
    costs=costs,
    fully_connected=True,
    sampling=(
        np.mean(np.diff(north_edges_zoomed)),
        np.mean(np.diff(east_edges_zoomed)),
    )
)
start = np.array([-100, 100])
end = np.array([100, -100])
cumulative_costs, traceback_array = g.find_costs(
    starts=[start],
    ends=[end],
)
traceback = np.array(g.traceback(end))

fig, ax = plt.subplots(
    figsize=(16, 6)
)
plt.imshow(
    costs,
    cmap='Reds',
    origin="lower",
    extent=(
        east_edges_zoomed[0],
        east_edges_zoomed[-1],
        north_edges_zoomed[0],
        north_edges_zoomed[-1],
    ),
    alpha=1,
    zorder=2.5
)
plt.plot(
    east_edges_zoomed[traceback[:, 1]],
    north_edges_zoomed[traceback[:, 0]],
    "-",
    color="red",
    linewidth=3,
    zorder=4
)

plt.imshow(
    terrain_data_zoomed,
    cmap='terrain',
    origin="lower",
    extent=(
        east_edges_zoomed[0],
        east_edges_zoomed[-1],
        north_edges_zoomed[0],
        north_edges_zoomed[-1],
    ),
    alpha=1,
    zorder=2
)

# p.plot_color_by_value(
#     dyn.y_e,
#     dyn.x_e,
#     c=dyn.speed,
#     cmap="Reds",
#     colorbar=True,
#     colorbar_label="Speed [m/s]",
# )
# plt.title(f"Resolution: {resolution}")
p.equal()
p.show_plot(
    rotate_axis_labels=False,
    savefig=[
        # f"./figures/trajectory_{resolution}.svg",
    ]
)
