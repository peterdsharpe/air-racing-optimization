import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from get_task_info import terrain_data_zoomed
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from airplane import airplane  # See cessna152.py for details.
import skimage as ski
from scipy import ndimage

costs = (
        terrain_data_zoomed["elev"]
        - ndimage.gaussian_filter(
    terrain_data_zoomed["elev"],
    500
)
)
costs = np.exp((costs - costs.mean()) / costs.std())

g = ski.graph.MCP_Geometric(
    costs=costs,
    fully_connected=True,
    sampling=(
        terrain_data_zoomed["dx_north"],
        terrain_data_zoomed["dx_east"],
    )
)
start = np.array([
    terrain_data_zoomed["north_start_index"],
    terrain_data_zoomed["east_start_index"],
])
end = np.array([
    terrain_data_zoomed["north_end_index"],
    terrain_data_zoomed["east_end_index"],
])
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
        terrain_data_zoomed["east_edges"][0],
        terrain_data_zoomed["east_edges"][-1],
        terrain_data_zoomed["north_edges"][0],
        terrain_data_zoomed["north_edges"][-1],
    ),
    alpha=0.2,
    zorder=2.5
)
plt.plot(
    terrain_data_zoomed["east_edges"][traceback[:, 1]],
    terrain_data_zoomed["north_edges"][traceback[:, 0]],
    "-",
    color="red",
    linewidth=3,
    zorder=4
)

plt.imshow(
    terrain_data_zoomed["elev"],
    cmap='terrain',
    origin="lower",
    extent=(
        terrain_data_zoomed["east_edges"][0],
        terrain_data_zoomed["east_edges"][-1],
        terrain_data_zoomed["north_edges"][0],
        terrain_data_zoomed["north_edges"][-1],
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
