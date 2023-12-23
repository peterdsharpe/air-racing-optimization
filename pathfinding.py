import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from get_task_info import terrain_data_zoomed, terrain_cost_heuristic
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
import skimage as ski
from scipy import ndimage

g = ski.graph.MCP_Geometric(
    costs=terrain_cost_heuristic,
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
print("Finding discrete path...")
cumulative_costs, traceback_array = g.find_costs(
    starts=[start],
    ends=[end],
)
index_path = np.array(g.traceback(end))
path = np.stack([
    terrain_data_zoomed["east_edges"][index_path[:, 1]],
    terrain_data_zoomed["north_edges"][index_path[:, 0]],
], axis=1)

if __name__ == '__main__':

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    plt.imshow(
        terrain_cost_heuristic,
        cmap='Reds',
        origin="lower",
        extent=(
            terrain_data_zoomed["east_edges"][0],
            terrain_data_zoomed["east_edges"][-1],
            terrain_data_zoomed["north_edges"][0],
            terrain_data_zoomed["north_edges"][-1],
        ),
        alpha=0.0,
        zorder=2.5
    )
    plt.plot(
        path[:, 0],
        path[:, 1],
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
    p.equal()
    p.show_plot(
        "Discrete Pathfinding",
        rotate_axis_labels=False,
        savefig=[
            # f"./figures/trajectory_{resolution}.svg",
        ]
    )
