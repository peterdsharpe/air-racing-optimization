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
    # ends=[end],
)
index_path = np.array(g.traceback(end))
path = np.stack([
    terrain_data_zoomed["east_edges"][index_path[:, 1]],
    terrain_data_zoomed["north_edges"][index_path[:, 0]],
], axis=1)

if __name__ == '__main__':

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )
    # plt.imshow(
    #     cumulative_costs,
    #     cmap='Reds',
    #     origin="lower",
    #     extent=(
    #         terrain_data_zoomed["east_edges"][0],
    #         terrain_data_zoomed["east_edges"][-1],
    #         terrain_data_zoomed["north_edges"][0],
    #         terrain_data_zoomed["north_edges"][-1],
    #     ),
    #     alpha=1,
    #     zorder=2.5
    # )
    p.contour(
        terrain_data_zoomed["east_edges"],
        terrain_data_zoomed["north_edges"],
        np.log(cumulative_costs),
        cmap="Reds",
        alpha=1,
        levels=61,
        linelabels=False,
        # zorder=3,
        colorbar_label=r"ln(Cumulative Costs)"
    )
    plt.clim(
        *np.quantile(np.log(cumulative_costs), [0.01, 0.8])
    )

    plt.plot(
        path[:, 0],
        path[:, 1],
        "-",
        color="red",
        linewidth=3,
        zorder=4
    )

    from matplotlib import patheffects

    plt.annotate(
        "Start",
        xy=(terrain_data_zoomed["east_start"], terrain_data_zoomed["north_start"]),
        xytext=(0, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="k",
        path_effects=[
            patheffects.withStroke(
                linewidth=3,
                foreground="w",
            )
        ]
    )
    plt.annotate(
        "Finish",
        xy=(terrain_data_zoomed["east_end"], terrain_data_zoomed["north_end"]),
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        color="k",
        path_effects=[
            patheffects.withStroke(
                linewidth=3,
                foreground="w",
            )
        ]
    )

    # plt.imshow(
    #     terrain_data_zoomed["elev"],
    #     cmap='terrain',
    #     origin="lower",
    #     extent=(
    #         terrain_data_zoomed["east_edges"][0],
    #         terrain_data_zoomed["east_edges"][-1],
    #         terrain_data_zoomed["north_edges"][0],
    #         terrain_data_zoomed["north_edges"][-1],
    #     ),
    #     alpha=1,
    #     zorder=2
    # )
    plt.xlabel("Position East of Datum [m]")
    plt.ylabel("Position North of Datum [m]")

    p.equal()
    p.show_plot(
        "Discrete Pathfinding",
        rotate_axis_labels=False,
        savefig=[
            f"./figures/pathfinding_trajectory.png",
        ]
    )
