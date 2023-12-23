import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from get_task_info import terrain_data_zoomed

dyn = asb.load(
    "./dyn.asb"
)

fig, ax = plt.subplots(
    figsize=(16, 6)
)
plt.plot(
    dyn.y_e,
    dyn.x_e,
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
    "3D Continuous Optimization",
    rotate_axis_labels=False,
    savefig=[
        # f"./figures/trajectory_{resolution}.svg",
    ]
)

N = len(dyn)

plotter = dyn.draw(
    backend="pyvista",
    show=False,
    n_vehicles_to_draw=N // 200,
    scale_vehicle_model=800 / (N / 100),
    trajectory_line_color="red"
)

grid = pv.RectilinearGrid(
    terrain_data_zoomed["north_edges"],
    terrain_data_zoomed["east_edges"],
)
grid["elev"] = terrain_data_zoomed["elev"].T.flatten()
grid = grid.warp_by_scalar("elev", factor=-1)
plotter.add_mesh(
    grid.extract_geometry(),
    scalars="elev",
    cmap='terrain',
    specular=0.5,
    specular_power=15,
    smooth_shading=True,
)
plotter.enable_terrain_style()
plotter.show()