import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from get_task_info import terrain_data_zoomed

dyn: asb.DynamicsPointMass3DSpeedGammaTrack = asb.load(
    "./dyn.asb"
)

time: np.ndarray = dyn.other_fields["time"]

# speedup_over_realtime = 1
# video_fps = 60
#
# frame_time = speedup_over_realtime / video_fps
#
# ### Resample
# time_video = np.arange(0, time[-1], frame_time)
# from scipy import interpolate
#
# state_interpolators = {
#     k: interpolate.InterpolatedUnivariateSpline(
#         x=time,
#         y=v,
#     )
#     for k, v in dyn.state.items()
# }
# control_interpolators = {
#     k: interpolate.InterpolatedUnivariateSpline(
#         x=time,
#         y=v,
#     )
#     for k, v in dyn.control_variables.items()
# }
# dyn = dyn.get_new_instance_with_state({
#     k: v(time_video)
#     for k, v in state_interpolators.items()
# })
# for k, v in control_interpolators.items():
#     setattr(dyn, k, v(time_video))

N = len(dyn)


fig, ax = plt.subplots(
    figsize=(11, 5)
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
plt.xlabel("Position East of Datum [m]")
plt.ylabel("Position North of Datum [m]")

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

plt.colorbar(label="Terrain Elevation [m]")
p.equal()
p.show_plot(
    # "3D Continuous Optimization",
    rotate_axis_labels=False,
    savefig=[
        f"../figures/trajectory.svg",
    ]
)

N = len(dyn)


# plotter = pv.Plotter()
# plotter.window_size = 1920, 1080
# plotter.set_background(
#     color="#FFFFFF",
#     top="#A5B8D7",
# )
#
# dyn.draw(
#     backend="pyvista",
#     plotter=plotter,
#     show=False,
#     n_vehicles_to_draw=N // 100,
#     scale_vehicle_model=14 * 5,
#     # scale_vehicle_model=1,
#     trajectory_line_color="red",
#     draw_global_grid=False,
#     draw_axes=False,
# )
#
# grid = pv.RectilinearGrid(
#     terrain_data_zoomed["north_edges"],
#     terrain_data_zoomed["east_edges"],
# )
# grid["elev"] = np.ravel(terrain_data_zoomed["elev"], order="F")
#
# dx, dy = np.gradient(
#     terrain_data_zoomed["elev"],
#     terrain_data_zoomed["north_edges"],
#     terrain_data_zoomed["east_edges"],
#     edge_order=2,
#     n=1
# )
# ddx, ddy = np.gradient(
#     terrain_data_zoomed["elev"],
#     terrain_data_zoomed["north_edges"],
#     terrain_data_zoomed["east_edges"],
#     edge_order=2,
#     n=2
# )
# mag = np.sqrt(dx ** 2 + dy ** 2)
# slope_deg = np.arctan2d(mag, 1)
#
# lap = ddx + ddy
# curvature = np.sqrt(ddx ** 2 + ddy ** 2)
#
# def normalize(x, q1=0.1, q2=0.9):
#     low, med, high = np.quantile(x, [q1, 0.5, q2])
#     from scipy import interpolate
#     norm = interpolate.interp1d(
#         [low, med, high],
#         [-1, 0, 1],
#         fill_value="extrapolate",
#     )(x)
#     return np.tanh(norm)
#
# color = (
#     np.clip(slope_deg, 0, 30) / 30
#     - 0.5 * normalize(lap)
#     + 0.2 * normalize(curvature)
#     + 0.2 * normalize(terrain_data_zoomed["elev"])
# )
# from scipy import ndimage
#
# color = ndimage.gaussian_filter(color, sigma=0.5)
# # color = normalize(color, q1=0.01, q2=0.9)
#
# grid["color"] = np.ravel(color, order="F")
#
# grid = grid.warp_by_scalar("elev", factor=-1)
# plotter.add_mesh(
#     grid.extract_geometry(),
#     scalars="color",
#     cmap='terrain',
#     # cmap=cmocean.cm.topo,
#     specular=0.5,
#     specular_power=15,
#     # smooth_shading=True,
#     show_scalar_bar=False,
# )
# plotter.enable_terrain_style()
# plotter.show()