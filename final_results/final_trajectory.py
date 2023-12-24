import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from get_task_info import terrain_data_zoomed

dyn: asb.DynamicsPointMass3DSpeedGammaTrack = asb.load(
    "./dyn.asb"
)

time: np.ndarray = np.load("./time.npy")

speedup_over_realtime = 1
video_fps = 60

frame_time = speedup_over_realtime / video_fps

### Resample
time_video = np.arange(0, time[-1], frame_time)
from scipy import interpolate

state_interpolators = {
    k: interpolate.InterpolatedUnivariateSpline(
        x=time,
        y=v,
    )
    for k, v in dyn.state.items()
}
control_interpolators = {
    k: interpolate.InterpolatedUnivariateSpline(
        x=time,
        y=v,
    )
    for k, v in dyn.control_variables.items()
}
dyn = dyn.get_new_instance_with_state({
    k: v(time_video)
    for k, v in state_interpolators.items()
})
for k, v in control_interpolators.items():
    setattr(dyn, k, v(time_video))

# dyn = dyn.get_new_instance_with_state(
#     {
#         k: interpolate.interp1d(
#             time,
#             v,
#             kind="cubic"
#         )(time_video)
#         for k, v in dyn.state.items()
#     }
# )


N = len(dyn)


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


plotter = pv.Plotter()
plotter.window_size = 1920, 1080
plotter.set_background(
    color="#FFFFFF",
    top="#A5B8D7",
)

dyn.draw(
    backend="pyvista",
    plotter=plotter,
    show=False,
    n_vehicles_to_draw=int(N / (video_fps / 4)),
    scale_vehicle_model=14 * 5,
    # scale_vehicle_model=1,
    trajectory_line_color="red",
    draw_global_grid=False,
    draw_axes=False,
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
    show_scalar_bar=False,
)
plotter.enable_terrain_style()
plotter.show()