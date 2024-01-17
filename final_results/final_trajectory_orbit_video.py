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

plotter = pv.Plotter(off_screen=True)
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
    scale_vehicle_model=14,
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
# plotter.enable_terrain_style()

# plotter.show(auto_close=False)

# plotter.open_gif("trajectory.gif")
plotter.open_movie("trajectory.mp4", framerate=video_fps)

# i = 0


up_vectors = np.stack(
    dyn.convert_axes(0, 0, -1, "body", "earth"),
    axis=1
)

terrain_altitude_interpolator = interpolate.RegularGridInterpolator(
    points=(
        terrain_data_zoomed["north_edges"],
        terrain_data_zoomed["east_edges"],
    ),
    values=terrain_data_zoomed["elev"],
    method="linear",
    bounds_error=False,
    fill_value=np.nan,

)

from tqdm import tqdm

# for i in tqdm(range(1)):
for i in tqdm(range((N - 1))):
    # plotter.camera.azimuth = np.degrees(dyn.track[0])
    # plotter.camera.elevation = np.degrees(dyn.gamma[0]) * -1
    # plotter.camera.roll = np.degrees(dyn.bank[0]) + 180
    # plotter.camera.up = (0, 0, -1)
    pos = np.array([
        dyn.x_e[i],
        dyn.y_e[i],
        dyn.z_e[i] - 5,
    ])
    next_pos = np.array([
        dyn.x_e[i + 1],
        dyn.y_e[i + 1],
        dyn.z_e[i + 1] - 5,
    ])
    dir = next_pos - pos
    dir /= np.linalg.norm(dir)

    plotter.camera.position = pos
    plotter.camera.focal_point = pos + dir * 10
    plotter.camera.up = (0, 0, -1)
    plotter.camera.roll = plotter.camera.roll + np.degrees(dyn.bank[i]) * 0.8
    plotter.camera.view_angle = 90 # FOV
    plotter.camera.clipping_range = (1, 200000)

    text = "\n".join([
        f"Time: {time_video[i]:.1f} s",
        f"Airspeed: {dyn.speed[i]:.0f} m/s",
        f"Altitude: {dyn.altitude[i]:.0f} m",
        f"Altitude AGL: {dyn.altitude[i] - terrain_altitude_interpolator((dyn.x_e[i], dyn.y_e[i])):.0f} m",
        f"AoA: {np.degrees(dyn.alpha[i]):.1f} deg",
        f"Bank: {np.degrees(dyn.bank[i]):.0f} deg",
        f"Gamma: {np.degrees(dyn.gamma[i]):.0f} deg",
        f"Bearing: {np.degrees(dyn.track[i]):.0f} deg",
        ])

    plotter.add_text(
        text,
        position="upper_left",
        font_size=18,
        color="white",
        font="courier",
        name="time_text",
        shadow=True,
    )

    plotter.write_frame()

plotter.close()