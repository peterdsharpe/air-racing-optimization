import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from get_task_info import terrain_data_zoomed
from scipy import interpolate

dyn: asb.DynamicsPointMass3DSpeedGammaTrack = asb.load(
    "./dyn.asb"
)

time: np.ndarray = dyn.other_fields["time"]

speedup_over_realtime = 1  # How much faster than realtime to render the video
video_fps = 60  # Frames per second
video_fraction = (0, 1)  # Only render part of the video

frame_time = speedup_over_realtime / video_fps

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

### Resample
time_video = np.arange(
    time[-1] * video_fraction[0],
    time[-1] * video_fraction[1],
    frame_time
)

state_interpolators = {
    k: interpolate.InterpolatedUnivariateSpline(
        x=time,
        y=v,
    )
    for k, v in dyn.state.items()
}
control_interpolators = {
    k: interpolate.PchipInterpolator(
        x=time,
        y=v,
    )
    for k, v in dyn.control_variables.items()
}
other_fields_interpolators = {
    k: interpolate.PchipInterpolator(
        x=time,
        y=v,
    )
    for k, v in dyn.other_fields.items()
    if np.length(v) == np.length(time)
}
dyn = dyn.get_new_instance_with_state({
    k: v(time_video)
    for k, v in state_interpolators.items()
})
for k, v in control_interpolators.items():
    setattr(dyn, k, v(time_video))
dyn.other_fields = {
    k: v(time_video)
    for k, v in other_fields_interpolators.items()
}

terrain_altitude = terrain_altitude_interpolator(
    np.stack([
        dyn.x_e,
        dyn.y_e,
    ], axis=1)
)

altitude_agl = dyn.altitude - terrain_altitude

N = len(dyn)

plotter = pv.Plotter(off_screen=True)
plotter.window_size = 1920, 1080
plotter.set_background(
    color="#FFFFFF",
    top="#A5B8D7",
)
plotter.enable_anti_aliasing('ssaa')

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

dx, dy = np.gradient(
    terrain_data_zoomed["elev"],
    terrain_data_zoomed["north_edges"],
    terrain_data_zoomed["east_edges"],
    edge_order=2,
    n=1
)
ddx, ddy = np.gradient(
    terrain_data_zoomed["elev"],
    terrain_data_zoomed["north_edges"],
    terrain_data_zoomed["east_edges"],
    edge_order=2,
    n=2
)
mag = np.sqrt(dx ** 2 + dy ** 2)
slope_deg = np.arctan2d(mag, 1)

lap = ddx + ddy
curvature = np.sqrt(ddx ** 2 + ddy ** 2)

def normalize(x, q1=0.1, q2=0.9):
    low, med, high = np.quantile(x, [q1, 0.5, q2])
    from scipy import interpolate
    norm = interpolate.interp1d(
        [low, med, high],
        [-1, 0, 1],
        fill_value="extrapolate",
    )(x)
    return np.tanh(norm)

color = (
    np.clip(slope_deg, 0, 30) / 30
    - 0.5 * normalize(lap)
    + 0.2 * normalize(curvature)
    + 0.2 * normalize(terrain_data_zoomed["elev"])
)
from scipy import ndimage

color = ndimage.gaussian_filter(color, sigma=0.5)
# color = normalize(color, q1=0.01, q2=0.9)

grid["color"] = np.ravel(color, order="F")

grid = grid.warp_by_scalar("elev", factor=-1)
plotter.add_mesh(
    grid.extract_geometry(),
    scalars="color",
    cmap='terrain',
    # cmap=cmocean.cm.topo,
    specular=0.5,
    specular_power=15,
    # smooth_shading=True,
    show_scalar_bar=False,
)

plotter.open_movie("trajectory.mp4", framerate=video_fps, quality=5)

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
    plotter.camera.roll = plotter.camera.roll + np.degrees(dyn.bank[i])
    plotter.camera.view_angle = 90  # FOV
    plotter.camera.clipping_range = (1, 200000)

    text = "\n".join([
        f"Time: {time_video[i]:.1f} s",
        f"Airspeed: {dyn.speed[i]:.0f} m/s",
        f"Altitude: {dyn.altitude[i]:.0f} m",
        f"Altitude AGL: {altitude_agl[i]:.0f} m",
        # f"AoA: {dyn.alpha[i]:.1f} deg",
        # f"Throttle: {dyn.other_fields['throttle'][i]:.0%}",
        f"G-Force: {dyn.other_fields['accel_G'][i]:.1f} G",
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
