import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from terrain_model.load_raw_data import lat_lon_to_north_east, terrain_data
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pyvista as pv
from airplane import airplane  # See cessna152.py for details.

lat_i = 46 + 32 / 60 + 53.84 / 3600
lon_i = -(122 + 28 / 60 + 8.98 / 3600)
north_i, east_i = lat_lon_to_north_east(lat_i, lon_i)

north_f, east_f = lat_lon_to_north_east(
    lat=46 + 16 / 60 + 37.56 / 3600,
    lon=-(121 + 34 / 60 + 38.70 / 3600),
)

initial_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    x_e=north_i,
    y_e=east_i,
    # z_e=-(730 * u.foot + 3000 * u.foot),
    z_e=None,
    speed=67 * u.knot,
    gamma=0,
    track=None,
    alpha=None,
    beta=None,
    bank=0
)

final_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    x_e=north_f,
    y_e=east_f,
    # z_e=-(8000 * u.foot),
    z_e=None,
    speed=None,
    gamma=None,
    track=None,
    alpha=None,
    beta=None,
    bank=None,
)

d_position = np.array([
    final_state.x_e - initial_state.x_e,
    final_state.y_e - initial_state.y_e,
    # final_state.z_e - initial_state.z_e,
])

### Initialize the problem
opti = asb.Opti()

### Define time. Note that the horizon length is unknown.
duration = opti.variable(init_guess=60, lower_bound=0)

N = 200  # Number of discretization points

time = np.linspace(
    0,
    duration,
    N
)

### Create a dynamics instance

dyn = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=1151.8 * u.lbm),
    x_e=opti.variable(
        init_guess=np.linspace(
            initial_state.x_e,
            final_state.x_e,
            N
        )
    ),
    y_e=opti.variable(
        init_guess=np.linspace(
            initial_state.y_e,
            final_state.y_e,
            N
        )
    ),
    z_e=opti.variable(
        # init_guess=np.linspace(
        #     initial_state.z_e,
        #     final_state.z_e,
        #     N
        # )
        init_guess=np.ones(N) * (-5000 * u.foot),
        n_vars=N,
    ),
    speed=opti.variable(
        init_guess=initial_state.speed,
        n_vars=N,
        lower_bound=10,
    ),
    gamma=opti.variable(
        init_guess=0,
        scale=0.1,
        n_vars=N,
    ),
    track=opti.variable(
        init_guess=np.arctan2(
            d_position[1],
            d_position[0],
        ),
        n_vars=N,
    ),
    alpha=opti.variable(
        init_guess=5,
        n_vars=N,
        lower_bound=-15,
        upper_bound=15
    ),
    beta=np.zeros(N),
    bank=opti.variable(
        init_guess=np.radians(0),
        n_vars=N,
        lower_bound=np.radians(-90),
        upper_bound=np.radians(90),
    )
)

for state in dyn.state.keys():
    if initial_state.state[state] is not None:
        print(f"Constraining initial state for '{state}'...")
        opti.subject_to(
            dyn.state[state][0] == initial_state.state[state]
        )

    if final_state.state[state] is not None:
        print(f"Constraining final   state for '{state}'...")
        opti.subject_to(
            dyn.state[state][-1] == final_state.state[state]
        )

# Add some constraints on rate of change of inputs (alpha and bank angle)
pitch_rate = np.diff(dyn.alpha) / np.diff(time)  # deg/sec
roll_rate = np.diff(np.degrees(dyn.bank)) / np.diff(time)  # deg/sec
opti.subject_to([
    np.diff(dyn.alpha) < 3,
    np.diff(dyn.alpha) > -3,
    np.diff(np.degrees(dyn.bank)) < 15,
    np.diff(np.degrees(dyn.bank)) > -15,
    np.diff(np.degrees(dyn.track)) < 20,
    np.diff(np.degrees(dyn.track)) > -20,
    dyn.speed[-1] > dyn.speed[0]
])

### Add in forces
dyn.add_gravity_force(g=9.81)

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point,
    model_size="xsmall",
).run()

dyn.add_force(
    *aero["F_w"],
    axes="wind"
)

thrust = 0.5 * dyn.mass_props.mass * 9.81

dyn.add_force(
    *[thrust, 0, 0],
    axes="wind"
)

# Add some extra drag to make the trajectory steeper and more interesting
# extra_drag = dyn.op_point.dynamic_pressure() * 0.3
# dyn.add_force(
#     Fx=-extra_drag, axes="wind"
# )

### Constrain the altitude to be above ground at all times
# opti.subject_to(
#     dyn.altitude / 1000 > 0
# )

from terrain_model.interpolated_model import get_elevation_interpolated_north_east
terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=dyn.x_e,
    query_points_east=dyn.y_e,
    resolution=(200, 600)
)
altitude_agl = dyn.altitude - terrain_altitude

opti.subject_to([
    altitude_agl / 1e3 > 30 * u.foot / 1e3,
    altitude_agl / 1e3 < 1500 * u.foot / 1e3
])

### Finalize the problem
dyn.constrain_derivatives(opti, time)  # Apply the dynamics constraints created up to this point

opti.minimize(
    duration / 500 +
    2.5 * np.mean(dyn.altitude) / 1300
)  # Minimize the starting altitude

### Add G-force constraints
accel_G = -aero["F_w"][2] / dyn.mass_props.mass / 9.81
opti.subject_to([
    accel_G < 6,
    accel_G > -3
])

### Solve it
sol = opti.solve(behavior_on_failure="return_last")

### Substitute the optimization variables in the dynamics instance with their solved values (in-place)
dyn = sol(dyn)

print("Plotting...")
fig, ax = plt.subplots(
    figsize=(16,6)
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


plt.imshow(
    terrain_data["elev"][i_lims[0]:i_lims[1], j_lims[0]:j_lims[1]],
    cmap='terrain',
    origin="lower",
    extent=(
        terrain_data["east_edges"][j_lims[0]],
        terrain_data["east_edges"][j_lims[-1]],
        terrain_data["north_edges"][i_lims[0]],
        terrain_data["north_edges"][i_lims[-1]],
    ),
)
plt.plot(
    dyn.y_e,
    dyn.x_e,
    color="red",
    alpha=0.8,
    linewidth=2,
)
p.equal()

p.show_plot()

plotter = dyn.draw(
    backend="pyvista",
    show=False,
    n_vehicles_to_draw=100,
    scale_vehicle_model=1e3,
)

grid = pv.RectilinearGrid(
    terrain_data["north_edges"],
    terrain_data["east_edges"],
)
grid["elev"] = terrain_data["elev"].T.flatten()
grid = grid.warp_by_scalar("elev", factor=-1)
plotter.add_mesh(
    grid,
    scalars=terrain_data["elev"].T,
    cmap='terrain',
    specular=0.5,
    specular_power=15,
)
plotter.enable_terrain_style()
plotter.show()
