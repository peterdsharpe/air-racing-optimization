from twoD_continuous_optimization import terrain_data_zoomed, solution_quantities
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from scipy import interpolate
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from airplane import airplane  # See cessna152.py for details.
import pyvista as pv

N = solution_quantities["N"]

initial_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0), # Placeholder
    x_e=terrain_data_zoomed["north_start"],
    y_e=terrain_data_zoomed["east_start"],
    z_e=None,
    speed=67 * u.knot,
    gamma=0,
    track=None,
    alpha=None,
    beta=None,
    bank=0
)

final_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0), # Placeholder
    x_e=terrain_data_zoomed["north_end"],
    y_e=terrain_data_zoomed["east_end"],
    z_e=None,
    speed=None,
    gamma=None,
    track=None,
    alpha=None,
    beta=None,
    bank=None,
)


### Initialize the problem
print("Solving 3D problem...")
opti = asb.Opti()

### Define time. Note that the horizon length is unknown.
duration = opti.variable(init_guess=solution_quantities["duration"], lower_bound=0)

time = np.linspace(
    0,
    duration,
    N
)

### Create a dynamics instance

dyn = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=1151.8 * u.lbm),
    x_e=opti.variable(
        init_guess=solution_quantities["north"]
    ),
    y_e=opti.variable(
        init_guess=solution_quantities["east"]
    ),
    z_e=opti.variable(
        init_guess=-solution_quantities["assumed_altitude"]
    ),
    speed=opti.variable(
        init_guess=solution_quantities["assumed_airspeed"],
        n_vars=N,
        lower_bound=10,
    ),
    gamma=opti.variable(
        init_guess=solution_quantities["gamma"],
        n_vars=N,
    ),
    track=opti.variable(
        init_guess=solution_quantities["track"],
        n_vars=N,
    ),
    alpha=opti.variable(
        init_guess=1,
        n_vars=N,
        lower_bound=-15,
        upper_bound=15
    ),
    beta=np.zeros(N),
    bank=opti.variable(
        init_guess=solution_quantities["bank"],
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
    np.diff(dyn.alpha) < 10,
    np.diff(dyn.alpha) > -10,
    np.diff(np.degrees(dyn.bank)) < 180,
    np.diff(np.degrees(dyn.bank)) > -180,
    np.diff(np.degrees(dyn.track)) < 60,
    np.diff(np.degrees(dyn.track)) > -60,
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

thrust = 1.0 * dyn.mass_props.mass * 9.81
dyn.add_force(
    *[thrust, 0, 0],
    axes="wind"
)

# Add extra drag from roll rate inputs
roll_rate_drag_multiplier = 2 * 0.2 * ( # You go 20% slower at max roll rate
    (np.diff(np.degrees(dyn.bank)) / np.diff(time)) # degrees / sec
    / 360 # Max roll rate is 360 deg/sec
) ** 2
roll_rate_drag_multiplier = np.concatenate([
    [0],
    roll_rate_drag_multiplier
])
dyn.add_force(
    aero["F_w"][0] * roll_rate_drag_multiplier, 0, 0,
    axes="wind"
)

from terrain_model.interpolated_model import get_elevation_interpolated_north_east

terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=dyn.x_e,
    query_points_east=dyn.y_e,
    resolution=(600, 1800),
    terrain_data=terrain_data_zoomed
)
altitude_agl = dyn.altitude - terrain_altitude

opti.subject_to([
    altitude_agl / (50 * u.foot) > 1,
])

### Add G-force constraints
accel_G = -aero["F_w"][2] / dyn.mass_props.mass / 9.81
opti.subject_to([
    accel_G < 9,
    accel_G > -0.5
])

### Finalize the problem
dyn.constrain_derivatives(opti, time)  # Apply the dynamics constraints created up to this point

opti.minimize(
    duration / 500 +
    5 * np.mean(dyn.altitude) / 1300
)

sol = opti.solve()

dyn=sol(dyn)

if __name__ == '__main__':

    dyn.save(
        "./final_results/dyn.asb"
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    plt.plot(
        sol(dyn.y_e),
        sol(dyn.x_e),
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

    plotter = dyn.draw(
        backend="pyvista",
        show=False,
        n_vehicles_to_draw=N // 200,
        scale_vehicle_model=800 / (N / 100),
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
