from twoD_continuous_optimization import terrain_data_zoomed, solution_quantities, terrain_resolution
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from scipy import interpolate
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from airplane import airplane  # See cessna152.py for details.
import pyvista as pv
from aerosandbox.numpy import integrate_discrete as nid

N = solution_quantities["N"]

initial_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0),  # Placeholder
    x_e=terrain_data_zoomed["north_start"],
    y_e=terrain_data_zoomed["east_start"],
    z_e=-1201.11 * u.foot,
    speed=800 * u.foot / u.sec,
    gamma=None,
    track=np.radians(115.68559831535446847),
    alpha=None,
    beta=None,
    bank=0
)

final_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0),  # Placeholder
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

time = np.linspace(0, duration, N)
dt = duration / (N - 1)

### Create a dynamics instance

dyn = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=12000),
    x_e=opti.variable(
        init_guess=solution_quantities["north"]
    ),
    y_e=opti.variable(
        init_guess=solution_quantities["east"]
    ),
    z_e=opti.variable(
        init_guess=-solution_quantities["assumed_altitude"] - 100 * u.foot
    ),
    speed=opti.variable(
        init_guess=solution_quantities["assumed_airspeed"],
        n_vars=N,
        lower_bound=10,
    ),
    gamma=opti.variable(
        init_guess=solution_quantities["gamma"],
        n_vars=N,
        lower_bound=np.radians(-45),
        upper_bound=np.radians(45)
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
        lower_bound=np.radians(-150),
        upper_bound=np.radians(150),
    )
)

for state in list(dyn.state.keys()) + ["alpha", "beta", "bank"]:
    if getattr(initial_state, state) is not None:
        print(f"Constraining initial state for '{state}'...")
        if getattr(initial_state, state) == 0:
            opti.subject_to(
                getattr(dyn, state)[0] == getattr(initial_state, state)
            )
        else:
            opti.subject_to(
                getattr(dyn, state)[0] / getattr(initial_state, state) == 1
            )

    if getattr(final_state, state) is not None:
        print(f"Constraining final   state for '{state}'...")
        if getattr(final_state, state) == 0:
            opti.subject_to(
                getattr(dyn, state)[-1] == getattr(final_state, state)
            )
        else:
            opti.subject_to(
                getattr(dyn, state)[-1] / getattr(final_state, state) == 1
            )

# Add some constraints on rate of change of inputs (alpha and bank angle)
pitch_rate = np.diff(dyn.alpha) / dt  # deg/sec
roll_rate = np.diff(np.degrees(dyn.bank)) / dt  # deg/sec
opti.subject_to([
    np.diff(dyn.alpha) / 10 < 1,
    np.diff(dyn.alpha) / 10 > -1,
    np.diff(np.degrees(dyn.bank)) / 90 < 1,
    np.diff(np.degrees(dyn.bank)) / 90 > -1,
    np.diff(np.degrees(dyn.track)) / 60 < 1,
    np.diff(np.degrees(dyn.track)) / 60 > -1,
    roll_rate / 180 < 1,
    roll_rate / 180 > -1,
])

### Add in forces
dyn.add_gravity_force(g=9.81)
#
# aero = asb.AeroBuildup(
#     airplane=airplane,
#     op_point=dyn.op_point,
#     model_size="xsmall",
#     include_wave_drag=False,
# ).run()

wing_area = 38
CL = (2 * np.pi * np.radians(dyn.alpha))  # Very crude model
lift = dyn.op_point.dynamic_pressure() * wing_area * CL
accel_G = lift / dyn.mass_props.mass / 9.81
max_thrust = (79e3 * 2) * dyn.op_point.atmosphere.density() / 1.225
drag = max_thrust * (dyn.speed / (1.0 * 343)) ** 2  # Very crude model, such that sea-level equilibrium at M1.0
drag *= 1 + 1 * (
    (CL - 0.05)
) ** 2
# Add extra drag from roll rate inputs
roll_rate_drag_multiplier = 4 * 0.2 * (  # You go 20% slower at max roll rate
        np.gradient(np.degrees(dyn.bank), time)  # degrees / sec
        / 180  # Max roll rate is 360 deg/sec
) ** 2
drag *= 1 + roll_rate_drag_multiplier

throttle = opti.variable(
    init_guess=1,
    n_vars=N,
    lower_bound=0,
    upper_bound=1,
)
# throttle = 1
thrust = throttle * max_thrust

dyn.add_force(
    thrust - drag, 0, -lift,
    axes="wind"
)

from terrain_model.interpolated_model import get_elevation_interpolated_north_east

terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=dyn.x_e,
    query_points_east=dyn.y_e,
    resolution=terrain_resolution,
    terrain_data=terrain_data_zoomed
)
opti.subject_to([
    (dyn.y_e - 1) / 1e4 > terrain_data_zoomed["east_edges"][0] / 1e4,
    (dyn.y_e + 1) / 1e4 < terrain_data_zoomed["east_edges"][-1] / 1e4,
    (dyn.x_e - 1) / 1e4 > terrain_data_zoomed["north_edges"][0] / 1e4,
    (dyn.x_e + 1) / 1e4 < terrain_data_zoomed["north_edges"][-1] / 1e4,
])
altitude_agl = dyn.altitude - terrain_altitude

altitude_agl_limit = 50 * u.foot

opti.subject_to([
    altitude_agl / altitude_agl_limit > 1,
    nid.integrate_discrete_intervals(
        altitude_agl,
        multiply_by_dx=False,
        method="cubic"
    ) / altitude_agl_limit > 1,
])


# goal_direction = np.array([
#     terrain_data_zoomed["east_end"] - terrain_data_zoomed["east_start"],
#     terrain_data_zoomed["north_end"] - terrain_data_zoomed["north_start"]
# ])
# goal_direction /= np.linalg.norm(goal_direction)
#
# # opti.subject_to([
# #     np.dot(
# #         [np.sin(dyn.track), np.cos(dyn.track)],
# #         goal_direction,
# #         manual=True
# #     ) > np.cosd(75)  # Max deviation from goal direction
# # ])


### Add G-force constraints
# accel_G = -aero["F_w"][2] / dyn.mass_props.mass / 9.81
opti.subject_to([
    accel_G < 9,
    accel_G > -0.5
])

### Finalize the problem
dyn.constrain_derivatives(
    opti,
    time,
    method="trapz",
)  # Apply the dynamics constraints created up to this point

wiggliness = np.mean(
    nid.integrate_discrete_squared_curvature(
        dyn.alpha / 5,
        time,
    )
    + nid.integrate_discrete_squared_curvature(
        dyn.bank / np.radians(90),
        time,
    )
    + nid.integrate_discrete_squared_curvature(
        throttle / 1,
        time,
    )
)

opti.minimize(
    (np.maximum(duration, 0) / 240) ** 2
    + 15 * np.mean(dyn.altitude) / 440
    + 1e-3 * wiggliness
    # + 1e-3 * np.mean(dyn.bank ** 2)
)

sol = opti.solve(
    behavior_on_failure="return_last",
    max_iter=1000000,
    options={
        # "ipopt.mu_strategy": "monotone"
    }
)

dyn.other_fields = {
    "throttle": throttle,
    "thrust": thrust,
    "lift": lift,
    "drag": drag,
    "terrain_altitude": terrain_altitude,
    "altitude_agl": altitude_agl,
    "accel_G": accel_G,
    "wiggliness": wiggliness,
    "time": time,
}

dyn = sol(dyn)

if __name__ == '__main__':

    dyn.save(
        "./final_results/dyn.asb"
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    plt.plot(
        solution_quantities["east"],
        solution_quantities["north"],
        ":",
        color="lime",
        linewidth=1,
        zorder=4
    )
    plt.plot(
        sol(dyn.y_e),
        sol(dyn.x_e),
        "-",
        color="red",
        linewidth=2,
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
