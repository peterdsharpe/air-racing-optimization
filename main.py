import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u

from airplane import airplane  # See cessna152.py for details.

initial_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    x_e=0,
    y_e=0,
    z_e=-500,
    speed=67 * u.knot,
    gamma=0,
    track=0,
    alpha=None,
    beta=None,
    bank=0
)

final_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    x_e=2000,
    y_e=0,
    z_e=0,
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
    final_state.z_e - initial_state.z_e,
])

### Initialize the problem
opti = asb.Opti()

### Define time. Note that the horizon length is unknown.
duration = opti.variable(init_guess=60, lower_bound=0)

N = 100  # Number of discretization points

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
        init_guess=np.linspace(
            initial_state.z_e,
            final_state.z_e,
            N
        )
    ),
    speed=opti.variable(
        init_guess=initial_state.speed,
        n_vars=N,
        lower_bound=1,
    ),
    gamma=opti.variable(
        init_guess=np.arctan(
            -d_position[2] /
            (d_position[0] ** 2 + d_position[1] ** 2) ** 0.5
        ),
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
    np.diff(dyn.alpha) < 2,
    np.diff(dyn.alpha) > -2,
    np.diff(np.degrees(dyn.bank)) < 20,
    np.diff(np.degrees(dyn.bank)) > -20,
    np.diff(np.degrees(dyn.track)) < 20,
    np.diff(np.degrees(dyn.track)) > -20,
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

# Add some extra drag to make the trajectory steeper and more interesting
# extra_drag = dyn.op_point.dynamic_pressure() * 0.3
# dyn.add_force(
#     Fx=-extra_drag, axes="wind"
# )

### Constrain the altitude to be above ground at all times
opti.subject_to(
    dyn.altitude / 1000 > 0
)

### Finalize the problem
dyn.constrain_derivatives(opti, time)  # Apply the dynamics constraints created up to this point

# opti.minimize(dyn.altitude[0] / 1000)  # Minimize the starting altitude
opti.minimize(duration)  # Minimize the starting altitude

### Solve it
sol = opti.solve(behavior_on_failure="return_last")

### Substitute the optimization variables in the dynamics instance with their solved values (in-place)
dyn = sol(dyn)

import pyvista as pv
plotter = dyn.draw(backend="pyvista", show=False)

from terrain_model.load_raw_data import terrain_data
terrain = pv.StructuredGrid(
    terrain_data["north_edges"],
    terrain_data["east_edges"],
)
plotter.add_mesh(terrain, color="green", opacity=0.5)
