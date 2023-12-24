from pathfinding import terrain_data_zoomed, terrain_cost_heuristic, path
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

N = 1000

### Downsample the initial path
ds_path = (
                  np.diff(path[:, 0]) ** 2 +
                  np.diff(path[:, 1]) ** 2
          ) ** 0.5
s_path = np.concatenate([
    [0],
    np.cumsum(ds_path)
])
path_downsampled = interpolate.interp1d(
    s_path,
    path,
    kind="cubic",
    axis=0
)(np.linspace(0, s_path[-1], N))

path_downsampled = ndimage.gaussian_filter(
    path_downsampled,
    sigma=1,
    axes=[0],
)

path_downsampled_length = np.sum(
    np.sum(np.diff(path_downsampled, axis=0) ** 2, axis=1) ** 0.5
)

assumed_airspeed = 0.8 * 343

### Initialize the problem
print("Solving 2D problem...")
opti = asb.Opti()

duration = opti.variable(
    init_guess=path_downsampled_length / assumed_airspeed,
    lower_bound=0
)

time = np.linspace(0, duration, N)

east = opti.variable(
    init_guess=path_downsampled[:, 0],
)
north = opti.variable(
    init_guess=path_downsampled[:, 1],
)
opti.subject_to([
    east[0] == terrain_data_zoomed["east_start"],
    north[0] == terrain_data_zoomed["north_start"],
    east[-1] == terrain_data_zoomed["east_end"],
    north[-1] == terrain_data_zoomed["north_end"],
])

dt = np.diff(time)
ds = (
        np.diff(east) ** 2 +
        np.diff(north) ** 2
) ** 0.5
opti.subject_to(
    ds == (assumed_airspeed * dt)
)

from terrain_model.interpolated_model import get_elevation_interpolated_north_east

terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=north,
    query_points_east=east,
    resolution=(800, 2400),
    terrain_data=terrain_data_zoomed
)
opti.subject_to([
    east  / 1e4 > terrain_data_zoomed["east_edges"][0] / 1e4,
    east  / 1e4 < terrain_data_zoomed["east_edges"][-1] / 1e4,
    north / 1e4  > terrain_data_zoomed["north_edges"][0] / 1e4,
    north / 1e4  < terrain_data_zoomed["north_edges"][-1] / 1e4,
])

# from terrain_model.fourier_model import get_elevation_fourier_north_east
#
# terrain_altitude = get_elevation_fourier_north_east(
#     query_points_north=north,
#     query_points_east=east,
#     resolution=(200, 800)
# )

opti.minimize(
    duration / 500 +
    15 * np.mean(terrain_altitude) / 1300
)

sol = opti.solve(
    options={
        # "ipopt.hessian_approximation": "limited-memory",
        # "ipopt.limited_memory_max_history": (2 * N),
        # "ipopt.derivative_test": "first-order",
    },
    max_iter=10000
)
# opti.set_initial_from_sol(sol)
#
# ### G constraint
# allowed_G = 9
# allowable_discrete_track_change = (
#         (allowed_G * 9.81) / assumed_airspeed * (duration / N)
# )
#
# track = np.arctan2(
#     np.diff(east),
#     np.diff(north),
# )
# opti.subject_to([
#     np.diff(track) < allowable_discrete_track_change,
#     np.diff(track) > -allowable_discrete_track_change,
# ])
#
# sol = opti.solve()

solution_quantities = {
    "N": N,
    "east": sol(east),
    "north": sol(north),
    "ds": sol(ds).mean(), # Valid since all are constrained to be equal
    "dt": sol(dt).mean(), # Valid since all are constrained to be equal
    "assumed_airspeed": assumed_airspeed,
    "duration": sol(duration),
    "terrain_altitude": sol(terrain_altitude),
}
solution_quantities["path"] = np.stack(
    [
        solution_quantities["east"],
        solution_quantities["north"]
    ],
    axis=1
)
solution_quantities["assumed_altitude"] = ndimage.gaussian_filter(
    solution_quantities["terrain_altitude"],
    2,
    mode="nearest"
)

solution_quantities["track"] = np.arctan2(
    np.gradient(solution_quantities["east"]),
    np.gradient(solution_quantities["north"]),
)

solution_quantities["gamma"] = np.arctan2(
    np.gradient(solution_quantities["assumed_altitude"]),
    solution_quantities["ds"],
)
solution_quantities["load_factor_vertical"] = 1 + (
    solution_quantities["assumed_airspeed"] * np.gradient(solution_quantities["gamma"]) / solution_quantities["dt"]
) / 9.81
solution_quantities["load_factor_horizontal"] = (
    solution_quantities["assumed_airspeed"] * np.gradient(solution_quantities["track"]) / solution_quantities["dt"]
) / 9.81
solution_quantities["bank"] = np.arctan2(
    solution_quantities["load_factor_horizontal"],
    solution_quantities["load_factor_vertical"],
)


path = np.stack(
    sol([
        east,
        north
    ]),
    axis=1
)
duration = sol(duration)

if __name__ == '__main__':

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    plt.plot(
        path[:, 0],
        path[:, 1],
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
        "2D Continuous Optimization",
        rotate_axis_labels=False,
        savefig=[
            # f"./figures/trajectory_{resolution}.svg",
        ]
    )
