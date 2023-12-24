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

### Initialize the problem
print("Solving 2D problem...")
opti = asb.Opti()

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

ds_squared = (
        np.diff(east) ** 2 +
        np.diff(north) ** 2
)

opti.subject_to(
    np.diff(ds_squared) == 0
)

assumed_airspeed = 0.8 * 343
duration = np.sum(ds_squared ** 0.5) / assumed_airspeed

from terrain_model.interpolated_model import get_elevation_interpolated_north_east

terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=north,
    query_points_east=east,
    resolution=(500, 1500),
    terrain_data=terrain_data_zoomed
)

# from terrain_model.fourier_model import get_elevation_fourier_north_east
#
# terrain_altitude = get_elevation_fourier_north_east(
#     query_points_north=north,
#     query_points_east=east,
#     resolution=(200, 800)
# )

opti.minimize(
    duration / 500 +
    5 * np.mean(terrain_altitude) / 1300
)

sol = opti.solve(
    options={
        # "ipopt.hessian_approximation": "limited-memory",
        # "ipopt.limited_memory_max_history": (2 * N),
        # "ipopt.derivative_test": "first-order",
    }
)

solution_quantities = {
    "N": N,
    "east": sol(east),
    "north": sol(north),
    "ds_squared": sol(ds_squared),
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

solution_quantities["ds"] = np.ones(N) * solution_quantities["ds_squared"].mean() ** 0.5
solution_quantities["dt"] = solution_quantities["ds"] / solution_quantities["assumed_airspeed"]

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
