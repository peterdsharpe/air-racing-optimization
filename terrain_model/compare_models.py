from interpolated_model import get_elevation_interpolated_north_east, get_elevation_interpolated_lat_lon
from fourier_model import get_elevation_fourier_north_east, get_elevation_fourier_lat_lon
from load_raw_data import terrain_data
import numpy as np
import casadi as cas
from aerosandbox.tools.code_benchmarking import Timer

N = 200

lat_query = np.linspace(
    47,
    46,
    N
)
lon_query = np.linspace(
    -123,
    -120,
    N
)

# lat_query = cas.MX(lat_query)
# lon_query = cas.MX(lon_query)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()

# print("Printing true data...")
# from scipy import interpolate
#
# interpolate.

with Timer("Interpolation"):
    elev_interpolated = get_elevation_interpolated_lat_lon(
        lat_query,
        lon_query,
        resolution=(100, 300)
    )
    elev_interpolated = np.array(cas.evalf(elev_interpolated))
plt.plot(
    np.array(cas.evalf(lon_query)),
    elev_interpolated,
    label="Interpolated",
)

with Timer("Fourier"):
    elev_fourier = get_elevation_fourier_lat_lon(
        lat_query,
        lon_query,
        resolution=(1000, 1000)
    )
plt.plot(
    lon_query,
    elev_fourier,
    label="Fourier",
)

print("Done")

p.show_plot()
