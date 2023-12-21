from terrain_model.load_raw_data import terrain_data
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(
    figsize=(16, 6)
)
plt.imshow(
    terrain_data["elev"],
    cmap='terrain',
    origin="lower",
    extent=(
        terrain_data["east_edges"][0],
        terrain_data["east_edges"][-1],
        terrain_data["north_edges"][0],
        terrain_data["north_edges"][-1],
    ),
)
plt.colorbar(
    label="Elevation [m]"
)
p.equal()
p.show_plot(
    rotate_axis_labels=False,
    savefig=[
        "../figures/terrain.svg",
    ]
)