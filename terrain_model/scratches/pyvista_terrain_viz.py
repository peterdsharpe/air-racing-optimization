from terrain_model.load_raw_data import terrain_data

import pyvista as pv

plotter = pv.Plotter()
grid = pv.RectilinearGrid(
    terrain_data["north_edges"],
    terrain_data["east_edges"],
)
grid["elev"] = terrain_data["elev"].T.flatten()
grid = grid.warp_by_scalar("elev", factor=-1)
plotter.add_mesh(
    grid,
    scalars=terrain_data["elev"].T,
    cmap='terrain'
)
# pv.enable_terrain_style()
plotter.show_grid()
plotter.show_axes()

plotter.camera.up = (0, 0, -1)
plotter.camera.Azimuth(90)
plotter.camera.Elevation(60)

plotter.show()
