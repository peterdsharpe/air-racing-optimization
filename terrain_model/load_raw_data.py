from pathlib import Path
import rasterio
import numpy as np

terrain_folder = Path(__file__).parent / "raw_data"

terrain_files = terrain_folder.glob("*.tif")


def assert_close_to_integer_and_round(x, atol=1e-3):
    rounded_x = np.round(x)
    assert np.allclose(x, rounded_x, atol=atol)
    return int(rounded_x)


tile_data = []

### Load the tiles, which could be in any order
for terrain_file in terrain_files:
    with rasterio.open(terrain_file) as dataset:

        # Read the dataset's data into a 2D NumPy array
        elev = dataset.read(1)

        # Calculate latitudes and longitudes from the dataset's metadata
        transform = dataset.transform

    lat_edges = np.linspace(transform[5], transform[5] + dataset.width * transform[4], elev.shape[0])
    lon_edges = np.linspace(transform[2], transform[2] + dataset.height * transform[0], elev.shape[1])

    lat_argsort = np.argsort(lat_edges)
    lon_argsort = np.argsort(lon_edges)

    lat_edges = lat_edges[lat_argsort]
    lon_edges = lon_edges[lon_argsort]
    elev = elev[lat_argsort, :][:, lon_argsort]

    # Trim the edges to those strictly inside the lon/lat bounds
    if elev.shape == (3612, 3612):
        lat_edges = np.linspace(
            assert_close_to_integer_and_round(transform[5] + 6 * transform[4]),
            assert_close_to_integer_and_round(transform[5] + 3606 * transform[4]),
            3600
        )
        lon_edges = np.linspace(
            assert_close_to_integer_and_round(transform[2] + 6 * transform[0]),
            assert_close_to_integer_and_round(transform[2] + 3606 * transform[0]),
            3600
        )
        elev = elev[6:3606, 6:3606]
    else:
        raise ValueError(
            "Currently, this logic only works for 1-arcsecond data on 1deg x 1deg grids from USGS, which have 6 extra datapoints on each axis. It seems you have a different data format; add your logic to the code to make this work.")

    tile_data.append({
        "elev"     : elev,
        "lat_edges": lat_edges,
        "lon_edges": lon_edges,
        "min_lat"  : assert_close_to_integer_and_round(lat_edges.min()),
        "min_lon"  : assert_close_to_integer_and_round(lon_edges.min()),
    })

### Merge the tiles into one big grid
unique_lats = np.unique([tile["min_lat"] for tile in tile_data])
unique_lons = np.unique([tile["min_lon"] for tile in tile_data])

tile_array = np.empty((
    len(unique_lats),
    len(unique_lons),
), dtype="O")

for tile in tile_data:
    tile_array[
        np.where(unique_lats == tile["min_lat"])[0][0],
        np.where(unique_lons == tile["min_lon"])[0][0],
    ] = tile

elev = np.empty((
    3599 * len(unique_lats) + 1,
    3599 * len(unique_lons) + 1
))
elev[:] = np.nan

for i in range(tile_array.shape[0]):
    for j in range(tile_array.shape[1]):
        tile = tile_array[i, j]

        elev[
        i * 3599:i * 3599 + 3600,
        j * 3599:j * 3599 + 3600,
        ] = tile["elev"]

lat_edges = np.unique([tile["lat_edges"] for tile in sorted(tile_data, key=lambda tile: tile["min_lat"])])
lon_edges = np.unique([tile["lon_edges"] for tile in sorted(tile_data, key=lambda tile: tile["min_lon"])])

if not (
        (len(lat_edges) == elev.shape[0]) and
        (len(lon_edges) == elev.shape[1]) and
        (np.sort(lat_edges) == lat_edges).all() and
        (np.sort(lon_edges) == lon_edges).all()
):
    raise ValueError(
        "Something went wrong with the merging of the tiles, likely because different tiles have different gridding."
    )

datum_lat = lat_edges.mean()
datum_lon = lon_edges.mean()
cos_lat = np.cos(np.deg2rad(datum_lat))

north_edges = (lat_edges - datum_lat) * 1e6 / 9
east_edges = (lon_edges - datum_lon) * 1e6 / 9 * cos_lat

terrain_data = {
    "elev"       : elev,
    "lat_edges"  : lat_edges,
    "lon_edges"  : lon_edges,
    "north_edges": north_edges,
    "east_edges" : east_edges,
}

def lat_lon_to_north_east(lat, lon):
    return (
        (lat - datum_lat) * 1e6 / 9,
        (lon - datum_lon) * 1e6 / 9 * cos_lat,
    )

def north_east_to_lat_lon(north, east):
    return (
        north * 9 / 1e6 + datum_lat,
        east * 9 / 1e6 / cos_lat + datum_lon,
    )

def lat_lon_to_normalized_coordinates(lat, lon):
    return (
        (lat - lat_edges.min()) / (lat_edges.max() - lat_edges.min()),
        (lon - lon_edges.min()) / (lon_edges.max() - lon_edges.min()),
    )

def north_east_to_normalized_coordinates(north, east):
    return (
        (north - north_edges.min()) / (north_edges.max() - north_edges.min()),
        (east - east_edges.min()) / (east_edges.max() - east_edges.min()),
    )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(
        figsize=(
            1 + 5 * len(unique_lons),
            1 + 5 * len(unique_lats)
        )
    )
    plt.imshow(
        elev,
        cmap='terrain',
        origin="lower",
        extent=[
            lon_edges.min(),
            lon_edges.max(),
            lat_edges.min(),
            lat_edges.max()
        ]
    )
    p.equal()
    plt.colorbar(label='Elevation [m]')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('Contour Plot of Elevation around Riffe Lake')
    p.show_plot(
        rotate_axis_labels=False,
    )
