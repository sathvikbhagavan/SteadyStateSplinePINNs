import trimesh
import numpy as np

def get_binary_mask(model, grid_resolution):
    bounds = model.bounds
    min_bound, max_bound = bounds[0], bounds[1]
    x = np.linspace(min_bound[0], max_bound[0], grid_resolution[0])
    y = np.linspace(min_bound[1], max_bound[1], grid_resolution[1])
    z = np.linspace(min_bound[2], max_bound[2], grid_resolution[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    inside_points = grid_points[model.contains(grid_points)]
    binary_volume = np.zeros(grid_resolution, dtype=np.uint8)
    for point in inside_points:
        # Compute the index of the point in the grid
        index_x = np.searchsorted(x, point[0])
        index_y = np.searchsorted(y, point[1])
        index_z = np.searchsorted(z, point[2])
        # Set the corresponding position in the binary volume to 1
        if (
            0 <= index_x < grid_resolution[0]
            and 0 <= index_y < grid_resolution[1]
            and 0 <= index_z < grid_resolution[2]
        ):
            binary_volume[index_x, index_y, index_z] = 1
    return binary_volume
