import trimesh
import numpy as np
import torch


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


def get_inlet_surface_points(obj, num_points):
    threshold = 1e-5
    faces_x_zero = [
        i
        for i, face in enumerate(obj.faces)
        if np.all(
            np.abs(obj.vertices[face, 0]) < threshold
        )  # Check if all vertices' x-coordinates are 0
    ]
    subset_mesh = obj.submesh([faces_x_zero], only_watertight=False)[0]
    points, _ = trimesh.sample.sample_surface(subset_mesh, count=num_points)
    inlet_surface_points = torch.tensor(points, dtype=torch.float64)
    inlet_surface_labels = torch.ones(inlet_surface_points.size(0), dtype=torch.int64)
    return inlet_surface_points, inlet_surface_labels


def get_other_surface_points(obj, num_points):
    threshold = 1e-5
    points, _ = trimesh.sample.sample_surface(obj, count=num_points)
    filtered_points = points[
        (np.abs(points[:, 0]) > threshold) & 
        ~((953 - threshold <= points[:, 0]) & (points[:, 0] <= 953 + threshold))
    ]
    other_surface_points = torch.tensor(filtered_points, dtype=torch.float64)
    other_surface_labels = 2 * torch.ones(
        other_surface_points.size(0), dtype=torch.int64
    )
    return other_surface_points, other_surface_labels


def get_volume_points(obj, num_points):
    volume_points = torch.tensor(
        trimesh.sample.volume_mesh(obj, num_points), dtype=torch.float64
    )
    volume_labels = torch.zeros(volume_points.size(0), dtype=torch.int64)
    return volume_points, volume_labels
