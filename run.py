import torch
import argparse
from src.unet import *
from src.utils import *
from src.spline_pinn import *

parser = argparse.ArgumentParser(description="Process model type")
parser.add_argument(
    "--model",
    type=str,
    choices=["pinn", "splinepinn"],
    required=True,
    help="Specify the model type: 'pinn' or 'splinepinn'",
)

args = parser.parse_args()
print(f"Selected model: {args.model}")

# This can be "dp0" or "dp11" which corresponds to 1000 or 100k Reynolds number
REYNOLDS = "dp11"
DATA_FOLDER = f"./preprocessedData/with_T/{REYNOLDS}"

device = "cpu"
torch.set_default_device(device)

if args.model == "splinepinn":
    unet_model = UNet3D().to(device)
    unet_model.load_state_dict(
        torch.load(
            f"./best_models/{REYNOLDS}/unet_model.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )

    obj = trimesh.load("./src/Baseline_ML4Science.stl")
    grid_resolution = np.array([512, 64, 16])
    binary_mask = get_binary_mask(obj, grid_resolution)
    step = obj.bounding_box.extents / (grid_resolution - 1)

    unet_input = prepare_mesh_for_unet(binary_mask).to(device)
    spline_coeff = unet_model(unet_input)[0]

    all_points = torch.tensor(
        np.concatenate(
            (
                np.load(os.path.join(DATA_FOLDER, "vel_x_inlet.npy"))[:, :3],
                np.load(os.path.join(DATA_FOLDER, "vel_x.npy"))[:, :3],
            )
        )
        * 1000
    )

    x, y, z, x_supports, y_supports, z_supports = get_support_points(
        all_points, step, grid_resolution
    )

    vx_pred, vy_pred, vz_pred, p_pred, T_pred = get_fields(
        spline_coeff, all_points, step, grid_resolution
    )

    vx_pred = vx_pred.cpu().detach().numpy()
    vy_pred = vy_pred.cpu().detach().numpy()
    vz_pred = vz_pred.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy() * 10**5
    T_pred = T_pred.cpu().detach().numpy()
    plot_aginast_data(DATA_FOLDER, vx_pred, vy_pred, vz_pred, p_pred, T_pred)
