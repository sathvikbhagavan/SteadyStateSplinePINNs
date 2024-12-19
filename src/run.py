import torch
import argparse
from unet import *
from utils import *
from spline_pinn import *
from pinn import *

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Process model type")
parser.add_argument(
    "--model",
    type=str,
    choices=["pinn", "sssplinepinn"],
    required=True,
    help="Specify the model type: 'pinn' or 'sssplinepinn'",
)

args = parser.parse_args()

# This can be "dp0" or "dp11" which corresponds to 1000 or 100k Reynolds number
REYNOLDS = "dp11"
DATA_FOLDER = f"./preprocessedData/with_T/{REYNOLDS}"
MODEL_FOLDER = f"../best_models/{args.model}_{REYNOLDS}"

print(f"Selected model: {args.model}, folder: {REYNOLDS}")

device = "cpu"
torch.set_default_device(device)

if args.model == "sssplinepinn":
    unet_model = UNet3D().to(device)
    unet_model.load_state_dict(
        torch.load(
            f"{MODEL_FOLDER}/unet_model.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )

    obj = trimesh.load("./Baseline_ML4Science.stl")
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
    fields = [
        ("vx", vx_pred),
        ("vy", vy_pred),
        ("vz", vz_pred),
        ("P", p_pred * 10**5),
        ("T", T_pred),
    ]
    plot_fields(fields, all_points)

if args.model == "pinn":
    hidden_dim = 128
    num_layer = 4
    pinn_model = PINNs(
        in_dim=3, hidden_dim=hidden_dim, out_dim=5, num_layer=num_layer
    ).to(device)
    pinn_model.load_state_dict(
        torch.load(
            f"{MODEL_FOLDER}/pinn_model.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )

    all_points = torch.tensor(
        np.concatenate(
            (
                np.load(os.path.join(DATA_FOLDER, "vel_x_inlet.npy"))[:, :3],
                np.load(os.path.join(DATA_FOLDER, "vel_x.npy"))[:, :3],
            )
        )
        * 1000
    )
    all_fields = pinn_model(all_points)
    vx_pred = all_fields[:, 0].cpu().detach().numpy()
    vy_pred = all_fields[:, 1].cpu().detach().numpy()
    vz_pred = all_fields[:, 2].cpu().detach().numpy()
    p_pred = all_fields[:, 3].cpu().detach().numpy()
    T_pred = (all_fields[:, 4] * 1000).cpu().detach().numpy()
    fields = [
        ("vx", vx_pred),
        ("vy", vy_pred),
        ("vz", vz_pred),
        ("P", p_pred * 10**5),
        ("T", T_pred),
    ]
    plot_fields(fields, all_points)
