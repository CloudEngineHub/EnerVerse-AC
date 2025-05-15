import sys
import os
import torch
import numpy as np
import math
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.general_utils import load_checkpoints, instantiate_from_config
from lvdm.data.get_actions import get_actions
from lvdm.data.statistics import StatisticInfo


def load_model(config):
    model = instantiate_from_config(config.model)
    model = load_checkpoints(model, config.model, ignore_mismatched_sizes=False)
    return model


def load_config(args):
    config_file = args.config_path
    config = OmegaConf.load(config_file)
    config.model.pretrained_checkpoint = args.ckp_path
    return config


def get_image(img_path, n):
    img = np.array(Image.open(img_path))
    img = torch.from_numpy(img).float().permute(2,0,1)/255.0
    img = img.unsqueeze(1).repeat(1,n,1,1)
    return img

def get_action_bias_std(domain_name):
    return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)

def get_action(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
):
    abs_act = np.load(action_path)

    if (abs_act.shape[0] < n_chunk * chunk + n_previous):
        raise ValueError(f"Num of Action Timestep {abs_act.shape[0]} smaller than {n_previous}+{n_chunk}*{n_chunk}")
    assert(abs_act.shape[1] == 16)

    abs_act = abs_act[:n_chunk*chunk+n_previous, :]

    action, delta_action = get_actions(
        gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
        all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
        all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),
        slices=None,
        delta_act_sidx=n_previous,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)

    delta_action[:, :6] = (delta_action[:, :6] - sep*delta_act_meanv[:, :6]) / (sep*delta_act_stdv[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep*delta_act_meanv[:, 6:]) / (sep*delta_act_stdv[:, 6:])
    return action, delta_action


def get_caminfo(extrinsic_path, intrinsic_path, n):
    c2w = torch.from_numpy(np.load(extrinsic_path))
    w2c = torch.linalg.inv(c2w).float()
    intrinsic = torch.from_numpy(np.load(intrinsic_path)).float()
    w2c = w2c.unsqueeze(0).repeat(n,1,1)
    c2w = c2w.unsqueeze(0).repeat(n,1,1)
    return c2w, w2c, intrinsic



def main(args):

    seed_everything(args.seed)
    device = torch.device(args.device)

    ### load config
    config = load_config(args)

    chunk = config.chunk
    n_previous = config.n_previous

    ### 
    img = get_image(
        args.input_path, n_previous
    )

    ###
    action, delta_action = get_action(
        args.action_path, args.n_chunk, chunk, n_previous,
        sep=config.data.params.train.params.max_sep,
        domain_name="agibotworld"
    )

    ###
    c2w, w2c, intrinsic = get_caminfo(
        args.extrinsic_path,
        args.intrinsic_path,
        n_previous+args.n_chunk*chunk
    )

    ###
    model = load_model(config).to(device=device)
    model.eval()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        model.inference(
            config, img, action, delta_action,
            c2w, w2c, intrinsic,
            args.save_root, args.n_chunk,
            chunk=chunk, n_previous=n_previous,
            unconditional_guidance_scale=args.cfg,
            guidanc_erescale=args.gr,
            ddim_steps=args.ddim_steps,
            saving_tag="", saving_fps=10
        )
        torch.cuda.empty_cache()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help document")

    parser.add_argument(
        "--input_path", "-i", type=str,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--action_path", "-a", type=str,
        help="Path to the .npy file including the ABSOLUTE actions of end-effector. The file should contain a {T x 16} numpy array: T x [xyz_left(3), quat_xyzw_left(4), gripper_left(1), xyz_right(3), quat_xyzw_right(4), gripper_right(1)]"
    )
    parser.add_argument(
        "--extrinsic_path", "-ex", type=str,
        help="Path to the .npy file of camera extrinsics {4 x 4}"
    )
    parser.add_argument(
        "--intrinsic_path", "-in", type=str,
        help="Path to the .npy file of camera intrinsics {3 x 3}"
    )
    parser.add_argument(
        "--save_root", "-s", type=str,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--ckp_path", type=str,
    )
    parser.add_argument(
        "--config_path", type=str,
    )

    parser.add_argument(
        "--n_chunk", type=int, default=20,
        help="number of chunks to predict"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=27,
    )
    parser.add_argument(
        "--cfg", type=float, default=1,
        help="unconditional guidance scale ",
    )
    parser.add_argument(
        "--gr", type=float, default=0.7,
        help="guidance rescale",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--seed", type=int,
        default=12345
    )

    args = parser.parse_args()

    main(args)