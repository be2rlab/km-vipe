import argparse
import glob
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import torch
import numpy as np
import open3d as o3d


@dataclass
class Intrinsic:
    """Camera intrinsics"""

    def __init__(self, width, height, fx, fy, cx, cy, depth_scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def __repr__(self):
        return f"Intrinsic(\
            width={self.width}, \
            height={self.height}, \
            fx={self.fx}, \
            fy={self.fy}, \
            cx={self.cx}, \
            cy={self.cy}, \
            depth_scale={self.depth_scale}, \
        )"


class MappingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        load_semantics: bool = False,
        frame_subpath = 'results/frame*.jpg',
        depth_subpath = 'results/depth*.png',
        semantic_subpath = 'results/semantic*.png',
        traj_subpath = "traj.txt",
        camera_params_subpath = "camera_params.yaml"
    ):
        self._data_path = data_path
        self._slice = slice(start, end, stride)
        self._load_semantics = load_semantics

        self._rgb_paths = sorted(glob.glob(os.path.join(self._data_path, frame_subpath)))
        self._depth_paths = sorted(glob.glob(os.path.join(self._data_path, depth_subpath)))
        
        assert len(self._rgb_paths) == len(self._depth_paths)
        
        if self._load_semantics:
            self._semantic_paths = sorted(glob.glob(os.path.join(self._data_path, semantic_subpath)))
            
            assert len(self._rgb_paths) == len(self._semantic_paths)
            
        self._poses = self._load_poses(traj_subpath)
        
        assert len(self._poses) == len(self._rgb_paths)
        
        self._intrinsics = self._load_intrinsics(camera_params_subpath)


    def __len__(self):
        return len(self._rgb_paths[self._slice])
    
    
    def __getitem__(self, index):
        rgb = o3d.io.read_image(self._rgb_paths[self._slice][index])
        depth = o3d.io.read_image(self._depth_paths[self._slice][index])
        
        if self._load_semantics:
            semantics = o3d.io.read_image(self._semantic_paths[self._slice][index])
        else:
            semantics = None
            
        pose = self._poses[self._slice][index]
        intrinsics = self._intrinsics
        
        return rgb, depth, semantics, pose, intrinsics
    
    
    def _load_poses(self, traj_subpath):
        with open(os.path.join(self._data_path, traj_subpath), "r") as file:
            poses = []
            for line in file:
                pose = np.fromstring(line, dtype=float, sep=" ")
                pose = np.reshape(pose, (4, 4))
                # pose[:, [0, 1]] = pose[:, [1, 0]]
                # pose[[0, 1], :] = pose[[1, 0], :]
                poses.append(pose)
                
        return poses
    
    
    def _load_intrinsics(self, camera_params_subpath):
        yaml_file = os.path.join(self._data_path, camera_params_subpath)
        
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        camera_params = data["camera_params"]

        intrinsic = Intrinsic(
            width = camera_params["image_width"],
            height = camera_params["image_height"],
            fx = camera_params["fx"],
            fy = camera_params["fy"],
            cx = camera_params["cx"],
            cy = camera_params["cy"],
            depth_scale = camera_params["png_depth_scale"],
        )

        return intrinsic


def get_parser_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=Path, required=True,
    )
    parser.add_argument(
        "--scene_id", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
    )
    
    # parser.add_argument(
    #     "--dataset_config", type=str, required=True,
    #     help="This path may need to be changed depending on where you run this script. "
    # )
    # parser.add_argument("--image_height", type=int, default=480)
    # parser.add_argument("--image_width", type=int, default=640)
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    
    parser.add_argument("--downsample_rate", type=int, default=1)
    
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_pcd", action="store_true", default=True)
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_h5", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--load_semseg", action="store_true",
                        help="Load GT semantic segmentation and run fusion on them.")
    
    
    parser.add_argument("--frame_subpath", type=Path, default="results/frame*.jpg")
    parser.add_argument("--depth_subpath", type=Path, default="results/depth*.png")
    parser.add_argument("--semantic_subpath", type=Path, default="results/semantic*.png")
    
    parser.add_argument("--traj_subpath", type=Path, default="traj.txt")
    parser.add_argument("--camera_params_subpath", type=Path, default="camera_params.yaml")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.dataset_root
    
    return args


def create_semantic_point_cloud(
    rgb, depth, intrinsic, pose, semantics=None
):
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb,
        depth,
        depth_scale=intrinsic.depth_scale,
        depth_trunc=np.inf,
        convert_rgb_to_intensity=False,
    )

    # Create point cloud
    color_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        ),
    )

    color_pcd.transform(pose)
        
    if semantics is not None:       
        semantic_d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            semantics,
            depth,
            depth_scale=intrinsic.depth_scale,
            depth_trunc=np.inf,
        )

        # Create semantic point cloud
        semantic_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            semantic_d,
            o3d.camera.PinholeCameraIntrinsic(
                width = intrinsic.width,
                height = intrinsic.height,
                fx = intrinsic.fx,
                fy = intrinsic.fy,
                cx = intrinsic.cx,
                cy = intrinsic.cy,
            ),
        )
        
        semantic_pcd.transform(pose)
    else:
        semantic_pcd = None

    return color_pcd, semantic_pcd


def main():
    args = get_parser_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dataset = MappingDataset(
        data_path = os.path.join(args.dataset_root, args.scene_id),
        stride = args.stride,
        start = args.start,
        end = args.end,
        load_semantics = args.load_semseg,
        frame_subpath = args.frame_subpath,
        depth_subpath = args.depth_subpath,
        semantic_subpath = args.semantic_subpath,
        traj_subpath = args.traj_subpath,
        camera_params_subpath = args.camera_params_subpath
    )
    
    color_map = o3d.geometry.PointCloud()
    semantic_map = o3d.geometry.PointCloud() if args.load_semseg else None

    geometries = []
    is_first_point = True
    for (rgb, depth, semantics, pose, intrinsics) in tqdm(dataset):
        color_pcd, semantic_pcd = create_semantic_point_cloud(
            rgb, depth, intrinsics, pose, semantics
        )
        
        color_map += color_pcd
        
        if semantic_map is not None:
            semantic_map += semantic_pcd
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=(0.2 if is_first_point else 0.1))
        frame.transform(pose)
        geometries.append(frame)
        
        is_first_point = False
        
    
    color_map = color_map.uniform_down_sample(every_k_points=args.downsample_rate)
    
    if semantic_map is not None:
        semantic_map = semantic_map.uniform_down_sample(every_k_points=args.downsample_rate)

    if args.visualize:
        o3d.visualization.draw_geometries([color_map] + geometries)
        
        if semantic_map is not None:
            o3d.visualization.draw_geometries([semantic_map])
            
    dir_to_save_map = os.path.join(args.output_dir, args.scene_id)
    
    if args.save_pcd or args.save_ply:
        try:
            os.makedirs(dir_to_save_map, exist_ok=False)
        except Exception as _:
            pass
            
    if args.save_pcd:
        saving_path = os.path.join(dir_to_save_map, "pointcloud.pcd")
        print(f'Saving .pcd file to "{saving_path}"')
        
        o3d.io.write_point_cloud(
            saving_path, 
            color_map
        )
            
    if args.save_ply:
        saving_path = os.path.join(dir_to_save_map, "pointcloud.ply")
        print(f'Saving .ply file to "{saving_path}"')
        
        o3d.io.write_point_cloud(
            saving_path, 
            color_map
        )
        
    if (args.save_pcd or args.save_ply) and semantic_map is not None:
        saving_path = os.path.join(dir_to_save_map, "semantic.npy")
        print(f'Saving semantics to "{saving_path}"')
        
        np_semantics = np.asarray(semantic_map.colors).round().astype(int)
        
        assert np.all(np_semantics[..., 0] == np_semantics[..., 1]) and \
               np.all(np_semantics[..., 0] == np_semantics[..., 2])
        
        np.save(saving_path, np_semantics[..., 0])

if __name__ == "__main__":  
    main()