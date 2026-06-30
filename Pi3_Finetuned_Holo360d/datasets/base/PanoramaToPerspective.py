import os
import cv2
import math
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import json
from tqdm import tqdm
import time
import utils3d  # 
from .panorama import get_panorama_cameras, get_panorama_cameras_8views, get_panorama_cameras_6views,get_panorama_cameras_10views, split_panorama_image,precompute_mapping_matrices,get_panorama_cameras_4views  
from typing import Iterable, Sequence 

def write_ply_xyz_ascii(path: str, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for row in xyz:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")
def _ensure_rgb_u8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {x.shape}")
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float64)
    if xf.max() <= 1.0:
        xf = np.clip(xf, 0.0, 1.0)
    else:
        xf = np.clip(xf / 255.0, 0.0, 1.0)
    return (xf * 255.0).astype(np.uint8)
def _depth_vis_u8(depth: np.ndarray, ignore_invalid: bool = True) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    if ignore_invalid:
        valid = np.isfinite(d) & (d > 1e-6)
    else:
        valid = np.isfinite(d)
    out = np.zeros(d.shape[:2], dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(d[valid], [2.0, 98.0])
    if hi <= lo + 1e-12:
        hi = lo + 1e-6
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = (t[valid] * 255.0).astype(np.uint8)
    return out
def save_split_multiview_assets(
    splitted_rgbs: Sequence[np.ndarray],
    splitted_depths: Sequence[np.ndarray] | None,
    out_dir: str,
    *,
    save_depth_npy: bool = False,
    depth_colormap: int | None = cv2.COLORMAP_MAGMA,
) -> None:
    """
    Args:
        splitted_rgbs: Each item has shape (H, W, 3) in RGB order.
        splitted_depths: Must match the RGB sequence length. If None, only RGB files are saved.
        out_dir: Output directory.
        save_depth_npy: If True, also write depth_view_XXXX.npy as float values.
        depth_colormap: cv2.COLORMAP_*；If None, only write the grayscale depth visualization.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(splitted_rgbs)
    if splitted_depths is not None and len(splitted_depths) != n:
        raise ValueError(
            f"RGB count {n} does not match depth count {len(splitted_depths)}"
        )
    for i in range(n):
        tag = f"view_{i:04d}"
        rgb_u8 = _ensure_rgb_u8(splitted_rgbs[i])
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"rgb_{tag}.png"), bgr)
        if splitted_depths is None:
            continue
        dm = np.asarray(splitted_depths[i])
        if save_depth_npy:
            np.save(os.path.join(out_dir, f"depth_{tag}.npy"), dm.astype(np.float32))
        vis = _depth_vis_u8(dm, ignore_invalid=True)
        path_vis = os.path.join(out_dir, f"depth_vis_{tag}.png")
        if depth_colormap is not None:
            color = cv2.applyColorMap(vis, depth_colormap)
            cv2.imwrite(path_vis, color)
        else:
            cv2.imwrite(path_vis, vis)
class PanoramaToPerspective:
    def __init__(self, output_resolution: int = 518,pano_w: int = 2880, pano_h: int = 1440, view_type:str='8views'):
        self.output_resolution = output_resolution
        self.pano_height = pano_h
        self.pano_width = pano_w
        if view_type=='8views':
            self.splitted_extrinsics, self.splitted_intrinsics = get_panorama_cameras_8views()
        elif view_type=='6views':
            self.splitted_extrinsics, self.splitted_intrinsics = get_panorama_cameras_6views()
        elif view_type=='10views':
            self.splitted_extrinsics, self.splitted_intrinsics = get_panorama_cameras_10views()
        elif view_type=='4views':
            self.splitted_extrinsics, self.splitted_intrinsics = get_panorama_cameras_4views()
        else:
            raise ValueError(f"Invalid view type: {view_type}")
        # Default to the 8-view panorama configuration.
        self.np_extrinsics=np.array(self.splitted_extrinsics, dtype=np.float32)
        self.np_intrisics=self.splitted_intrinsics[0]*output_resolution
        self.np_intrisics[2,2]=1

        self.mapping,self.occupancy_vectors = precompute_mapping_matrices(self.splitted_extrinsics, self.splitted_intrinsics,[self.pano_height,self.pano_width],self.output_resolution)
        self.overlaps = self.precompute_view_overlaps()
        self.pano_pointmap=None

    def pano_pixel_to_world_rays(self):
        H, W = self.pano_height, self.pano_width

        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing='xy')

        # Keep the same convention as depth_to_pointcloud.
        u_raw = (1.5 * W - uu) % W
        theta = u_raw * (2.0 * np.pi) / W - np.pi
        phi = np.pi / 2.0 - vv * np.pi / H

        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)

        rays = np.stack([x, y, z], axis=-1).astype(np.float32)
        return rays

    def bilinear_sample_pointmap(self, point_map: np.ndarray, x: np.ndarray, y: np.ndarray):
        Hv, Wv = point_map.shape[:2]

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        inside = (x0 >= 0) & (x1 < Wv) & (y0 >= 0) & (y1 < Hv)

        sampled = np.zeros((len(x), 3), dtype=np.float32)
        valid_out = np.zeros((len(x),), dtype=bool)

        if np.any(inside):
            idx = np.where(inside)[0]

            xv = x[idx]
            yv = y[idx]
            x0v, x1v = x0[idx], x1[idx]
            y0v, y1v = y0[idx], y1[idx]

            wa = (x1v - xv) * (y1v - yv)
            wb = (xv - x0v) * (y1v - yv)
            wc = (x1v - xv) * (yv - y0v)
            wd = (xv - x0v) * (yv - y0v)

            Ia = point_map[y0v, x0v]
            Ib = point_map[y0v, x1v]
            Ic = point_map[y1v, x0v]
            Id = point_map[y1v, x1v]

            # 💡 Reject NaN, Inf, and near-zero samples.
            # 1. Compute the norm of the four corner samples.
            norm_a = np.linalg.norm(Ia, axis=-1)
            norm_b = np.linalg.norm(Ib, axis=-1)
            norm_c = np.linalg.norm(Ic, axis=-1)
            norm_d = np.linalg.norm(Id, axis=-1)

            # 2. A sample is valid only if it is finite and sufficiently far from zero.
            va = np.isfinite(Ia).all(axis=-1) & (norm_a > 1e-5)
            vb = np.isfinite(Ib).all(axis=-1) & (norm_b > 1e-5)
            vc = np.isfinite(Ic).all(axis=-1) & (norm_c > 1e-5)
            vd = np.isfinite(Id).all(axis=-1) & (norm_d > 1e-5)

            wsum = np.zeros_like(wa, dtype=np.float32)
            out = np.zeros((len(idx), 3), dtype=np.float32)

            if np.any(va):
                out[va] += Ia[va] * wa[va, None]
                wsum[va] += wa[va]
            if np.any(vb):
                out[vb] += Ib[vb] * wb[vb, None]
                wsum[vb] += wb[vb]
            if np.any(vc):
                out[vc] += Ic[vc] * wc[vc, None]
                wsum[vc] += wc[vc]
            if np.any(vd):
                out[vd] += Id[vd] * wd[vd, None]
                wsum[vd] += wd[vd]

            good = wsum > 1e-8
            out[good] /= wsum[good, None]

            sampled[idx[good]] = out[good]
            valid_out[idx[good]] = True

        # Fall back to nearest-neighbor sampling when bilinear sampling fails.
        margin = 1.0
        inside_fov = (x > -margin) & (x < (Wv - 1 + margin)) & \
                     (y > -margin) & (y < (Hv - 1 + margin))
        
        bad = (~valid_out) & inside_fov
        
        if np.any(bad):
            # np.rint rounds to the nearest integer.
            xn = np.rint(x[bad]).astype(np.int32)
            yn = np.rint(y[bad]).astype(np.int32)

            # Clip again to protect against numerical drift.
            xn = np.clip(xn, 0, Wv - 1)
            yn = np.clip(yn, 0, Hv - 1)

            nearest = point_map[yn, xn]
            
            # Filter zeros and NaNs as well.
            nearest_norm = np.linalg.norm(nearest, axis=-1)
            nearest_valid = np.isfinite(nearest).all(axis=-1) & (nearest_norm > 1e-5)

            bad_idx = np.where(bad)[0]
            sampled[bad_idx[nearest_valid]] = nearest[nearest_valid]
            valid_out[bad_idx[nearest_valid]] = True

        return sampled, valid_out

    def fuse_perspective_pointmaps_to_pano(self, pointmaps_world, center_weighted=False):
        """
        Does not depend on mapping_matrices.
        Projects panorama pixels into each perspective view geometrically, bilinearly samples the world point map, and averages across views.

        Args:
            pointmaps_world: list[np.ndarray]
                Length is N_views.
                Each element has shape [Hv, Wv, 3].
                Represents the world point associated with each pixel in the perspective view.
            center_weighted: bool
                Whether to weight by distance from the view center. Defaults to False for simple averaging.

        Returns:
            pano_points: [Hp, Wp, 3]
            pano_count:  [Hp, Wp]
            pano_mask:   [Hp, Wp] bool
        """
        Hp, Wp = self.pano_height, self.pano_width
        rays_world = self.pano_pixel_to_world_rays().reshape(-1, 3)  # [P,3]
        P = rays_world.shape[0]

        pano_sum = np.zeros((P, 3), dtype=np.float32)
        pano_wsum = np.zeros((P,), dtype=np.float32)

        K = self.np_intrisics.astype(np.float32)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for view_idx, pointmap_world in enumerate(pointmaps_world):
            if pointmap_world is None:
                continue

            Hv, Wv = pointmap_world.shape[:2]
            ext = self.np_extrinsics[view_idx].astype(np.float32)  # world -> camera
            R = ext[:3, :3]

            # For concentric rotating views, only the direction vector needs rotation.
            rays_cam = rays_world @ R.T   # [P,3]
            z = rays_cam[:, 2]

            front = z > 1e-6
            if not np.any(front):
                continue

            rc = rays_cam[front]
            x = fx * (rc[:, 0] / z[front]) + cx
            y = fy * (rc[:, 1] / z[front]) + cy

            sampled, valid = self.bilinear_sample_pointmap(pointmap_world, x, y)

            front_idx = np.where(front)[0]
            hit_idx = front_idx[valid]
            if len(hit_idx) == 0:
                continue

            if center_weighted:
                xv = x[valid]
                yv = y[valid]
                nx = (xv - cx) / max(cx, 1e-6)
                ny = (yv - cy) / max(cy, 1e-6)
                r2 = nx * nx + ny * ny
                w = np.exp(-2.0 * r2).astype(np.float32)
            else:
                w = np.ones((len(hit_idx),), dtype=np.float32)

            pano_sum[hit_idx] += sampled[valid] * w[:, None]
            pano_wsum[hit_idx] += w

        pano_points = np.zeros((P, 3), dtype=np.float32)
        valid_mask = pano_wsum > 1e-8
        pano_points[valid_mask] = pano_sum[valid_mask] / pano_wsum[valid_mask, None]

        pano_points = pano_points.reshape(Hp, Wp, 3)
        pano_count = valid_mask.reshape(Hp, Wp).astype(np.int32)
        pano_mask = valid_mask.reshape(Hp, Wp)

        return pano_points, pano_count, pano_mask

    def precompute_view_overlaps(self, save_path: str = None):
        """
        Precompute pixel correspondences between views.
        Output: overlaps[(i, j)] = {'idx_i': idx_i, 'idx_j': idx_j}.
        idx_* are 1D indices into the flattened (H * W) pixel grid.
        """
        mapping = self.mapping  # list of length N, each entry has shape (H, W, 2)
        N = len(mapping)
        H, W = mapping[0].shape[:2]
        pano_w, pano_h = self.pano_width, self.pano_height

        # View-to-panorama index table: view_pano2idx[j][pano_lin] = hw_idx or -1.
        view_pano2idx = []
        for m in mapping:
            u = np.rint(m[..., 0]).astype(np.int64)
            v = np.rint(m[..., 1]).astype(np.int64)
            u = np.clip(u, 0, pano_w - 1)
            v = np.clip(v, 0, pano_h - 1)
            pano_lin = v * pano_w + u  # (H,W)
            table = -np.ones(pano_h * pano_w, dtype=np.int64)
            hw_idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
            table[pano_lin.reshape(-1)] = hw_idx.reshape(-1)
            view_pano2idx.append(table)

        overlaps = {}
        for i in range(N):
            u_i = np.rint(mapping[i][..., 0]).astype(np.int64)
            v_i = np.rint(mapping[i][..., 1]).astype(np.int64)
            u_i = np.clip(u_i, 0, pano_w - 1)
            v_i = np.clip(v_i, 0, pano_h - 1)
            pano_lin_i = v_i * pano_w + u_i  # (H,W)
            lin_i = np.arange(H * W, dtype=np.int64).reshape(H, W)

            for j in range(i + 1, N):
                table_j = view_pano2idx[j]  # (pano_h * pano_w,)
                lin_j = table_j[pano_lin_i.reshape(-1)].reshape(H, W)
                mask = lin_j >= 0  # valid overlap
                idx_i = lin_i.reshape(-1)[mask.reshape(-1)]
                idx_j = lin_j.reshape(-1)[mask.reshape(-1)]
                overlaps[(i, j)] = {'idx_i': idx_i, 'idx_j': idx_j}
                # Only store adjacent views if needed.

        # Optional persistence.
        if save_path is not None:
            # Expand keys into strings for serialization.
            out = {}
            for (i, j), d in overlaps.items():
                out[f'{i}_{j}_idx_i'] = d['idx_i']
                out[f'{i}_{j}_idx_j'] = d['idx_j']
            np.savez_compressed(save_path, **out)
            print(f"Saved overlap indices to {save_path}")

        return overlaps

    def load_pose(self, pose_file: Path) -> np.ndarray:
        """Load pose.txt and return a 4x4 pose matrix."""
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
            
            # Skip the header.
            data_lines = [line.strip().split() for line in lines[1:] if line.strip()]
            
            poses = []
            for line in data_lines:
                if len(line) >= 13:  # image + 3 translation values + 9 rotation values
                    # Extract the 3x3 rotation matrix and 3D translation.
                    rotation = np.array([float(x) for x in line[4:13]]).reshape(3, 3)
                    translation = np.array([float(x) for x in line[1:4]])
                    
                    # Assemble the 4x4 pose matrix.
                    pose = np.eye(4)
                    pose[:3, :3] = rotation.T
                    pose[:3, 3] = translation
                    poses.append(pose)
            
            return poses
            
        except Exception as e:
            print(f"Error loading pose file {pose_file}: {e}")
            return None
    
    def depth_to_pointcloud(self, depth_map: np.ndarray) -> np.ndarray:
        """Reconstruct a point cloud from a panorama depth map."""
        height, width = depth_map.shape
        
        # Generate the pixel grid.
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v, indexing='xy')
        
        # Invert the horizontal coordinate transform u = (1.5 * width - u) % width.
        u_raw = (1.5 * width - u) % width
        
        # Recover spherical angles.
        theta = u_raw * (2 * np.pi) / width - np.pi
        phi = np.pi/2 - v * np.pi / height
        
        # Build unit direction vectors.
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        directions = np.stack([x, y, z], axis=-1)
        
        # Multiply by depth to recover the point cloud.
        depths = depth_map[..., np.newaxis]
        points_cam = directions * depths
        
        return points_cam.reshape(-1, 3)
    def convert_pointcloud_and_c2w(self,points_unproject: np.ndarray, c2w: np.ndarray):
        """
        Convert the point cloud from unproject_pano_depth_to_camera_coords together with its c2w matrix
        into the depth_to_pointcloud coordinate system.

        Args:
            points_unproject: (N, 3) original unprojected point cloud
            c2w: (4, 4) original camera-to-world matrix

        Returns:
            points_depth2pc: (N, 3) converted point cloud
            c2w_new: (4, 4) converted c2w matrix
        """
        # Rotation matrix from the unproject coordinate system to the depth_to_pointcloud coordinate system.
        R_convert = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0,-1, 0]], dtype=points_unproject.dtype)
        
        # Transform the point cloud.
        points_depth2pc = (R_convert @ points_unproject.T).T

        # Build the 4x4 homogeneous matrix.
        R_hom = np.eye(4, dtype=points_unproject.dtype)
        R_hom[:3,:3] = R_convert

        # Transform the c2w matrix accordingly.
        c2w_new = c2w @ np.linalg.inv(R_hom)

        return points_depth2pc, c2w_new
    def unproject_pano_depth_to_camera_coords(self,depth: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Convert panorama depth to 3D points in the OpenCV camera frame (X right, Y down, Z forward).
        depth: (1, H, W) or (H, W); shape: (H, W). Returns (H, W, 3).
        """
        h, w = shape
        if depth.ndim == 3:
            if depth.shape[0] != 1:
                raise ValueError(f"depth Expected shape (1, H, W), got {depth.shape}")
            depth = depth[0]
        if depth.shape != (h, w):
            raise ValueError(f"depth spatial size {depth.shape} and shape {(h, w)} does not match")

        v_den = max(h - 1, 1)
        u_den = max(w - 1, 1)
        v_normalized = np.arange(h, dtype=np.float64) / v_den
        u_normalized = np.arange(w, dtype=np.float64) / u_den
        v_grid, u_grid = np.meshgrid(v_normalized, u_normalized, indexing="ij")

        theta = (u_grid - 0.5) * (2 * math.pi)
        phi = -(v_grid - 0.5) * math.pi
        x = np.cos(phi) * np.sin(theta)
        y = -np.sin(phi)
        z = np.cos(phi) * np.cos(theta)
        directions = np.stack([x, y, z], axis=-1)
        camera_coords = directions * depth[..., np.newaxis]
        # return camera_coords.astype(depth.dtype, copy=False)
        return camera_coords.reshape(-1, 3)

    def  project_points_to_camera(self,points_world, extrinsic, intrinsic, image_size):
        """
        Project world-space points into a perspective camera view.
        """
        H= image_size
        W = image_size

        # 1. Convert to homogeneous coordinates. (N, 4)
        points_world_homo = np.hstack([points_world, np.ones((len(points_world), 1))])  # (N, 4)

        # 2. Transform into camera coordinates with the extrinsic matrix.: [R | t] @ [x, y, z, 1]
        points_camera_homo = points_world_homo @ extrinsic.T  # (N, 4)
        points_camera = points_camera_homo[:, :3]  # (N, 3), taking x, y, z

        # 3. Filter out points with z <= 0 behind the camera.
        valid_mask = points_camera[:, 2] > 1e-3
        points_camera = points_camera[valid_mask]

        # 4. Project onto the image plane with the intrinsics.: [u, v, depth] = K @ [x, y, z].T
        # Note: intrinsic has shape (3, 3) and points_camera has shape (N, 3).
        points_image = points_camera @ intrinsic.T  # (N, 3) @ (3,3) → (N, 3)
        # Each row of points_image is now [u * z, v * z, z].

        uv = points_image[:, :2] / points_image[:, 2:3]  # normalize to pixel coordinates (N, 2)
        depth = points_image[:, 2]  # depth z (N,)

        # 5. Filter points that fall inside the image bounds.
        u_valid = (uv[:, 0] >= 0) & (uv[:, 0] < W)
        v_valid = (uv[:, 1] >= 0) & (uv[:, 1] < H)
        inside_mask = u_valid & v_valid

        uv = uv[inside_mask]
        depth = depth[inside_mask]

        # 6. Create the depth map and mask map.
        depth_map = np.zeros((H, W), dtype=np.float32)

        # uv is floating point, so round or interpolation may be preferable here.
        u_coords = np.round(uv[:, 0]).astype(int)
        v_coords = np.round(uv[:, 1]).astype(int)

        # Prevent out-of-bounds indices after rounding.
        valid_coords = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        u_coords = u_coords[valid_coords]
        v_coords = v_coords[valid_coords]
        depth = depth[valid_coords]

        depth_map[v_coords, u_coords] = depth

        return depth_map

    def remap_depth(self,points_world, extrinsic, mapping_matrice):
        """
        Project world-space points into a perspective camera view.
        """
        # 1. Convert to homogeneous coordinates. (N, 4)
        points_world_homo = np.hstack([points_world, np.ones((len(points_world), 1))])  # (N, 4)

        # 2. Transform into camera coordinates with the extrinsic matrix.: [R | t] @ [x, y, z, 1]
        points_camera_homo = points_world_homo @ extrinsic.T  # (N, 4)
        # points_camera = points_camera_homo[:, :3]  # (N, 3), taking x, y, z

        pano_depth_in_camera = points_camera_homo[:, 2]

        pano_depth_in_camera=pano_depth_in_camera.reshape(self.pano_height,self.pano_width)


        splitted_image = cv2.remap(pano_depth_in_camera, mapping_matrice[..., 0], mapping_matrice[..., 1], 
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE) 

        return splitted_image

    def depth_to_pointcloud_perspective(self, depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
        """
        Reconstruct a world-space point cloud from a perspective depth map, extrinsics, and intrinsics.

        Args:
            depth_map (np.ndarray): Depth map with shape (H, W) in meters.
            extrinsic (np.ndarray): Extrinsic matrix of shape (4, 4) or (3, 4) representing [R | t] from world to camera.
            intrinsic (np.ndarray): Intrinsic matrix of shape (3, 3):
                                [[fx,  0, cx],
                                    [ 0, fy, cy],
                                    [ 0,  0,  1]]

        Returns:
            np.ndarray: World-space point cloud with shape (H * W, 3).
        """
        H, W = depth_map.shape

        # Unscaled 3D points in the camera coordinate frame.
        # 1. Generate the image coordinate grid (u, v).
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v, indexing='xy')  # (H, W)

        # 2. Back-project ray directions into the camera frame.
        #    Use the inverse intrinsics to map pixels into normalized camera coordinates.
        fx, fy = intrinsic[0,0], intrinsic[1,1]
        cx, cy = intrinsic[0,2], intrinsic[1,2]

        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        z_cam = np.ones_like(u)  # normalized direction

        # Stack direction vectors.
        directions = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

        # 3. Multiply by depth to recover camera-frame 3D points.
        depths = depth_map[..., np.newaxis]  # (H, W, 1)
        points_cam = directions * depths  # (H, W, 3)

        # 4. Flatten to shape (N, 3).
        points_cam_flat = points_cam.reshape(-1, 3)  # (H*W, 3)

        # 5. Transform into world coordinates.
        # Assume extrinsic maps from world to camera: P_cam = R @ P_world + t.
        # Then P_world = R.T @ (P_cam - t), or equivalently a homogeneous transform.
        # R = extrinsic[:3, :3]   # (3, 3)
        # t = extrinsic[:3, 3]    # (3,)

        # # Using P_world = R.T @ (P_cam - t) is often more intuitive.
        # # Equivalent to P_world = R.T @ P_cam - R.T @ t.
        # points_world = (R.T @ points_cam_flat.T).T - (R.T @ t)

        # Alternatively, use homogeneous coordinates for a more general implementation.
        points_hom = np.hstack([points_cam_flat, np.ones((len(points_cam_flat), 1))])
        points_world_hom = np.linalg.inv(extrinsic) @ points_hom.T
        points_world = points_world_hom[:3].T

        return points_world  # shape: (H*W, 3)

    def process_single_image(self, rgb,depth,pose):

        
        splitted_rgbs = split_panorama_image(rgb, self.mapping,True)
        if depth is None:
            return splitted_rgbs, None, None
        # splitted_masks = split_panorama_image(mask, self.mapping,False)

        points = self.depth_to_pointcloud(depth.astype(np.float32))
        self.pano_pointmap=points.reshape(self.pano_height,self.pano_width,3)
        splitted_depths = []
        splitted_poses = []

        for view_idx in range(len(self.splitted_extrinsics)):

            view_extrinsic = self.np_extrinsics[view_idx]
            global_extrinsic = view_extrinsic@np.linalg.inv(pose)  

            valid_points=points[self.occupancy_vectors[view_idx]]

            depth_map = self.project_points_to_camera(valid_points, view_extrinsic, self.np_intrisics, self.output_resolution)
            splitted_depths.append(depth_map)
            splitted_poses.append(global_extrinsic)

        return splitted_rgbs, splitted_depths, splitted_poses
    
    def process_single_image_mp3d(self, rgb,depth,pose):

        
        splitted_rgbs = split_panorama_image(rgb, self.mapping,True)
        if depth is None:
            return splitted_rgbs, None, None
        # splitted_masks = split_panorama_image(mask, self.mapping,False)

        points_mp3d = self.unproject_pano_depth_to_camera_coords(depth.astype(np.float32),depth.shape)
        points,pose=self.convert_pointcloud_and_c2w(points_mp3d,pose)
        # points = self.convert_unproject_to_depth2pc(points_mp3d)
        # write_ply_xyz_ascii("./results_multiview/all.ply",points)
        
        self.pano_pointmap=points.reshape(self.pano_height,self.pano_width,3)
        splitted_depths = []
        splitted_poses = []

        for view_idx in range(len(self.splitted_extrinsics)):

            view_extrinsic = self.np_extrinsics[view_idx]
            global_extrinsic = view_extrinsic@np.linalg.inv(pose)  

            # valid_points=points[self.occupancy_vectors[view_idx]]

            depth_map = self.project_points_to_camera(points, view_extrinsic, self.np_intrisics, self.output_resolution)
            splitted_depths.append(depth_map)
            splitted_poses.append(global_extrinsic)

            # point_pes=self.depth_to_pointcloud_perspective(depth_map,np.linalg.inv(global_extrinsic),self.np_intrisics)
            # point_pes=self.depth_to_pointcloud_perspective(depth_map,global_extrinsic,self.np_intrisics)
            # path = os.path.join("./results_multiview", f"view_{view_idx:04d}.ply")
            # write_ply_xyz_ascii(path,point_pes)

        save_split_multiview_assets(
            splitted_rgbs,
            splitted_depths,
            "./results_multiview/split_vis",
            save_depth_npy=True,
            depth_colormap=cv2.COLORMAP_MAGMA,
        )

        return splitted_rgbs, splitted_depths, splitted_poses

    
    def process_single_scene(self, scene_path: Path):
        """Process a single scene."""
        print(f"Processing scene: {scene_path.name}")
        
        # Create output directories.
        output_dirs = {
            'rgb': scene_path /'perspective_8/rgb',
            'depth': scene_path / 'perspective_8/depth', 
            'mask': scene_path / 'perspective_8/mask',
            'depth_gray': scene_path / 'perspective_8/depth_gray',
            'pose': scene_path / 'perspective_8/poses'
        }
        
        for dir_path in output_dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Load pose data.
        pose_file = scene_path / 'cam4' / 'poses' / 'pose.txt'
        poses = self.load_pose(pose_file)
        if poses is None:
            print(f"No valid poses found in {pose_file}")
            return
        
        # Collect all image files.
        rgb_files = sorted((scene_path /'cam4' /'rgb').glob('*.jpg'))
        depth_files = sorted((scene_path /'cam4' / 'pointcloud_depth').glob('*.exr'))
        mask_files = sorted((scene_path /'cam4' / 'mask_merge').glob('*.jpg'))
        
        # Ensure the number of files matches.
        min_files = min(len(rgb_files), len(depth_files), len(mask_files), len(poses))
        print(len(rgb_files), len(depth_files), len(mask_files), len(poses))
        if min_files == 0:
            print(f"No files found in {scene_path}")
            return
        
        print(f"Found {min_files} frames to process")
        
        # Process each frame.
        start_idx = int(min_files * 0.1) # Skip the first 80% of frames and start from there.
        end_idx = int(min_files * 0.9) 

        global_poses=[]
        for i in tqdm(range(min_files), desc=f"Processing {scene_path.name}"):
            if(i%2==0):
                continue
            # print(rgb_files[i])
            # Load the frame data.
            rgb = cv2.imread(str(rgb_files[i]))
            depth = cv2.imread(str(depth_files[i]), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
            pose = poses[i]
            
            if rgb is None or depth is None or mask is None:
                print(f"Skipping frame {i} due to loading error")
                continue

            rgb = cv2.resize(rgb, [self.pano_width , self.pano_height])

            splitted_rgbs = split_panorama_image(rgb, self.mapping,True)
            splitted_masks = split_panorama_image(mask, self.mapping,False)

            base_name = Path(rgb_files[i]).stem

            # splitted_rgbs=split_panorama_image(rgb, self.splitted_extrinsics, self.splitted_intrinsics, self.output_resolution)
            # splitted_masks=split_panorama_image(mask, self.splitted_extrinsics, self.splitted_intrinsics, self.output_resolution)
            
            # Convert to a point cloud.
            points = self.depth_to_pointcloud(depth.astype(np.float32))

            # Generate perspective images for each view.
            for view_idx in range(len(self.splitted_extrinsics)):
                # Compute the global pose for the current view.
                view_extrinsic = self.np_extrinsics[view_idx]
                global_extrinsic = view_extrinsic@np.linalg.inv(pose)  # Perspective pose = view pose * scene pose * world coordinates.
                
                # Project into the current view.

                valid_points=points[self.occupancy_vectors[view_idx]]

                depth_map = self.project_points_to_camera(valid_points, view_extrinsic, self.np_intrisics, self.output_resolution)
          
        
                # points_world=self.depth_to_pointcloud_perspective(depth_map,view_extrinsic,self.np_intrisics)
                
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(valid_points)

                
                
                # o3d.io.write_point_cloud("test_clud_1.ply", pcd)

                # pcd.points = o3d.utility.Vector3dVector(points_world)

                
                
                # o3d.io.write_point_cloud("test_clud_2.ply", pcd)
                # break
            


                frame_base = f"{base_name}_view{view_idx:02d}"
                
                # Save the depth map.
                cv2.imwrite(str(output_dirs['depth'] / f"{frame_base}.exr"), depth_map.astype(np.float32))
                
                # Save the mask.
                cv2.imwrite(str(output_dirs['mask'] / f"{frame_base}.jpg"), splitted_masks[view_idx])
                cv2.imwrite(str(output_dirs['rgb'] / f"{frame_base}.jpg"), splitted_rgbs[view_idx])
                cv2.imwrite(str(output_dirs['depth_gray'] / f"{frame_base}.jpg"), depth_map*20.0)
                
                t = global_extrinsic[:3, 3]        # translation component: (tx, ty, tz)
                R = global_extrinsic[:3, :3]       # rotation component: 3x3

                # Flatten R into r1..r9 in row-major order.
                R_flat = R.flatten()  # [r11, r12, r13, r21, r22, r23, r31, r32, r33]

                # Build the output format: filename.jpg tx ty tz r1 r2 ... r9.
                pose_line = f"{frame_base}.jpg {' '.join(map(str, t))} {' '.join(map(str, R_flat))}"
                                
                # Append to the pose list.
                global_poses.append(pose_line)
                
                # end_time = time.time()

                # # Compute and print runtime.
                # duration = end_time - start_time
                

        pose_output_path = output_dirs['pose'] / "pose.txt"

        # Ensure the directory exists.
        pose_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file with one view per line.
        with open(pose_output_path, 'w') as f:
            for line in global_poses:
                f.write(line + '\n')
                

    
    def process_all_scenes(self, base_dir: Path, sence_list:List):
        """Process all scenes."""
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Base directory {base_dir} does not exist")
            return
        
        folders_file_path = base_path / 'folders.txt'
        if not folders_file_path.exists():
            print(f"Missing folders_perspective.txt in the base directory.")
            return
        all_dirs=None
        with open(folders_file_path, 'r') as f:
            all_dirs = [line.strip() for line in f.readlines()]
        print(f"Found {len(all_dirs)} scenes to process")


        scene_dirs=[]

        for i in range(sence_list[0],sence_list[1]):
            scene_dirs.append(all_dirs[i])
            print(f"run {all_dirs[i]} scenes ")
        
        for scene_dir in scene_dirs:

            # Check whether the required folders exist.
            required_dirs = ['rgb', 'mask_merge', 'pointcloud_depth', 'poses']
            
            for d in required_dirs:
                print(base_path/scene_dir / 'cam4' / d)
            if all((base_path/scene_dir / 'cam4' / d).exists() for d in required_dirs):
                self.process_single_scene(base_path/scene_dir)
            else:
                print(f"Skipping {scene_dir}: missing required directories")

def main():
    parser = argparse.ArgumentParser(description='Convert panorama data to perspective views')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing scene folders')
    parser.add_argument('--output_resolution', type=int, default=518,
                       help='Output resolution for perspective images')
    
    parser.add_argument('--scene_list', type=int, nargs='+', help='List of scene indices to process')
    
    args = parser.parse_args()
    
    converter = PanoramaToPerspective(output_resolution=args.output_resolution)
    converter.process_all_scenes(args.input_dir,args.scene_list)

if __name__ == "__main__":
    main()


