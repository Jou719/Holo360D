import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import *
import itertools
import json
import warnings

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm, trange
from scipy.sparse import csr_array, hstack, vstack
from scipy.ndimage import convolve
from scipy.sparse.linalg import lsmr
from concurrent.futures import ProcessPoolExecutor, as_completed

import utils3d
import time
from utils3d.numpy import (
    intrinsics_from_fov,
    extrinsics_look_at,
    icosahedron,
    image_uv,
    project_cv,
    uv_to_pixel
)


def process_view(extrinsics, intrinsics, cube_image, spherical_directions):
    panorama_part = np.zeros_like(panorama)
    weight_sum_part = np.zeros_like(weight_sum[..., 0])
    
    # Project into the current view.
    projected_uv, projected_depth = utils3d.numpy.project_cv(
        spherical_directions, 
        extrinsics=extrinsics, 
        intrinsics=intrinsics
    )
    
    # Validate the projected coordinates.
    valid_mask = (
        (projected_depth > 0) & 
        (projected_uv >= 0).all(axis=-1) & 
        (projected_uv <= 1).all(axis=-1)
    )
    
    if not np.any(valid_mask):
        return panorama_part, weight_sum_part
    
    # Convert to pixel coordinates.
    projected_pixels = utils3d.numpy.uv_to_pixel(
        projected_uv, 
        width=cube_image.shape[1], 
        height=cube_image.shape[0]
    ).astype(np.float32)
    
    # Remap the image.
    remapped = cv2.remap(
        cube_image, 
        projected_pixels[..., 0], 
        projected_pixels[..., 1], 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    # Compute weights based on projection depth and border distance.
    depth_weight = 1.0 / (projected_depth + 1e-6)
    border_dist = np.minimum(
        np.minimum(projected_uv[..., 0], 1 - projected_uv[..., 0]),
        np.minimum(projected_uv[..., 1], 1 - projected_uv[..., 1])
    )
    border_weight = np.exp(-10 * (1 - border_dist))  # Reduce the weight near borders.
    
    weights = depth_weight * border_weight * valid_mask
    
    # Accumulate weights and colors.
    panorama_part += remapped.astype(np.float32) * weights[..., np.newaxis]
    weight_sum_part += weights
    
    return panorama_part, weight_sum_part

def convert_cubemap_to_panorama(splitted_extrinsics: List[np.ndarray], 
                                        splitted_intrinsics: List[np.ndarray],
                                        splitted_images: List[np.ndarray],
                                        panorama_width: int = 1440,
                                        panorama_height: int = 720) -> np.ndarray:

    # Create the panorama and weight maps.
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.float32)
    weight_sum = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    
    # Build the panorama UV grid.
    uv = utils3d.numpy.image_uv(width=panorama_width, height=panorama_height)
    spherical_directions = spherical_uv_to_directions(uv)
    
    # Build remapping coordinates for each view.
    for i in range(len(splitted_images)):
        start_time = time.time()
        extrinsics = splitted_extrinsics[i]
        intrinsics = splitted_intrinsics[i]
        cube_image = splitted_images[i]

        
        # Project into the current view.
        projected_uv, projected_depth = utils3d.numpy.project_cv(
            spherical_directions, 
            extrinsics=extrinsics, 
            intrinsics=intrinsics
        )

        # Validate the projected coordinates.
        valid_mask = (
            (projected_depth > 0) & 
            (projected_uv >= 0).all(axis=-1) & 
            (projected_uv <= 1).all(axis=-1)
        )
        
        if not np.any(valid_mask):
            continue
        
        # Convert to pixel coordinates.
        projected_pixels = utils3d.numpy.uv_to_pixel(
            projected_uv, 
            width=cube_image.shape[1], 
            height=cube_image.shape[0]
        ).astype(np.float32)
        
        # Remap the image.
        remapped = cv2.remap(
            cube_image, 
            projected_pixels[..., 0], 
            projected_pixels[..., 1], 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )

        # Compute weights based on projection depth and border distance.
        depth_weight = 1.0 / (projected_depth + 1e-6)
        border_dist = np.minimum(
            np.minimum(projected_uv[..., 0], 1 - projected_uv[..., 0]),
            np.minimum(projected_uv[..., 1], 1 - projected_uv[..., 1])
        )
        border_weight = np.exp(-10 * (1 - border_dist))  # Reduce the weight near borders.
        
        weights = depth_weight * border_weight * valid_mask
        
        # Accumulate weights and colors.
        panorama += remapped.astype(np.float32) * weights[..., np.newaxis]
        weight_sum += weights
        # Normalize the accumulated result.
    valid_mask = weight_sum > 1e-6
    panorama[valid_mask] /= weight_sum[valid_mask, np.newaxis]
    panorama[~valid_mask] = 0
    
    return panorama.astype(np.uint8)

def normalize_vertices(vertices):
    """Normalize vertices onto the unit sphere."""
    norms = np.linalg.norm(vertices, axis=1)
    return vertices / norms[:, np.newaxis]

# Subdivision helper.
def subdivide(vertices, faces):
    new_vertices = vertices.tolist()
    new_faces = []
    vertex_cache = {}

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute and normalize edge midpoints.
        mid01 = normalize_vertices(np.array([(v0 + v1) / 2]))[0]
        mid12 = normalize_vertices(np.array([(v1 + v2) / 2]))[0]
        mid20 = normalize_vertices(np.array([(v2 + v0) / 2]))[0]

        # Add a new vertex if it does not already exist.
        def get_vertex_index(v):
            key = tuple(v)
            if key not in vertex_cache:
                vertex_cache[key] = len(new_vertices)
                new_vertices.append(v)
            return vertex_cache[key]

        i_mid01 = get_vertex_index(mid01)
        i_mid12 = get_vertex_index(mid12)
        i_mid20 = get_vertex_index(mid20)

        # Generate four new triangles.
        new_faces.append([face[0], i_mid01, i_mid20])
        new_faces.append([face[1], i_mid12, i_mid01])
        new_faces.append([face[2], i_mid20, i_mid12])
        new_faces.append([i_mid01, i_mid12, i_mid20])

    return np.array(new_vertices), np.array(new_faces)

def get_panorama_cameras():
    vertices, faces = utils3d.numpy.icosahedron()

    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)


    return extrinsics, [intrinsics] * len(vertices)

def get_panorama_cameras_8views():
    vertices = np.array([
        [0.5, 0.5,0],  [0, 0.5, 0],  [-0.5, 0.5, 0], [-0.5, 0.0, 0],   [-0.5, -0.5, 0], [0, -0.5, 0],   [0.5, -0.5, 0], [0.5, 0, 0] # v4-v5-v6-v7
    ], dtype=np.float32).reshape((-1, 3))
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)
    
def get_panorama_cameras_10views():
    """
    Ten-view panorama camera setup.
    Assume the panorama uses a Z-up convention, which is common for indoor scenes.
    Includes eight horizontal views plus one upward and one downward view.
    """
    face_configs = [
        # Eight horizontal views around the Z axis at 45-degree intervals.
        ([0.5, 0.5, 0],   [0, 0, 1]),    # northeast (45 deg)
        ([0, 0.5, 0],     [0, 0, 1]),    # north (90 deg)
        ([-0.5, 0.5, 0],  [0, 0, 1]),    # northwest (135 deg)
        ([-0.5, 0, 0],    [0, 0, 1]),    # west (180 deg)
        ([-0.5, -0.5, 0], [0, 0, 1]),    # southwest (225 deg)
        ([0, -0.5, 0],    [0, 0, 1]),    # south (270 deg)
        ([0.5, -0.5, 0],  [0, 0, 1]),    # southeast (315 deg)
        ([0.5, 0, 0],     [0, 0, 1]),    # east (0 deg)
        
        # Up/down views.
        ([0, 0, 0.5],     [0, 1, 0]),    # +Z (up)
        ([0, 0, -0.5],    [0, -1, 0]),   # -Z (down)
    ]
    
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    
    extrinsics_list = []
    for look_at, up in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])
    
    extrinsics = np.array(extrinsics_list, dtype=np.float32)
    
    return extrinsics, [intrinsics] * len(extrinsics)

def get_panorama_cameras_6views():
    """
    Six-view panorama camera setup.
    Assume the panorama uses a Z-up convention, which is common for indoor scenes.
    """
    face_configs = [
        # (look-at direction, up vector)
        ([1, 0, 0],   [0, 0, 1]),    # +X (right)
        ([-1, 0, 0],  [0, 0, 1]),    # -X (left)
        ([0, 1, 0],   [0, 0, 1]),    # +Y (front)
        ([0, -1, 0],  [0, 0, 1]),    # -Y (back)
        ([0, 0, 1],   [0, 1, 0]),    # +Z (up)
        ([0, 0, -1],  [-0, -1, 0]),  # -Z (down)
    ]
    
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    
    extrinsics_list = []
    for look_at, up in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])
    
    extrinsics = np.array(extrinsics_list, dtype=np.float32)
    
    return extrinsics, [intrinsics] * len(extrinsics)

def get_panorama_cameras_4views():
    """
    Four-view panorama camera setup.
    Uses front, back, left, and right views with a shared 10-degree upward pitch.
    Assume the panorama uses a Z-up convention.
    """
    pitch_deg = 10.0
    pitch = np.deg2rad(pitch_deg)

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    face_configs = [
        # (look-at direction, up vector)
        ([0,  cos_pitch, sin_pitch], [0, 0, 1]),   # front (+Y)
        ([0, -cos_pitch, sin_pitch], [0, 0, 1]),   # back (-Y)
        ([-cos_pitch, 0, sin_pitch], [0, 0, 1]),   # left (-X)
        ([ cos_pitch, 0, sin_pitch], [0, 0, 1]),   # right (+X)
    ]

    intrinsics = utils3d.numpy.intrinsics_from_fov(
        fov_x=np.deg2rad(120),
        fov_y=np.deg2rad(120)
    )

    extrinsics_list = []
    for look_at, up in face_configs:
        look_at_arr = np.array(look_at, dtype=np.float32).reshape(1, 3)
        up_arr = np.array(up, dtype=np.float32)
        ext = utils3d.numpy.extrinsics_look_at([0, 0, 0], look_at_arr, up_arr)
        extrinsics_list.append(ext[0])

    extrinsics = np.array(extrinsics_list, dtype=np.float32)

    return extrinsics, [intrinsics] * len(extrinsics)


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)



#         splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)    
#         splitted_images.append(splitted_image)
#     return splitted_images

    
#     for i in range(len(extrinsics)):
#         occupancy_vector=np.zeros(height * width, dtype=bool) 
#         spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
#         pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
#         mapping_matrices.append(pixels)
        
#         valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
#         indices = (pixels[valid_mask, 1] * width + pixels[valid_mask, 0]).astype(int)
#         occupancy_vector[indices]=True
#         occupancy_vectors.append(occupancy_vector)

#     return mapping_matrices,occupancy_vectors

def precompute_mapping_matrices(extrinsics: np.ndarray, intrinsics: list[np.ndarray], 
                               panorama_size, output_resolution: int):
    """
    Precompute mapping matrices and occupancy vectors.
    """
    occupancy_vectors = []
    mapping_matrices = []
    height, width = panorama_size[0], panorama_size[1]  # panorama resolution
    H_out, W_out = output_resolution, output_resolution

    # Generate output UV coordinates. (H_out, W_out, 2)
    uv = utils3d.numpy.image_uv(width=W_out, height=H_out)

    for i in range(len(extrinsics)):
        # 1. Create the occupancy vector.
        occupancy_vector = np.zeros(height * width, dtype=bool)

        # 2. Compute the mapping from output view to panorama.
        spherical_uv = directions_to_spherical_uv(
            utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        )
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)  # (H_out, W_out, 2)

        # 3. Store the mapping matrix.
        mapping_matrices.append(pixels)  # Keep the (H_out, W_out, 2) layout.

        # 4. Flatten pixels for index computation.
        pixels_flat = pixels.reshape(-1, 2)  # (H_out * W_out, 2)

        # 5. Filter valid pixels.
        valid_mask = (pixels_flat[:, 0] >= 0) & (pixels_flat[:, 0] < width) & \
                     (pixels_flat[:, 1] >= 0) & (pixels_flat[:, 1] < height)

        # 6. Compute linear indices.
        # indices = (pixels_flat[valid_mask, 1] * width + pixels_flat[valid_mask, 0]).astype(int)
        u = pixels_flat[valid_mask, 0]
        v = pixels_flat[valid_mask, 1]

        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)

        # Consider neighboring pixels as well.
        offsets = np.arange(-5, 6)  # [-2, -1, 0, 1, 2]

        all_us = []
        all_vs = []

        for du in offsets:
            for dv in offsets:
                
                temp_u = u0 + du
                temp_v = v0 + dv
                
                valid_mask = (temp_u >= 0) & (temp_u < width) & (temp_v >= 0) & (temp_v < height)
                
                all_us.append(temp_u[valid_mask])
                all_vs.append(temp_v[valid_mask])
        # Merge all valid coordinates.
        all_u = np.concatenate(all_us)

        all_v = np.concatenate(all_vs)
        valid_int = (all_u >= 0) & (all_u < width) & (all_v >= 0) & (all_v < height)
        indices = (all_v[valid_int] * width + all_u[valid_int]).astype(int)
        indices = np.unique(indices)
        # 7. Update the occupancy vector.
        occupancy_vector[indices] = True
        occupancy_vectors.append(occupancy_vector)
    return mapping_matrices, occupancy_vectors

def split_panorama_image(image: np.ndarray, mapping_matrices: list[np.ndarray],is_rgb):
    """
    Split a panorama image using precomputed mapping matrices.
    """
    splitted_images = []
    
    for pixels in mapping_matrices:
        splitted_image=None
        if is_rgb:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], 
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)   
        else:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], 
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE) 
        splitted_images.append(splitted_image)
    
    return splitted_images


def poisson_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, ndarray]:
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode='wrap' if wrap_y else 'edge')
    
    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1],         # up
        grid_index[2:, 1:-1],          # down
        grid_index[1:-1, :-2],         # left
        grid_index[1:-1, 2:]           # right
    ], axis=-1).reshape(-1)                                                                 
    indptr = np.arange(0, height * width * 5 + 1, 5) 
    A = csr_array((data, indices, indptr), shape=(height * width, height * width))
    
    return A


def grad_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, np.ndarray]:
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]           
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]           
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    A = csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))

    return A


def merge_panorama_depth(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray]):
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = utils3d.numpy.image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)
        
        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask, cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]
        
        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        
        panorama_pred_masks.append(panorama_pred_mask)  
        
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # Solve overdetermined system
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask], 
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    x, *_ = lsmr(
        A, b, 
        atol=1e-5, btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None, 
        show=False,
    )
    
    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask
         


