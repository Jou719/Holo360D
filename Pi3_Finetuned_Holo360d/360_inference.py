import argparse
import glob
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from datasets.base.PanoramaToPerspective import PanoramaToPerspective
from pi3.models.pi3 import Pi3
from pi3.utils.geometry import depth_edge


MODEL_CACHE: dict[tuple[str, str], Pi3] = {}


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct RGB point clouds from panoramas and save a merged PLY.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint or folder")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Panorama RGB directory")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional panorama mask directory")
    parser.add_argument("--output_dir", type=str, default="./ply_outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--view_type", type=str, default="8views", choices=["8views", "10views"])
    parser.add_argument("--resize_w", type=int, default=2880)
    parser.add_argument("--resize_h", type=int, default=1440)
    parser.add_argument("--conf_keep_percent", type=float, default=0.20)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--voxel_downsample", type=float, default=0.0)
    parser.add_argument("--mask_sky", action="store_true")
    return parser.parse_args()


def load_model(device: torch.device, ckpt_path: str) -> Pi3:
    cache_key = (str(device), ckpt_path)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    print(f"Loading model from {ckpt_path}")
    model = Pi3().to(device).eval()

    if os.path.isdir(ckpt_path):
        candidates = [
            os.path.join(ckpt_path, "pytorch_model.safetensors"),
            os.path.join(ckpt_path, "pytorch_model.bin"),
            os.path.join(ckpt_path, "pytorch_model.pt"),
        ]
        candidates = [path for path in candidates if os.path.exists(path)]
        if not candidates:
            found = sorted(glob.glob(os.path.join(ckpt_path, "**", "pytorch_model.*"), recursive=True))
            if not found:
                raise FileNotFoundError(f"No pytorch_model.* found in: {ckpt_path}")
            ckpt_path = found[0]
        else:
            ckpt_path = candidates[0]
        print(f"Using checkpoint file: {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(ckpt_path)
    else:
        weight = torch.load(ckpt_path, map_location=device, weights_only=False)

    if any(key.startswith("module.") for key in weight):
        weight = {key.replace("module.", ""): value for key, value in weight.items()}

    model.load_state_dict(weight, strict=False)
    MODEL_CACHE[cache_key] = model
    return model


def collect_panorama_paths(rgb_dir: str) -> list[str]:
    paths: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(rgb_dir, "**", ext), recursive=True))
    return sorted(paths)


def read_rgb_and_mask(impath: str, resize_w: int, resize_h: int, mask_dir: str | None = None):
    rgb = np.array(Image.open(impath).convert("RGB"))
    rgb = cv2.resize(rgb, (resize_w, resize_h))

    panorama_mask = None
    if mask_dir is not None:
        img_basename = os.path.basename(impath)
        maskpath = os.path.join(mask_dir, img_basename)
        if os.path.exists(maskpath):
            panorama_mask = np.array(Image.open(maskpath))
            if panorama_mask.ndim == 3:
                panorama_mask = panorama_mask[:, :, 0]
            panorama_mask = cv2.resize(panorama_mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
            panorama_mask = (panorama_mask > 0).astype(np.float32)
            rgb = rgb * panorama_mask[:, :, np.newaxis]
            print(f"  Loaded panorama mask: {maskpath}")

    return rgb, panorama_mask


def build_sky_masks_for_perspectives(colors_all: np.ndarray, out_dir: str) -> np.ndarray:
    from visual_util import download_file_from_url, segment_sky
    import onnxruntime

    os.makedirs(out_dir, exist_ok=True)
    sky_onnx_path = "skyseg.onnx"
    if not os.path.exists(sky_onnx_path):
        print("Downloading skyseg.onnx ...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", sky_onnx_path)

    sess = onnxruntime.InferenceSession(sky_onnx_path)
    tmp_dir = os.path.join(out_dir, "_tmp_skyseg")
    os.makedirs(tmp_dir, exist_ok=True)

    total_n, height, width, _ = colors_all.shape
    sky_keep_mask = np.ones((total_n, height, width), dtype=np.float32)

    for idx in range(total_n):
        img_uint8 = (np.clip(colors_all[idx], 0.0, 1.0) * 255).astype(np.uint8)
        tmp_path = os.path.join(tmp_dir, f"persp_{idx:06d}.png")
        mask_path = os.path.join(tmp_dir, f"persp_{idx:06d}_sky.png")
        cv2.imwrite(tmp_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        sky_mask = segment_sky(tmp_path, sess, mask_path)
        if sky_mask.shape[:2] != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        sky_keep_mask[idx] = (sky_mask > 0).astype(np.float32)

    return sky_keep_mask


def run_inference(
    ckpt: str,
    rgb_dir: str,
    mask_dir: str | None = None,
    output_dir: str = "./ply_outputs",
    device_name: str = "cuda",
    view_type: str = "8views",
    resize_w: int = 2880,
    resize_h: int = 1440,
    conf_keep_percent: float = 0.20,
    indices: str | None = None,
    voxel_downsample: float = 0.0,
    mask_sky: bool = False,
):
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    p2p = PanoramaToPerspective(output_resolution=518, view_type=view_type)
    model = load_model(device, ckpt)
    pano_paths = collect_panorama_paths(rgb_dir)
    if not pano_paths:
        raise FileNotFoundError(f"No panorama images found in {rgb_dir}")

    if indices is not None:
        parsed_indices = [int(item.strip()) for item in indices.split(",") if item.strip()]
        pano_paths = [path for i, path in enumerate(sorted(pano_paths)) if i in parsed_indices]
        print(f"Selected {len(pano_paths)} images by indices: {parsed_indices}")
        if not pano_paths:
            raise ValueError(
                f"No images matched indices {parsed_indices}. "
                f"Available image count: {len(collect_panorama_paths(rgb_dir))}."
            )

    all_imgs = []
    all_masks = []
    per_pano_names = []
    per_pano_counts = []

    print(f"Found {len(pano_paths)} panoramas. Splitting all first...")
    for impath in pano_paths:
        base = os.path.splitext(os.path.basename(impath))[0]
        per_pano_names.append(base)

        rgb_image, panorama_mask = read_rgb_and_mask(impath, resize_w, resize_h, mask_dir)
        splitted_rgbs, _, _ = p2p.process_single_image(rgb_image, None, np.eye(4, dtype=np.float32))
        if not splitted_rgbs:
            raise RuntimeError(f"Splitter returned empty for {impath}")

        all_imgs.extend([torch.from_numpy(view).permute(2, 0, 1).float() / 255.0 for view in splitted_rgbs])
        per_pano_counts.append(len(splitted_rgbs))

        if panorama_mask is not None:
            mask_3ch = np.stack([panorama_mask, panorama_mask, panorama_mask], axis=-1)
            splitted_masks_rgb, _, _ = p2p.process_single_image(mask_3ch, None, np.eye(4, dtype=np.float32))
            all_masks.extend([mask_rgb[:, :, 0] if mask_rgb.ndim == 3 else mask_rgb for mask_rgb in splitted_masks_rgb])
        else:
            all_masks.extend([None] * len(splitted_rgbs))

    images_all = torch.stack(all_imgs, dim=0).to(device, non_blocking=True)
    print(f"Total perspectives: {images_all.shape[0]}. Running one-shot inference...")

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                res = model(images_all[None])
        else:
            res = model(images_all[None])

    pred_points_all = res["points"][0]
    pred_local_points_all = res["local_points"][0]
    pred_conf_all = res["conf"][0]
    pred_c2w_all = res["camera_poses"][0]
    colors_all = images_all.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    sky_keep_all = build_sky_masks_for_perspectives(colors_all, output_dir) if mask_sky else None

    start = 0
    all_pts = []
    all_cols = []
    for pano_name, cnt in zip(per_pano_names, per_pano_counts):
        end = start + cnt
        pred_points = pred_points_all[start:end]
        pred_local_points = pred_local_points_all[start:end]
        pred_conf = pred_conf_all[start:end]
        colors = colors_all[start:end]
        masks_slice = all_masks[start:end]

        conf_sigmoid = torch.sigmoid(pred_conf[..., 0]).cpu().numpy()
        flat = conf_sigmoid.reshape(-1)
        if conf_keep_percent <= 0:
            conf_mask = np.ones_like(conf_sigmoid, dtype=bool)
        elif conf_keep_percent >= 1:
            conf_mask = np.zeros_like(conf_sigmoid, dtype=bool)
        else:
            threshold = np.percentile(flat, (1.0 - conf_keep_percent) * 100.0)
            conf_mask = conf_sigmoid > threshold

        non_edge = (~depth_edge(pred_local_points[..., 2], rtol=0.1)).cpu().numpy()
        panorama_mask_slice = np.stack([
            np.ones((conf_mask.shape[1], conf_mask.shape[2]), dtype=bool) if mask is None else ((mask.astype(np.float32) / 255.0 > 0.5) if mask.max() > 1.0 else mask > 0.5)
            for mask in masks_slice
        ], axis=0)

        valid_mask = conf_mask & non_edge & panorama_mask_slice
        if sky_keep_all is not None:
            valid_mask = valid_mask & (sky_keep_all[start:end] > 0.5)

        pts = pred_points.cpu().numpy()[valid_mask].reshape(-1, 3)
        cols = colors[valid_mask].reshape(-1, 3)
        if pts.shape[0] > 0:
            all_pts.append(pts)
            all_cols.append(cols)

        print(f"{pano_name}: kept {pts.shape[0]} points")
        start = end

    if not all_pts:
        raise RuntimeError("No valid points after masking for all panoramas.")

    pts_merged = np.concatenate(all_pts, axis=0)
    cols_merged = np.concatenate(all_cols, axis=0)

    poses = pred_c2w_all.cpu().numpy()
    if voxel_downsample > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64, copy=False))
        pcd.colors = o3d.utility.Vector3dVector(cols_merged.astype(np.float64, copy=False))
        pcd = pcd.voxel_down_sample(voxel_downsample)
        pts_merged = np.asarray(pcd.points)
        cols_merged = np.asarray(pcd.colors)

    return {
        "points": pts_merged,
        "colors": cols_merged,
        "poses": poses,
        "view_type": view_type,
        "per_pano_counts": per_pano_counts,
    }


def main() -> None:
    args = setup_args()
    result = run_inference(
        ckpt=args.ckpt,
        rgb_dir=args.rgb_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        view_type=args.view_type,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        conf_keep_percent=args.conf_keep_percent,
        indices=args.indices,
        voxel_downsample=args.voxel_downsample,
        mask_sky=args.mask_sky,
    )
    pts_merged = result["points"]
    cols_merged = result["colors"]
    poses = result["poses"]

    os.makedirs(args.output_dir, exist_ok=True)
    out_ply = os.path.join(args.output_dir, "merged_all_panoramas.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(cols_merged.astype(np.float64, copy=False))
    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False, compressed=False, print_progress=False)
    print(f"Saved merged point cloud: {out_ply} (points: {np.asarray(pcd.points).shape[0]})")

    pose_path = os.path.join(args.output_dir, "pose.txt")
    with open(pose_path, "w", encoding="utf-8") as handle:
        for pose in poses:
            rotation = pose[:3, :3].reshape(-1)
            translation = pose[:3, 3]
            values = [*translation.tolist(), *rotation.tolist()]
            handle.write(" ".join(f"{value:.6f}" for value in values) + "\n")
    print(f"Saved camera poses: {pose_path}")


if __name__ == "__main__":
    main()

