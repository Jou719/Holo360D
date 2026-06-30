export CUDA_VISIBLE_DEVICES=0
python 360_inference.py \
  --ckpt ./ckpt/10views.bin \
  --rgb_dir ./assets/singleview_outdoor/rgb \
  --mask_dir ./assets/singleview_outdoor/mask \
  --output_dir ./ply_outputs/10views/ \
  --view_type 10views \
  --voxel_downsample 0.0 \
  --conf_keep_percent 0.5 \
  --mask_sky  \
  # --indices "15" \