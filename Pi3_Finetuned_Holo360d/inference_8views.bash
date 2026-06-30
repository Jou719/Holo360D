export CUDA_VISIBLE_DEVICES=0
python 360_inference.py \
  --ckpt ./ckpt/8views.bin \
  --rgb_dir ./assets/multiview_outdoor/rgb \
  --mask_dir ./assets/multiview_outdoor/mask \
  --output_dir ./ply_outputs/8views/ \
  --view_type 8views \
  --voxel_downsample 0.0 \
  --conf_keep_percent 0.8 \
  --mask_sky  \
  # --indices "1,2" \
