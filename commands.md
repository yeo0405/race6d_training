## Transfer
python tools/prepare_pose_rgbd_dataset.py \
  --source /home/yeo/Downloads/POSE/dataset_ft \
  --output my_data/pose_rgbd_race6d \
  --category-yaml configs/race6d/include/pose_dataset2coco_category.yml \
  --max-depth-mm 2000 \
  --copy-rgb

python tools/prepare_pose_rgbd_dataset_FT.py \
  --source /home/yeo/Downloads/POSE/dataset_ft \
  --output my_data/pose_rgbd_race6d \
  --category-yaml configs/race6d/include/pose_dataset2coco_category.yml \
  --crop-left 240 \
  --crop-right 240 \
  --max-depth-mm 2000 \
  --copy-rgb

- pose_dataset2coco_category.yml
- include/race6d_r50vd_pose_rgbd.yml

## Visualize
python tools/preview_race6d_dataset.py \
  --dataset my_data/pose_rgbd_race6d \
  --category-yaml configs/race6d/include/pose_dataset2coco_category.yml \
  --split all \
  --num-images 100 \
  --seed 42 \
  --output-dir output/pose_rgbd_preview

## Check
python tools/check_race6d_dataset.py \
  --dataset /home/yeo/Downloads/race_6d/my_data/pose_rgbd_race6d \
  --split train

## Train
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_pose_rgbd.yml --use-amp

## Tune
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_pose_rgbd.yml --use-amp -t SEP10/race6d_r50vd_pose_rgbd/last.pth

## Test
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_pose_rgbd.yml --test-only -r output/last.pth --overlay-dir output/race6d_r50vd_pose_rgbd/overlays --overlay-limit 300 --score-threshold 0.25 --pose-iou-threshold 0.5

## Predict Once
python tools/predict_single.py \
  -c configs/race6d/r50vd/race6d_r50vd_pose_rgbd.yml \
  -r output/last.pth \
  --rgb /home/yeo/Downloads/POSE/Testing_videos/1/color.jpg \
  --depth /home/yeo/Downloads/POSE/Testing_videos/1/depth.png \
  --camera-k 1070.7897900136718 0 962.3093872070312 0 1070.5269785615235 512.0372314453125 0 0 1 \
  --output output/single_prediction.json \
  --debug-image output/single_prediction.jpg

## Check Training process
 python tools/analyze_log.py output/log.txt 

## check K
 python tools/check_intrinsic.py --json /home/yeo/Downloads/race_6d/my_data/pose_rgbd_race6d/annotations/instances_train.json

 ## dataset merge
python tools/dataset_merg.py --dataset-a dataset_blender/ --dataset-b dataset_foundationpose/ --output my_data/pose_rgbd_race6d

## upload to thor (passwrod: tomo)
rsync -avh --info=progress2 --partial \
  /home/yeo/Downloads/race_6d/my_data/pose_rgbd_race6d/ \
  tomo@tt1:/home/tomo/training/race_thor/my_data/pose_rgbd_race6d/

## Tmux

```
tmux new -s race6d
tmux ls
tmux attach -t race6d
```