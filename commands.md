## Transfer
```bash
python tools/prepare_blenderproc_race6d.py \
    --source ~/Downloads/BlenderporcGenaratedDataset/iphone_only/bop_data/hb/train_pbr \
    --output iphone_only/pose_rgbd_race6d \
    --models-dir ~/Downloads/DTTD_obj/Apple_pick \
    --width 640 \
    --height 480 \
    --val-ratio 0.1 \
    --seed 42 \
    --max-depth-mm 2000
```

## Merged

```bash
python tools/merge_race6d_datasets.py \
    --inputs \
        iphone_only/pose_rgbd_race6d  \
        box_condition/pose_rgbd_race6d \
    --output my_data/pose_rgbd_race6d 
```

## Visualize

```bash
python tools/preview_race6d_dataset.py \
  --dataset my_data/pose_rgbd_race6d \
  --category-yaml configs/race6d/include/pose_dataset2coco_category.yml \
  --split all \
  --num-images 100 \
  --seed 42 \
  --output-dir output/pose_rgbd_preview
```

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

## Evaluation
python tools/eval_per_class.py \
  -c configs/race6d/r50vd/race6d_r50vd_pose_rgbd.yml \
  -r Oct26/race6d_1026.pth \
  --output-dir output/race6d_per_class_050 \
  --score-threshold 0.50 \
  --pose-iou-threshold 0.5

# Visualize Evalutaion
python tools/visualize_evaluation.py \
  --evaluation-stats output/race6d_per_class/evaluation_stats.json \
  --per-class-metrics output/race6d_per_class/per_class_metrics.json \
  --output-dir output/evaluation_visualization

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