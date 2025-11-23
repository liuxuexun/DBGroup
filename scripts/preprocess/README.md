# 2D & 3D Data Preprocessing

## Prerequisites

### Environment

Make sure the following package installed:

- `torch`
- `numpy`
- `plyfile`
- `opencv-python`
- `imageio`
- `pandas`
- `tqdm`

### Download the original dataset

- **ScanNet**: Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet). (You should also download the .sens file)

## Run the code

For preprocessing 3D point clouds with GT labels (except for Replica, no semantic labels), one can simply run:

```bash
python preprocess_3d_scannet.py
```

For preprocessing 2D RGB-D images, one can also simply run:

```bash
python preprocess_2d_scannet.py
```

**Note**: In the code, you can modify the following (but we **recommend** you to use the default setting):

- `in_path`: path to the original downloaded dataset
- `out_dir`: output directory to save your processed data
- `scene_list`: path to the list containing all scenes
- `split`: choose from `train`/`val`/`test` data to process

## Folder structure

Once running the pre-processing code above, you should have a data structure like below. Here we choose the processed ScanNet as an example:

```
data/
│
├── scannet_2d
│   │
│   ├── scene0000_00
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │
│   ├── scene0000_01
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │
│   └── ...
|   |
|   └── intrinsics.txt (fixed intrinsic parameters for all scenes)
│
└── scannet_3d
    │
    ├── train
    │   ├── scene0000_00.pth
    │   ├── scene0000_01.pth
    │   ├── ...
    │   ├── scene0706_00.pth
    │
    └── val
        ├── scene0011_00.pth
        └── ...
  
```
