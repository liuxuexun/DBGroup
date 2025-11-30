# Multi-view Feature Fusion

## Prerequisites

### Data preprocessing

Follow [this instruction](../preprocess/README.md) to obtain the processed 2D and 3D data.

- **3D**: Point clouds in the pytorch format `.pth`
- **2D**: RGB-D images with their intrinsic and extrinsic parameters

## Run the code

Take ScanNet as an example, to perform multi-view feature fusion your can run:

```bash
python scannet_openseg.py \
            --data_dir PATH/TO/scannet_processed \
            --output_dir PATH/TO/OUTPUT_DIR \
            --openseg_model PATH/TO/OPENSEG_MODEL \
            --process_id_range 0,100\
            --split train

or you can use the default settings:

python scannet_openseg.py


```

where:

- `data_dir`: path to the pre-processed 2D&3D data from [here](../../openscene#datasets)
- `output_dir`: output directory to save your fused features
- `openseg_model`: path to the OpenSeg model
- `process_id_range`: only process scenes within the range
- `split`: choose from `train`/`val`/`test` data to process

## Suggestions & Troubleshooting

- For using the OpenSeg model, you'd better make sure you have an NVIDIA GPU with **>30G memory**.
