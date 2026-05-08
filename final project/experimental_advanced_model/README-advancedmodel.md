# Experimental Advanced Model Files

This folder contains the code pulled from the RunPod workspace for the experimental advanced model (which has some advantages to the other model but is mostly worse):

`timm_geometry_outputs_b3_textgeom_gridgrad`

The runnable conventional filenames are:

- `model.py`
- `train.py`
- `evaluate.py`
- `inference.py`
- `dataset_loader.py`

These files correspond to the more geometry-heavy experimental model that used stronger grid supervision, grid-gradient loss, edge-weighted grid loss, decoder refinement blocks, and a deeper flow head.

Important training configuration for this advanced model:

- Backbone: `tf_efficientnet_b3_ns`
- Pretrained encoder: `True`
- Image size: `384 x 384`
- Batch size: `16`
- Learning rate: `1e-4`
- Encoder learning-rate scale: `0.1`
- Grid loss weight: `80.0`
- Grid-gradient loss weight: `120.0`
- Edge-grid loss weight: `30.0`
- Decoder refinement blocks: `3`
- Head channels: `96`
- Head depth: `3`
