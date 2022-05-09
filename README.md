# Nuclei segmentation

## Objective  

- Compare, Dist-U-net, Dist-Res-U-Net
- make public an efficient segmentation algorithm
- pre-trained weights

## Models to compare

- U-Net - Dist
- HoverNet
- Pre-trained U-Net

## Metrics

- F1 pixel classification
- AJI
- IoU different thresholds

## Datasets

- CONSEP
- MoNuSAC
- TNBC

## Evaluation procedure

Train, validation and test split

# For local debuging: 

'''singularity shell --bind /data:/data ./environment/segmentation-unet.sif'''

