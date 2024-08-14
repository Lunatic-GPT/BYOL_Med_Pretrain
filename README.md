# BYOL_Med_Pretrain
 
A simple framework to pretrain medical images with BYOL, inspired by [this reop](https://github.com/lucidrains/byol-pytorch).

## difference
### medical image support
- support loading dcm/mhd/nii directly with SimpleITK
- support window apply to images
### dual crop

1. crop1: crop in dataloader to get the same size
2. crop2: data aug

### data aug

- medical image friendly data aug (for small ROI)