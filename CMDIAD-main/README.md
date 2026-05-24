# Incomplete Multimodal Industrial Anomaly Detection via Cross-Modal Distillation

This repository is the official implementation of 
[Incomplete Multimodal Industrial Anomaly Detection via Cross-Modal Distillation](https://doi.org/10.1016/j.inffus.2025.103572). 

## PIRN-CMPT Adaptation

This fork changes the default pipeline into a PIRN-based RGB/SN reconstruction system with CMPT missing-modality completion:

- the original point-cloud / Point-MAE branch is no longer used by default;
- the geometric branch is an SN (surface-normal) image branch generated from the organized XYZ tiff;
- CMPT predicts missing-modality features/prototypes, not RGB/SN images;
- normal training samples are summarized into RGB-specific, SN-specific, and shared prototypes;
- SPA uses balanced Sinkhorn prototype assignment to reconstruct local tokens from normal prototypes;
- APR refines the learned prototypes from normal-token assignment statistics;
- MNC mixes own-modality reconstruction with cross-modal prototype reconstruction and outputs reconstruction-error maps;
- `--method_name PIRN_CMPT` is the default entry point. `CMPT` remains as a backward-compatible alias.

Training data flow:

1. RGB image -> frozen DINOv2 RGB encoder -> RGB patch tokens.
2. Organized XYZ -> SN image -> frozen DINOv2 SN encoder -> SN patch tokens.
3. Paired RGB/SN tokens train the CMPT feature-transfer MLP.
4. Normal tokens build structured prototype memory: RGB-specific, SN-specific, and shared prototypes.
5. CMPT transfers RGB-specific prototypes to pseudo SN prototypes and SN-specific prototypes to pseudo RGB prototypes.

Missing-modality inference flow:

- `--main_modality rgb`: RGB tokens are extracted; SN tokens are not encoded from input. CMPT predicts pseudo SN tokens, and RGB-specific prototypes are transferred to pseudo SN prototypes. RGB and pseudo SN reconstruction-error maps are fused into the final heatmap.
- `--main_modality sn`: SN tokens are extracted; RGB tokens are not encoded from input. CMPT predicts pseudo RGB tokens, and SN-specific prototypes are transferred to pseudo RGB prototypes. SN and pseudo RGB reconstruction-error maps are fused into the final heatmap.
- `--main_modality ''`: both RGB and SN are available; both real branches are reconstructed, with pseudo branches kept as additional CMPT consistency maps.

Run RGB available, SN missing:

```bash
python main.py \
--method_name PIRN_CMPT \
--main_modality rgb \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--sn_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_size 518 \
--experiment_note cmpt_rgb_to_sn
```

Run SN available, RGB missing:

```bash
python main.py \
--method_name PIRN_CMPT \
--main_modality sn \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--sn_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_size 518 \
--experiment_note cmpt_sn_to_rgb
```

By default CMPT does not download pretrained weights, which avoids failures on servers without
HuggingFace access. For real experiments, pass local DINOv2 checkpoints:

```bash
--rgb_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
--sn_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth
```

If your checkpoint has another filename, pass that exact file path. When checkpoint paths are empty,
CMPT also searches common local locations such as `checkpoints/`, `weights/`, `pretrained/`,
`~/.cache/torch/hub/checkpoints`, `~/.cache/huggingface/hub`, and `/root/autodl-tmp`.

For a smoke test only, you may omit checkpoints; the encoders will be randomly initialized and the
metrics are not meaningful. If the server can access HuggingFace, add `--allow_pretrained_download`
to let `timm` download weights automatically.

Minimum Python packages for `--method_name PIRN_CMPT`:

```bash
pip install torch torchvision timm==0.9.12 scikit-learn scipy pandas tabulate tifffile
```

The legacy Point-MAE/CMDIAD paths still require their original CUDA extensions. PIRN_CMPT does not require
`cupy`, `knn_cuda`, or `pointnet2_ops`.

## Accepted paper
This work has been accepted by *Information Fusion* (2025.08) :blush:

## Visualization of Some Prediction Results
![fig1](./figures/fig1.png)
## Requirements
We implement this repo with the following environment:

* Ubuntu 22.04
* CUDA 12.1
* Python 3.11
* Pytorch 2.2.0

To install requirements:

```setup
# Please install Pytorch first before other packages

# Install KNN_CUDA
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# Install Pointnet2_PyTorch(pointnet2_ops)
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
# You may encounter compilation issues for Pointnet2_PyTorch (see attached note). 

# Now you can go back and install other packages for CMDIAD :)
pip install -r requirements.txt
```

>📋  Sometimes conda's version control will cause the installation failure. We recommend using venv or conda to create 
> a virtual environment and then use pip to install all packages. 
> If you encountered compilation issues for Pointnet2_PyTorch, please modify `pointnet2_ops_lib/setup.py` with my attempts [Pull request](https://github.com/erikwijmans/Pointnet2_PyTorch/pull/177/files)

## Dataset and Pre-trained Models
### Dataset
The `MVTec 3D-AD` dataset can be downloaded from  [MVTec3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). 
It should be unzipped and placed under the `datasets` folder.

### Data Pre-processing
```Pre-processing
python utils/preprocessing.py --dataset_path datasets/mvtec_3d/ 
```
>📋  It is recommended to use the default value for the path to the dataset to prevent problems in subsequent training and evaluation, but you can change the number of threads used according to your configuration. Please note that the pre-processing is performed in place.
### Checkpoints
| Purpose                               | Checkpoint                                                                                          |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| Point Clouds (PCs) feature extractor  | ~~Point-MAE~~     |
| RGB Images feature extractor          | DINOv2 ViT-B/14   |
| Feature-to-Feature network (main PCs) | ~~MTFI_FtoF_PCs~~ |
| Feature-to-Input network (main PCs)   | ~~MTFI_FtoI_PCs~~ |
| Input-to-Feature network (main PCs)   | ~~MTFI_ItoF_PCs~~ |
| Feature-to-Feature network (main RGB) | ~~MTFI_FtoF_RGB~~ |
| Feature-to-Input network (main RGB)   | ~~MTFI_FtoI_RGB~~ |
| Input-to-Feature network (main RGB)   | ~~MTFI_ItoF_RGB~~ |
(Updated 2026.03.17) All the checkpoints can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.18456013), old share links expired.
>📋  Please put all checkpoints in folder `checkpoints`. 

## Training

To train the models in the paper, run these commands:
### MTFI pipeline with Feature-to-Feature distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_size 518 \
--experiment_note <your_note> \
--save_feature_for_fusion \
--save_path datasets/patch_lib \
```
> The results are saved in the `results` folder.
> If you need to output the raw anomaly scores at image or pixel level to a file, add `--save_raw_results` or `--save_seg_results`. You can use `utils/heatmap` to generate similar visualized results.
> You can define the maximum number of threads with `--cpu_core_num` and leave your note through `--experiment_note`.  

To train MTFI pipeline with Feature-to-Feature distillation network:
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/patch_lib \
--output_dir <your_output_dir_path> \
--train_method HallucinationCrossModality \
--num_workers 2 \
```
>📋 For MTFI pipeline with Feature-to-Feature distillation network, PCs or RGB images as the main modality are trained simultaneously.
> If you think your GPU memory is really not enough, maybe try with `--accum_iter 2` for Gradient Accumulation and change `--batch_size 16` correspondingly.
> The data is loaded into GPU memory in advance to speed up the training, you can change it through dataset and dataloader.   


### MTFI pipeline with Feature-to-Input distillation network:
To save the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_size 518 \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```
For PCs as main modality.
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--train_method XYZFeatureToRGBInputConv \
```
For RGB images as main modality.
```
python hallucination_network_pretrain.py \
--lr 0.0005 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--train_method RGBFeatureToXYZInputConv \
```
### MTFI pipeline with Input-to-Feature distillation network:
Similarly, you need to store the features for distillation network training:
```
python main.py \
--method_name DINO+Point_MAE \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_size 518 \
--experiment_note <your_note> \
--save_frgb_xyz \
--save_path_frgb_xyz datasets/frgb_xyz \
--save_rgb_fxyz \
--save_path_rgb_fxyz datasets/rgb_fxyz \
```

For PCs as main modality.
```
python -u hallucination_network_pretrain.py \
--lr 0.0003 \
--batch_size 32 \
--data_path datasets/frgb_xyz \
--output_dir <your_output_dir_path> \
--train_method XYZInputToRGBFeatureHRNET \
--c_hrnet 128 \
--pin_mem \
```
For RGB images as main modality.
```
python -u hallucination_network_pretrain.py \
--lr 0.0002 \
--batch_size 32 \
--data_path datasets/rgb_fxyz \
--output_dir <your_output_dir_path> \
--train_method XYZInputToRGBFeatureHRNET \
--c_hrnet 192 \
--pin_mem \
```

## Evaluation

### Evaluate the model on MVTec 3D-AD with single and dual memory bank method
For single PCs memory bank:
```single PCs memory bank
python main.py \
--method_name Point_MAE \
--experiment_note <your_note> \
```

>📋 The RGB branch now uses DINOv2 by default. If the machine cannot download timm weights automatically, pass `--rgb_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth`.
> For single RGB memory bank and dual memory bank, please replace `Point_MAE` with `DINO` and `DINO+Point_MAE`, respectively.

### MTFI pipeline with Feature-to-Feature distillation network:
For PCs as main modality.
```MTFI PCs
python main.py \
--method_name WithHallucination \
--use_hn \
--main_modality xyz \
--fusion_module_path checkpoints/MTFI_FtoF_PCs.pth \
--experiment_note <your_note> \
```

>📋 For RGB images as main modality, please replace `xyz` with `rgb` for `--main_modality` and give the new checkpoint path `checkpoints/MTFI_FtoF_RGB.pth` to the model.

### MTFI pipeline with Feature-to-Input distillation network:
For PCs as main modality.
```
python main.py \
--method_name WithHallucinationFromFeature \
--use_hn_from_rgb_conv \
--main_modality xyz \
--fusion_module_path checkpoints/MTFI_FtoI_PCs.pth \
--experiment_note <your_note> \
```

>📋 For RGB images as main modality, replace `xyz` with `rgb` and give model the new checkpoint path.

### MTFI pipeline with Input-to-Feature distillation network:
For PCs as main modality.
```
python main.py \
--method_name WithHallucination \
--use_hrnet \
--main_modality xyz \
--c_hrnet 128 \
--fusion_module_path checkpoints/MTFI_ItoF_PCs.pth \
--experiment_note <your_note> \
```

For RGB images as main modality.
```
python main.py \
--method_name WithHallucination \
--use_hrnet \
--main_modality rgb \
--c_hrnet 192 \
--fusion_module_path checkpoints/MTFI_ItoF_RGB.pth \
--experiment_note <your_note> \
```

## Citation
If you think this repository is helpful for your project, please use the following.
```
@article{SUI2025103572,
title = {Incomplete multimodal industrial anomaly detection via cross-modal distillation},
journal = {Information Fusion},
pages = {103572},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103572},
url = {https://www.sciencedirect.com/science/article/pii/S156625352500644X},
author = {Wenbo Sui and Daniel Lichau and Josselin Lefèvre and Harold Phelippeau},
}
```
## Acknowledgement
We appreciate the following github repos for their valuable code:
- [M3DM](https://github.com/nomewang/M3DM/)
- [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS)
- [Shape-Guided](https://github.com/jayliu0313/Shape-Guided)

