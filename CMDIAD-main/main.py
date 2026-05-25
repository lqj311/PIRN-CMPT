import argparse
from cmdiad_runner import CMDIAD
from dataset import eyecandies_classes, mvtec3d_classes
from utils.utils import set_multithreading
import pandas as pd
import os
import gc


def _default_classes(args):
    if args.dataset_type == 'eyecandies':
        return eyecandies_classes()
    if args.dataset_type == 'mvtec3d':
        return mvtec3d_classes()
    raise ValueError(f'Unsupported dataset_type: {args.dataset_type}')


def resolve_classes_and_dataset_path(args):
    known_classes = _default_classes(args)
    dataset_path = os.path.normpath(args.dataset_path)
    leaf = os.path.basename(dataset_path)

    if args.classes:
        classes = [cls.strip() for cls in args.classes.split(',') if cls.strip()]
    elif leaf in known_classes:
        classes = [leaf]
        args.dataset_path = os.path.dirname(dataset_path)
        print(f'[Dataset] Detected class directory `{dataset_path}`. '
              f'Using --dataset_path `{args.dataset_path}` and --classes `{leaf}`.')
    else:
        classes = known_classes

    unknown = [cls for cls in classes if cls not in known_classes]
    if unknown:
        raise ValueError(f'Unknown classes for {args.dataset_type}: {unknown}. Known classes: {known_classes}')
    return classes


def append_result_table(path, experiment_note, dataframe):
    results_dir = os.path.dirname(path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    if os.path.isdir(path):
        legacy_dir = path
        legacy_backup = f'{path}.legacy_dir'
        suffix = 1
        while os.path.exists(legacy_backup):
            legacy_backup = f'{path}.legacy_dir_{suffix}'
            suffix += 1
        os.rename(legacy_dir, legacy_backup)
        print(f'[Results] Renamed legacy result directory `{legacy_dir}` to `{legacy_backup}`.')
    with open(path, "a", encoding="utf-8") as tf:
        tf.write('\n\n' + experiment_note + '\n')
        tf.write(dataframe.to_markdown(index=False))


def cleanup_runtime_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_3d_ads(args):
    classes = resolve_classes_and_dataset_path(args)

    METHOD_NAMES = [args.method_name]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_001_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        model = CMDIAD(args)
        model.fit(cls)
        image_rocaucs, pixel_rocaucs, au_pros, au_pros_001 = model.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)
        au_pros_001_df[cls.title()] = au_pros_001_df['Method'].map(au_pros_001)

        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")
        del model
        cleanup_runtime_cache()

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1), 3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1), 3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1), 3)
    au_pros_001_df['Mean'] = round(au_pros_001_df.iloc[:, 1:].mean(axis=1), 3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))

    # print("\n\n##########################################################################")
    # print("############################ AU PRO 0.01 Results #########################")
    # print("##########################################################################\n")
    # print(au_pros_001_df.to_markdown(index=False))

    if args.save_results:
        append_result_table("results/image_rocauc_results.md", args.experiment_note, image_rocaucs_df)
        append_result_table("results/pixel_rocauc_results.md", args.experiment_note, pixel_rocaucs_df)
        append_result_table("results/aupro_results.md", args.experiment_note, au_pros_df)
        # append_result_table("results/aupro_001_results.md", args.experiment_note, au_pros_001_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--method_name', default='PIRN_CMPT', type=str,
                        choices=['PIRN_CMPT', 'CMPT', 'DINO', 'Point_MAE', 'DINO+Point_MAE',
                                 'WithHallucination', 'WithHallucinationFromFeature'],
                        help='Anomaly detection modal name.')
    parser.add_argument('--max_sample', default=500, type=int,
                        help='Max sample number.')
    parser.add_argument('--few_shot_k', default=0, type=int,
                        help='Use a deterministic K-shot subset of normal training samples when > 0.')
    parser.add_argument('--shot_seed', default=0, type=int,
                        help='Random seed for deterministic K-shot sample selection.')
    parser.add_argument('--save_fewshot_list', default=False, action='store_true',
                        help='Save selected K-shot training sample paths for reproducibility.')
    parser.add_argument('--fewshot_list_dir', default='results/fewshot_lists', type=str,
                        help='Directory used by --save_fewshot_list.')
    parser.add_argument('--allow_true_missing_modality', default=False, action='store_true',
                        help='Allow test samples with only RGB or only XYZ/SN when main_modality is rgb/sn.')
    parser.add_argument('--memory_bank', default='multiple', type=str,
                        choices=["multiple"])
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch14_dinov2.lvd142m', type=str,
                        choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k',
                                 'vit_small_patch8_224_dino','vit_base_patch14_dinov2.lvd142m'],
                        help='Timm checkpoints name of RGB backbone.')
    parser.add_argument('--rgb_checkpoint_path', default='', type=str,
                        help='Optional local RGB backbone checkpoint path.')
    parser.add_argument('--sn_backbone_name', default='vit_base_patch14_dinov2.lvd142m', type=str,
                        help='Timm checkpoint name of SN backbone.')
    parser.add_argument('--sn_checkpoint_path', default='', type=str,
                        help='Optional local SN backbone checkpoint path.')
    parser.add_argument('--allow_pretrained_download', default=False, action='store_true',
                        help='Allow timm to download pretrained DINOv2 weights when checkpoint paths are empty.')
    parser.add_argument('--feature_dim', default=768, type=int)
    parser.add_argument('--cmpt_checkpoint_path', default='', type=str)
    parser.add_argument('--cmpt_save_path', default='', type=str)
    parser.add_argument('--cmpt_epochs', default=10, type=int)
    parser.add_argument('--cmpt_batch_size', default=4096, type=int)
    parser.add_argument('--cmpt_max_train_tokens', default=200000, type=int)
    parser.add_argument('--cmpt_feature_grid', default=56, type=int,
                        help='Token grid used by CMPT/PIRN_CMPT prototype modules. Default 56 keeps prior behavior; 16 approximates PIRN N=256.')
    parser.add_argument('--cmpt_lr', default=1e-4, type=float)
    parser.add_argument('--cmpt_weight_decay', default=1e-4, type=float)
    parser.add_argument('--cmpt_hidden_ratio', default=2.5, type=float)
    parser.add_argument('--cmpt_mlp_depth', default=2, type=int)
    parser.add_argument('--cmpt_cycle_loss_weight', default=0.25, type=float,
                        help='Cycle-consistency loss weight for CMPT RGB->SN->RGB and SN->RGB->SN training.')
    parser.add_argument('--cmpt_cosine_loss_weight', default=0.1, type=float,
                        help='Cosine alignment loss weight for paired CMPT feature transfer.')
    parser.add_argument('--cmpt_relation_loss_weight', default=0.05, type=float,
                        help='Relational consistency loss weight for preserving normal-token geometry across modalities.')
    parser.add_argument('--cmpt_relation_tokens', default=512, type=int,
                        help='Maximum tokens per batch used by CMPT relational consistency loss.')
    parser.add_argument('--num_rgb_prototypes', default=64, type=int)
    parser.add_argument('--num_sn_prototypes', default=64, type=int)
    parser.add_argument('--num_shared_prototypes', default=64, type=int)
    parser.add_argument('--prototype_kmeans_iters', default=20, type=int)
    parser.add_argument('--prototype_max_tokens', default=200000, type=int)
    parser.add_argument('--spa_assignment', default='structured_ot', type=str,
                        choices=['structured_ot', 'softmax', 'topk', 'nearest'],
                        help='Prototype assignment used by SPA reconstruction.')
    parser.add_argument('--spa_topk', default=5, type=int,
                        help='Top-k prototypes used when --spa_assignment topk.')
    parser.add_argument('--disable_spa', default=False, action='store_true',
                        help='Ablation: replace SPA structured assignment with nearest prototype reconstruction.')
    parser.add_argument('--spa_temperature', default=0.05, type=float)
    parser.add_argument('--spa_sinkhorn_iters', default=5, type=int)
    parser.add_argument('--apr_update_rate', default=0.15, type=float)
    parser.add_argument('--apr_confidence_threshold', default=0.25, type=float)
    parser.add_argument('--apr_inference_refine', default=False, action='store_true')
    parser.add_argument('--apr_memory_update_iters', default=0, type=int)
    parser.add_argument('--cmpt_ot_temperature', default=0.05, type=float)
    parser.add_argument('--cmpt_ot_iters', default=5, type=int)
    parser.add_argument('--cmpt_proto_mlp_weight', default=0.5, type=float)
    parser.add_argument('--shared_proto_gate', default=0.25, type=float,
                        help='Maximum contribution of shared prototypes when mixed with modality-specific reconstruction.')
    parser.add_argument('--shared_proto_confidence_threshold', default=0.35, type=float,
                        help='Confidence threshold for gated shared prototype reconstruction.')
    parser.add_argument('--cmpt_aux_weight', default=0.35, type=float,
                        help='Maximum auxiliary contribution of CMPT pseudo-modality maps in missing-modality scoring.')
    parser.add_argument('--cmpt_aux_confidence_threshold', default=0.50, type=float,
                        help='Pseudo-modality reliability threshold for CMPT auxiliary scoring.')
    parser.add_argument('--cmpt_aux_mode', default='both', type=str, choices=['pseudo_error', 'cycle', 'both'],
                        help='How CMPT contributes under missing modality.')
    parser.add_argument('--cmpt_gate_max_tokens', default=20000, type=int,
                        help='Maximum normal training tokens used to estimate CMPT auxiliary reliability.')
    parser.add_argument('--cmpt_fusion_mode', default='consensus', type=str, choices=['add', 'consensus', 'max'],
                        help='How real and CMPT auxiliary error maps are fused under missing modality.')
    parser.add_argument('--cmpt_consensus_power', default=1.5, type=float,
                        help='Power applied to normalized real-modality map for consensus CMPT fusion.')
    parser.add_argument('--cmpt_full_consistency_weight', default=0.25, type=float,
                        help='Maximum contribution of RGB<->SN CMPT consistency maps when both modalities are available.')
    parser.add_argument('--cmpt_full_calibration_std', default=-1.0, type=float,
                        help='Normal-training std multiplier used to threshold full-modality CMPT consistency maps; negative disables calibration.')
    parser.add_argument('--cmpt_full_map_gain', default=1.0, type=float,
                        help='Gain applied to normal-calibrated full-modality CMPT consistency residual maps.')
    parser.add_argument('--cmpt_replace_mnc1', default=False, action='store_true',
                        help='Use CMPT as the first-stage cross-modal normality communication block.')
    parser.add_argument('--classic_pirn_mnc1', default=False, action='store_true',
                        help='Keep the previous PIRN-style MNC stage1 when --paper_mnc is used.')
    parser.add_argument('--cmpt_nc_weight', default=0.20, type=float,
                        help='Maximum gate weight of CMPT normality communication when replacing MNC stage1.')
    parser.add_argument('--cmpt_nc_confidence_threshold', default=0.55, type=float,
                        help='Confidence threshold for CMPT normality communication.')
    parser.add_argument('--cmpt_nc_safe_margin', default=0.02, type=float,
                        help='Allow CMPT-NC candidate reconstruction only when normal-prototype compatibility does not decrease beyond this margin.')
    parser.add_argument('--branch_fusion_mode', default='consensus', type=str, choices=['sum', 'mean', 'consensus', 'max'],
                        help='Fusion strategy for RGB and SN maps when both modalities are available.')
    parser.add_argument('--branch_consensus_weight', default=0.25, type=float,
                        help='Weight of RGB/SN consensus term when branch_fusion_mode=consensus.')
    parser.add_argument('--mnc_cross_weight', default=0.5, type=float)
    parser.add_argument('--mnc_stages', default=2, type=int)
    parser.add_argument('--mnc_strong', default=False, action='store_true')
    parser.add_argument('--mnc_temperature', default=0.05, type=float)
    parser.add_argument('--mnc_stage1_weight', default=0.5, type=float)
    parser.add_argument('--mnc_stage2_weight', default=0.5, type=float)
    parser.add_argument('--mnc_learnable', default=False, action='store_true',
                        help='Train and use learnable gated cross-attention MNC with a feature reconstruction decoder.')
    parser.add_argument('--mnc_checkpoint_path', default='', type=str,
                        help='Optional checkpoint for the learnable MNC/decoder.')
    parser.add_argument('--mnc_save_path', default='', type=str,
                        help='Optional save path for the learnable MNC/decoder checkpoint.')
    parser.add_argument('--mnc_epochs', default=20, type=int)
    parser.add_argument('--mnc_batch_size', default=2048, type=int)
    parser.add_argument('--mnc_max_train_tokens', default=50000, type=int)
    parser.add_argument('--mnc_lr', default=1e-4, type=float)
    parser.add_argument('--mnc_weight_decay', default=1e-4, type=float)
    parser.add_argument('--mnc_num_heads', default=8, type=int)
    parser.add_argument('--mnc_decoder_hidden_ratio', default=2.0, type=float)
    parser.add_argument('--mnc_dropout', default=0.0, type=float)
    parser.add_argument('--mnc_train_pseudo', default=False, action='store_true',
                        help='Also train learnable MNC on CMPT pseudo RGB/SN tokens for missing-modality routes.')
    parser.add_argument('--mnc_pseudo_loss_weight', default=0.5, type=float)
    parser.add_argument('--paper_mnc', default=False, action='store_true',
                        help='Use the explicit two-stage PIRN-style MNC approximation for paper-system runs.')
    parser.add_argument('--disable_cmpt', default=False, action='store_true',
                        help='Ablation: disable CMPT training and pseudo-modality scoring.')
    parser.add_argument('--disable_shared_proto', default=False, action='store_true',
                        help='Ablation: remove shared prototypes from all prototype banks.')
    parser.add_argument('--only_shared_proto', default=False, action='store_true',
                        help='Ablation: use only shared prototypes and remove modality-specific prototype banks at scoring time.')
    parser.add_argument('--disable_prototypes', default=False, action='store_true',
                        help='Ablation: remove structured prototypes and reconstruct by direct normal token memory matching.')
    parser.add_argument('--no_proto_memory_tokens', default=4096, type=int,
                        help='Maximum normal training tokens per modality for --disable_prototypes.')
    parser.add_argument('--disable_apr', default=False, action='store_true',
                        help='Ablation: use raw k-means prototypes without APR refinement/update.')
    parser.add_argument('--disable_mnc', default=False, action='store_true',
                        help='Ablation: disable cross-modal normality communication after SPA.')
    parser.add_argument('--disable_pseudo_proto', default=False, action='store_true',
                        help='Ablation: skip pseudo prototype/feature completion in missing-modality routes.')
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str)
    parser.add_argument('--fusion_module_path', default='', type=str)

    parser.add_argument('--save_preds', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=1024, type=int,
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    parser.add_argument('--dataset_type', default='mvtec3d', type=str, choices=['mvtec3d', 'eyecandies'],
                        help='Dataset type for training or testing')
    parser.add_argument('--dataset_path', default='datasets/mvtec_3d', type=str,
                        help='Dataset store path')
    parser.add_argument('--classes', default='', type=str,
                        help='Comma-separated class names to run, e.g. rope or bagel,rope. '
                             'If --dataset_path points to a class directory, it is inferred automatically.')
    parser.add_argument('--xyz_s_lambda', default=1.0, type=float,
                        help='xyz_s_lambda')
    parser.add_argument('--xyz_smap_lambda', default=1.0, type=float,
                        help='xyz_smap_lambda')
    parser.add_argument('--rgb_s_lambda', default=0.1, type=float,
                        help='rgb_s_lambda')
    parser.add_argument('--rgb_smap_lambda', default=0.1, type=float,
                        help='rgb_smap_lambda')
    parser.add_argument('--sn_s_lambda', default=1.0, type=float,
                        help='sn_s_lambda')
    parser.add_argument('--sn_smap_lambda', default=1.0, type=float,
                        help='sn_smap_lambda')
    parser.add_argument('--cmpt_s_lambda', default=1.0, type=float,
                        help='cmpt_s_lambda')
    parser.add_argument('--cmpt_smap_lambda', default=1.0, type=float,
                        help='cmpt_smap_lambda')
    parser.add_argument('--fusion_s_lambda', default=1.0, type=float,
                        help='fusion_s_lambda')
    parser.add_argument('--fusion_smap_lambda', default=1.0, type=float,
                        help='fusion_smap_lambda')
    parser.add_argument('--share_s_lambda', default=1.0, type=float,
                        help='share_s_lambda')
    parser.add_argument('--share_smap_lambda', default=1.0, type=float,
                        help='non_share_smap_lambda')
    parser.add_argument('--non_share_s_lambda', default=1.0, type=float,
                        help='share_s_lambda')
    parser.add_argument('--non_share_smap_lambda', default=1.0, type=float,
                        help='non_share_smap_lambda')

    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--asy_memory_bank', default=None, type=int,
                        help='build an asymmetric memory bank for point clouds')
    parser.add_argument('--ocsvm_nu', default=0.5, type=float,
                        help='ocsvm nu')
    parser.add_argument('--ocsvm_maxiter', default=1000, type=int,
                        help='ocsvm maxiter')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')

    parser.add_argument('--img_process_method', default='cpu_v1', type=str)
    parser.add_argument('--cpu_core_num', default=6, type=int)
    parser.add_argument('--experiment_note', default='', type=str)
    parser.add_argument('--coreset_dtype', default='FP16', type=str)
    parser.add_argument('--similarity_only', default=False, action='store_true')
    parser.add_argument('--difference_only', default=False, action='store_true')
    parser.add_argument('--concat_only', default=False, action='store_true')
    parser.add_argument('--need_detection_head', default=False, type=bool)
    parser.add_argument('--train_with_validation', default=False, action='store_true')
    parser.add_argument('--dist_method_s', default='l2', type=str, choices=['l1', 'l2', 'cos_dist'])
    parser.add_argument('--dist_method_coreset', default='l2', type=str, choices=['l1', 'l2', 'cos_dist'])
    parser.add_argument('--main_modality', default='rgb', type=str, choices=['', 'rgb', 'sn', 'xyz'])

    parser.add_argument('--use_hn', default=False, action='store_true')
    parser.add_argument('--use_hn_conv', default=False, action='store_true')
    parser.add_argument('--use_hn_from_rgb_mlp', default=False, action='store_true')
    parser.add_argument('--use_hn_from_rgb_conv', default=False, action='store_true')
    parser.add_argument('--use_depth', default=False, action='store_true')
    parser.add_argument('--use_hrnet', default=False, action='store_true')
    parser.add_argument('--use_uff', default=False, action='store_true')

    parser.add_argument('--with_norm', default=True, type=bool)
    parser.add_argument('--rgb_size', default=518, type=int,
                        help='Images size for model')
    parser.add_argument('--xyz_size', default=224, type=int,
                        help='XYZ size for model')
    parser.add_argument('--gt_size', default=224, type=int,
                        help='gt size for model')
    parser.add_argument('--sn_mean', nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--sn_std', nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--sn_background', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help='RGB value used for invalid/background pixels in surface-normal maps.')
    parser.add_argument('--sn_resize_mode', default='nearest', type=str,
                        choices=['nearest', 'bilinear', 'bicubic'],
                        help='Resize mode for surface-normal maps before DINOv2 encoding.')
    parser.add_argument('--sn_smooth_kernel', default=1, type=int,
                        help='Average smoothing kernel for surface-normal maps; set 1 to disable.')
    parser.add_argument('--sn_foreground_only', default=False, type=bool,
                        help='Use only valid foreground SN tokens when fitting prototypes and CMPT.')
    parser.add_argument('--sn_foreground_threshold', default=0.5, type=float,
                        help='Foreground threshold for downsampled valid-point masks.')
    parser.add_argument('--sn_mask_error_map', default=False, type=bool,
                        help='Mask SN-available error maps with the valid 3D foreground mask.')

    parser.add_argument('--save_feature_for_fusion', default=False, action='store_true')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_path_frgb_xyz', type=str)
    parser.add_argument('--save_path_rgb_fxyz', type=str)
    parser.add_argument('--save_frgb_xyz', default=False, action='store_true')
    parser.add_argument('--save_rgb_fxyz', default=False, action='store_true')
    parser.add_argument('--save_results', default=True, type=bool)
    parser.add_argument('--c_hrnet', default=48, type=int)

    parser.add_argument('--save_raw_results', default=False, action='store_true')
    parser.add_argument('--save_seg_results', default=False, action='store_true')
    parser.add_argument('--save_component_maps', default=False, action='store_true',
                        help='With PIRN_CMPT, save branch-level RGB/SN/pseudo error maps besides fused maps.')
    parser.add_argument('--save_heatmaps', default=False, action='store_true',
                        help='Save RGB, GT, anomaly heatmap and overlay PNGs for test samples.')
    parser.add_argument('--save_heatmap_num', default=20, type=int,
                        help='Maximum number of test samples to visualize per class/method run.')
    parser.add_argument('--heatmap_dir', default='visualization', type=str,
                        help='Root directory for saved heatmap visualizations.')

    args = parser.parse_args()
    if args.paper_mnc and not args.disable_apr and args.apr_memory_update_iters == 0:
        args.apr_memory_update_iters = 1
    if args.paper_mnc and not args.disable_mnc:
        args.mnc_strong = True
        args.mnc_stages = max(2, args.mnc_stages)
        if not args.classic_pirn_mnc1:
            args.cmpt_replace_mnc1 = True
    elif args.paper_mnc and args.disable_mnc and not args.disable_cmpt and not args.classic_pirn_mnc1:
        args.cmpt_replace_mnc1 = True
    if args.disable_shared_proto and args.only_shared_proto:
        raise ValueError('--disable_shared_proto and --only_shared_proto cannot be used together.')
    if args.disable_prototypes:
        args.disable_mnc = True
        args.mnc_strong = False
        args.mnc_learnable = False
        args.mnc_stages = 0
    if args.disable_mnc:
        args.mnc_strong = False
        args.mnc_learnable = False
        args.mnc_stages = 0
    if args.few_shot_k > 0:
        args.max_sample = max(args.max_sample, args.few_shot_k)
    cpu_num = args.cpu_core_num
    set_multithreading(cpu_num)

    run_3d_ads(args)
