import os
import sys
import glob
import argparse
import logging
import numpy as np
import imageio
from tqdm import tqdm, trange
from os.path import join, exists

import torch
import torch.nn.functional as F

# TensorFlow for OpenSeg model
import tensorflow as tf2
import tensorflow.compat.v1 as tf

# CLIP for text features
import clip

# Local dependencies
sys.path.append('./')
try:
    from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, adjust_intrinsic, make_intrinsic
except ImportError:
    print("错误: 找不到 'fusion_util' 模块。请确保 fusion_util.py 在当前目录。")

# ==========================================
# 常量定义
# ==========================================

SCANNET_LABELS_20 = (
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
    'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 
    'desk', 'curtain', 'refrigerator', 'shower curtain',
    'toilet', 'sink', 'bathtub', 'otherfurniture'
)

# ==========================================
# 参数设置
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline: OpenSeg Fusion -> Memory -> CLIP Logits')

    # --- 路径参数 ---
    parser.add_argument('--data_dir', default='../../data/', type=str, help='数据根目录')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.add_argument('--result_dir', default='../../data/logit/', type=str, help='结果保存路径 (.npy)')
    parser.add_argument('--openseg_model', type=str, default='../../checkpoint/openseg_exported_clip', help='OpenSeg saved_model 路径')
    
    # --- 融合与采样参数 (源自 Code 2) ---
    parser.add_argument('--process_id_range', nargs='+', default=None, help='处理范围: start,end')
    parser.add_argument('--depth_scale', type=float, default=1000.0)
    parser.add_argument('--n_split_points', type=int, default=2000000, help='点云采样上限')
    parser.add_argument('--feat_dim', type=int, default=768, help='特征维度')
    
    # --- CLIP 参数 (源自 Code 1) ---
    parser.add_argument('--clip_model_name', type=str, default="ViT-L/14@336px", help='CLIP 模型版本')
    
    args = parser.parse_args()
    return args

# ==========================================
# Part 1: CLIP 文本特征准备 (Code 1 逻辑)
# ==========================================

def get_text_embeddings(model_name):
    print(f"Loading CLIP {model_name} for text embeddings...")
    clip_model, _ = clip.load(model_name, device='cuda', jit=False)
    
    labelset = list(SCANNET_LABELS_20)
    # 按照 Code 1 的 Prompt Engineering
    prompts = []
    for label in labelset:
        if label == 'otherfurniture':
            prompts.append("other") # 对应 Code 1: labelset[-1] = 'other'
        else:
            prompts.append(f"a {label} in a scene")
    
    text = clip.tokenize(prompts).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    return text_features # [20, 768]

# ==========================================
# Part 2: 融合计算与处理主逻辑 (Code 2 + Code 1)
# ==========================================

def process_scene_pipeline(scene_id, file_path, args, mapper, text_feat):
    """
    核心流水线：
    1. 读取 3D 点云
    2. 读取 2D 图片并提取 OpenSeg 特征
    3. 融合特征 (Memory Only)
    4. 应用 save_fused_feature 的采样逻辑 (Memory Only)
    5. 计算 Logit 并应用 Code 1 的 Mask 逻辑
    6. 保存最终 .npy
    """
    
    save_path = os.path.join(args.result_dir, f'{scene_name_clean(file_path)}_fusion.npy')
    if os.path.exists(save_path):
        # print(f"{save_path} exists, skipping.")
        return

    # --- Step A: Load 3D Data ---
    try:
        data = torch.load(file_path)
        locs_in = data[0]      # [N, 3]
        point_labels = data[2] # [N]
        n_points = locs_in.shape[0]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # --- Step B: Feature Fusion (Code 2 Logic) ---
    # if you have a 32G-GPU, you can use cuda.
    device = torch.device('cpu')
    counter = torch.zeros((n_points, 1), device=device)
    sum_features = torch.zeros((n_points, args.feat_dim), device=device)
    
    scene_2d_dir = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob.glob(join(scene_2d_dir, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    
    if len(img_dirs) == 0: return
    vis_id = torch.zeros((n_points, len(img_dirs)), dtype=torch.bool, device=device)

    # 遍历图片进行融合
    for img_id, img_dir in enumerate(tqdm(img_dirs, desc=f"Fusion {scene_id}", leave=False)):
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        depthpath = img_dir.replace('color', 'depth').replace('jpg', 'png')
        if not exists(posepath) or not exists(depthpath): continue
        
        pose = np.loadtxt(posepath)
        depth = imageio.v2.imread(depthpath) / args.depth_scale
        
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: continue
        
        mapping_torch = torch.from_numpy(mapping).to(device)
        
        # OpenSeg Extract
        feat_2d = extract_openseg_img_feature(img_dir, args.openseg_model, args.text_emb, img_size=[240, 320])
        feat_2d = feat_2d.to(device) # [1, C, H, W] or [C, H, W]
        if len(feat_2d.shape) == 4: feat_2d = feat_2d.squeeze(0)
        
        # Sample Features
        # mapping: [idx, u, v, mask] -> feat_2d[:, u, v]
        feat_sampled = feat_2d[:, mapping_torch[:, 1], mapping_torch[:, 2]].permute(1, 0) # [N, C]
        
        # Accumulate
        mask = mapping[:, 3].astype(bool)
        mask_torch = torch.from_numpy(mask).to(device)
        feat_sampled_cpu = feat_sampled.to(device)
        
        counter[mask_torch] += 1
        sum_features[mask_torch] += feat_sampled_cpu[mask_torch]
        vis_id[mask_torch, img_id] = True

    # Average Features
    valid_points_mask = (counter > 0).squeeze() # [N]
    counter[counter == 0] = 1e-5
    feat_bank = sum_features / counter # [N, C] (Full Point Cloud)
    
    # Subsampling Logic
    if n_points < args.n_split_points:
        n_points_cur = n_points
    else:
        n_points_cur = args.n_split_points

    # 随机选择 n_points_cur 个点
    rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)
    
    mask_entire = torch.zeros(n_points, dtype=torch.bool)
    mask_entire[rand_ind] = True # 随机选中的点
    
    # mask_valid (对应代码中的 mask[point_ids] = True)
    # 这里复用 valid_points_mask 即可
    mask_entire = mask_entire & valid_points_mask
    
    # 最终筛选出的特征 (Code 2 保存到 .pt 的就是这部分)
    final_feats = feat_bank[mask_entire] # [M, C]
    
    if final_feats.shape[0] == 0:
        print(f"Warning: No valid features after sampling for {scene_id}")
        return

    # --- Step D: Logit Calculation (Code 1 Logic) ---
    
    # 1. 准备 Scene Labels (用于 Mask 掉无关类)
    scene_label_idx = np.unique(point_labels)
    # 过滤 ignore labels
    if len(scene_label_idx) > 0 and scene_label_idx[0] == -100: scene_label_idx = scene_label_idx[1:]
    if len(scene_label_idx) > 0 and scene_label_idx[-1] == 255: scene_label_idx = scene_label_idx[:-1]
    
    onehot_scene_label = torch.zeros(len(SCANNET_LABELS_20)).cuda()
    if len(scene_label_idx) > 0:
        onehot_scene_label[scene_label_idx.astype(np.int32)] = 1.0
        
    # 2. 计算余弦相似度 (Logits)
    # final_feats: [M, 768], text_feat: [20, 768]
    final_feats_cuda = final_feats.half().cuda() # Code 1 使用 fp16
    
    # [M, 20] = [M, C] @ [20, C].T
    pred = final_feats_cuda @ text_feat.t()
    
    # 3. 应用 Scene Filter
    pred = pred.float() * onehot_scene_label # [M, 20]
    
    # 4. Map back to Full Point Cloud (N)
    # Code 1: true_logits_pred[mask_full] = pred
    pred_cpu = pred.detach().cpu().numpy()
    
    true_logits_pred = np.zeros([n_points, pred_cpu.shape[1]], dtype=np.float32)
    true_logits_pred[mask_entire] = pred_cpu
    
    # --- Step E: Save Result ---
    np.save(save_path, true_logits_pred)
    # print(f"Saved logits for {scene_id}")

def scene_name_clean(path):
    if '_vh' in path:
        return path.split('/')[-1].split('_vh')[0]
    return os.path.splitext(os.path.basename(path))[0]

# ==========================================
# 主函数
# ==========================================

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    data_root_3d = join(args.data_dir, 'scannet_3d')
    args.data_root_2d = join(args.data_dir, 'scannet_2d')
    os.makedirs(args.result_dir, exist_ok=True)

    # 1. Load OpenSeg
    print(">>> Loading OpenSeg...")
    args.openseg_model = tf2.saved_model.load(args.openseg_model, tags=[tf.saved_model.tag_constants.SERVING])
    args.text_emb = tf.zeros([1, 1, args.feat_dim])

    # 2. Init Mapper
    img_dim = (320, 240)
    intrinsic = make_intrinsic(fx=577.870605, fy=577.870605, mx=319.5, my=239.5)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim)
    mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=intrinsic,
        visibility_threshold=0.25, cut_bound=10
    )

    # 3. Precompute Text Features
    print(">>> Computing Text Embeddings...")
    text_feat = get_text_embeddings(args.clip_model_name) # [20, 768]

    # 4. File List
    data_paths = sorted(glob.glob(join(data_root_3d, args.split, '*.pth')))
    
    id_range = [0, len(data_paths)]
    if args.process_id_range:
        id_range = [int(x) for x in args.process_id_range[0].split(',')]

    # 5. Loop
    for i in trange(len(data_paths), desc="Processing Scenes"):
        if i < id_range[0] or i > id_range[1]: continue
        
        file_path = data_paths[i]
        scene_id = scene_name_clean(file_path)
        
        process_scene_pipeline(scene_id, file_path, args, mapper, text_feat)

    print(">>> All Done.")

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)