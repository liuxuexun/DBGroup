import os
import torch
import numpy as np
from utils.utils import num_to_natural
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from utils.ap_tool.evaluate_semantic_instance import evaluate
import numpy as np

def fuse_instance_labels(points, labels_a, labels_b, ignore=1000):
    """
    融合两种不同粒度的实例分割标签
    
    参数:
    points: np.array (N,3) - 点云坐标
    labels_a: np.array (N,) - 粗粒度实例标签
    labels_b: np.array (N,) - 细粒度实例标签
    
    返回:
    np.array (N,) - 融合后的实例标签
    """
    
    def compute_overlap_matrix():
        """计算标签A和标签B之间的重叠矩阵"""
        unique_a = np.unique(labels_a)
        unique_b = np.unique(labels_b)
        overlap = np.zeros((len(unique_a), len(unique_b)))
        
        for i, label_a in enumerate(unique_a):
            if label_a == ignore:
                continue
            mask_a = labels_a == label_a
            for j, label_b in enumerate(unique_b):
                if label_b == ignore:
                    continue
                mask_b = labels_b == label_b
                overlap[i,j] = np.sum(mask_a & mask_b)
                
        return overlap, unique_a, unique_b
    
    def compute_spatial_connectivity(points, labels):
        """计算实例间的空间连通性"""
        # 构建KD树用于近邻搜索
        tree = KDTree(points)
        unique_labels = np.unique(labels)
        connectivity = defaultdict(set)
        
        # 为每个实例找到质心
        centroids = {}
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            centroids[label] = np.mean(points[mask], axis=0)
        
        # 计算实例间的连通性
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            label_points = points[mask]
            
            # 查找邻近点
            # neighbors = tree.query_radius(label_points, r=0.03)  # 根据点云尺度调整半径
            neighbors = tree.query(label_points, k=8)[1]           # 这个速度快点
            for n_idx in neighbors:
                neighbor_labels = np.unique(labels[n_idx])
                for n_label in neighbor_labels:
                    if n_label == -1:
                        continue
                    if n_label != label:
                        connectivity[label].add(n_label)
        
        return connectivity, centroids
    
    def merge_small_instances(labels, points, min_points=200):
        """合并过小的实例到相邻的大实例中"""
        unique_labels = np.unique(labels)
        connectivity, centroids = compute_spatial_connectivity(points, labels)
        
        # 计算每个实例的点数
        instance_sizes = {label: np.sum(labels == label) for label in unique_labels}
        
        # 标记需要合并的小实例
        small_instances = {label for label, size in instance_sizes.items() 
                         if size < min_points}
        
        new_labels = labels.copy()
        
        # 合并小实例
        for small_label in small_instances:
            if small_label == -1:
                continue
            if small_label not in connectivity:
                continue
                
            # 找到最近的大实例
            neighbors = connectivity[small_label]
            if not neighbors:
                continue
                
            best_neighbor = None
            min_dist = float('inf')
            
            for neighbor in neighbors:
                if neighbor in small_instances:
                    continue
                    
                dist = np.linalg.norm(
                    centroids[small_label] - centroids[neighbor]
                )
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
            
            if best_neighbor is not None:
                new_labels[labels == small_label] = best_neighbor
                
        return new_labels
    
    # 1. 计算重叠矩阵
    # import pdb;pdb.set_trace()
    overlap_matrix, unique_a, unique_b = compute_overlap_matrix()
    
    # 2. 对于每个粗糙的A标签，判断是否需要细分
    new_labels = np.ones_like(labels_a) * -1
    try:
        next_label = max(np.sort(np.unique(labels_a))[-2], np.sort(np.unique(labels_b))[-2]) + 1
    except:
        next_label = np.sort(np.unique(labels_b))[-2] + 1
    
    for i, label_a in enumerate(unique_a):
        if label_a == ignore:
            continue
        mask_a = labels_a == label_a
        overlapping_b = overlap_matrix[i]
        
        # 如果A标签主要与一个B标签重叠，保持原样
        if np.sum(overlapping_b) != 0 and np.max(overlapping_b) / np.sum(overlapping_b) > 0.4:
            new_labels[mask_a] = label_a
            continue
            
        # 否则，采用B的细分结果
        # import pdb;pdb.set_trace()
        significant_b = unique_b[overlapping_b > 0]  # 这个参数是 当A要拆分成多个B中的集合时，要过滤掉B中哪些cluster（现在是过滤掉点数少于100的）
        for label_b in significant_b:
            mask_b = labels_b == label_b
            combined_mask = mask_a & mask_b
            if np.sum(combined_mask) > 0:
                new_labels[combined_mask] = next_label
                next_label += 1
    
    # 3. 合并过小的实例
    final_labels = merge_small_instances(new_labels, points, 400)
    
    return final_labels

if __name__ == '__main__':

    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]    
    coarse_pred_path = './data/coarseinslabel'
    fine_pred_path = './data/fineinslabel'
    pc_path = './data/scannet_3d/train'
    name_path = './dataset/scannet/scannetv2_train.txt'
    save_path = './data/InsPseuLabel'
    gt_path = './data/gt_scannetv2'
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scene_name = [line.rstrip() for line in open(name_path)]
    matches = {}
    for scene_name in tqdm(scene_name):
        
        xyz = torch.load(os.path.join(pc_path, f"{scene_name}.pth"))[0]
        coarse = np.load(os.path.join(coarse_pred_path, scene_name + '.npy'))
        fine = np.load(os.path.join(fine_pred_path, scene_name + '.npy'))

        coarse = num_to_natural(coarse).copy()      # num_to_natural 在这里没啥用，因为本来我pg cluster保存的时候ins id就是按顺序保存的
        coarse[coarse==-1] = 1000
        fine[fine==-2] = -1
        fine[fine==-1] = 1000

        new_group = fuse_instance_labels(xyz, coarse, fine, ignore=1000)
        np.save(os.path.join(save_path, scene_name + '.npy'), new_group)
        
        # eval
        scene_dic = {}
        proposal_list = []
        # 组织成proposal的形式
        # import pdb;pdb.set_trace()
        for ins_id in np.unique(new_group):
            if ins_id <= -1:
                continue
            # 获得当前实例的mask
            mask = new_group == ins_id
            proposal_list.append(np.expand_dims(mask,-1))
        if proposal_list == []:
            proposal_arr = np.ones([new_group.shape[0], 1])
            scene_dic['pred_masks'] = proposal_arr
            scene_dic['pred_scores'] = np.ones([proposal_arr.shape[1]])
            scene_dic['pred_classes'] = np.ones([proposal_arr.shape[1]]) * 4
        else:
            proposal_arr = np.concatenate(proposal_list, axis=-1)
            sem_arr = np.ones([proposal_arr.shape[1]]) * 4
            score_arr = np.ones([proposal_arr.shape[1]])
            scene_dic['pred_masks'] = proposal_arr
            scene_dic['pred_scores'] = score_arr
            scene_dic['pred_classes'] = sem_arr
        
        
        matches[scene_name] = scene_dic
    labelset_name = 'scannet'
    evaluate(matches, gt_path, save_path, dataset=labelset_name)
    