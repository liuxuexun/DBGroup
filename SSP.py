import numpy as np
import glob, os
from os.path import join
import utils.iou_tool.metric as metric
import torch
from scipy.sparse import coo_matrix
import time


def align_superpoint_label(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint.cpu().numpy() # superpoint has been compression
    col = labels.cpu().numpy()
    col[col < 0] = num_label
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)  # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = torch.Tensor(label_map.max(1) / label_map.sum(axis=1)).to(labels.device) # [num_superpoint, num_label + 1]

    return label, label_scores


def superpoint_label_propagate(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint.cpu().numpy() # superpoint has been compression
    col = labels.cpu().numpy()
    newcol = col[col >= 0]
    newrow = row[col >= 0]
    data = np.ones(len(newcol))
    shape = (len(np.unique(row)), num_label)
    label_map = coo_matrix((data, (newrow, newcol)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)
    label[label_map.sum(1) == 0] = ignore_label
    
    return label


logit_path = './data/logit'
pc_path = './data/scannet_3d/train'
save_path = './data/SemPseuLabel'

if not os.path.exists(save_path):
    os.makedirs(save_path)

pseudo_label_files = sorted(glob.glob(join(logit_path, '*.npy')))

preds = []
gts = []
total_time = 0
count = 0
for i in range(len(pseudo_label_files)):
 
    # sub cloud 10%
    # 选择点的条件：1.某类置信度大 2.其他类置信度低
    scene_name = pseudo_label_files[i].split('/')[-1][:12]
    gt_label = torch.load(join(pc_path, scene_name + '.pth'))[2]
    sp = torch.load(join(pc_path, scene_name + '.pth'))[-1]
    pseudo_label_logit = np.load(pseudo_label_files[i])
    pseudo_label = pseudo_label_logit.argmax(1)
    ratio = 0.3
    
    start = time.time()
    selected_label = np.ones_like(pseudo_label) * -1
    
    # 遍历每个类
    for i in range(pseudo_label_logit.shape[1]): 
        
        this_class_mask = pseudo_label == i
        this_class_logit = pseudo_label_logit[:, i]
        this_class_score = this_class_mask * this_class_logit
        topk_index_10 = np.argsort(this_class_score)[::-1][:int(np.sum(pseudo_label==i)*ratio)]  
        selected_label[topk_index_10] = i 
    
    gt_label[gt_label==-100] = 255
    selected_index = selected_label==-1
    gt_label[selected_index] = 255
    selected_label[selected_index] = 255        # 其他没有选择的点，语义标签为255

    # superpoint ensemble
    selected_label[selected_index] = -100
    sp_label, _ = align_superpoint_label(torch.from_numpy(selected_label).cuda(), torch.from_numpy(sp).cuda(), num_label=20, ignore_label=-100)
    semantic_pred = sp_label[torch.from_numpy(sp).cuda()].cpu().numpy()
    selected_label = semantic_pred
    gt_label[selected_label==-100] = 255
    selected_label[selected_label==-100] = 255

    
    np.save(os.path.join(save_path, scene_name + '.npy'), selected_label)
    
    gts.append(torch.from_numpy(gt_label).int())
    preds.append((torch.from_numpy(selected_label).int()))



gt = torch.cat(gts)
pred = torch.cat(preds)
pred_logit = pred
labelset_name = 'scannet_3d'
current_iou = metric.evaluate(pred_logit.numpy(),
                            gt.numpy(),
                            dataset=labelset_name,
                            stdout=True)