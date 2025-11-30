import torch
from lib.pointgroup_ops.functions import pointgroup_ops
import glob
import os
import numpy as np
from scipy.sparse import coo_matrix

def scene_name_clean(path):
    if '_vh' in path:
        return path.split('/')[-1].split('_vh')[0]
    return os.path.splitext(os.path.basename(path))[0]

def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

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


def getInstLabel(instance_label):
    j = 0
    while (j < instance_label.max()):
        if (len(np.where(instance_label == j)[0]) == 0):
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label

def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def main():
    
    pc_path = './data/scannet_3d/train'
    opensc_logit_path = './data/logit'
    result_dir = './data/coarseinslabel'
    
    os.makedirs(result_dir, exist_ok=True)
    
    test_file_names = sorted(glob.glob(os.path.join(pc_path, '*.pth')))
    
    
    for i, file_path in enumerate(test_file_names):
    
        scene_name = scene_name_clean(file_path)
        
        scene_data = torch.load(file_path)
        opensc_logit = torch.from_numpy(np.load(os.path.join(opensc_logit_path, scene_name + '_fusion.npy'))).cuda()

        # extract data
        opensc_sem_preds = opensc_logit.max(1)[1]
        coords = torch.from_numpy(scene_data[0]).cuda()
        batch_idxs = torch.from_numpy(np.zeros(coords.shape[0])).cuda()
        
        # # superpoint smooth
        sp = scene_data[-1]
        sp_label, _ = align_superpoint_label(opensc_sem_preds, torch.from_numpy(sp).cuda(), num_label=20, ignore_label=-100)
        opensc_sem_preds = sp_label[torch.from_numpy(sp).cuda()]

        # delete uncare class
        object_idxs = torch.nonzero(opensc_sem_preds > 1).view(-1)        # 这里应该是把地板和墙壁筛掉
        batch_idxs_ = batch_idxs[object_idxs]
        batch_offsets_ = get_batch_offsets(batch_idxs_, 1)
        coords_ = coords[object_idxs]
        semantic_preds_cpu = opensc_sem_preds[object_idxs].int().cpu()

        # config
        cluster_radius = 0.04; cluster_meanActive = 50; cluster_npoint_thre = 50
        
        # begin grouping
        if coords_.shape[0] == 0:
            print("no instance")
            opensc_sem_preds += 2
            object_idxs = torch.nonzero(opensc_sem_preds > 1).view(-1)
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = get_batch_offsets(batch_idxs_, 1)
            coords_ = coords[object_idxs]
            semantic_preds_cpu = opensc_sem_preds[object_idxs].int().cpu()
        idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_.int(), batch_offsets_.int(), cluster_radius, cluster_meanActive)
        proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), cluster_npoint_thre)
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
        
        # for scnen0000, the distence of points is too far
        if(proposals_offset.shape[0] < len(torch.unique(semantic_preds_cpu))):
            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_.int(), batch_offsets_.int(), 0.08, cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
    
        scores = torch.ones([proposals_offset.shape[0] - 1, 1])
        
        # package result
        preds = {}
        # preds['semantic'] = opensc_logit
        preds['semantic'] = opensc_sem_preds
        preds['score'] = scores
        preds['proposals'] = (proposals_idx, proposals_offset)
        

        # semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
        # semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
        semantic_pred = preds['semantic']
                
        # semantic_pred = semantic_scores.long()
        scores = preds['score']   # (nProposal, 1) float, cuda
        # scores = semantic_scores

        # scores_pred = torch.sigmoid(scores.view(-1))    
        scores_pred = scores.view(-1)   # 所有proposal的分数为1

        proposals_idx, proposals_offset = preds['proposals']
        N = coords.shape[0]
        proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores.device) # (nProposal, N), int, cuda
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        semantic_id = torch.tensor(semantic_label_idx, device=scores.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()].cpu()] # (nProposal), long
        semantic_id_20 = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]
        semantic_id = semantic_id_20 - 1
        

        #### score threshold
        TEST_SCORE_THRESH = 0.09
        TEST_NPOINT_THRESH = 100
        score_mask = (scores_pred > TEST_SCORE_THRESH)      # 过滤掉分数太低的（0.09）
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        semantic_id = semantic_id[score_mask]

        ##### npoint threshold
        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = (proposals_pointnum > TEST_NPOINT_THRESH)     # 过滤掉点数太少的（100） 这里改成300点会高一点，这个参数可以调调
        scores_pred = scores_pred[npoint_mask]
        proposals_pred = proposals_pred[npoint_mask]
        semantic_id = semantic_id[npoint_mask]

        ##### save files
        # make instance label
        ins_label = torch.ones_like(semantic_pred) * -1
        
        areas = torch.sum(proposals_pred, axis=1)
        sorted_idx = np.argsort(-areas)
        for ins_id, idx in enumerate(sorted_idx):
            current_mask = proposals_pred[idx]
            ins_label[current_mask==1] = ins_id
        
        ins_label = getInstLabel(ins_label.cpu().numpy())
        np.save(os.path.join(result_dir, scene_name + '.npy'), ins_label)
        print("instance iter: {}/{} point_num: {} ncluster: {} ".format(i + 1, len(test_file_names), N, proposals_pred.shape[0]))



if __name__ == '__main__':
    main()