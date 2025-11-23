import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import cv2
import argparse
import torch
import numpy as np
import open3d as o3d
# import pointops
from utils.main_utils import *
from utils.sam_utils import *
from segment_anything import sam_model_registry, SamPredictor
from tqdm import trange
from tqdm import tqdm

# from natsort import natsorted 
from utils.vis_utils import *
from scipy.sparse import coo_matrix


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
    try:
        label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    except:
        import pdb;pdb.set_trace()
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)  # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = torch.Tensor(label_map.max(1) / label_map.sum(axis=1)).to(labels.device) # [num_superpoint, num_label + 1]

    return label, label_scores


def prompt_filter(init_prompt, data_original_list, frame_id_list, predictor, args):
    device = torch.device(args.device)
    # gap = 1  # number of skipped frames
    stop_limit = 10  # we find that not considering all frames for filter is better

    keep_score = torch.zeros(len(init_prompt), device=device)
    counter = torch.zeros(len(init_prompt), device=device)
    del_score = torch.zeros(len(init_prompt), device=device)

    for i, (data_original) in enumerate(tqdm(data_original_list)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(data_original['points']).to(device)
        iou_preds_data = torch.from_numpy(data_original['iou_preds']).to(device)
        masks_data = torch.from_numpy(data_original['masks']).to(device)
        corre_3d_ins_data = torch.from_numpy(data_original['corre_3d_ins']).to(device)  # the valid (i.e., has mapped pixels at the current frame) prompt ID  in the original 3D point cloud of initial prompts
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data 
            )

        corr_ins_idx = data['corre_3d_ins']
        # ins_flag[corr_ins_idx] = 1 # set the valid ins value to 1
        counter[corr_ins_idx] += 1  # only count if it is not the stopped instances
        stop_id = torch.where(counter >= stop_limit)[0]

        ############ start filter:
        # Filter by predicted IoU
        if args.pred_iou_thres > 0.0:
            keep_mask = data["iou_preds"] > args.pred_iou_thres
            data.filter(keep_mask)
        #     print(data['points'].shape)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            masks=data["masks"], mask_threshold=predictor.model.mask_threshold, threshold_offset=1.0
        )

        if args.stability_score_thres > 0.0:
            keep_mask = data["stability_score"] >= args.stability_score_thres
            data.filter(keep_mask)
    #     print(data['points'].shape)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        
        # Remove duplicates within this crop.
        from torchvision.ops.boxes import batched_nms, box_area 
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=args.box_nms_thres,
        )
        data.filter(keep_by_nms)

        keep_ins_idx = data["corre_3d_ins"]
        del_ins_idx = corr_ins_idx[torch.isin(corr_ins_idx, keep_ins_idx, invert=True)]

        if stop_id.shape[0] > 0:
            keep_ins_idx = keep_ins_idx[torch.isin(keep_ins_idx, stop_id, invert=True)]
            del_ins_idx = del_ins_idx[torch.isin(del_ins_idx, stop_id, invert=True)]
        keep_score[keep_ins_idx] += 1
        del_score[del_ins_idx] += 1 

    # make all selected frames happy:
    counter[torch.where(counter >= stop_limit)] = stop_limit
    counter[torch.where(counter == 0)] = -1  #  avoid that the the score is divided by counter of 0
    # keep prompts whose score is larger than a threshold:
    keep_score_mean = keep_score / counter
    keep_idx = torch.where(keep_score_mean >= args.keep_thres)[0]

    print("the number of prompts after filter", keep_idx.shape[0])

    return keep_idx


def perform_3dsegmentation(xyz, keep_idx, data_original_list, frame_id_list, args):
    device = torch.device(args.device)
    # gap = 1  # number of skipped frames
    n_points = xyz.shape[0]
    num_ins = keep_idx.shape[0]
    pt_score = torch.zeros([n_points, num_ins], device=device)  # All input points have a score
    counter_final = torch.zeros([n_points, num_ins], device=device)

    for i, (data_original) in enumerate(tqdm(data_original_list)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(data_original['points']).to(device)
        iou_preds_data = torch.from_numpy(data_original['iou_preds']).to(device)
        masks_data = torch.from_numpy(data_original['masks']).to(device)
        corre_3d_ins_data = torch.from_numpy(data_original['corre_3d_ins']).to(device)   
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data)

        frame_id = frame_id_list[i]

        # calculate the 3d-2d mapping on ALL input points (not just prompt)
        mapping = compute_mapping(xyz, args.data_path, args.scene_name, str(frame_id))
        if mapping[:, 2].sum() == 0: # no points corresponds to this image, skip
            continue
        mapping = torch.from_numpy(mapping).to(device)

        keep_mask = torch.isin(data["corre_3d_ins"], keep_idx)  # only keep the mask that has been kept during previous prompt filter process
        data.filter(keep_mask)

        masks_logits = data["masks"]
        masks = masks_logits > 0.

        ins_idx_all = []
        for actual_idx in data["corre_3d_ins"]:  # the actual prompt ID in the original 3D point cloud of initial prompts, \
            # for calculating pt_score later (since pt_score is considered on all initial prompts)
            ins_idx = torch.where(keep_idx == actual_idx)[0]
            ins_idx_all.append(ins_idx.item())
        
        # import pdb;pdb.set_trace()
        # when both a point i and a prompt j is found in this frame, counter[i, j] + 1
        counter_point = mapping[:, 2]   # the found points
        counter_point = counter_point.reshape(-1, 1).repeat(1, num_ins)
        counter_ins = torch.zeros(num_ins, device=device)              
        counter_ins[ins_idx_all] += 1   # the found prompts                 
        counter_ins = counter_ins.reshape(1, -1).repeat(n_points, 1)
        counter_final += (counter_point * counter_ins)

        # caculate the score on mask area:
        for index, (mask) in enumerate(masks):  # iterate over each mask area segmented by different prompts
            ins_id = ins_idx_all[index]  # get the actual instance id  # ins_idx_al
            mask = mask.int()
        
            mask_2d_3d = mask[mapping[:, 0], mapping[:, 1]]
            mask_2d_3d = mask_2d_3d * mapping[:, 2]  # set the score to 0 if no mapping is found
            
            pt_score[:, ins_id] += mask_2d_3d  # For each individual input point in the scene, \
            # if it is projected within the mask area segmented by a prompt k at current frame, we assign its prediction as the prompt ID k
    
    # import pdb;pdb.set_trace()
    pt_score_cpu = pt_score.cpu().numpy()
    counter_final_cpu = counter_final.cpu().numpy()
    counter_final_cpu[np.where(counter_final_cpu==0)] = -1  # avoid divided by zero

    pt_score_mean = pt_score_cpu / counter_final_cpu  # mean score denotes the probability of a point assigned to a specified prompt ID, and is only used for later thresholding
    pt_score_abs = pt_score_cpu
    max_score = np.max(pt_score_mean, axis=-1)  # the actual scores that has been segmented into one instance
    max_score_abs = np.max(pt_score_abs, axis=-1)

    # if pt_score_mean has the max value on more than one instance，we use the instance with higher pt_score as the pred
    max_indices_mean = np.where(pt_score_mean == max_score[:, np.newaxis])
    pt_score_mean_new = pt_score_mean.copy()   # only for calculate label, merge will still use pt_score_mean
    pt_score_mean_new[max_indices_mean] += pt_score_cpu[max_indices_mean]
    pt_pred_mean = np.argmax(pt_score_mean_new, axis=-1) # the ins index

    pt_pred_abs = np.argmax(pt_score_abs, axis=-1)

    low_pt_idx_mean = np.where(max_score <= 0.)[0]  # assign ins_label=-1 (unlabelled) if its score=0 (i.e., no 2D mask assigned)
    pt_score_mean[low_pt_idx_mean] = 0.
    pt_pred_mean[low_pt_idx_mean] = -1

    low_pt_idx_abs = np.where(max_score_abs <= 0.)[0]
    pt_score_abs[low_pt_idx_abs] = 0.
    pt_pred_abs[low_pt_idx_abs] = -1

    return pt_score_abs, pt_pred_abs, pt_score_mean


def prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean):
    pt_pred_final = pt_pred_abs.copy()
    # for each segmentated space, we first use DBSCAN to separate noisy predictions that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_pred(xyz, pt_pred_abs.copy(), pt_score_abs.copy())
    pt_score_mean_ori = pt_score_mean.copy()
    pt_score_merge_ori = pt_score_merge.copy()

    # for each segmentated space, we again use DBSCAN to separate noisy score-level predictions (indicating a point has been segmented to a label at one frame) \
    # that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_score(xyz, pt_score_mean_ori, pt_score_merge_ori)

    # only regard "confident" (label probability > 0.5) points as valid points belonging to an instance (or prompt) for consolidation:
    valid_thres = 0.5
    ins_areas = []
    ins_ids = []
    ins_score_mean = pt_score_mean.T
    ins_score = pt_score_merge.T
    for ins_id in range(ins_score.shape[0]):
        ins_area_mean = np.where(ins_score_mean[ins_id] >= valid_thres)[0]  # mean_score (probability) is only for thresholding more easily
        ins_area_abs = np.where(ins_score[ins_id] > 0)[0]
        ins_area = ins_area_abs[np.isin(ins_area_abs, ins_area_mean)]
        if ins_area.shape[0] > 0:
            ins_areas.append(ins_area)  # append the valid point idx of each instance/prompt
            ins_ids.append(ins_id)

    # import pdb;pdb.set_trace()  # pt_score_abs.shape[0]==376 len(ins_ids)==359
    
    inter_all = []  # the intersection list to denote which prompts are segmenting the same 3D object
    for i in range(len(ins_areas)):
        inter_ins = [ins_ids[i]]
        for j in range(i+1, len(ins_areas)):
            inter = np.intersect1d(ins_areas[i], ins_areas[j])
            inter_ratio = inter.shape[0] / ins_areas[i].shape[0]
            if inter_ratio > 0.1:  # consider i and j are segmenting the same 3D object if have a certain overlap \
                # and append together in a sublist which are started from i:
                inter_ins.append(ins_ids[j])
            inter_all.append(inter_ins)
            
    # import pdb;pdb.set_trace()
    
    consolidated_list = merge_common_values(inter_all)  # consolidate all prompts (i, j, k, ...) that are segmenting the same 3D object
    print("number of instances after Prompt Consolidation", len(consolidated_list))
        
    # Consolidate the result:
    for sublist in consolidated_list:
        for consolidate_id in sublist:
            mask = np.isin(pt_pred_final, sublist)
            pt_pred_final[mask] = sublist[0]  # regard the first prompt id as the pseudo prompt id

    return pt_pred_final, pt_score_merge, pt_score_mean


def process_batch(
    predictor,
    points: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    transformed_points = predictor.transform.apply_coords_torch(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    masks, iou_preds, _ = predictor.predict_torch(
        in_points[:, None, :],
        in_labels[:, None],
        multimask_output=False,
        return_logits=True,
    )
    
    # Serialize predictions and store in MaskData  
    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points, 
        corre_3d_ins=ins_idxs 
    )

    return data_original
    

def sam_seg(predictor, frame_id_init, frame_id_end, init_prompt, args):
    data_original_list = []
    frame_id_list = []
    for i in trange(frame_id_init, frame_id_end, 1):
        frame_id = i * 20
        image = cv2.imread(os.path.join(args.data_path, args.scene_name, 'color', str(frame_id) + '.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load the intrinsic
        depth_intrinsic = torch.tensor(np.loadtxt(os.path.join(args.data_path, 'intrinsics.txt')), dtype=torch.float64).to(device=predictor.device)
        # Load the depth, and pose
        depth = cv2.imread(os.path.join(args.data_path, args.scene_name, 'depth', str(frame_id) + '.png'), -1) # read 16bit grayscale 
        depth = torch.from_numpy(depth.astype(np.float64)).to(device=predictor.device)
        pose = torch.tensor(np.loadtxt(os.path.join(args.data_path, args.scene_name, 'pose', str(frame_id) + '.txt')), dtype=torch.float64).to(device=predictor.device)
        
        if str(pose[0, 0].item()) == '-inf': # skip frame with '-inf' pose
            print(f'skip frame {frame_id}')
            continue

        # 3D-2D projection
        input_point_pos, corre_ins_idx = transform_pt_depth_scannet_torch(init_prompt, depth_intrinsic, depth, pose, predictor.device)  # [valid, 2], [valid]
        if input_point_pos.shape[0] == 0 or input_point_pos.shape[1] == 0:
            print(f'skip frame {frame_id}')
            continue

        image_size = image.shape[:2]
        predictor.set_image(image)
        # SAM segmetaion on image
        data_original = MaskData()
        for (points, ins_idxs) in batch_iterator(64, input_point_pos, corre_ins_idx):
            batch_data_original = process_batch(predictor, points, ins_idxs, image_size)
            data_original.cat(batch_data_original)
            del batch_data_original
        predictor.reset_image()
        data_original.to_numpy()
        
        frame_id_list.append(frame_id)
        data_original_list.append(data_original)
    return data_original_list, frame_id_list

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3d prompt proposal on ScanNet.")
    # for voxelization to decide the number of fps-sampled points:
    parser.add_argument('--voxel_size', default=0.2, type=float, help='Size of voxels.')
    # path arguments:
    parser.add_argument('--data_path', default="./data/scannet_2d", type=str, help='Path to the dataset containing ScanNet 2d frames and 3d .ply files.')
    parser.add_argument('--pred_path', default="./data/fineinslabel", type=str, help='Path to save the predicted per-point segmentation.')

    parser.add_argument('--model_type', default="vit_h", type=str, help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument('--sam_checkpoint', default="./checkpoint/sam_vit_h_4b8939.pth", type=str, help='The path to the SAM checkpoint to use for mask generation.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")
    # arguments for prompt filter:
    parser.add_argument('--pred_iou_thres', type=float, default=0.7, help='Exclude masks with a predicted score from the model that is lower than this threshold.')
    parser.add_argument('--stability_score_thres', type=float, default=0.6, help='Exclude masks with a stability score lower than this threshold.')
    parser.add_argument('--box_nms_thres', type=float, default=0.8, help='The overlap threshold for excluding a duplicate mask.')
    parser.add_argument('--keep_thres', type=float, default=0.4, help='The keeping threshold for keeping a prompt.')
    
    
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    # Initialize SAM:
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    
    
    # Load all 3D points of the input scene:
    
    scene_list = sorted([line.rstrip() for line in open('./dataset/scannet/scannetv2_train.txt')])
    
    for scene_name in tqdm(scene_list): # [0 500 1000 1500]
        
        args.scene_name = scene_name
        
        if os.path.exists(os.path.join(args.pred_path, scene_name+'.npy')):
            print(f'{scene_name} exists')
            continue
        
        data = torch.load(f'./data/scannet_3d/train/{scene_name}.pth')
        xyz, rgb = data[0], data[1]
        sp_label = data[-1]
        
        init_prompt_list = []
        init_color_list = []
        sp_id_list = []
        for ins_id in np.unique(sp_label):
            if ins_id == -100:
                continue
            index = sp_label == ins_id
            ins_points = xyz[index]
            ins_color = rgb[index]
            ins_points_mean = np.mean(ins_points, axis=0)
            # 每个超点中选一个作为init prompt
            closest_index = np.argsort(np.sum((ins_points - ins_points_mean) * (ins_points - ins_points_mean), axis=1))[0]
            init_prompt_list.append(ins_points[closest_index])
            init_color_list.append(ins_color[closest_index])
            sp_id_list.append(ins_id)
        init_prompt = np.vstack(init_prompt_list)
        init_color = np.vstack(init_color_list)
        init_sp_id = np.vstack(sp_id_list)
        print("init_prompt num:", init_prompt.shape[0])
        
        frame_id_init = 0
        frame_id_end = len(os.listdir(os.path.join(args.data_path, scene_name, 'depth'))) 
        # You can define frame_id_init and frame_id_end by yourself for segmenting partial point clouds from limited frames. Sometimes partial result is better!
        print("Start performing SAM segmentations on {} 2D frames...".format(frame_id_end))
        data_original_list, frame_id_list = sam_seg(predictor, frame_id_init, frame_id_end, torch.from_numpy(init_prompt).cuda(), args)
        print("the number of initial prompts", init_prompt.shape[0])
        keep_idx = prompt_filter(init_prompt, data_original_list, frame_id_list, predictor, args)
        pt_score_abs, pt_pred_abs, pt_score_mean = perform_3dsegmentation(xyz, keep_idx, data_original_list, frame_id_list, args) 
        pt_pred, pt_score_abs, pt_score_mean = prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean)

        
        pt_pred = num_to_natural(pt_pred)
        pt_pred[pt_pred==-1] = -100
        if np.unique(pt_pred)[0] == -100:
            nproposal = np.unique(pt_pred).shape[0] - 1
        else:
            nproposal = np.unique(pt_pred).shape[0]
        new_group, _ = align_superpoint_label(torch.from_numpy(pt_pred), torch.from_numpy(sp_label), nproposal)
        pt_pred = new_group[sp_label].numpy()

        pt_pred[pt_pred==-100] = -1
        pt_pred = num_to_natural(pt_pred)
        create_folder(args.pred_path)
        
        # save the prediction result:
        pred_file = os.path.join(args.pred_path, args.scene_name + '.npy')
        np.save(pred_file, pt_pred)
        print(f'{scene_name} finish')