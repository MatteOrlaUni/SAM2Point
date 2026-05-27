from PIL import Image
import numpy as np
import torch, os
import sam2point.utils as utils
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
MODELCFG = "sam2_hiera_l.yaml"
RESOLUTION = 256

def grid_to_frames(grid, foldpath, args):
    if not utils.build_fold(foldpath):
        utils.visualize_per_frame(grid, foldpath=foldpath, resolution=RESOLUTION, args=args)
    
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(foldpath)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    for i in range(len(frame_names)):
        frame_names[i] = os.path.join(foldpath, frame_names[i])
    
    return frame_names


def segment_point(frame_paths, point):
    sam2_checkpoint = CHECKPOINT
    model_cfg = MODELCFG
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(frame_paths=frame_paths)
    
    predictor.reset_state(inference_state)
    
    ann_frame_idx = 0  
    ann_obj_id = 1  

    points = np.array([point], dtype=np.float32)
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    video_segments = {}  
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    masks = []
    for out_frame_idx in range(0, len(frame_paths)):       
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = torch.from_numpy(out_mask * 1.0)
            masks.append(out_mask)
    masks = torch.cat(masks, dim=0)
    return masks
    
                            
def seg_point(locs, feats, prompt, args):
    num_voxels = locs.max().astype(int)
    grid = np.ones((num_voxels + 5, num_voxels+5, num_voxels+5, 3))
    
    locs = locs.astype(int)
    for v in range(locs.shape[0]):
        grid[locs[v][0]+2,locs[v][1]+2,locs[v][2]+2] = feats[v]
    
    X, Y, Z, _ = grid.shape
    grid = torch.from_numpy(grid)
    
    base_name = os.path.splitext(args.file_name)[0]
    name = f"./video/{base_name}"
    
    os.makedirs(name + '_frames', exist_ok=True)
    axis0, axis1, axis2 = name + "_frames/x", name + "_frames/y", name + "_frames/z"
    grid0, grid1, grid2 = grid.permute(0,3,1,2), grid.permute(1,3,0,2), grid.permute(2,3,0,1)

    a0_frame_paths = grid_to_frames(grid0, axis0, args)
    a1_frame_paths = grid_to_frames(grid1, axis1, args)
    a2_frame_paths = grid_to_frames(grid2, axis2, args)
    
    voxel_coords = np.array(prompt) / args.voxel_size + 2
    voxel_coords = voxel_coords.astype(int) 

    pixel = voxel_coords * 1.0 / X * RESOLUTION + args.theta * RESOLUTION / X
    pixel = pixel.astype(int)

    idx = 0
    a0_paths_0, a0_paths_1 = a0_frame_paths[:voxel_coords[idx, 0]+1][::-1], a0_frame_paths[voxel_coords[idx, 0]:]
    a1_paths_0, a1_paths_1 = a1_frame_paths[:voxel_coords[idx, 1]+1][::-1], a1_frame_paths[voxel_coords[idx, 1]:]
    a2_paths_0, a2_paths_1 = a2_frame_paths[:voxel_coords[idx, 2]+1][::-1], a2_frame_paths[voxel_coords[idx, 2]:]
    
    a0_mask_0 = torch.flip(segment_point(a0_paths_0, [pixel[idx, 2], pixel[idx, 1]]), dims=[0])
    a0_mask_1 = segment_point(a0_paths_1, [pixel[idx, 2], pixel[idx, 1]])[1:, :, :]
    a0_mask = torch.cat([a0_mask_0, a0_mask_1], dim=0)
    a0_mask = torch.nn.functional.interpolate(a0_mask.unsqueeze(0).unsqueeze(0), size=(X, X, X), mode='trilinear').squeeze(0)
    
    a1_mask_0 = torch.flip(segment_point(a1_paths_0, [pixel[idx, 2], pixel[idx, 0]]), dims=[0])
    a1_mask_1 = segment_point(a1_paths_1, [pixel[idx, 2], pixel[idx, 0]])[1:, :, :]
    a1_mask = torch.cat([a1_mask_0, a1_mask_1], dim=0)
    a1_mask = torch.nn.functional.interpolate(a1_mask.unsqueeze(0).unsqueeze(0), size=(X, X, X), mode='trilinear').squeeze(0)
    
    a2_mask_0 = torch.flip(segment_point(a2_paths_0, [pixel[idx, 1], pixel[idx, 0]]), dims=[0])
    a2_mask_1 = segment_point(a2_paths_1, [pixel[idx, 1], pixel[idx, 0]])[1:, :, :]
    a2_mask = torch.cat([a2_mask_0, a2_mask_1], dim=0)
    a2_mask = torch.nn.functional.interpolate(a2_mask.unsqueeze(0).unsqueeze(0), size=(X, X, X), mode='trilinear').squeeze(0)
    
    a0_mask, a1_mask, a2_mask = a0_mask.transpose(0, 1), a1_mask.transpose(0, 1), a2_mask.transpose(0, 1)
    
    mask = a0_mask.permute(0, 2, 3, 1) + a1_mask.permute(2, 0, 3, 1) + a2_mask.permute(2, 3, 0, 1)
    mask = (mask > 1.5).squeeze()[2:, 2:, 2:]
    return mask
