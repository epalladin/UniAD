#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import pickle

from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map
from .data_utils.data_utils import obtain_map_info
from .nuscenes_e2e_dataset import NuScenesE2EDataset


@DATASETS.register_module()
class NuScenesE2EDatasetLidar(NuScenesE2EDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    Lidar Version
    """
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args['backend'] == 'disk':
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        elif self.file_client_args['backend'] == 'petrel':
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'
        return data_infos

    def union2one(self, queue):
        """
        convert sample dict into one single sample.
        """
        # imgs_list = [each['img'].data for each in queue]
        pts_list = [each['points'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_sdc_label_list = [each['gt_sdc_label'].data for each in queue]
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
        gt_past_traj_mask_list = [
            to_tensor(each['gt_past_traj_mask']) for each in queue]
        gt_sdc_bbox_list = [each['gt_sdc_bbox'].data for each in queue]
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        timestamp_list = [to_tensor(each['timestamp']) for each in queue]
        gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])
        gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])
        gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])
        gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])
        gt_future_boxes_list = queue[-1]['gt_future_boxes']
        gt_future_labels_list = [to_tensor(each)
                                 for each in queue[-1]['gt_future_labels']]

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['points'] = pts_list
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        queue['gt_labels_3d'] = DC(gt_labels_3d_list)
        queue['gt_sdc_label'] = DC(gt_sdc_label_list)
        queue['gt_inds'] = DC(gt_inds_list)
        queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)
        queue['l2g_r_mat'] = DC(l2g_r_mat_list)
        queue['l2g_t'] = DC(l2g_t_list)
        queue['timestamp'] = DC(timestamp_list)
        queue['gt_fut_traj'] = DC(gt_fut_traj)
        queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)
        queue['gt_past_traj'] = DC(gt_past_traj_list)
        queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
        queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
        queue['gt_future_labels'] = DC(gt_future_labels_list)
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # semantic format
        lane_info = self.lane_infos[index] if self.lane_infos else None
        # panoptic format
        location = self.nusc.get('log', self.nusc.get(
            'scene', info['scene_token'])['log_token'])['location']
        vectors = self.vector_map.gen_vectorized_samples(location,
                                                         info['ego2global_translation'],
                                                         info['ego2global_rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(vectors,
                                                                                       self.patch_size,
                                                                                       self.canvas_size,
                                                                                       self.map_num_classes,
                                                                                       self.thickness,
                                                                                       self.angle_class)
        instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
        instance_masks = torch.tensor(instance_masks.copy())
        gt_labels = []
        gt_bboxes = []
        gt_masks = []
        for cls in range(self.map_num_classes):
            for i in np.unique(instance_masks[cls]):
                if i == 0:
                    continue
                gt_mask = (instance_masks[cls] == i).to(torch.uint8)
                ys, xs = np.where(gt_mask)
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
                gt_labels.append(cls)
                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)
        map_mask = obtain_map_info(self.nusc,
                                   self.nusc_maps,
                                   info,
                                   patch_size=self.patch_size,
                                   canvas_size=self.canvas_size,
                                   layer_names=['lane_divider', 'road_divider'])
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = torch.tensor(map_mask.copy())
        for i, gt_mask in enumerate(map_mask[:-1]):
            ys, xs = np.where(gt_mask)
            gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            gt_labels.append(i + self.map_num_classes)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)
        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = torch.tensor(np.stack(gt_bboxes))
        gt_masks = torch.stack(gt_masks)
        for sweep in info['sweeps']:
            sweep['data_path'] =  sweep['data_path'].replace('/mnt/petrelfs/yangjiazhi/e2e_proj/', '')
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'].replace('/mnt/petrelfs/yangjiazhi/e2e_proj/', ''),
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            map_filename=lane_info['maps']['map_mask'] if lane_info else None,
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
        )

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32)))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        if 'sdc_planning' in input_dict['ann_info'].keys():
            input_dict['sdc_planning'] = input_dict['ann_info']['sdc_planning']
            input_dict['sdc_planning_mask'] = input_dict['ann_info']['sdc_planning_mask']
            input_dict['command'] = input_dict['ann_info']['command']

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        # TODO: Warp all those below occupancy-related codes into a function
        prev_indices, future_indices = self.occ_get_temporal_indices(
            index, self.occ_receptive_field, self.occ_n_future)

        # ego motions of all frames are needed
        all_frames = prev_indices + [index] + future_indices

        # whether invalid frames is present
        # 
        has_invalid_frame = -1 in all_frames[:self.occ_only_total_frames]
        # NOTE: This can only represent 7 frames in total as it influence evaluation
        input_dict['occ_has_invalid_frame'] = has_invalid_frame
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0

        # might have None if not in the same sequence
        future_frames = [index] + future_indices

        # get lidar to ego to global transforms for each curr and fut index
        occ_transforms = self.occ_get_transforms(
            future_frames)  # might have None
        input_dict.update(occ_transforms)

        # for (current and) future frames, detection labels are needed
        # generate detection labels for current + future frames
        input_dict['occ_future_ann_infos'] = \
            self.get_future_detection_infos(future_frames)
        return input_dict
