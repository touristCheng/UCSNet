import torch
from torch.utils.data import Dataset

from PIL import Image
from utils.utils import read_pfm

import numpy as np
import cv2
import glob
import os, sys
import re


def scale_inputs(img, intrinsics, max_w, max_h, base=32):
    h, w = img.shape[:2]
    if h > max_h or w > max_w:
        scale = 1.0 * max_h / h
        if scale * w > max_w:
            scale = 1.0 * max_w / w
        new_w, new_h = scale * w // base * base, scale * h // base * base
    else:
        new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

    scale_w = 1.0 * new_w / w
    scale_h = 1.0 * new_h / h
    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h
    img = cv2.resize(img, (int(new_w), int(new_h)))
    return img, intrinsics

class MVSTrainSet(Dataset):
    def __init__(self, root_dir, data_list, lightings=range(7), num_views=4):
        super(MVSTrainSet, self).__init__()

        self.root_dir = root_dir
        scene_names = open(data_list, 'r').readlines()
        self.scene_names = list(map(lambda x: x.strip(), scene_names))
        self.lightings = lightings
        self.num_views = num_views
        self.generate_pairs()

    def generate_pairs(self, ):
        data_pairs = []
        pair_list = open('{}/Cameras/pair.txt'.format(self.root_dir), 'r').readlines()

        pair_list = list(map(lambda x: x.strip(), pair_list))
        cnt = int(pair_list[0])
        for i in range(cnt):
            ref_id = int(pair_list[i*2+1])
            candidates = pair_list[i*2+2].split()

            nei_id = [int(candidates[2*j+1]) for j in range(self.num_views)]
            for scene_name in self.scene_names:
                for light in self.lightings:
                    data_pairs.append({'scene_name': scene_name,
                                       'frame_idx': [ref_id, ]+nei_id,
                                       'light': light
                                       })
        self.data_pairs = data_pairs

    def parse_cameras(self, path):
        cam_txt = open(path).readlines()
        f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

        extr_mat = f(cam_txt[1:5])
        intr_mat = f(cam_txt[7:10])

        extr_mat = np.array(extr_mat, np.float32)
        intr_mat = np.array(intr_mat, np.float32)

        min_dep, delta = list(map(float, cam_txt[11].strip().split()))
        max_dep = 1.06 * 191.5 * delta + min_dep

        intr_mat[:2] *= 4.
        # note the loaded camera model is for 1/4 original resolution

        return extr_mat, intr_mat, min_dep, max_dep

    def load_depths(self, path):
        depth_s3 = np.array(read_pfm(path)[0], np.float32)
        h, w = depth_s3.shape
        depth_s2 = cv2.resize(depth_s3, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        depth_s1 = cv2.resize(depth_s3, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        return {'stage1': depth_s1, 'stage2': depth_s2, 'stage3': depth_s3}

    def make_masks(self, depths:dict, min_d, max_d):
        masks = {}
        for k, v in depths.items():
            m = np.ones(v.shape, np.uint8)
            m[v>max_d] = 0
            m[v<min_d] = 0
            masks[k] = m
        return masks

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair_dict = self.data_pairs[idx]

        scene_name = pair_dict['scene_name']
        frame_idx = pair_dict['frame_idx']
        light = pair_dict['light']

        images = []
        proj_mats_s3 = []
        res = {}

        for i, idx in enumerate(frame_idx):
            img_path = '{}/Rectified/{}_train/rect_{:03d}_{}_r5000.png'.format(self.root_dir, scene_name, idx+1, light)
            image = Image.open(img_path)
            image = np.array(image, dtype=np.float32) / 255.
            images.append(image)

            cam_path = '{}/Cameras/train/{:08d}_cam.txt'.format(self.root_dir, idx)
            extr_mat, intr_mat, min_dep, max_dep = self.parse_cameras(cam_path)

            proj_mat = np.zeros((2, 4, 4), np.float32)
            proj_mat[0, :4, :4] = extr_mat
            proj_mat[1, :3, :3] = intr_mat
            proj_mats_s3.append(proj_mat)

            if i == 0:
                dep_path = '{}/Depths_4/{}_train/depth_map_{:04d}.pfm'.format(self.root_dir, scene_name, idx)
                depth_gts = self.load_depths(dep_path)
                masks = self.make_masks(depth_gts, min_d=min_dep, max_d=max_dep)

                res['depth_labels'] = depth_gts
                res['masks'] = masks
                res['depth_values'] = np.array([min_dep, max_dep], np.float32)

        proj_mats_s3 = np.stack(proj_mats_s3)
        proj_mats_s2 = proj_mats_s3.copy()
        proj_mats_s2[:, 1, :2, :3] = proj_mats_s3[:, 1, :2, :3] / 2.
        proj_mats_s1 = proj_mats_s3.copy()
        proj_mats_s1[:, 1, :2, :3] = proj_mats_s3[:, 1, :2, :3] / 4.
        proj_mats = {'stage1': proj_mats_s1, 'stage2': proj_mats_s2, 'stage3': proj_mats_s3}

        images = np.stack(images).transpose([0, 3, 1, 2])

        res['imgs'] = images
        res['proj_matrices'] = proj_mats
        res['scene_name'] = scene_name
        res['frame_idx'] = frame_idx
        return res

class MVSTestSet(Dataset):
    def __init__(self, root_dir, data_list, max_h, max_w, num_views=4):
        super(MVSTestSet, self).__init__()

        self.root_dir = root_dir
        scene_names = open(data_list, 'r').readlines()
        self.scene_names = list(map(lambda x: x.strip(), scene_names))
        self.num_views = num_views
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pairs()

    def generate_pairs(self, ):
        data_pairs = []
        for scene_name in self.scene_names:
            pair_list = open('{}/{}/pair.txt'.format(self.root_dir, scene_name), 'r').readlines()
            pair_list = list(map(lambda x: x.strip(), pair_list))
            cnt = int(pair_list[0])
            for i in range(cnt):
                ref_id = int(pair_list[i * 2 + 1])
                candidates = pair_list[i * 2 + 2].split()
                nei_id = [int(candidates[2 * j + 1]) for j in range(self.num_views)]

                data_pairs.append({'scene_name': scene_name,
                                   'frame_idx': [ref_id, ] + nei_id,})
        self.data_pairs = data_pairs

    def parse_cameras(self, path):
        cam_txt = open(path).readlines()
        f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

        extr_mat = f(cam_txt[1:5])
        intr_mat = f(cam_txt[7:10])

        extr_mat = np.array(extr_mat, np.float32)
        intr_mat = np.array(intr_mat, np.float32)

        min_dep, delta = list(map(float, cam_txt[11].strip().split()))
        max_dep = 1.06 * 191.5 * delta + min_dep

        return extr_mat, intr_mat, min_dep, max_dep

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair_dict = self.data_pairs[idx]

        scene_name = pair_dict['scene_name']
        frame_idx = pair_dict['frame_idx']

        images = []
        proj_mats_s3 = []
        res = {}

        for i, idx in enumerate(frame_idx):
            img_path = '{}/{}/images/{:08d}.jpg'.format(self.root_dir, scene_name, idx)

            image = Image.open(img_path)
            image = np.array(image, dtype=np.float32) / 255.

            cam_path = '{}/{}/cams/{:08d}_cam.txt'.format(self.root_dir, scene_name, idx)
            extr_mat, intr_mat, min_dep, max_dep = self.parse_cameras(cam_path)

            image, intr_mat = scale_inputs(image, intr_mat, max_h=self.max_h, max_w=self.max_w)

            images.append(image)


            proj_mat = np.zeros((2, 4, 4), np.float32)
            proj_mat[0, :4, :4] = extr_mat
            proj_mat[1, :3, :3] = intr_mat
            proj_mats_s3.append(proj_mat)

            if i == 0:
                res['depth_values'] = np.array([min_dep, max_dep], np.float32)

        proj_mats_s3 = np.stack(proj_mats_s3)
        proj_mats_s2 = proj_mats_s3.copy()
        proj_mats_s2[:, 1, :2, :3] = proj_mats_s3[:, 1, :2, :3] / 2.
        proj_mats_s1 = proj_mats_s3.copy()
        proj_mats_s1[:, 1, :2, :3] = proj_mats_s3[:, 1, :2, :3] / 4.
        proj_mats = {'stage1': proj_mats_s1, 'stage2': proj_mats_s2, 'stage3': proj_mats_s3}

        images = np.stack(images).transpose([0, 3, 1, 2])

        res['imgs'] = images
        res['proj_matrices'] = proj_mats
        res['scene_name'] = scene_name
        res['frame_idx'] = frame_idx
        return res





