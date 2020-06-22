
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataloader.mvs_dataset import MVSTestSet
from networks.ucsnet import UCSNet
from utils.utils import dict2cuda, dict2numpy, mkdir_p, save_cameras, write_pfm

import numpy as np
import argparse, os, time, gc, cv2
from PIL import Image
import os.path as osp
from collections import *
import sys

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Test UCSNet.')

parser.add_argument('--root_path', type=str, help='path to root directory.')
parser.add_argument('--test_list', type=str, help='testing scene list.')
parser.add_argument('--save_path', type=str, help='path to save depth maps.')

#test parameters
parser.add_argument('--max_h', type=int, help='image height', default=1080)
parser.add_argument('--max_w', type=int, help='image width', default=1920)
parser.add_argument('--num_views', type=int, help='num of candidate views', default=3)
parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')
parser.add_argument('--ckpt', type=str, help='the path for pre-trained model.', default='./checkpoints/model.ckpt')

args = parser.parse_args()


def main(args):
    # dataset, dataloader
    testset = MVSTestSet(root_dir=args.root_path, data_list=args.test_list,
                         max_h=args.max_h, max_w=args.max_w, num_views=args.num_views)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=4, drop_last=False)

    # build model
    model = UCSNet(stage_configs=list(map(int, args.net_configs.split(","))),
                   lamb=args.lamb)

    # load checkpoint file specified by args.loadckpt
    print("Loading model {} ...".format(args.ckpt))
    state_dict = torch.load(args.ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    print('Success!')

    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    tim_cnt = 0

    for batch_idx, sample in enumerate(test_loader):
        scene_name = sample["scene_name"][0]
        frame_idx = sample["frame_idx"][0][0]
        scene_path = osp.join(args.save_path, scene_name)

        print('Process data ...')
        sample_cuda = dict2cuda(sample)

        print('Testing {} frame {} ...'.format(scene_name, frame_idx))
        start_time = time.time()
        outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        end_time = time.time()

        outputs = dict2numpy(outputs)
        del sample_cuda

        tim_cnt += (end_time - start_time)

        print('Finished {}/{}, time: {:.2f}s ({:.2f}s/frame).'.format(batch_idx+1, len(test_loader), end_time-start_time,
                                                               tim_cnt / (batch_idx + 1.)))

        rgb_path = osp.join(scene_path, 'rgb')
        mkdir_p(rgb_path)
        depth_path = osp.join(scene_path, 'depth')
        mkdir_p(depth_path)
        cam_path = osp.join(scene_path, 'cam')
        mkdir_p(cam_path)
        conf_path = osp.join(scene_path, 'confidence')
        mkdir_p(conf_path)


        ref_img = sample["imgs"][0, 0].numpy().transpose(1, 2, 0) * 255
        ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
        Image.fromarray(ref_img).save(rgb_path+'/{:08d}.png'.format(frame_idx))

        cam = sample["proj_matrices"]["stage3"][0, 0].numpy()
        save_cameras(cam, cam_path+'/cam_{:08d}.txt'.format(frame_idx))

        for stage_id in range(3):
            cur_res = outputs["stage{}".format(stage_id+1)]
            cur_dep = cur_res["depth"][0]
            cur_conf = cur_res["confidence"][0]

            write_pfm(depth_path+"/dep_{:08d}_{}.pfm".format(frame_idx, stage_id+1), cur_dep)
            write_pfm(conf_path+'/conf_{:08d}_{}.pfm'.format(frame_idx, stage_id+1), cur_conf)

        print('Saved results for {}/{} (resolution: {})'.format(scene_name, frame_idx, cur_dep.shape))

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
	with torch.no_grad():
		main(args)