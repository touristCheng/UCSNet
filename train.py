import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from dataloader.mvs_dataset import MVSTrainSet, MVSTestSet
from networks.ucsnet import UCSNet
from utils.utils import *

import argparse, os, sys, time, gc, datetime
import os.path as osp

cudnn.benchmark = True
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


parser = argparse.ArgumentParser(description='Deep stereo using adaptive cost volume.')
parser.add_argument('--root_path', type=str, help='path to root directory.')
parser.add_argument('--train_list', type=str, help='train scene list.', default='./dataloader/datalist/dtu/train.txt')
parser.add_argument('--val_list', type=str, help='val scene list.', default='./dataloader/datalist/dtu/val.txt')
parser.add_argument('--save_path', type=str, help='path to save checkpoints.')

parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.0016)
parser.add_argument('--lr_idx', type=str, default="10,12,14:0.5")
parser.add_argument('--loss_weights', type=str, default="0.5,1.0,2.0")
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--num_views', type=int, help='num of candidate views', default=2)
parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')

parser.add_argument('--log_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency.')
parser.add_argument('--eval_freq', type=int, default=1, help='evaluate frequency.')

parser.add_argument('--sync_bn', action='store_true',help='Sync BN.')
parser.add_argument('--opt_level', type=str, default="O0")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)


args = parser.parse_args()

if args.sync_bn:
	import apex
	import apex.amp as amp

on_main = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def print_func(data: dict, prefix: str= ''):
	for k, v in data.items():
		if isinstance(v, dict):
			print_func(v, prefix + '.' + k)
		elif isinstance(v, list):
			print(prefix+'.'+k, v)
		else:
			print(prefix+'.'+k, v.shape)

def main(args, model:nn.Module, optimizer, train_loader, val_loader):
	milestones = list(map(lambda x: int(x) * len(train_loader), args.lr_idx.split(':')[0].split(',')))
	gamma = float(args.lr_idx.split(':')[1])
	scheduler = get_step_schedule_with_warmup(optimizer=optimizer, milestones=milestones, gamma=gamma)

	loss_weights = list(map(float, args.loss_weights.split(',')))

	for ep in range(args.epochs):
		model.train()
		for batch_idx, sample in enumerate(train_loader):

			tic = time.time()
			sample_cuda = dict2cuda(sample)

			# print_func(sample_cuda)

			optimizer.zero_grad()
			outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

			# print_func(outputs)

			loss = multi_stage_loss(outputs, sample_cuda["depth_labels"], sample_cuda["masks"], loss_weights)
			if is_distributed and args.sync_bn:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			scheduler.step()

			log_index = (len(train_loader)+len(val_loader)) * ep + batch_idx
			if log_index % args.log_freq == 0:

				image_summary, scalar_summary = collect_summary(sample_cuda, outputs)
				if on_main:
					add_summary(image_summary, 'image', logger, index=log_index, flag='train')
					add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='train')
					print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.2f}, eval 4mm ({:.2f}, {:.2f}), time = {:.2f}".format(
						ep+1, args.epochs, batch_idx+1, len(train_loader),
						optimizer.param_groups[0]["lr"], loss,
						scalar_summary["4mm_abs"], scalar_summary["4mm_acc"],
						time.time() - tic))

				del scalar_summary, image_summary

		gc.collect()
		if on_main and (ep + 1) % args.save_freq == 0:
			torch.save({"epoch": ep+1,
			            "model": model.module.state_dict(),
			            "optimizer": optimizer.state_dict()},
			            "{}/model_{:06d}.ckpt".format(args.save_path, ep+1))

		if (ep + 1) % args.eval_freq == 0 or (ep+1) == args.epochs:
			with torch.no_grad():
				test(args, model, val_loader, ep)

def test(args, model, test_loader, epoch):
	model.eval()
	avg_scalars = DictAverageMeter()
	for batch_idx, sample in enumerate(test_loader):
		sample_cuda = dict2cuda(sample)
		outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

		image_summary, scalar_summary = collect_summary(sample_cuda, outputs)
		avg_scalars.update(scalar_summary)

		log_index = len(train_loader) * (epoch + 1) + len(val_loader) * epoch + batch_idx
		if log_index % args.log_freq == 0 and on_main:
			add_summary(image_summary, 'image', logger, index=log_index, flag='val')
			add_summary(scalar_summary, 'scalar', logger, index=log_index, flag='val')

		del scalar_summary, image_summary

	if on_main:
		print("Epoch {}/{}: {}".format(epoch + 1, args.epochs, avg_scalars.mean()))
		add_summary(avg_scalars.mean(), 'scalar', logger, index=epoch + 1, flag='brief')

	gc.collect()

def collect_summary(inputs, outputs):
	depth = outputs["stage3"]["depth"]
	label = inputs["depth_labels"]["stage3"]
	mask = inputs["masks"]["stage3"].bool()

	err_map = torch.abs(label - depth) * mask.float()
	rgb = inputs["imgs"][:, 0]

	image_summary = {"depth": depth,
	                "label": label,
	                "mask": mask,
	                "error": err_map,
	                "ref_view": rgb
	                }

	scalar_summary = {}
	for thresh in [2, 3, 4, 20]:
		abs_err, acc = evaluate(depth, mask, label, thresh)
		scalar_summary["{}mm_abs".format(thresh)] = abs_err
		scalar_summary["{}mm_acc".format(thresh)] = acc
	scalar_summary = reduce_tensors(scalar_summary)
	return dict2numpy(image_summary), dict2float(scalar_summary)

def distribute_model(args):
	def sync():
		if not dist.is_available():
			return
		if not dist.is_initialized():
			return
		if dist.get_world_size() == 1:
			return
		dist.barrier()

	if is_distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(
			backend="nccl", init_method="env://"
		)
		sync()

	model: torch.nn.Module = UCSNet(stage_configs=list(map(int, args.net_configs.split(","))),
	                                lamb=args.lamb)
	model.to(torch.device("cuda"))

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
	                       weight_decay=args.wd)

	train_set = MVSTrainSet(root_dir=args.root_path, data_list=args.train_list, num_views=args.num_views)

	val_set = MVSTrainSet(root_dir=args.root_path, data_list=args.val_list, num_views=args.num_views)

	if is_distributed:
		if args.sync_bn:
			model = apex.parallel.convert_syncbn_model(model)
			model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, )
			print('Convert BN to Sync_BN successful.')

		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank,)

		train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
		                                                    rank=dist.get_rank())
		val_sampler = torch.utils.data.DistributedSampler(val_set, num_replicas=dist.get_world_size(),
		                                                   rank=dist.get_rank())
	else:
		model = nn.DataParallel(model)
		train_sampler, val_sampler = None, None

	train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler, num_workers=1,
	                          drop_last=True, shuffle=not is_distributed)
	val_loader = DataLoader(val_set, args.batch_size, sampler=val_sampler, num_workers=1,
	                         drop_last=False, shuffle=False)

	return model, optimizer, train_loader, val_loader

def multi_stage_loss(outputs, labels, masks, weights):
	tot_loss = 0.
	for stage_id in range(3):
		depth_i = outputs["stage{}".format(stage_id+1)]["depth"]
		label_i = labels["stage{}".format(stage_id+1)]
		mask_i = masks["stage{}".format(stage_id+1)].bool()
		depth_loss = F.smooth_l1_loss(depth_i[mask_i], label_i[mask_i], reduction='mean')
		tot_loss += depth_loss * weights[stage_id]
	return tot_loss

if __name__ == '__main__':

	model, optimizer, train_loader, val_loader = distribute_model(args)

	on_main = (not is_distributed) or (dist.get_rank() == 0)

	if on_main:
		mkdir_p(args.save_path)
		logger = SummaryWriter(args.save_path)
		print(args)

	main(args=args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader)