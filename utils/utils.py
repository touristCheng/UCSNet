import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
import torchvision.utils as vutils
import torch.distributed as dist


import errno
import os
import re
import sys
import numpy as np
from bisect import bisect_right

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic

def dict2numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2numpy(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic

def dict2float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic

def metric_with_thresh(depth, label, mask, thresh):
    err = torch.abs(depth - label)
    valid = err <= thresh
    mean_abs = torch.mean(err[valid])
    acc = valid.sum(dtype=torch.float) / mask.sum(dtype=torch.float)
    return mean_abs, acc

def evaluate(depth, mask, label, thresh):
    batch_abs_err = []
    batch_acc = []
    for d, m, l in zip(depth, mask, label):
        abs_err, acc = metric_with_thresh(d, l, m, thresh)
        batch_abs_err.append(abs_err)
        batch_acc.append(acc)

    tot_abs = torch.stack(batch_abs_err)
    tot_acc = torch.stack(batch_acc)
    return tot_abs.mean(), tot_acc.mean()

def save_cameras(cam, path):
    cam_txt = open(path, 'w+')

    cam_txt.write('extrinsic\n')
    for i in range(4):
        for j in range(4):
            cam_txt.write(str(cam[0, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.write('\n')

    cam_txt.write('intrinsic\n')
    for i in range(3):
        for j in range(3):
            cam_txt.write(str(cam[1, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.close()

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def add_summary(data_dict: dict, dtype: str, logger, index: int, flag: str):
    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        if img.dtype == np.bool:
            img = img.astype(np.float32)
        img = torch.from_numpy(img[:1])
        if 'depth' in name or 'label' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(450, 850))
        elif 'mask' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(0, 1))
        elif 'error' in name:
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True, range=(0, 4))
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True,)

    on_main = (not is_distributed) or (dist.get_rank() == 0)
    if not on_main:
        return

    if dtype == 'image':
        for k, v in data_dict.items():
            logger.add_image('{}/{}'.format(flag, k), preprocess(k, v), index)

    elif dtype == 'scalar':
        for k, v in data_dict.items():
            logger.add_scalar('{}/{}'.format(flag, k), v, index)
    else:
        raise NotImplementedError

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input: dict):
        self.count += 1
        for k, v in new_input.items():
            assert isinstance(v, float), type(v)
            self.data[k] = self.data.get(k, 0) + v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

def reduce_tensors(datas: dict):
    if not is_distributed:
        return datas
    world_size = dist.get_world_size()
    with torch.no_grad():
        keys = list(datas.keys())
        vals = []
        for k in keys:
            vals.append(datas[k])
        vals = torch.stack(vals, dim=0)
        dist.reduce(vals, op=dist.reduce_op.SUM, dst=0)
        if dist.get_rank() == 0:
            vals /= float(world_size)
        reduced_datas = {k: v for k, v in zip(keys, vals)}
    return reduced_datas

