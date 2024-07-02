import sys
sys.path.append('core')
import math
import os
import random
import time
import yaml
import cupy as cp
from raft import RAFT
from utils.utils import InputPadder
import cv2
import numba
import argparse
# import apex
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from data_parallel import BalancedDataParallel
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from torchvision.transforms import ToPILImage
from PIL import Image
from DatasetLidarCamera import DatasetLidarCameraKittiOdometry
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from calib_raft import RaftCalib
from torchvision import transforms
from quaternion_distances import quaternion_distance
import torchvision.transforms.functional as TTF
from tensorboardX import SummaryWriter
from utils.utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat, lidar_project_depth_label, point_fill)
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("LCCNet")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = './checkpoints/'
    savemodel_log = 'seg_calib_kitti'
    dataset = 'kitti/odom' # 'kitti/raw'
    data_folder = '/data/xc/LCCNet/LCCNet-main/odometry'
    use_reflectance = False
    val_sequence = 0
    epochs = 200
    BASE_LEARNING_RATE = 1e-3  # 1e-4
    loss = 'combined'
    max_t = 1.0 # 1.5, 1.0,  0.5,  0.2,  0.1
    max_r = 10.0 # 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 52  # 120
    num_worker = 12
    network = 'Res_f1'
    optimizer = 'adam'
    resume = True
    weights = './checkpoints/kitti/odom/val_seq_00/models/seg_calib_kitti_checkpoint_r20.00_t1.50_e24_0.991_1.tar'
    # weights = None
    rescale_rot = 1.0
    rescale_transl = 2.0
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 80.
    weight_point_cloud = 0.5
    log_frequency = 10
    print_frequency = 50
    starting_epoch = 0
    label_path = '/data/xc/Codes-for-PVKD/out_label/semantic-kitti'


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, loss):
    model.train()

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    losses['total_loss'].backward()
    optimizer.step()

    return losses, rot_err, transl_err


# CNN test
@ex.capture
def val(model, rgb_img, refl_img, target_transl, target_rot, loss_fn, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_err, rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j].cpu() - transl_err[j].cpu()) * 100.

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err, transl_err


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))

    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        _config['val_sequence'] = f"{_config['val_sequence']:02d}"
        print("Val Sequence: ", _config['val_sequence'])
        dataset_class = DatasetLidarCameraKittiOdometry
    input_size = (376, 1248)
    _config["checkpoints"] = os.path.join(_config["checkpoints"], _config['dataset'])

    dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                  split='train', use_reflectance=_config['use_reflectance'],
                                  val_sequence=_config['val_sequence'], label_path=_config['label_path'])

    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='val', use_reflectance=_config['use_reflectance'],
                                val_sequence=_config['val_sequence'], label_path=_config['label_path'])
    model_savepath = os.path.join(_config['checkpoints'], 'val_seq_' + _config['val_sequence'], 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print("len(TrainImgLoader): ", len(TrainImgLoader))
    print("len(ValImgLoader): ", len(ValImgLoader))

    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='models/raft-kitti.pth')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--small', action='store_true', help='use small model', default='False')
    args = parser.parse_args()

    model = RaftCalib(image_size=input_size, args=args)

    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    model = BalancedDataParallel(16, model, device_ids=[0, 1, 2])
    # model = nn.DataParallel(model)
    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=0)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])
    to_pil_image = ToPILImage()
    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            _run.log_scalar("LR", scheduler.get_lr()[0])


        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                depth_img = sample['point_cloud'][idx].cuda()
                depth_img = depth_img.permute(2, 0, 1)
                # depth_img = point_fill(depth_img)
                rgb = sample['rgb'][idx].cuda()
                # rgb_o = torch.tensor(np.array(sample['img_org'][idx]), dtype=torch.float32).cuda().permute(2, 0, 1)
                shape_pad = [0, 0, 0, 0]
                shape_pad[3] = (input_size[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (input_size[1] - rgb.shape[2])  # // 2 + 1
                #
                rgb = F.pad(rgb, shape_pad)
                # # rgb_o = F.pad(rgb_o, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                # depth_gt = F.pad(depth_gt, shape_pad)

                rgb_input.append(rgb)
                # rgb_org.append(rgb_o)
                lidar_input.append(depth_img)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            local_loss += loss['total_loss'].item()

            if batch_idx % 100 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/100:.6f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for 100 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/100, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.6f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)

        ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.
        errors_r = []
        errors_t = []
        errors_t2 = []
        errors_xyz = []
        errors_rpy = []
        RT_inv = []
        true_r = []
        true_t = []
        RTs = []
        local_loss = 0.0
        for batch_idx, sample in enumerate(ValImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                # depth_img = point_fill(depth_img)
                depth_img = sample['point_cloud'][idx].cuda()
                depth_img = depth_img.permute(2, 0, 1)
                rgb = sample['rgb'][idx].cuda()
                shape_pad = [0, 0, 0, 0]
                shape_pad[3] = (input_size[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (input_size[1] - rgb.shape[2])  # // 2 + 1
                rgb = F.pad(rgb, shape_pad)
                # # rgb_o = F.pad(rgb_o, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                # # depth_gt = F.pad(depth_gt, shape_pad)

                rgb_input.append(rgb)
                # rgb_org.append(rgb_o)
                lidar_input.append(depth_img)
                # RT_inv.append(sample['RT_inv'][idx])

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            loss, trasl_e, rot_e, R_predicted,  T_predicted = val(model, rgb_input, lidar_input,
                                                                  sample['tr_error'], sample['rot_error'],
                                                                  loss_fn, _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            total_val_t += trasl_e
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                print('Iter %d val loss = %.6f , time = %.2f' % (batch_idx, local_loss/10.,
                                                                  (time.time() - start_time)/lidar_input.shape[0]))
                local_loss = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.6f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

        # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            #_run.result = BEST_VAL_LOSS
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/{_config["savemodel_log"]}_checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}_1.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                # 'state_dict': model.state_dict(), # single gpu
                'state_dict': model.module.state_dict(), # multi gpu
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result
