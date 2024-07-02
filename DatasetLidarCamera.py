import sys
sys.path.append('core')
import csv
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import invert_pose, rotate_forward, quat2mat, tvector2mat, quaternion_from_matrix, rotate_back, lidar_project_depth_label
from pykitti import odometry
import pykitti


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=10., split='val', device='cpu', val_sequence='00', suf='.png', label_path=''):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf
        self.label_path = label_path

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '13', '14', '15', '16']
        self.sequence = 'sequences'
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2 # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(dataset_dir, self.sequence, seq, 'velodyne'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, self.sequence, seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, self.sequence, seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, sequence,
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = os.path.join(dataset_dir, self.sequence,
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, self.sequence, seq, 'image_2', rgb_name+self.suf)
        img_seg_path = os.path.join(r'/data/xc/datasets/semantic_image', seq, rgb_name+self.suf)
        label_path = os.path.join(self.label_path, self.sequence, seq, 'predictions', rgb_name+'.label')
        lidar_path = os.path.join(self.root_dir, self.sequence, seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        # pc = np.load(lidar_path)    # load .npy
        # valid_indices = pc[:, 0] < -3.
        # valid_indices = valid_indices | (pc[:, 0] > 3.)
        # valid_indices = valid_indices | (pc[:, 1] < -3.)
        # valid_indices = valid_indices | (pc[:, 1] > 3.)
        # valid_indices_x = pc[:, 0] > -80.
        # valid_indices_x = valid_indices_x & (pc[:, 0] < 80.)
        # valid_indices_y = pc[:, 1] > -80.
        # valid_indices_y = valid_indices_y & (pc[:, 1] < 80.)
        # valid_indices = valid_indices_x & valid_indices_y & valid_indices
        # pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        # h_mirror = False
        # img = Image.open(img_path)
        # img = Image.open(img_seg_path)
        img = cv2.imread(img_seg_path)
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        # img_rotation = 0.
        #
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        calib = self.K[seq]

        real_shape = [img.shape[1], img.shape[2], img.shape[0]]
        R0 = mathutils.Quaternion(R).to_matrix()
        R0.resize_4x4()
        T0 = mathutils.Matrix.Translation(T)
        RT0 = T0 * R0
        pc_rotated = rotate_back(pc_in, RT0) # Pc` = RT * Pc
        depth_img, uv = lidar_project_depth_label(pc_rotated, calib, real_shape, label_path)
        # cv2.imwrite(os.path.join(r'/data/xc/seg_raft_calib/view', '000.png'), depth_img.numpy().astype(np.uint8))

        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': depth_img, 'calib': calib, 'pc_in': pc_in,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT, 'label_path': label_path}
        else:
            sample = {'rgb': img, 'point_cloud': depth_img, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT,
                      'label_path': label_path}

        return sample