import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)



import math
import cupy as cp
from scipy.spatial import Delaunay
import cv2
import yaml
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data.dataloader import default_collate


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T*R
    else:
        RT=R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    # img_org = []
    reflectances = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' and key != 'img_org'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        # img_org.append(input['img_org'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    # returns['img_org'] = img_org
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns


def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    #plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth_label(pc_rotated, cam_calib, img_shape, label_path):
    pc_rotated = pc_rotated[:3, :].detach().numpy()
    colors = point_color(label_path)
    colors = colors[:, [2, 1, 0]]
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    color = colors[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    depth_img = np.zeros((img_shape[0], img_shape[1], 3))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = color
    # depth_img = depth_img.permute(2, 0, 1)
    depth_img = torch.from_numpy(depth_img.astype(np.float32))

    return depth_img, pcl_uv


def lidar_depth_label(pc_rotated, cam_calib, img_shape, label_path):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    colors = point_color(label_path)
    colors = colors[:, [2, 1, 0]]
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    color = colors[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    depth_img = np.zeros((img_shape[0], img_shape[1], 3))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = color
    # depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


def point_color(label_path):
    # 读取semantic-kitti.yaml文件
    with open("./semantic-kitti.yaml", "r") as file:
        semantic_config = yaml.safe_load(file)

    # 获取labels和color_map
    color_map = semantic_config.get("color_map", {})

    # 读取语义标签数据
    with open(label_path, "rb") as file:
        # 以字节方式读取文件内容
        content = file.read()
    # 解析字节数据为NumPy数组
    semantic_labels = np.frombuffer(content, dtype=np.uint32)
    # 获取语义标签对应的颜色
    colors = np.array([color_map.get(label, [0, 0, 0]) for label in semantic_labels]).astype(np.float32)
    return colors


def point_fill(image_org):
    alpha_shape = np.zeros_like(image_org).astype(np.uint8)
    image_org = cp.asarray(image_org).astype(cp.uint8)
    image_result = image_org.copy()
    # time0 = time.time()
    # 找出所有非零点
    non_black_pixels = cp.argwhere(cp.any(image_org != 0, axis=-1))
    alpha = 5
    while cp.size(non_black_pixels):
        # time2 = time.time()
        alpha_shape = np.zeros_like(alpha_shape)
        # 在有颜色值的点中随机选择一个点作为种子点
        pixel = image_org[non_black_pixels[0][0], non_black_pixels[0][1]]
        # 调用函数找到相似颜色的点
        similar_color_points = cp.argwhere((image_org == pixel).all(axis=-1))
        if similar_color_points.shape[0] <= 10:
            image_org[similar_color_points[:, 0], similar_color_points[:, 1]] = 0
            non_black_pixels = cp.argwhere(cp.any(image_org != 0, axis=-1))
            continue
        if similar_color_points.shape[0] > 6000:
            similar_color_points = similar_color_points[:6000, :]
        # 计算pi与点集内其他各个点间的距离，并将所有与点pi间距离小于2alpha的点记为点集Qi
        distances = cp.linalg.norm(similar_color_points[cp.newaxis, :, :] - similar_color_points[:, cp.newaxis, :], axis=-1)
        q_index = cp.argpartition(distances, 10, axis=-1)[:, :10]
        q_points = similar_color_points[q_index]
        # 计算两点之间的距离
        d = cp.partition(distances, 10, axis=-1)[:, :10]
        # 计算中点坐标
        mid_x = (similar_color_points[:, cp.newaxis, 0] + q_points[:, :, 0]) / 2
        mid_y = (similar_color_points[:, cp.newaxis, 1] + q_points[:, :, 1]) / 2
        # 计算垂直线的长度
        h = cp.sqrt(alpha ** 2 - (d / 2) ** 2)
        # 计算圆心坐标
        x1 = mid_x + h * (q_points[:, :, 1] - similar_color_points[:, cp.newaxis, 1]) / d
        y1 = mid_y - h * (q_points[:, :, 0] - similar_color_points[:, cp.newaxis, 0]) / d
        x2 = mid_x - h * (q_points[:, :, 1] - similar_color_points[:, cp.newaxis, 1]) / d
        y2 = mid_y + h * (q_points[:, :, 0] - similar_color_points[:, cp.newaxis, 0]) / d
        circle1 = cp.transpose(cp.array([x1, y1]), (1, 2, 0))
        circle2 = cp.transpose(cp.array([x2, y2]), (1, 2, 0))
        # 计算Qi中的点到两个圆心的距离
        q_distances1 = cp.linalg.norm(q_points[:, cp.newaxis, :, :] - circle1[:, :, cp.newaxis, :], axis=-1)
        q_distances2 = cp.linalg.norm(q_points[:, cp.newaxis, :, :] - circle2[:, :, cp.newaxis, :], axis=-1)
        # 计算Qi中每个点产生的圆是否是边界圆（边界圆：Qi中所有点到任一圆心的距离大于等于alpha）
        flag = cp.logical_or(cp.all(q_distances1 >= alpha - 0.5, axis=-1), cp.all(q_distances2 >= alpha - 0.5, axis=-1))
        # 找到边界点
        # line_points = q_points[flag]
        color = tuple([int(x) for x in pixel])  # 设置为整数
        # print("cupy: ", time.time() - time2)
        # time2 = time.time()
        similar_color_points_np = similar_color_points.get()
        q_points_np = q_points.get()
        flag_np = flag.get()
        for first_point_index in range(similar_color_points_np.shape[0]):
            first_point = similar_color_points_np[first_point_index]
            line_points = q_points_np[first_point_index][flag_np[first_point_index]]
            for point in line_points:
                cv2.line(alpha_shape, (first_point[1], first_point[0]), (point[1], point[0]), (255, 255, 255), 1)
        # print("cv2.line: ", time.time() - time2)
        pixel_image_obj = fill(alpha_shape, color)
        image_result[pixel_image_obj[:, 0], pixel_image_obj[:, 1]] = pixel
        image_org[similar_color_points[:, 0], similar_color_points[:, 1]] = 0
        non_black_pixels = cp.argwhere(cp.any(image_org != 0, axis=-1))
    # cv2.imwrite(os.path.join(r'/data/xc/seg_calib', '000.png'), image_result.get())
    image_result = torch.from_numpy(image_result.get().astype(np.float32))
    image_result = image_result.cuda()
    # time1 = time.time()
    # print('point_fill_time: ', time1 - time0)
    image_result = image_result.permute(2, 0, 1)
    return image_result


def point_fill_cpu(image_org):
    alpha_shape = np.zeros_like(image_org).astype(np.uint8)
    image_result = image_org.copy()
    # time0 = time.time()
    # 找出所有非零点
    non_black_pixels = np.argwhere(np.any(image_org != 0, axis=-1))
    alpha = 5
    while np.size(non_black_pixels):
        # time2 = time.time()
        alpha_shape = np.zeros_like(alpha_shape)
        # 在有颜色值的点中随机选择一个点作为种子点
        pixel = image_org[non_black_pixels[0][0], non_black_pixels[0][1]]
        # 调用函数找到相似颜色的点
        similar_color_points = np.argwhere((image_org == pixel).all(axis=-1))
        if similar_color_points.shape[0] <= 10:
            image_org[similar_color_points[:, 0], similar_color_points[:, 1]] = 0
            non_black_pixels = np.argwhere(np.any(image_org != 0, axis=-1))
            continue
        if similar_color_points.shape[0] > 6000:
            similar_color_points = similar_color_points[:6000, :]
        # 计算pi与点集内其他各个点间的距离，并将所有与点pi间距离小于2alpha的点记为点集Qi
        distances = np.linalg.norm(similar_color_points[np.newaxis, :, :] - similar_color_points[:, np.newaxis, :], axis=-1)
        q_index = np.argpartition(distances, 10, axis=-1)[:, :10]
        q_points = similar_color_points[q_index]
        # 计算两点之间的距离
        d = np.partition(distances, 10, axis=-1)[:, :10]
        # 计算中点坐标
        mid_x = (similar_color_points[:, np.newaxis, 0] + q_points[:, :, 0]) / 2
        mid_y = (similar_color_points[:, np.newaxis, 1] + q_points[:, :, 1]) / 2
        # 计算垂直线的长度
        h = np.sqrt(alpha ** 2 - (d / 2) ** 2)
        # 计算圆心坐标
        x1 = mid_x + h * (q_points[:, :, 1] - similar_color_points[:, np.newaxis, 1]) / d
        y1 = mid_y - h * (q_points[:, :, 0] - similar_color_points[:, np.newaxis, 0]) / d
        x2 = mid_x - h * (q_points[:, :, 1] - similar_color_points[:, np.newaxis, 1]) / d
        y2 = mid_y + h * (q_points[:, :, 0] - similar_color_points[:, np.newaxis, 0]) / d
        circle1 = np.transpose(np.array([x1, y1]), (1, 2, 0))
        circle2 = np.transpose(np.array([x2, y2]), (1, 2, 0))
        # 计算Qi中的点到两个圆心的距离
        q_distances1 = np.linalg.norm(q_points[:, np.newaxis, :, :] - circle1[:, :, np.newaxis, :], axis=-1)
        q_distances2 = np.linalg.norm(q_points[:, np.newaxis, :, :] - circle2[:, :, np.newaxis, :], axis=-1)
        # 计算Qi中每个点产生的圆是否是边界圆（边界圆：Qi中所有点到任一圆心的距离大于等于alpha）
        flag = np.logical_or(np.all(q_distances1 >= alpha - 0.5, axis=-1), np.all(q_distances2 >= alpha - 0.5, axis=-1))
        # 找到边界点
        # line_points = q_points[flag]
        color = tuple([int(x) for x in pixel])  # 设置为整数
        # print("cupy: ", time.time() - time2)
        # time2 = time.time()
        for first_point_index in range(similar_color_points.shape[0]):
            first_point = similar_color_points[first_point_index]
            line_points = q_points[first_point_index][flag[first_point_index]]
            for point in line_points:
                cv2.line(alpha_shape, (first_point[1], first_point[0]), (point[1], point[0]), (255, 255, 255), 1)
        # print("cv2.line: ", time.time() - time2)
        pixel_image_obj = fill(alpha_shape, color)
        image_result[pixel_image_obj[:, 0], pixel_image_obj[:, 1]] = pixel
        image_org[similar_color_points[:, 0], similar_color_points[:, 1]] = 0
        non_black_pixels = np.argwhere(np.any(image_org != 0, axis=-1))
    # cv2.imwrite(os.path.join(r'/data/xc/seg_calib', '000.png'), image_result.get())
    image_result = torch.from_numpy(image_result.astype(np.float32))
    # time1 = time.time()
    # print('point_fill_time: ', time1 - time0)
    image_result = image_result.permute(2, 0, 1)
    return image_result


def find_similar_color_points(image, seed_point, max_distance):
    # 创建一个空列表，用于存储相似颜色的点
    result = np.expand_dims(np.array(seed_point), 0)
    similar_points = np.expand_dims(np.array(seed_point), 0)

    # 将种子点转换为 Numpy 数组
    seed_point = np.array(seed_point)

    # 找到与种子点相同颜色的点
    color_points = np.argwhere((image == image[seed_point[0], seed_point[1]]).all(axis=-1))

    while True:
        # 计算 similar_points 中任意点到 color_points 中所有点的距离
        distances = np.linalg.norm(similar_points[:, np.newaxis, :] - color_points[np.newaxis, :, :], axis=-1)

        # 找到 color_points 中距离 similar_points 中任意点小于等于阈值的点
        close_points_mask = distances <= max_distance
        mask = np.any(close_points_mask, axis=0)
        # 检查是否还有符合条件的点
        if not np.any(mask):
            break

        # 将符合条件的点添加到 result 中
        close_points = color_points[mask]
        result = np.vstack([result, close_points])
        similar_points = close_points
        color_points = color_points[~mask]

        # 找到与种子点相同颜色的点
    # similar_color_indices = [i for i, color in enumerate(colors) if np.linalg.norm(color - image[seed_point[1], seed_point[0]]) <= color_threshold]

    # 返回最终找到的相似颜色的点
    return result


def delaunay(similar_color_points, image):
    # 创建一个空白图像
    img = np.zeros_like(image)

    # 计算 Delaunay 三角化
    try:
        tri = Delaunay(similar_color_points)
    except BaseException:
        return img

    # 获取三角形的边缘
    edges = set()  # 保存每个三角形的三条边
    for simplex in tri.simplices:
        for i in range(3):
            edge = (simplex[i], simplex[(i + 1) % 3])
            edge = tuple(sorted(edge))
            edges.add(edge)

    # 计算每个边的长度
    edge_lengths = []
    for edge in edges:
        p1, p2 = edge
        length = np.linalg.norm(similar_color_points[p1] - similar_color_points[p2])
        edge_lengths.append((p1, p2, length))

    # 根据一定的 alpha 参数选择边界
    alpha = 10  # 调整此参数来控制边界的凹凸程度
    boundary_edges = [edge for edge in edge_lengths if edge[2] < alpha]

    # 绘制边界
    for edge in boundary_edges:
        p1, p2 = edge[:2]
        cv2.line(img, (int(similar_color_points[p1, 1]), int(similar_color_points[p1, 0])),
                 (int(similar_color_points[p2, 1]), int(similar_color_points[p2, 0])), (255, 255, 255), 1)
    return img


def fill(img_delaunay, pixel):
    # 转换为灰度图
    gray = img_delaunay[:, :, 0]

    # 二值化处理
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建与原图像相同大小的全0图像
    filled_image = np.zeros_like(img_delaunay)

    # 在白色图像上绘制轮廓（填充白色区域）
    cv2.drawContours(filled_image, contours, -1, pixel, thickness=cv2.FILLED)
    pixel_image_obj = np.argwhere(np.any(filled_image != 0, axis=-1))

    return pixel_image_obj


def point_fill_de(image_org):
    # gpu_image_org = cv2.cuda_GpuMat()
    # gpu_image_org.upload(image_org)
    image = image_org.copy()
    # gpu_image = cv2.cuda_GpuMat()
    # gpu_image.upload(image)
    image_result = image_org.copy()
    # gpu_image_result = cv2.cuda_GpuMat()
    # gpu_image_result.upload(image_result)
    # 有颜色值的点
    non_black_pixels = np.argwhere(np.any(image != 0, axis=-1))
    while np.size(non_black_pixels):
        # 在有颜色值的点中随机选择一个点作为种子点
        seed_point = non_black_pixels[0]
        pixel = image_org[seed_point[0], seed_point[1]]

        # 定义最大距离和颜色阈值
        max_distance = 10
        # color_threshold = 10

        # 调用函数找到相似颜色的点
        similar_color_points = find_similar_color_points(image, seed_point, max_distance)
        # 计算 Delaunay 三角化
        img_delaunay = delaunay(similar_color_points, image)
        if not np.size(np.argwhere(np.any(img_delaunay != 0, axis=-1))):
            image[similar_color_points[:, 0], similar_color_points[:, 1]] = 0

        # 填充图像
        pixel_image_obj = fill(img_delaunay, tuple(pixel))
        image_result[pixel_image_obj[:, 0], pixel_image_obj[:, 1]] = pixel
        image[pixel_image_obj[:, 0], pixel_image_obj[:, 1]] = 0
        non_black_pixels = np.argwhere(np.any(image != 0, axis=-1))
    # image_result = np.array(image_result.cpu())
    # cv2.imwrite(os.path.join(r'/data/xc/seg_calib', '000.png'), image_result)
    image_result = torch.from_numpy(image_result.astype(np.float32))
    # image_result = image_result.cuda()
    return image_result
