import os
import glob
import numpy as np
import math
import json
import trimesh
import argparse

from load_llfff import load_llff_data

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="root directory to the LLFF dataset (contains images/ and pose_bounds.npy)")
    parser.add_argument('--images', type=str, default='images_8', help="images folder (do not include full path, e.g., just use `images_4`)")
    parser.add_argument('--downscale', type=int, default=1, help="image size down scale, e.g., 4")
    parser.add_argument('--hold', type=int, default=8, help="hold out for validation every $ images")
    parser.add_argument('--grid', type=int, default=3, help="hold out for validation every $ images")

    opt = parser.parse_args()
    print(f'[INFO] process {opt.path}')

    # path must end with / to make sure image path is relative
    if opt.path[-1] != '/':
        opt.path += '/'
    images_test, poses, bds, render_poses, i_test,focal_depth = load_llff_data(opt.path, opt.downscale,
                                                                recenter=True, bd_factor=None)
    

    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
   

    H, W, focal = hwf
    H, W = int(H), int(W)

    # load data
    images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    

    N = poses.shape[0]
    print(f"NNNNN {N}")

    H = H // opt.downscale
    W = W // opt.downscale

    print(f'[INFO] H = {H}, W = {W}, fl = {focal} focal_depth = {focal_depth} (downscale = {opt.downscale})')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    x_min = poses[:,0,3].min()

    x_max = poses[:,0,3].max()

    y_min = poses[:,1,3].min()

    y_max = poses[:,1,3].max()

    z_min = poses[:,2,3].min()

    z_max = poses[:,2,3].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2


    ranges = [(x_range, 0), (y_range, 1), (z_range, 2)]

    # 차이가 큰 순서대로 정렬
    sorted_ranges = sorted(ranges, key=lambda x: x[0], reverse=True)

    # 상위 두 개 차원 인덱스를 설정
    length_max = sorted_ranges[0][0]
    
    x_index = sorted_ranges[0][1]
    y_index = sorted_ranges[1][1]



    def normalize(values, center, length):
            # 중심점 기준으로 정규화 실행
            return 2 * (values - center) / length

    poses[:, 0, 3] = normalize(poses[:, 0, 3], x_center, length_max)
    poses[:, 1, 3] = normalize(poses[:, 1, 3], y_center, length_max)
    poses[:, 2, 3] = normalize(poses[:, 2, 3], z_center, length_max)


    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 
    print("240513_test!!!!!!!!!!!!!!!!!!")

    poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
    poses_torch = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)

    # visualize_poses(poses)

    # the following stuff are from colmap2nerf... [flower fails, the camera must be in-ward...]
    # poses[:, 0:3, 1] *= -1
    # poses[:, 0:3, 2] *= -1
    # poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    # poses[:, 2, :] *= -1 # flip whole world upside down

    # up = poses[:, 0:3, 1].sum(0)
    # up = up / np.linalg.norm(up)
    # R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    # R = np.pad(R, [0, 1])
    # R[-1, -1] = 1

    # poses = R @ poses

    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # for i in range(N):
    #     mf = poses[i, :3, :]
    #     for j in range(i + 1, N):
    #         mg = poses[j, :3, :]
    #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
    #         #print(i, j, p, w)
    #         if w > 0.01:
    #             totp += p * w
    #             totw += w
    # totp /= totw
    # print(f'[INFO] totp = {totp}')
    # poses[:, :3, 3] -= totp
    # avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    # poses[:, :3, 3] *= 4.0 / avglen
    # print(f'[INFO] average radius = {avglen}')

#jw
    # positions = poses[:, :3, 3]

    # # 최대 및 최소 값 계산
    # max_pos = positions.max(axis=0)
    # min_pos = positions.min(axis=0)

    # # 분모가 0이 되는 것을 방지
    # scale = max_pos - min_pos
    # scale[scale == 0] = 1  # 0인 경우 1로 설정하여 나눗셈에서 0으로 나누는 것을 방지

    # # 각 축별로 정규화 수행
    # normalized_positions = 2 * (positions - min_pos) / scale - 1

    # # 정규화된 위치 벡터를 원래의 포즈에 다시 할당
    # poses[:, :3, 3] = normalized_positions
#jw


    # construct frames
    def print_min_max_values(poses):
        x_min = poses[:, 0, 3].min()
        x_max = poses[:, 0, 3].max()

        y_min = poses[:, 1, 3].min()
        y_max = poses[:, 1, 3].max()

        z_min = poses[:, 2, 3].min()
        z_max = poses[:, 2, 3].max()

        print(f"x_min: {x_min}")
        print(f"x_max: {x_max}")
        print(f"y_min: {y_min}")
        print(f"y_max: {y_max}")
        print(f"z_min: {z_min}")
        print(f"z_max: {z_max}")

# 설정된 모드에 따라 처리
    mode = opt.grid
    row = 0 
    col = 0
    inds = []
    if mode == 3:
        inds = [0, 8, 16]
        test_ind = [4]
    elif mode == 5:
        inds = [0, 4, 8, 12, 16]
        test_ind = [13]
    elif mode == 9:
        inds = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        test_ind = [41]

    whole_num = mode * mode
    all_ids = np.arange(whole_num)
    test_ids = [all_ids[i] for i in test_ind]
    train_ids = np.array([i for i in all_ids])
    images_resized = []
    poses_resized = []

    #breakpoint()
    for i in range(poses.shape[0]):
        row = i % 17
        col = i // 17
        if row in inds and col in inds:
            images_resized.append(images[i])
            poses_resized.append(poses[i,:,:])
    images = images_resized
    poses = np.array(poses_resized)

    # 최소 및 최대 값 계산
    x_min = poses[:, 0, 3].min()
    x_max = poses[:, 0, 3].max()
    y_min = poses[:, 1, 3].min()
    y_max = poses[:, 1, 3].max()
    z_min = poses[:, 2, 3].min()
    z_max = poses[:, 2, 3].max()

    # 중심 및 범위 계산
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # 범위 출력
    # print(f"x_min: {x_min}")
    # print(f"x_max: {x_max}")
    # print(f"y_min: {y_min}")
    # print(f"y_max: {y_max}")
    # print(f"z_min: {z_min}")
    # print(f"z_max: {z_max}")

    # print(f"x_range: {x_range}")
    # print(f"y_range: {y_range}")
    # print(f"z_range: {z_range}")

    # print(f"x_center: {x_center}")
    # print(f"y_center: {y_center}")
    # print(f"z_center: {z_center}")

    # 각 축에 대해 개별적으로 정규화 함수 적용
    def normalize(values, center, length):
        return 2 * (values - center) / length

    # 각 축에 대해 개별적으로 정규화 수행
    poses[:, 0, 3] = normalize(poses[:, 0, 3], x_center, x_range)
    poses[:, 1, 3] = normalize(poses[:, 1, 3], y_center, y_range)
    poses[:, 2, 3] = normalize(poses[:, 2, 3], z_center, z_range)

    # 정규화 후 최소 및 최대 값 출력
    print_min_max_values(poses)

    
    #breakpoint()
    frames_train = []
    frames_test = []
    for i in train_ids:
        frames_train.append({
            'file_path': images[i],
            'transform_matrix': poses[i].tolist(),
        })
    for i in test_ids:
        frames_test.append({
            'file_path': images[i],
            'transform_matrix': poses[i].tolist(),
        })

    def write_json(filename, frames):

        # construct a transforms.json
        out = {
            'w': W,
            'h': H,
            'fl_x': focal,
            'fl_y': focal,
            'cx': W // 2,
            'cy': H // 2,
            'aabb_scale': 2,
            'frames': frames,
            'focal_depth':focal_depth,
            'x_index' : x_index,
            'y_index' : y_index,
        }

        # write
        output_path = os.path.join(opt.path, filename)
        print(f'[INFO] write {len(frames)} images to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2 , default=convert_to_serializable)

    write_json('transforms_train.json', frames_train)
    write_json('transforms_val.json', frames_test[::10])
    write_json('transforms_test.json', frames_test)

