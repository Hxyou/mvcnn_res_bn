import os
import os.path as osp
import glob
from random import shuffle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_info(shape_dir):
    splits = shape_dir.split('/')
    class_name = splits[-3]
    set_name = splits[-2]
    file_name = splits[-1].split('.')[0]
    return class_name, set_name, file_name

def expend_triangle(a, b, c):
    pass

def normal_pc(pc, L):
    pc_new = []
    pc = np.array(pc)



def get_pc(shape, point_each):
    points = []
    points_expend = []
    faces = []
    with open(shape, 'r') as f:
        line = f.readline().strip()
        if line == 'OFF':
            num_verts, num_faces, num_edge = f.readline().split()
            num_verts = int(num_verts)
        else:
            num_verts, num_faces, num_edge = line[4:].split()
            num_verts = int(num_verts)

        for idx in range(num_verts):
            line = f.readline()
            point = [float(v) for v in line.split()]
            points.append(point)

        for idx in range(num_faces):
            line = f.readline()
            face = [int(f) for f in line.split()]
            faces.append(face[1:])



    shuffle(points)

    return points[: point_each]


def generate(cfg):
    shape_all = glob.glob(osp.join(cfg.data_3d_root, '*', '*', '*.off'))
    for shape in tqdm(shape_all):
        pc = get_pc(shape, cfg.point_each)
        class_name, set_name, file_name = get_info(shape)
        pc = np.array(pc)
        new_folder = osp.join(cfg.data_points_root, class_name, set_name)
        if not osp.exists(new_folder):
            os.makedirs(new_folder)
        new_dir = osp.join(new_folder, file_name+'.npy')
        np.save(new_dir, pc)



if __name__ == '__main__':
    file_name = '/home/fyf/code/data/pc_ModelNet40/table/test/table_0428.npy'
    ps = np.load(file_name)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2])
    plt.show()

