import sys
sys.path.append('../')
import os
import os.path as osp
import configparser

class config(object):
    def __init__(self, cfg_file='config/config.cfg'):
        super(config, self).__init__()
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)

        # default
        self.data_root = cfg.get('DEFAULT', 'data_root')
        self.data_3d_root = cfg.get('DEFAULT', 'data_3d_root')
        self.data_views_root = cfg.get('DEFAULT', 'data_views_root')
        self.data_points_root = cfg.get('DEFAULT', 'data_points_root')
        self.result_root = cfg.get('DEFAULT', 'result_root')
        self.class_num = cfg.getint('DEFAULT', 'class_num')
        self.point_each = cfg.getint('DEFAULT', 'point_each')
        self.model_type = cfg.get('DEFAULT', 'model_type')

        # train
        self.cuda = cfg.getboolean('TRAIN', 'cuda')
        self.batch_size = cfg.getint('TRAIN', 'batch_size')

        self.result_sub_folder = cfg.get('TRAIN', 'result_sub_folder')
        self.ckpt_folder = cfg.get('TRAIN', 'ckpt_folder')
        self.split_folder = cfg.get('TRAIN', 'split_folder')

        self.split_train = cfg.get('TRAIN', 'split_train')
        self.split_test = cfg.get('TRAIN', 'split_test')
        self.ckpt_model = cfg.get('TRAIN', 'ckpt_model')
        self.ckpt_optim = cfg.get('TRAIN', 'ckpt_optim')

        self.ckpt_view_model = cfg.get('TRAIN', 'ckpt_view_model')

        self.check_dirs()

    def check_dir(self, folder):
        if not osp.exists(folder):
            os.mkdir(folder)

    def check_dirs(self):
        self.check_dir(self.data_views_root)
        self.check_dir(self.data_points_root)
        self.check_dir(self.result_root)
        self.check_dir(self.result_sub_folder)
        self.check_dir(self.ckpt_folder)
        self.check_dir(self.split_folder)
