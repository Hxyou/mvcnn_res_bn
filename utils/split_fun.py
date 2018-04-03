import os
import sys
sys.path.append('../')
import utils.config
import os.path as osp
import pickle
import glob

def split_views():
    cfg = utils.config.config()
    train = get_filenames('train', cfg.data_views_root)
    test = get_filenames('test', cfg.data_views_root)
    train = get_filename_list(train)
    test = get_filename_list(test)

    print('train num: %d'% len(train))
    print('test num: %d'% len(test))

    with open(cfg.split_train, 'wb') as f:
        pickle.dump(train, f)
    with open(cfg.split_test, 'wb') as f:
        pickle.dump(test, f)


def get_filename_list(data_raw):
    data_list = []
    for c in data_raw:
        lbl = c['label_idx']
        lbl_name = c['label']
        for shape in c['shapes']:
            data_list.append({
                'label': lbl,
                'label_name':lbl_name,
                'imgs':list(shape.values())[0],
                'shape_name': list(shape.keys())[0]
            })
    return data_list

def get_d_list(d_root):
    """
    get all structed filenames in one class
    :param d_root:
    :param data_views:
    :return:
    -bench_0001--
               |-/home/fyf/code/mvcnn/data/12_ModelNet40/bench/train/bench_0001_001.jpg
               |-/home/fyf/code/mvcnn/data/12_ModelNet40/bench/train/bench_0001_002.jpg
               |-......
    -bench_0002--
               |-......
    """
    full_names = glob.glob(osp.join(d_root, '*.jpg'))
    raw_structed_data = {}
    names = [osp.split(name)[1] for name in full_names]
    for _idx, name in enumerate(names):
        shape_name = name[:name.rfind('_')]
        if shape_name not in raw_structed_data:
            raw_structed_data[shape_name] = [full_names[_idx]]
        else:
            raw_structed_data[shape_name].append(full_names[_idx])

    structed_data = [{k: v} for k, v in raw_structed_data.items()]

    return structed_data


def get_filenames(data_type, root):
    filename_list = []
    data_all = glob.glob(osp.join(root, '*'))
    data_all = sorted(data_all)
    data_all = [data for data in data_all if osp.isdir(data)]
    for _idx, d in enumerate(data_all):
        d_lbl = osp.split(d)[1]
        d_lbl_idx = _idx
        d_root = osp.join(root, d_lbl, data_type)
        d_list = get_d_list(d_root)
        filename_list.append({'label': d_lbl,
                              'label_idx': d_lbl_idx,
                              'shapes': d_list})
    return filename_list



if __name__ == '__main__':
    random_split()