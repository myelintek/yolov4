import os
import glob
import pathlib
import argparse
import subprocess
from string import Template
from shutil import copyfile

from mlsteam import stparams


DEFAULT_MODEL_CFG = '/mlsteam/lab/cfg/mls.cfg'
DEFAULT_DATA_CFG = '/mlsteam/lab/cfg/mls.data'
DEFAULT_DATA_DIR = '/mlsteam/data/yolo/training_data/yolo'  # container images labels folder and obj.names config
DEFAULT_VALID_FILE_LIST = '/mlsteam/lab/mls_valid_list.txt'
DEFAULT_OUTPUT_FOLDER = '/mlsteam/data/yolo/model_weights/traind' # will output weight and cfg file
TRAIN_LIST_FILENAME = 'train.txt' # if exists read it, else parse data_dir and generate
VALID_LIST_FILENAME = 'valid.txt' # if exists read it, else parse data_dir and genrate? or leave it blank.
MODEL_CFG_NAME ='mls.cfg'
DATA_CFG_NAME = 'obj.data'
MODEL_CFG_ORIG_TEMPLATE = '/mlsteam/lab/cfg/yolov4-custom.cfg.template'
MODEL_CFG_TINY_TEMPLATE = '/mlsteam/lab/cfg/yolov4-tiny-custom.cfg.template'
DATA_CFG_TEMPLATE = """classes = ${num_classes}
train  = ${train_list}
valid  = ${valid_list}
names = ${names}
backup = ${backup}
eval = ${eval}
"""
INFERENCE_CFG_TEMPLATE = """classes = ${num_classes}
names = ${names}
"""

def log_summary(msg):
    print(f'MLSTEAM_SUMMARY {msg}')

def gen_file_list(dir_name, file_name="file_list.txt", exts=['jpg', 'png']):
    with open(file_name, 'w') as f:
        for filename in pathlib.Path(dir_name).rglob('*'):
            if exts and os.path.splitext(filename)[1].replace('.', '', 1) not in exts:
                continue
            f.write('%s\n' % os.path.abspath(filename))
    return file_name


def write_cfg(template, output_cfg, **kwargs):
    if os.path.isfile(template):
        with open(template, 'r') as f:
            t = Template(f.read())
    else:
        t = Template(template)
    cfg = t.substitute(**kwargs)
    with open(output_cfg, 'w') as f:
        f.write(cfg)
    return output_cfg


def write_model_cfg(output_cfg=DEFAULT_MODEL_CFG, **kwargs):
    if stparams.get_value('version') == 'tiny':
        model_file = MODEL_CFG_TINY_TEMPLATE
    else:
        model_file = MODEL_CFG_ORIG_TEMPLATE
    return write_cfg(model_file, output_cfg, **kwargs)


def write_data_cfg(output_cfg=DEFAULT_DATA_CFG, **kwargs):
    return write_cfg(DATA_CFG_TEMPLATE, output_cfg, **kwargs)


def write_inf_cfg(output_cfg, **kwargs):
    return write_cfg(INFERENCE_CFG_TEMPLATE, output_cfg, **kwargs)


def train(data_cfg, model_cfg, pretrained_weights=None):
    if pretrained_weights not in [None, '']:
        if not os.path.exists(pretrained_weights):
            raise ValueError('Pretrained_weights {} not found.'.format(pretrained_weights))
        cmd = ['/mlsteam/lab/darknet', 'detector', 'train', data_cfg, model_cfg, pretrained_weights]
    else:
        cmd = ['/mlsteam/lab/darknet', 'detector', 'train', data_cfg, model_cfg]
    os.system(' '.join(cmd))

    
def num_classes(names_file):
    f = open(names_file, 'r')
    lines = f.readlines()
    return len(lines)


def get_input_dirs(data_dir, default_names='/mlsteam/data/yolo/obj.names'):
    if not data_dir or not os.path.exists(data_dir):
        raise ValueError('data_dir {} not exist'.format(data_dir))

    image_dir = data_dir
    if not os.path.exists(data_dir):
        raise ValueError('directory images not in {}'.format(data_dir))

    names_file = os.path.join(data_dir, 'obj.names')
    if os.path.exists(names_file):
        print(f'label names config exists, load config {names_file}')
    elif os.path.exists(default_names):
        print('label names config not in data_dir {} , read {}'.format(data_dir, default_names))
        names_file = default_names
    else:
        raise ValueError('label names config {} not exists'.format(names_file))
    return image_dir, names_file

def ensure_img_list(list_file, target_dir, img_exts):
    if not os.path.exists(list_file):
        print("Config {} not exist, try to generate from {}.".format(list_file, target_dir))
        gen_file_list(target_dir, list_file, img_exts)
    else:
        print("Config {} exists".format(list_file))

def train_run(pretrained_weights, output_dir, output_weights=None):
    # read obj names get num class
    train_dir = stparams.get_value('train_dir','/mlstea/data/yolo/training_data/yolo')
    param_names = stparams.get_value('names', '/mlsteam/data/yolo/obj.names')
    image_dir, names_file = get_input_dirs(train_dir, param_names)
    max_batches = int(stparams.get_value('max_batches', 500))
    number_classes = num_classes(names_file)
    filters = int((number_classes + 5) * 3)
    
    num_epoch_save = '1,000' if max_batches < 10000 else '10,000'
    log_summary(f"Number of classes: {number_classes}")
    log_summary(f"Output weight policy: every {num_epoch_save} batch")
    log_summary("   if max_batcheds < 10,000, save weights every 1,000 batch")
    log_summary("   if max_batcheds >= 10,000, save weights every 10,000 batch")
    
    # Prepare train_list.txt
    train_list = os.path.join(train_dir, TRAIN_LIST_FILENAME)
    ensure_img_list(train_list, image_dir, 
                    stparams.get_value('image_exts', 'jpg;png').split(';'))
    
    # Prepare valid_list.txt if valid_dir present
    valid_dir = stparams.get_value('valid_dir', None)
    valid_list = os.path.join(train_dir, VALID_LIST_FILENAME)
    if not os.path.exists(valid_list):
        print("Validate list not exist, try to scan valid_dir {}".format(valid_dir))
        if valid_dir in ['', None]:
            print ("Parameter valid_dir not specify! skip validation")
            valid_list='no_valid_list'
        else:
            image_dir, names_file = get_input_dirs(valid_list, param_names)
            valid_list = ensure_img_list(valid_list, image_dir)

    cfg_dir = os.path.join(output_dir, 'cfg')
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    model_cfg = write_model_cfg(
        output_cfg=os.path.join(cfg_dir, MODEL_CFG_NAME),
        batch=stparams.get_value('batch', 64),
        subdivisions=stparams.get_value('subdivisions', 64),
        learning_rate=stparams.get_value('learning_rate', 0.001),
        max_batches=max_batches,
        steps=stparams.get_value('steps', '400, 450').replace(';', ','), # comma(,) is mlsteam parameter reserved character
        scales=stparams.get_value('scales', '.1, .1').replace(';', ','), # comma(,) is mlsteam parameter reserved character
        num_classes=number_classes,
        filters=filters, # num_mask *(num_class+5) = 3*(1+5)
    )
    data_cfg = os.path.join(train_dir, DATA_CFG_NAME)
    data_cfg = write_data_cfg(
        output_cfg=data_cfg,
        num_classes=number_classes,
        train_list=train_list,
        valid_list=valid_list,
        names=names_file,
        backup=output_dir,
        eval=stparams.get_value('eval', 'coco')
    )
    
    #prepare config for inferencing
    write_inf_cfg(
        output_cfg=os.path.join(cfg_dir, DATA_CFG_NAME),
        num_classes=number_classes,
        names=os.path.join('cfg', os.path.basename(names_file))
    )
    copyfile(names_file, os.path.join(cfg_dir, os.path.basename(names_file)))

    train(data_cfg, model_cfg, pretrained_weights=pretrained_weights)
    
    if output_weights:
        weights_path = '%s/%s_final.weights' % (output_dir, os.path.basename(model_cfg).split('.')[0])
        copyfile(weights_path, output_weights)


def main():
    repeat = stparams.get_value('repeat', None)
    output_dir = stparams.get_value('outpu_dir', 
                                    stparams.get_value('backup', '/mlsteam/data/yolo/model_weights/trained'))
    pretrained_weights = stparams.get_value('weights_file', None)

    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if repeat:
        repeat = int(repeat)
        for i in range(1, repeat + 1):
            if i == 1:
                pretrained = pretrained_weights
            else:
                pretrained = '%s/mls_final_run_%d.weights' % (output_dir, i - 1)
                
            trained = '%s/mls_final_run_%d.weights' % (output_dir, i)
            train_run(pretrained, output_dir, trained)
    else:
        train_run(pretrained_weights, output_dir)

if __name__ == '__main__':
    main()

