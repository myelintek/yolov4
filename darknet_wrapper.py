import os
import glob
import argparse
import subprocess
from string import Template
from shutil import copyfile

from mlsteam import stparams

MODEL_CFG_TEMPLATE='/mlsteam/lab/cfg/yolov4-custom.cfg.template'
if stparams.get_value('version') == 'tiny':
    MODEL_CFG_TEMPLATE='/mlsteam/lab/cfg/yolov4-tiny-custom.cfg.template'

DATA_CFG_TEMPLATE="""classes = ${num_classes}
train  = ${train_list}
valid  = ${valid_list}
names = ${names}
backup = ${backup}
eval = ${eval}
"""
MODEL_CFG='/mlsteam/lab/cfg/mls.cfg'
DATA_CFG='/mlsteam/lab/cfg/mls.data'
TRAIN_LIST='/mlsteam/lab/mlst_train_list.txt'
DEFAULT_TRAIN_FILE_LIST='/mlsteam/lab/mls_train_list.txt'
DEFAULT_VALID_FILE_LIST='/mlsteam/lab/mls_valid_list.txt'
DEFAULT_TRAIN_DIR='/mlsteam/data/training_data/yolo/images'

def gen_file_list(dir_name, file_name="file_list.txt"):
    with open(file_name, 'w') as f:
        for filename in glob.iglob('%s/**/*' % dir_name, recursive=True):
            f.write('%s\n' % os.path.abspath(filename))
    return file_name

def write_model_cfg(batch, subdivisions, learning_rate, max_batches, steps, scales, num_classes, filters, output_cfg=MODEL_CFG):
    with open(MODEL_CFG_TEMPLATE, 'r') as f:
        t = Template(f.read())
    cfg = t.substitute(
        batch=batch,
        subdivisions=subdivisions,
        learning_rate=learning_rate,
        max_batches=max_batches,
        steps=steps.replace(';', ','),
        scales=scales.replace(';', ','),
        num_classes=num_classes,
        filters=filters,
    )
    with open(output_cfg, 'w') as f:
        f.write(cfg)
    return output_cfg

def write_data_cfg(num_classes, train_list, valid_list, names, backup, eval_method, output_cfg=DATA_CFG):
    t = Template(DATA_CFG_TEMPLATE)
    
    cfg = t.substitute(
        num_classes=num_classes,
        train_list=train_list,
        valid_list=valid_list,
        names=names,
        backup=backup,
        eval=eval_method,
    )
    with open(output_cfg, 'w') as f:
        f.write(cfg)
    return output_cfg

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
    
def train_run(pretrained_weights, output_weights=None):
    names_file = stparams.get_value('names', '/mlsteam/data/obj.names')
    number_classes = num_classes(names_file)
    filters = int((number_classes + 5) * 3)
    
    model_cfg = write_model_cfg(
        batch=stparams.get_value('batch', 64),
        subdivisions=stparams.get_value('subdivisions', 64),
        learning_rate=stparams.get_value('learning_rate', 0.001),
        max_batches=stparams.get_value('max_batches', 1),
        steps=stparams.get_value('steps', '90000, 95000'),
        scales=stparams.get_value('scales', '.1, .1'),
        num_classes=number_classes,
        filters=filters, # num_mask *(num_class+5) = 3*(1+5)
    )

    train_list = stparams.get_value('train_list', None)
    if train_list is None:
        train_dir = stparams.get_value('train_dir', DEFAULT_TRAIN_DIR)
        if train_dir is None:
            raise ValueError('Must define train_list or train_dir')
        train_list = gen_file_list(train_dir, DEFAULT_TRAIN_FILE_LIST)

    valid_list = stparams.get_value('valid_list', None)
    if valid_list is None:
        valid_dir = stparams.get_value('valid_dir', None)
        if valid_dir is None:
            valid_list = DEFAULT_VALID_FILE_LIST
        else:
            valid_list = gen_file_list(valid_dir, DEFAULT_VALID_FILE_LIST)

    data_cfg = write_data_cfg(
        num_classes=number_classes,
        train_list=train_list,
        valid_list=valid_list,
        names=names_file,
        backup=stparams.get_value('backup', '/mlsteam/data/model_weights/trained'),
        eval_method=stparams.get_value('eval', 'coco')
    )

    train(data_cfg, model_cfg, pretrained_weights=pretrained_weights)
    if output_weights:
        backup_path = stparams.get_value('backup', '/mlsteam/data/model_weights/trained')
        weights_path = '%s/%s_final.weights' % (backup_path, os.path.basename(model_cfg).split('.')[0])
        copyfile(weights_path, output_weights)

def main(): 
    repeat = stparams.get_value('repeat', None)
    keep_weights_when_repeat = stparams.get_value('keep_weights_when_repeat', True)
    pretrained_weights = stparams.get_value('weights_file', None)
    
    if repeat:
        backup_path = stparams.get_value('backup', '/mlsteam/data/model_weights/trained')
        for i in range(1, repeat + 1):
            if i == 1:
                pretrained = pretrained_weights
            else:
                pretrained = '%s/mls_final_run_%d.weights' % (backup_path, i - 1)
                
            trained = '%s/mls_final_run_%d.weights' % (backup_path, i)
            train_run(pretrained, trained)
    else:
        train_run(pretrained_weights)

if __name__ == '__main__':
    main()

