#! /usr/bin/env python

import argparse
import os,glob,logging
import numpy as np
# from preprocessing import parse_annotation
from frontend import YOLO
import json
logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):
   config_path = args.conf

   with open(config_path) as config_buffer:
      config = json.loads(config_buffer.read())

   glob_str = config['train']['train_image_folder'] + '/*.npz'
   filelist = glob.glob(glob_str)

   logger.info('train_image_folder =   %s',glob_str)
   logger.info('filelist =             %s',filelist)



   ###############################
   #   Parse the annotations
   ###############################
   '''
   # parse annotations of the training set
   train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                             config['train']['train_image_folder'],
                                             config['model']['labels'])

   # parse annotations of the validation set, if any, otherwise split the training set
   if os.path.exists(config['valid']['valid_annot_folder']):
     valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                 config['valid']['valid_image_folder'],
                                                 config['model']['labels'])
   else:'''
   train_valid_split = int(0.8 * len(filelist))
   np.random.shuffle(filelist)

   valid_imgs = filelist[train_valid_split:]
   train_imgs = filelist[:train_valid_split]
   
   logger.info('Length of filelist:      %s',len(filelist))
   logger.info('Length of train_images:  %s',len(train_imgs))
   logger.info('Length of valid_imgs:    %s',len(valid_imgs))

   ###############################
   #   Construct the model
   ###############################

   yolo = YOLO(backend          = config['model']['backend'],
            input_shape         = config['model']['input_shape'],
            labels              = config['model']['labels'],
            max_box_per_image   = config['model']['max_box_per_image'],
            anchors             = config['model']['anchors'])


   ###############################
   #   Start the training process
   ###############################

   yolo.train(train_imgs       = train_imgs,
            valid_imgs         = valid_imgs,
            evts_per_file      = config['train']['evts_per_file'],
            train_times        = config['train']['train_times'],
            valid_times        = config['valid']['valid_times'],
            nb_epochs          = config['train']['nb_epochs'],
            learning_rate      = config['train']['learning_rate'],
            batch_size         = config['train']['batch_size'],
            warmup_epochs      = config['train']['warmup_epochs'],
            object_scale       = config['train']['object_scale'],
            no_object_scale    = config['train']['no_object_scale'],
            coord_scale        = config['train']['coord_scale'],
            class_scale        = config['train']['class_scale'],
            use_caching        = config['train']['use_caching'],
            saved_weights_name = config['train']['saved_weights_name'],
            debug              = config['train']['debug'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    args = argparser.parse_args()
    _main_(args)
