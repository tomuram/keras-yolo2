#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from tqdm import tqdm
# from preprocessing import parse_annotation
from utils import draw_boxes, BoundBox
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend          = config['model']['backend'],
        input_shape         = config['model']['input_shape'],
        labels              = config['model']['labels'],
        max_box_per_image   = config['model']['max_box_per_image'],
        anchors             = config['model']['anchors'])


    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    fig,ax=plt.subplots(1)
# bbox x                    # globe eta
# bbox y                    # globe phi
# bbox width             # Gaussian sigma (required to be 3*sigma<pi)
# bbox height            # Gaussian sigma (required to be 3*sigma<pi)

    file_content = np.load(image_path)
    images = file_content['raw']
    truth_boxes = file_content['truth']
    for image_index in range(10):
        image = images[image_index]
        all_objs = truth_boxes[image_index]

        print(image.shape)
        boxes = yolo.predict(image)
        print(len(boxes), 'boxes are found')
        for i in range(len(boxes)):
            b = boxes[i]
            print('box:',i,b)
        draw_boxes(image, ax, boxes, config['model']['labels'],color='y',scale=True)

        obj_boxes=[]
        i=0
        for obj in all_objs:
            # x,y,w,h = obj[:4]
            y,x,h,w = obj[1:5]
            b = BoundBox(x-w/2,y-h/2,x+w/2,y+h/2)
            # print('box:',i,b,obj[5],obj[6],obj[7],obj[8],obj[9])
            print('box:',i,obj)
            obj_boxes.append( b )
            i+=1
        draw_boxes(image, ax, obj_boxes, config['model']['labels'],color='g',scale=False)

        #image = draw_boxes(image, boxes, config['model']['labels'])
        i=np.swapaxes(np.swapaxes(image,0,1),1,2)
        x=np.sum(i,axis=2)
        
        #plt.imshow(x,cmap='hot')
        plt.imshow(x,aspect='auto',extent=(0,256,0,9600),interpolation='nearest',cmap=cm.jet)        
        plt.savefig('out%d.png' % image_index ,dpi=200)
        #plt.show()


    #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
