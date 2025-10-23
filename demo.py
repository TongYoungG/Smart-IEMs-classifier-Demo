'''
Author: Eric
Date: 2023-01-29 16:32:37
LastEditors: Eric
LastEditTime: 2023-05-16 15:37:21
'''
#coding:utf-8
import argparse
import os
import time
import torch
import numpy as np

from loguru import logger
from easydict import EasyDict
from utils.demo_utils import Predictor,VOC_CLASSES,EAR_POS
from utils.demo_utils import get_model,get_model_info,extract_temporal_bone_img
from utils.setting import parse_opts
from utils.model import generate_model
from scipy import ndimage
import pdb

def init_tb_detection_model(model_path):
    ckpt_file = './weights/best_ckpt.pth'
    
    # attr
    attr = EasyDict()
    attr.num_classes = 2
    attr.test_conf = 0.25
    attr.nmsthre = 0.45
    attr.test_size = (640,640)
    attr.depth = 0.33
    attr.width = 0.50
    attr.act = 'silu' # 'relu'
    attr.fp16 = False
    attr.device = 'cpu'
    attr.save_result = True
    
    # init model
    model = get_model(attr)
    logger.info("Model Summary: {}".format(get_model_info(model, attr.test_size)))
    if torch.cuda.is_available():
        model.cuda()
        if attr.fp16:
            model.half()
        attr.device = 'gpu'
    model.eval()
    ckpt = torch.load(model_path,map_location='cpu')
    model.load_state_dict(ckpt['model'])
    logger.info('loaded checkpoint done.')
    
    # predictor
    cur_time = time.localtime()
    predictor = Predictor(model, attr, VOC_CLASSES)
    
    return predictor
    
def IEM_Cls_model_init(model_path,n_class=2):
    # settting
    sets = parse_opts()
    sets.task = 'cls'
    sets.phase = 'test'
    sets.batch_size = 1
    sets.input_D = 112
    sets.input_H = 112
    sets.input_W = 112
    sets.model_depth = 34
    sets.n_classes = n_class
    sets.gpu_id = '0'
    sets.resume_path = model_path
    
    checkpoint = torch.load(sets.resume_path)
    model, _ = generate_model(sets)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model,sets

def nii2tensorarray(data):
    [z, y, x] = data.shape
    new_data = np.reshape(data, [1,1, z, y, x])
    new_data = new_data.astype("float32")

    return new_data

def resize_data(data,sets):
    [depth, height, width] = data.shape
    scale = [sets.input_D * 1.0 / depth, sets.input_H * 1.0 / height, sets.input_W * 1.0 / width]
    data = ndimage.interpolation.zoom(data, scale, order=0)
    return data

if __name__ == '__main__':
    IEM_class_name = ['CA_RO_CLA','CC','CH','IP-1','IP-2','VSCM','CCAA']

    # test_imgfile
    nii_filepath = './nii_test_file/IP-2/BAI CHAO LE MEN_Y3840475_20181121_UIH_201.nii.gz'    
    # tb det model init 
    tb_det_model_path = "./weights/yolox-tb-detection.pth"
    predictor = init_tb_detection_model(tb_det_model_path)
    tb_psize = [50,50,50] # temporal bone phy size (mm)
    
    # IEM cls model init
    # IEM_2cls_model_path = "./weights/resnet_34_epoch_55_batch_0.pth.tar"
    # IEM_2Classfier, sets_2cls = IEM_Cls_model_init(IEM_2cls_model_path,n_class=2)
    
    IEM_multicls_model_path = "./weights/IEM-multi-classifier.tar"
    IEM_MlClassfier,sets_mcls = IEM_Cls_model_init(IEM_multicls_model_path,n_class=8)
    
    # start process
    print("Start process nii file: {}".format(nii_filepath))
    tb_nii_imgs = extract_temporal_bone_img(nii_filepath,predictor,tb_psize,topk=1)
    if len(tb_nii_imgs.keys()) == 0:
        print("No temporal bone detected.")
    for ear_side in tb_nii_imgs.keys():
        tb_nii_img = tb_nii_imgs[ear_side]
        if tb_nii_img is None:
            print("No {} temporal bone detected.".format(ear_side))
            continue
        # 2-classification
        resized_img = resize_data(tb_nii_img,sets_mcls)
        img_tensor = torch.tensor(nii2tensorarray(resized_img))
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        input_var = torch.autograd.Variable(img_tensor)  
        with torch.no_grad():
            # cls_2out = IEM_2Classfier(input_var)
            # cls_2pred = torch.argmax(cls_2out,dim=1).item()
            cls_2pred = 1  # For demo purpose, assume all are IEM
            if cls_2pred == 0:
                cls_pred_str = 'Non-IEM'
            else:
                cls_mout = IEM_MlClassfier(input_var)
                cls_mpred = torch.argmax(cls_mout,dim=1).item()
                cls_pred_str = IEM_class_name[cls_mpred]
                
        print("{} ear temporal bone IEM classification results: {}".format(ear_side,cls_pred_str))