'''
Author: Eric
Date: 2023-01-29 16:51:36
LastEditors: Eric
LastEditTime: 2023-05-16 16:10:50
'''
import os
import time
import cv2
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import SimpleITK as sitk

from enum import Enum
from loguru import logger
from copy import deepcopy
from typing import Sequence

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis

import pdb

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class EAR_POS(Enum):
    LEFT=0,
    RIGHT=1,

class COMPUTE_MODE(Enum):
    UNKNOWN=0,
    JOINT=1,
    HSC=2,
    IAC=3,

VOC_CLASSES = (
    "hsc", # 水平半规管
    "iac", # 内耳道
)

def get_nii_info(nii_filepath):
    data = sitk.ReadImage(nii_filepath)
    origin = data.GetOrigin()
    spacing = data.GetSpacing()
    dimensions = data.GetSize()
    
    origin_str = '{:.2f}x{:.2f}x{:.2f}'.format(origin[0],origin[1],origin[2])
    spacing_str = '{:.4f}x{:.4f}x{:.4f}'.format(spacing[0],spacing[1],spacing[2])
    dimensions_str = '{}x{}x{}'.format(dimensions[0],dimensions[1],dimensions[2])
    
    return origin_str,spacing_str,dimensions_str

def extract_nii_info(nii_root_dir,csv_file):
    cols = ['ID','Path','Left_Dimensions','Right_Dimensions','VoxelSpacing','Origin']
    rows_values = []
    nii_dirs = os.listdir(nii_root_dir)
    for nii_dir in nii_dirs:
        row_values = []
        
        nii_id = nii_dir
        nii_dirpath = os.path.join(nii_root_dir,nii_dir)
        
        l_nii_file = '{}_left.nii.gz'.format(nii_dir)
        r_nii_file = '{}_right.nii.gz'.format(nii_dir)

        l_nii_filepath = os.path.join(nii_dirpath,l_nii_file)
        r_nii_filepath = os.path.join(nii_dirpath,r_nii_file)
        # pdb.set_trace()
        origin_str,spacing_str,l_dimensions_str = get_nii_info(l_nii_filepath)
        _,_,r_dimensions_str = get_nii_info(r_nii_filepath)
        
        rows_values.append([nii_dir,nii_dirpath,l_dimensions_str,r_dimensions_str,spacing_str,origin_str])
    
    data = pd.DataFrame(rows_values,columns=cols)
    data.to_csv(csv_file,encoding='utf-8-sig',index=None)
    
def extract_nii_info2(nii_root_dir,csv_file):
    cols = ['ID','Path','Left_Dimensions','Right_Dimensions','VoxelSpacing','Origin']
    rows_values = []
    nii_dirs = os.listdir(nii_root_dir)
    for nii_dir in nii_dirs:
        row_values = []
        
        nii_id = nii_dir
        nii_dirpath = os.path.join(nii_root_dir,nii_dir)
        
        l_nii_file = '{}_left.nii.gz'.format(nii_dir)
        r_nii_file = '{}_right.nii.gz'.format(nii_dir)

        l_nii_filepath = os.path.join(nii_dirpath,l_nii_file)
        r_nii_filepath = os.path.join(nii_dirpath,r_nii_file)
        # pdb.set_trace()
        origin_str,spacing_str,l_dimensions_str = get_nii_info(l_nii_filepath)
        _,_,r_dimensions_str = get_nii_info(r_nii_filepath)
        
        rows_values.append([nii_dir,nii_dirpath,l_dimensions_str,r_dimensions_str,spacing_str,origin_str])
    
    data = pd.DataFrame(rows_values,columns=cols)
    data.to_csv(csv_file,encoding='utf-8-sig',index=None)
    
def get_center_img(nii_img):
    d,h,w = nii_img.shape
    t_cnt_img = nii_img[int(d/2)+1,:,:] # transverse 水平面
    c_cnt_img = nii_img[:,int(h/2)+1,:] # coronal 冠状面
    s_cnt_img = nii_img[:,:,int(w/2)+1] # sagittal 矢状面
    return t_cnt_img, c_cnt_img, s_cnt_img

def draw_joint_results(nii_array,idxs,outputs,pos,save_path):
    nii_array_copy = copy.deepcopy(nii_array)
    if pos == EAR_POS.LEFT:
        pos_name = 'left'
    else:
        pos_name = 'right'
    for idx in idxs:
        opt_slice = nii_array_copy[idx,:,:]
        opt_output = outputs[idx,:]
        hsc_bbox = opt_output[:7].astype(np.int32)
        iac_bbox = opt_output[7:].astype(np.int32)
        
        save_slice = np.zeros((opt_slice.shape[0],opt_slice.shape[1],3))
        save_slice[:,:,0] = opt_slice
        save_slice[:,:,1] = opt_slice
        save_slice[:,:,2] = opt_slice
        # pdb.set_trace()
        save_filepath = os.path.join(save_path,'{}_{}.jpg'.format(pos_name,idx))
        cv2.rectangle(save_slice,(hsc_bbox[0],hsc_bbox[1]),(hsc_bbox[2],hsc_bbox[3]),(255,0,0),thickness=2)
        cv2.rectangle(save_slice,(iac_bbox[0],iac_bbox[1]),(iac_bbox[2],iac_bbox[3]),(0,255,255),thickness=2)
        cv2.imwrite(save_filepath,save_slice)
    
def temporal_bone_detection(outputs,ear_pos,input_size,tb_pixelsize,topk=1):
    hsc_probs = outputs[:,4]*outputs[:,5] # 水平半规管
    iac_probs = outputs[:,11]*outputs[:,12] # 内听道
    joint_probs = hsc_probs*iac_probs # 联合概率
    # pdb.set_trace()
    compute_mode = COMPUTE_MODE.UNKNOWN
    # 若联合概率均为0，则使用hsc的概率和检测结果来确定tb区域
    if np.sum(joint_probs) != 0:
        det_probs = joint_probs
        compute_mode = COMPUTE_MODE.JOINT
    else:
        if np.sum(hsc_probs) != 0:
            det_probs = hsc_probs
            compute_mode = COMPUTE_MODE.HSC
        elif np.sum(iac_probs) != 0:
            det_probs = iac_probs
            compute_mode = COMPUTE_MODE.IAC

    print('\tear_pose: {} compute_mode: {}'.format(ear_pos,compute_mode))
    
    tb_bbox = []
    valid_topk_idxs = []
    if compute_mode != COMPUTE_MODE.UNKNOWN:
        argsort_idxs = (-det_probs).argsort()
        topk_idxs = argsort_idxs[:topk]
        topk_tb_bboxes = np.zeros((topk,4),dtype=np.float32)
        
        for i,idx in enumerate(topk_idxs):
            if det_probs[idx] == 0:
                continue
            hsc_bbox = outputs[idx,:4]
            iac_bbox = outputs[idx,7:11]
            topk_tb_bboxes[i,:] = get_tb_region(hsc_bbox,iac_bbox,tb_pixelsize,ear_pos,compute_mode)
            valid_topk_idxs.append(idx)

        if len(valid_topk_idxs) != 0:
            # 计算topk概率下的最大颞骨边界
            topk_tb_bboxes = topk_tb_bboxes[topk_tb_bboxes[:,2]!=0]
            tb_bbox_xmin = max(int(np.min(topk_tb_bboxes[:,0])),0)
            tb_bbox_ymin = max(int(np.min(topk_tb_bboxes[:,1])),0)
            tb_bbox_xmax = min(int(np.max(topk_tb_bboxes[:,2])),input_size[1])
            tb_bbox_ymax = min(int(np.max(topk_tb_bboxes[:,3])),input_size[0])
            
            tb_bbox_h = tb_bbox_ymax - tb_bbox_ymin
            tb_bbox_w = tb_bbox_xmax - tb_bbox_xmin
            
            tb_bbox_newh = max(tb_bbox_w,tb_bbox_h)
            tb_bbox_neww = tb_bbox_newh
            tb_bbox_nxmax = tb_bbox_xmin+tb_bbox_neww
            tb_bbox_nymax = tb_bbox_ymin+tb_bbox_newh
            
            tb_bbox = [tb_bbox_xmin,tb_bbox_ymin,tb_bbox_nxmax,tb_bbox_nymax]
    return tb_bbox,valid_topk_idxs

def nii_process(nii_filepath, predictor, tb_physize, topk = 1, draw_vis = 0):
    nii_array,spacing,origin = get_array_from_nii(nii_filepath)
    
    d,h,w = nii_array.shape
    nii_c_w = w/2
    input_size = (h,w)
    
    l_outputs = np.zeros((d,14),dtype=np.float32)
    r_outputs = np.zeros((d,14),dtype=np.float32)
    input = np.zeros((h,w,3),dtype=np.uint8)
    valid_slice_idxs = []
    valid_r_idxs = []
    for i in range(d):
        slice_img = nii_array[i,:,:]
        input[:,:,0] = slice_img
        input[:,:,1] = slice_img
        input[:,:,2] = slice_img
        # pdb.set_trace()
        outputs,img_info = predictor.inference(input)
        if outputs[0] == None:
            continue
        valid_slice_idxs.append(i)
        
        output = outputs[0].cpu().numpy()
        output[:,0:4] = output[:,0:4] / img_info['ratio']
        pred_num = output.shape[0]
        for j in range(pred_num):
            # pdb.set_trace()
            cur_pred = output[j,:]
            xmin = cur_pred[0]
            ymin = cur_pred[1]
            xmax = cur_pred[2]
            ymax = cur_pred[3]
            c_w = (xmin+xmax) / 2
            pred_label = int(cur_pred[-1])
            if c_w < nii_c_w:
                r_outputs[i,pred_label*7:pred_label*7+7] = cur_pred
                valid_r_idxs.append(i)
            else:
                l_outputs[i,pred_label*7:pred_label*7+7] = cur_pred

        
        # # 左右同时出现检测框结果认为是有效
        # if np.sum(r_outputs[i,:]!=0) and np.sum(l_outputs[i,:]!=0):
        #     valid_slice_idxs.append(i)
    
    l_tb_bbox = []
    r_tb_bbox = []
    l_idxs = []
    r_idxs = []
    if len(valid_slice_idxs) != 0:
    # compute pixel size
        tb_pixelsize = [0,0,0]
        tb_pixelsize[0] = tb_physize[0] / spacing[0] # w
        tb_pixelsize[1] = tb_physize[1] / spacing[1] # h
        tb_pixelsize[2] = tb_physize[2] / spacing[2] # d
        # print('\tvalid_slice_idxs: {}'.format(valid_slice_idxs))
        # # 取均值
        # valid_min_idx = np.min(valid_slice_idxs)
        # valid_max_idx = np.max(valid_slice_idxs)
        # tb_bbox_cz = (valid_min_idx+valid_max_idx)/2 
        # 取中位数
        valid_idx_num = len(valid_slice_idxs)
        if valid_idx_num % 2 == 0:
            median_idx = int(valid_idx_num/2)
            tb_bbox_cz = int((valid_slice_idxs[median_idx]+valid_slice_idxs[median_idx+1])/2)
        else:
            median_idx = int((valid_idx_num+1) / 2)
            tb_bbox_cz = valid_slice_idxs[median_idx]
        
        crop_depth = min(max(valid_idx_num,tb_pixelsize[2]),d)
        tb_bbox_zmin = max(0,int(tb_bbox_cz-crop_depth/2))
        tb_bbox_zmax = min(int(tb_bbox_cz+crop_depth/2),d)
        # pdb.set_trace()
        # slice temporal bone detction
        l_tb_bbox,l_idxs = temporal_bone_detection(l_outputs,EAR_POS.LEFT,input_size,tb_pixelsize,topk=topk)
        r_tb_bbox,r_idxs = temporal_bone_detection(r_outputs,EAR_POS.RIGHT,input_size,tb_pixelsize,topk=topk)
    
    # # draw respond result img
    # if draw_vis:
    #     draw_joint_results(nii_array,l_idxs,l_outputs,EAR_POS.LEFT,save_path)
    #     draw_joint_results(nii_array,r_idxs,r_outputs,EAR_POS.RIGHT,save_path)
    return l_tb_bbox,l_idxs,r_tb_bbox,r_idxs,tb_bbox_zmin,tb_bbox_zmax

def extract_temporal_bone_img(nii_filepath,predictor,tb_physize,topk=1,draw_vis=0):
    tb_nii_imgs={ }
    
    img,spacing,origin = get_array_from_nii(nii_filepath)
    l_tb_bbox,l_idxs,r_tb_bbox,r_idxs,zmin,zmax = nii_process(nii_filepath,predictor,tb_physize,topk=topk,draw_vis=draw_vis)
    
    if len(l_idxs) == 0 and  len(r_idxs) == 0:
        print('warning: abnormaly niifile!')
        return tb_nii_imgs   
    # crop and save
    # left
    if len(l_idxs) != 0:
        l_tb_img = img[zmin:zmax,l_tb_bbox[1]:l_tb_bbox[3],l_tb_bbox[0]:l_tb_bbox[2]]
        tb_nii_imgs['left'] = l_tb_img
    else:
        print('\t{} left detect failed!'.format(nii_filepath))
    # right
    if len(r_idxs) != 0:
        r_tb_img = img[zmin:zmax,r_tb_bbox[1]:r_tb_bbox[3],r_tb_bbox[0]:r_tb_bbox[2]]
        tb_nii_imgs['right'] = r_tb_img
    else:
        print('\t{} right detect failed!'.format(nii_filepath))
        
    return tb_nii_imgs

def batch_nii_process(nii_root,save_root,predictor,tb_physize,topk=1):
    nii_files = os.listdir(nii_root)
    for nii_file in nii_files:
        print('process {}...'.format(nii_file))
        nii_filename = nii_file.replace('.nii.gz','')
        nii_filepath = os.path.join(nii_root,nii_file)
        img,spacing,origin = get_array_from_nii(nii_filepath)
        
        # savepath
        tb_save_path = os.path.join(save_root,nii_filename)
        if not os.path.exists(tb_save_path):
            os.mkdir(tb_save_path)
        l_tb_filepath = os.path.join(tb_save_path,'{}_left.nii.gz'.format(nii_filename))
        r_tb_filepath = os.path.join(tb_save_path,'{}_right.nii.gz'.format(nii_filename))
        if os.path.exists(l_tb_filepath) or os.path.exists(r_tb_filepath):
            continue
        
        # process
        l_tb_bbox,l_idxs,r_tb_bbox,r_idxs,zmin,zmax = nii_process(nii_filepath,predictor,tb_save_path,tb_physize,topk=topk,draw_vis=draw_vis)
        if len(l_idxs) == 0 and  len(r_idxs) == 0:
            print('warning: abnormaly niifile!')
            continue
        
        # crop and save
        # left
        if len(l_idxs) != 0:
            l_tb_img = img[zmin:zmax,l_tb_bbox[1]:l_tb_bbox[3],l_tb_bbox[0]:l_tb_bbox[2]]
            savenii(l_tb_img,spacing,origin,l_tb_filepath,std=True)
            save_center_img(l_tb_img,tb_save_path,nii_filename,mode='left')
            # pdb.set_trace()
        else:
            print('\t{} left detect failed!'.format(nii_file))
        # right
        if len(r_idxs) != 0:
            r_tb_img = img[zmin:zmax,r_tb_bbox[1]:r_tb_bbox[3],r_tb_bbox[0]:r_tb_bbox[2]]
            save_center_img(r_tb_img,tb_save_path,nii_filename,mode='right')
            savenii(r_tb_img,spacing,origin,r_tb_filepath,std=True)
        else:
            print('\t{} right detect failed!'.format(nii_file))
    print('Done!')

def get_tb_region(hsc_bbox,iac_bbox,tb_pixelsize,ear_pos,compute_mode=1):
    preset_w = tb_pixelsize[0]
    preset_h = tb_pixelsize[1]

    # tb_w, tb_h
    if compute_mode == COMPUTE_MODE.JOINT:
        if ear_pos == EAR_POS.LEFT:
            tb_w = max(np.abs(hsc_bbox[2] - iac_bbox[0]) * 2, preset_w)
            tb_h = max((hsc_bbox[3] - iac_bbox[1] + (iac_bbox[3] - iac_bbox[1]))*2, preset_h)
        elif ear_pos == EAR_POS.RIGHT:
            tb_w = max((iac_bbox[2] - hsc_bbox[0])*2, preset_w)
            tb_h = max((hsc_bbox[3] - iac_bbox[1] + (iac_bbox[3] - iac_bbox[1]))*2,preset_h)
    else:
        tb_w = preset_w
        tb_h = preset_h

    tb_w = max(tb_w,tb_h)
    tb_h = tb_w

    # tb_cx, tb_cy
    if compute_mode != COMPUTE_MODE.IAC: # center hsc based
        if ear_pos == EAR_POS.LEFT:
            tb_cx = hsc_bbox[2] # hsc_xmax
        elif ear_pos == EAR_POS.RIGHT:
            tb_cx = hsc_bbox[0] # hsc_xmin
        tb_cy = hsc_bbox[3] # hsc_ymax
    else: # center iac based
        iac_cx = (iac_bbox[0]+iac_bbox[2]) / 2
        if ear_pos == EAR_POS.LEFT:
            tb_cx = iac_bbox[0] + 0.5*tb_w # iac_xmin
        elif ear_pos == EAR_POS.RIGHT:
            tb_cx = iac_bbox[2] - 0.5*tb_w
        tb_ymin = (iac_bbox[1] + iac_bbox[3])/2 - (1/3)*tb_h
        tb_cy = tb_ymin + 0.5*tb_h

    tb_xmin = int(tb_cx - tb_w/2)
    tb_xmax = int(tb_cx + tb_w/2)
    tb_ymin = int(tb_cy - tb_h/2)
    tb_ymax = int(tb_cy + tb_h/2)
    
    tb_bbox = [tb_xmin,tb_ymin,tb_xmax,tb_ymax]
    return tb_bbox

def save_center_img(nii_img,savepath,nii_filename,mode='left'):
    t_cnt_img,c_cnt_img,s_cnt_img = get_center_img(nii_img)
    t_cnt_imgfile = '{}_{}_{}.jpg'.format(nii_filename,mode,'transverse')
    c_cnt_imgfile = '{}_{}_{}.jpg'.format(nii_filename,mode,'coronal')
    s_cnt_imgfile = '{}_{}_{}.jpg'.format(nii_filename,mode,'sagittal')
    cv2.imwrite(os.path.join(savepath,t_cnt_imgfile),t_cnt_img)
    cv2.imwrite(os.path.join(savepath,c_cnt_imgfile),np.flip(c_cnt_img,0))
    cv2.imwrite(os.path.join(savepath,s_cnt_imgfile),np.flip(s_cnt_img,0))

def savenii(vol0,spacing,origin,outname,std=False):
    if not std:
        vol = np.transpose(vol0, (2, 0, 1))
        vol = vol[::-1]
    else:
        vol=vol0
    out = sitk.GetImageFromArray(vol)
    out.SetSpacing(spacing)
    out.SetOrigin(origin)
    sitk.WriteImage(out,'%s.nii.gz'%(outname))

def get_array_from_nii(nii_filepath,mask=0):
    data = sitk.ReadImage(nii_filepath)
    spacing = data.GetSpacing()
    origin = data.GetOrigin()

    img = sitk.GetArrayFromImage(data)
    if not mask:
        # 默认最大值和最小值的中间值为窗位，最大值和最小值的距离为窗宽
        imax = np.max(img)
        imin = np.min(img)
        
        if imax < 10000:
            center = (imax+imin) / 2 
            window = imax - imin           
        else:
            histogram,bins = np.histogram(img.flatten(),bins=100)
            idx = np.where(histogram==histogram[histogram>10000][-1])[0]
            new_imax = bins[idx+1][0]
            
            window = new_imax - imin
            img[img>new_imax] = new_imax
            
        img = ((img - imin) / window).astype(np.float32)
        img = (img * 255).astype(np.uint8)
        
    return img,spacing,origin

def get_model(attr):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(attr.depth, attr.width, in_channels=in_channels, act=attr.act)
    head = YOLOXHead(attr.num_classes, attr.width, in_channels=in_channels, act=attr.act)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    model.train()
    return model

def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

class Predictor(object):
    def __init__(
        self,
        model,
        attr,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = attr.num_classes
        self.confthre = attr.test_conf
        self.nmsthre = attr.nmsthre
        self.test_size = attr.test_size
        self.device = attr.device
        self.fp16 = attr.fp16
        self.preproc = ValTransform(legacy=legacy)
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, attr.test_size[0], attr.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
