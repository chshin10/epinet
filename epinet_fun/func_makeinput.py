# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:38:16 2018

@author: shinyonsei2
"""

import imageio
import numpy as np
import os

def make_epiinput(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
        
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % seq)) 
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp

def make_epiinput_lytro(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
        
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/%s_%02d_%02d.png' % (image_path.split("/")[-1],1+seq//9, 1+seq-(seq//9)*9)) )
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp
    


def make_multiinput(image_path,image_h,image_w,view_n):
    
    RGB = [0.299,0.587,0.114] ## RGB to Gray // 0.299 0.587 0.114
    
    ''' data from http://hci-lightfield.iwr.uni-heidelberg.de/
    Sample images "training/dino, training/cotton" Cam000~ Cam080.png  
    We select seq of images to get epipolar images.
    For example, seq90d: Cam076.png, 67, 58, 49, 40, 31, 22, 13, 4'''
    
    # 00          04          08
    #    10       13       16 
    #       20    22    24 
    #          30 31 32 
    # 36 37 38 39 40 41 42 43 44
    #          48 49 50 
    #       56    58    60 
    #    64       67       70 
    # 72          76          80    
    
    seq90d=list(range(4,81,9))[::-1] # 90degree:  [76, 67, 58, 49, 40, 31, 22, 13, 4 ]    
    seq0d=list(range(36,45,1)) # 0degree:  [36, 37, 38, 39, 40, 41, 42, 43, 44] 
    seq45d=list(range(8,80,8))[::-1] # 45degree:  [72, 64, 56, 48, 40, 32, 24, 16, 8 ]
    seqM45d=list(range(0,81,10)) # -45degree: [0, 10, 20, 30, 40, 50, 60, 70, 80] 
    
    if(image_path[:8]=='training' and os.listdir(image_path)[0][:9]=='input_Cam'):
        val_90d=make_epiinput(image_path,seq90d,image_h,image_w,view_n,RGB)    
        val_0d=make_epiinput(image_path,seq0d,image_h,image_w,view_n,RGB)
        val_45d=make_epiinput(image_path,seq45d,image_h,image_w,view_n,RGB)
        val_M45d=make_epiinput(image_path,seqM45d,image_h,image_w,view_n,RGB) 
    
    elif(image_path[:5]=='lytro'):
        val_90d=make_epiinput_lytro(image_path,seq90d,image_h,image_w,view_n,RGB)    
        val_0d=make_epiinput_lytro(image_path,seq0d,image_h,image_w,view_n,RGB)
        val_45d=make_epiinput_lytro(image_path,seq45d,image_h,image_w,view_n,RGB)
        val_M45d=make_epiinput_lytro(image_path,seqM45d,image_h,image_w,view_n,RGB) 
       
    
    return val_90d , val_0d, val_45d, val_M45d