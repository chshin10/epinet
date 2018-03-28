# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:05:35 2018

@author: shinyonsei2
"""
import imageio
import numpy as np

def make_epiinput(image_path,seq1,sz_input,sz_input2,view_n,RGB):
    traindata_tmp=np.zeros((1,sz_input,sz_input2,len(view_n)),dtype=np.float32)
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
        
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % seq)) 
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp


def make_multiinput(image_path,sz_input,sz_input2,view_n):
    
    RGB = [0.299,0.587,0.114] ## RGB to Gray // 0.299 0.587 0.114
    
    seqLtoR=list(range(36,45,1)) # [36, 37, 38, 39, 40, 41, 42, 43, 44]
    seqUtoD=list(range(4,81,9)) # [4, 13, 22, 31, 40, 49, 58, 67, 76]    
    seq5c=list(range(0,81,10)) # [0, 10, 20, 30, 40, 50, 60, 70, 80]
    seq7c=list(range(8,80,8)) # [8, 16, 24, 32, 40, 48, 56, 64, 72]
    
    val_LtoR=make_epiinput(image_path,seqLtoR,sz_input,sz_input2,view_n,RGB)    
    val_UtoD=make_epiinput(image_path,seqUtoD[::-1],sz_input,sz_input2,view_n,RGB)
    val_5c=make_epiinput(image_path,seq5c,sz_input,sz_input2,view_n,RGB)
    val_7c=make_epiinput(image_path,seq7c[::-1],sz_input,sz_input2,view_n,RGB) 
       
    return val_LtoR , val_UtoD, val_5c, val_7c