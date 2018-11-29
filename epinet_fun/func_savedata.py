# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:55:17 2018

@author: shinyonsei2
"""
import numpy as np
import imageio

def display_current_output(train_output, traindata_label, iter00, directory_save):
    '''
        display current results from EPINET 
        and save results in /current_output
    '''
    sz=len(traindata_label)
    train_output=np.squeeze(train_output)
    if(len(traindata_label.shape)>3 and traindata_label.shape[-1]==9): # traindata
        pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
        train_label482=traindata_label[:,15:-15,15:-15,4,4]
    else: # valdata
        pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
        train_label482=traindata_label[:,15:-15,15:-15]
        
    train_output482=train_output[:,15-pad1_half:482+15-pad1_half,15-pad1_half:482+15-pad1_half]
    
    train_diff=np.abs(train_output482-train_label482)
    train_bp=(train_diff>=0.07)
        
    train_output482_all=np.zeros((2*482,sz*482),np.uint8)        
    train_output482_all[0:482,:]=np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)
    train_output482_all[482:2*482,:]=np.uint8(25*np.reshape(np.transpose(train_output482,(1,0,2)),(482,sz*482))+100)
                      
    imageio.imsave(directory_save+'/train_iter%05d.jpg' % (iter00), np.squeeze(train_output482_all))
    
    return train_diff, train_bp
