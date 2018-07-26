# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:11:34 2018

@author: shinyonsei2
    
    requirement: numpy>=1.14

    if LF patch_size= 25x25, batch_size=16, 9x9 viewpoints
    
    seq_90d_batch: ( batch_size(16), patch_size(25), patch_size(25), 9(view stack) ) 
    seq_0d_batch: ( batch_size(16), patch_size(25), patch_size(25), 9(view stack) )  
    seq_45d_batch: ( batch_size(16), patch_size(25), patch_size(25), 9(view stack) )  
    seq_M45d_batch: ( batch_size(16), patch_size(25), patch_size(25), 9(view stack) ) 
    label_batch: : ( batch_size(16), patch_size(25-22), patch_size(25-22) ) 
    --> label patch size is decreased by convolutional layers with no-padding


    !!! Caution: The order of View stack is very important! 
    Please see this link for detail description. https://github.com/chshin10/epinet/issues/1
    
"""
import numpy as np

    
def rotation_augmentation(seq_90d_batch, seq_0d_batch,seq_45d_batch,seq_M45d_batch, label_batch, batch_size):


    for batch_i in range(batch_size):
            
        rot90_rand=np.random.randint(0,4)
        transp_rand=np.random.randint(0,2)

        if transp_rand==1: # transpose

            seq_90d_batch_tmp6=np.copy(np.transpose(np.squeeze(seq_90d_batch[batch_i,:,:,:]),(1, 0, 2)) )   
            seq_0d_batch_tmp6=np.copy(np.transpose(np.squeeze(seq_0d_batch[batch_i,:,:,:]),(1, 0, 2)) ) 
            seq_45d_batch_tmp6=np.copy(np.transpose(np.squeeze(seq_45d_batch[batch_i,:,:,:]),(1, 0, 2)) )
            seq_M45d_batch_tmp6=np.copy(np.transpose(np.squeeze(seq_M45d_batch[batch_i,:,:,:]),(1, 0, 2)) )

            seq_0d_batch[batch_i,:,:,:]=np.copy(seq_90d_batch_tmp6[:,:,::-1])
            seq_90d_batch[batch_i,:,:,:]=np.copy(seq_0d_batch_tmp6[:,:,::-1])
            seq_45d_batch[batch_i,:,:,:]=np.copy(seq_45d_batch_tmp6[:,:,::-1])
            seq_M45d_batch[batch_i,:,:,:]=np.copy(seq_M45d_batch_tmp6)

            label_batch[batch_i,:,:]=np.copy(np.transpose(label_batch[batch_i,:,:],(1, 0))) 
    
    
        if rot90_rand==1: # 90 degree

            seq_90d_batch_tmp3=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],1,(0,1)))
            seq_0d_batch_tmp3=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],1,(0,1)))
            seq_45d_batch_tmp3=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],1,(0,1)))
            seq_M45d_batch_tmp3=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],1,(0,1)))

            seq_90d_batch[batch_i,:,:,:]=seq_0d_batch_tmp3  
            seq_45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp3
            seq_0d_batch[batch_i,:,:,:]=seq_90d_batch_tmp3[:,:,::-1]  
            seq_M45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp3[:,:,::-1]
            
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],1,(0,1))) 

        if rot90_rand==2: # 180 degree

            seq_90d_batch_tmp4=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],2,(0,1)))
            seq_0d_batch_tmp4=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],2,(0,1)))
            seq_45d_batch_tmp4=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],2,(0,1)))
            seq_M45d_batch_tmp4=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],2,(0,1)))

            seq_90d_batch[batch_i,:,:,:]=seq_90d_batch_tmp4[:,:,::-1]
            seq_0d_batch[batch_i,:,:,:]=seq_0d_batch_tmp4[:,:,::-1] 
            seq_45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp4[:,:,::-1] 
            seq_M45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp4[:,:,::-1] 
            
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],2,(0,1)))

        if rot90_rand==3: # 270 degree
            seq_90d_batch_tmp5=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],3,(0,1)))
            seq_0d_batch_tmp5=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],3,(0,1)))
            seq_45d_batch_tmp5=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],3,(0,1)))
            seq_M45d_batch_tmp5=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],3,(0,1)))
            
            seq_90d_batch[batch_i,:,:,:]=seq_0d_batch_tmp5[:,:,::-1] 
            seq_0d_batch[batch_i,:,:,:]=seq_90d_batch_tmp5
            seq_45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp5[:,:,::-1]
            seq_M45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp5
            
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],3,(0,1)))  

    return seq_90d_batch, seq_0d_batch, seq_45d_batch, seq_M45d_batch, label_batch

