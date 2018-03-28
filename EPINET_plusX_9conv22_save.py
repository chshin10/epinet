# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:41:04 2018

@author: shinyonsei2
"""
#import numpy as np
import numpy as np
import os
import time
from epinet_fun.func_pfm import write_pfm
from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput
from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_epinetmodel import layer1_multistream
from epinet_fun.func_epinetmodel import layer2_merged
from epinet_fun.func_epinetmodel import layer3_last

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"    
    
    """ --- Directory setting --- """
    
    # Light field images: input_Cam000-080.png
    dir_LFimages=['training/dino','training/cotton']
#    dir_LFimages=['training/cotton']
    image_h=512
    image_w=512
    
    # Depth output : image_name.pfm
    dir_depth='epinet_output'
    
    if not os.path.exists(dir_depth):
        os.makedirs(dir_depth)    
        
    # Checkpoint (= Pretrained Weights)
#    path_weight='epinet_checkpoints/iter10000_mse1.504_bp3.65.hdf5' 
    path_weight='checkpoints/iter13140_mse1.467_bp5.93.hdf5' # it's sample weight.. not the best one.
    
    # number of views ( 0~8 for 9x9 ) 
    angular_views=[0,1,2,3,4,5,6,7,8] 
    


    """ --- Define model & initalization --- """
    
    # model parameters
    model_conv_depth=7
    model_filt_num=70
    model_learning_rate=0.1**5
    model_512=define_epinet(image_h,
                            image_w,
                            angular_views,
                            model_conv_depth, 
                            model_filt_num,
                            model_learning_rate)
    
    
    """  --- Initalization --- """
    
    model_512.load_weights(path_weight)
    dum_sz=model_512.input_shape[0]
    dum=np.zeros((1,dum_sz[1],dum_sz[2],dum_sz[3]),dtype=np.float32)
    dummy=model_512.predict([dum,dum, dum,dum],batch_size=1) 
    
    
    """  --- Test --- """
    for image_path in dir_LFimages:


        (val_LtoR, val_UtoD, val_5c, val_7c)=make_multiinput(image_path,
                                                             image_h,
                                                             image_w,
                                                             angular_views)

        start=time.clock() 
        
        # predict
        val_output_tmp=model_512.predict([val_UtoD, val_LtoR, 
                                    val_7c, val_5c], batch_size=1); 
        plt.imshow(val_output_tmp[0,:,:,0])
        runtime=time.clock() - start
        print("%.5f(s)" % runtime)
        
        # save .pfm file
        write_pfm(val_output_tmp[0,:,:,0], dir_depth+'/%s.pfm' % (image_path.split('/')[-1]))
        plt.imshow(val_output_tmp[0,:,:,0])

