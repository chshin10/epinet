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
from epinet_fun.func_epinetmodel import layer1_multistream
from epinet_fun.func_epinetmodel import layer2_merged
from epinet_fun.func_epinetmodel import layer3_last
from epinet_fun.func_epinetmodel import define_epinet

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"    # gtx 1080ti - 0
    
    ''' --- Directory setting --- '''
    
    ''' 
    The order of LF image files may be different with this file.
      (Top to Bottom, Left to Right, and so on..)
      
    If you use different LF images, you should change 'func_makeinput.py' file.
    
    '''
    # Light field images: input_Cam000-080.png
    
    # All viewpoints = 9x9(81)
    
    # 00 01 02 03 04 05 06 07 08
    # 09 10 11 12 13 14 15 16 17
    # 18 19 20 21 22 23 24 25 26
    # 27 28 29 30 31 32 33 34 35
    # 36 37 38 39 40 41 42 43 44
    # 45 46 47 48 49 50 51 52 53
    # 54 55 56 57 58 59 60 61 62
    # 63 64 65 66 67 68 69 70 71
    # 72 73 74 75 76 77 78 79 80
    
    # We select star-shape viewpoints
    
    # 00          04          08
    #    10       13       16 
    #       20    22    24 
    #          30 31 32 
    # 36 37 38 39 40 41 42 43 44
    #          48 49 50 
    #       56    58    60 
    #    64       67       70 
    # 72          76          80    
    
    dir_LFimages=['training/dino','training/cotton']
    image_h=512
    image_w=512
    
    # Depth output : image_name.pfm
    dir_depth='epinet_output'
    
    if not os.path.exists(dir_depth):
        os.makedirs(dir_depth)    
        
    # Checkpoint (= Pretrained Weights)
    path_weight='epinet_checkpoints/iter16320_mse1.496_bp3.55.hdf5' # sample weight.
    
    # number of views ( 0~8 for 9x9 ) 
    angular_views=[0,1,2,3,4,5,6,7,8] 
    


    ''' --- Define Model --- '''
    
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
    
    
    '''  --- Model Initalization --- '''
    
    model_512.load_weights(path_weight)
    dum_sz=model_512.input_shape[0]
    dum=np.zeros((1,dum_sz[1],dum_sz[2],dum_sz[3]),dtype=np.float32)
    dummy=model_512.predict([dum,dum, dum,dum],batch_size=1) 
    
    
    """  --- Test --- """
    for image_path in dir_LFimages:


        (val_90d , val_0d, val_45d, val_M45d)=make_multiinput(image_path,
                                                             image_h,
                                                             image_w,
                                                             angular_views)

        start=time.clock() 
        
        # predict
        val_output_tmp=model_512.predict([val_90d , val_0d, 
                                          val_45d, val_M45d], batch_size=1); 
        plt.imshow(val_output_tmp[0,:,:,0])
        runtime=time.clock() - start
        print("%.5f(s)" % runtime)
        
        # save .pfm file
        write_pfm(val_output_tmp[0,:,:,0], dir_depth+'/%s.pfm' % (image_path.split('/')[-1]))
#        plt.imshow(val_output_tmp[0,:,:,0])

