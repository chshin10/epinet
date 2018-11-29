# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:58 2017

@author: shinyonsei2
"""


from __future__ import print_function


from epinet_fun.func_generate_traindata import generate_traindata_for_train
from epinet_fun.func_generate_traindata import data_augmentation_for_train
from epinet_fun.func_generate_traindata import generate_traindata512

from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_pfm import read_pfm
from epinet_fun.func_savedata import display_current_output
from epinet_fun.util import load_LFdata

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os
import time
import imageio
import datetime
import threading



if __name__ == '__main__':	
    
    ''' 
    We use fit_generator to train EPINET, 
    so here we defined a generator function.
    '''
    
    class threadsafe_iter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """
        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()
    
        def __iter__(self):
            return self
    
        def __next__(self):
            with self.lock:
                return self.it.__next__()
    
    
    def threadsafe_generator(f):
        """A decorator that takes a generator function and makes it thread-safe.
        """
        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))
        return g

    
    @threadsafe_generator
    def myGenerator(traindata_all,traindata_label,
                    input_size,label_size,batch_size,
                    Setting02_AngualrViews,
                    boolmask_img4,boolmask_img6,boolmask_img15):  
        while 1:
            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d, 
             traindata_label_batchNxN)= generate_traindata_for_train(traindata_all,traindata_label,
                                                                     input_size,label_size,batch_size,
                                                                     Setting02_AngualrViews,
                                                                     boolmask_img4,boolmask_img6,boolmask_img15)                

            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d,traindata_batch_m45d, 
             traindata_label_batchNxN) =  data_augmentation_for_train(traindata_batch_90d, 
                                                                      traindata_batch_0d,
                                                                      traindata_batch_45d,
                                                                      traindata_batch_m45d, 
                                                                      traindata_label_batchNxN,
                                                                      batch_size) 

            traindata_label_batchNxN=traindata_label_batchNxN[:,:,:,np.newaxis] 


            yield([traindata_batch_90d,
                   traindata_batch_0d,
                   traindata_batch_45d,
                   traindata_batch_m45d],
                   traindata_label_batchNxN)
    


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    
    If_trian_is = True;  
    
    ''' 
    GPU setting ( Our setting: gtx 1080ti,  
                               gpu number = 0 ) 
    '''
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"    

    
    networkname='EPINET_train'
    


    iter00=0; 
    
    load_weight_is=False;   

        
    
    ''' 
    Define Model parameters    
        first layer:  3 convolutional blocks, 
        second layer: 7 convolutional blocks, 
        last layer:   1 convolutional block
    ''' 
    model_conv_depth=7 # 7 convolutional blocks for second layer
    model_filt_num=70
    model_learning_rate=0.1**4






    ''' 
    Define Patch-wise training parameters
    '''        
    input_size=23+2         # Input size should be greater than or equal to 23
    label_size=input_size-22 # Since label_size should be greater than or equal to 1
    Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 ) 
    
    batch_size=16       
    workers_num=2  # number of threads
    
    display_status_ratio=10000 

       
    
    
    ''' 
    Define directory for saving checkpoint file & disparity output image
    '''       
    directory_ckp="epinet_checkpoints/%s_ckp"% (networkname)     
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)
        
    if not os.path.exists('epinet_output/'):
        os.makedirs('epinet_output/')   
    directory_t='epinet_output/%s' % (networkname)    
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)     
        
    txt_name='epinet_checkpoints/lf_%s.txt' % (networkname)        
        
        
        
    ''' 
    Load Train data from LF .png files
    '''
    print('Load training data...')    
    dir_LFimages=[
            'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
            'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
            'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
            'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]


    traindata_all,traindata_label=load_LFdata(dir_LFimages)
     
    traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ =generate_traindata512(traindata_all,traindata_label,Setting02_AngualrViews)
    # (traindata_90d, 0d, 45d, m45d) to validation or test
    # traindata_90d, 0d, 45d, m45d:  16x512x512x9  float32

    print('Load training data... Complete')  
    
    '''load invalid regions from training data (ex. reflective region)'''    
    boolmask_img4= imageio.imread('hci_dataset/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
    boolmask_img6= imageio.imread('hci_dataset/additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
    boolmask_img15=imageio.imread('hci_dataset/additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
    
    boolmask_img4  = 1.0*boolmask_img4[:,:,3]>0
    boolmask_img6  = 1.0*boolmask_img6[:,:,3]>0
    boolmask_img15 = 1.0*boolmask_img15[:,:,3]>0    
    
    
        
    ''' 
    Load Test data from LF .png files
    '''
    print('Load test data...') 
    dir_LFimages=[
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
    
    
    valdata_all,valdata_label=load_LFdata(dir_LFimages)
        
    valdata_90d,valdata_0d,valdata_45d,valdata_m45d, valdata_label=generate_traindata512(valdata_all,valdata_label,Setting02_AngualrViews)
    # (valdata_90d, 0d, 45d, m45d) to validation or test      
    print('Load test data... Complete') 

  

    ''' 
    Model for patch-wise training  
    '''
    model=define_epinet(input_size,input_size,
                            Setting02_AngualrViews,
                            model_conv_depth, 
                            model_filt_num,
                            model_learning_rate)
  

    ''' 
    Model for predicting full-size LF images  
    '''
    image_w=512
    image_h=512
    model_512=define_epinet(image_w,image_h,
                            Setting02_AngualrViews,
                            model_conv_depth, 
                            model_filt_num,
                            model_learning_rate)


   
 
    """ 
    load latest_checkpoint
    """
    if load_weight_is:
        list_name=os.listdir(directory_ckp)
        if(len(list_name)>=1):
            list1=os.listdir(directory_ckp)
            list_i=0
            for list1_tmp in list1:
                if(list1_tmp ==  'checkpoint'):
                    list1[list_i]=0
                    list_i=list_i+1   
                else:
                    list1[list_i]=int(list1_tmp.split('_')[0][4:])
                    list_i=list_i+1            
            list1=np.array(list1) 
            iter00=list1[np.argmax(list1)]+1
            ckp_name=list_name[np.argmax(list1)].split('.hdf5')[0]+'.hdf5'
            model.load_weights(directory_ckp+'/'+ckp_name)
            print("Network weights will be loaded from previous checkpoints \n(%s)" % ckp_name)


    """ 
    Write date & time 
    """
    f1 = open(txt_name, 'a')
    now = datetime.datetime.now()
    f1.write('\n'+str(now)+'\n\n')
    f1.close()    


    
    my_generator = myGenerator(traindata_all,traindata_label,input_size,label_size,batch_size,Setting02_AngualrViews ,boolmask_img4,boolmask_img6,boolmask_img15)
    best_bad_pixel=100.0
    for iter02 in range(10000000):
        
        ''' Patch-wise training... start'''
        t0=time.time()
        
        model.fit_generator(my_generator, steps_per_epoch = int(display_status_ratio), 
                            epochs = iter00+1, class_weight=None, max_queue_size=10, 
                            initial_epoch=iter00, verbose=1,workers=workers_num)

        iter00=iter00+1
        
        
        ''' Test after N*(display_status_ratio) iteration.'''
        weight_tmp1=model.get_weights() 
        model_512.set_weights(weight_tmp1)
        train_output=model_512.predict([traindata_90d,traindata_0d,
                                        traindata_45d,traindata_m45d],batch_size=1)

            

        ''' Save prediction image(disparity map) in 'current_output/' folder '''    
        train_error, train_bp=display_current_output(train_output, traindata_label, iter00, directory_t)


        training_mean_squared_error_x100=100*np.average(np.square(train_error))
        training_bad_pixel_ratio=100*np.average(train_bp)


        save_path_file_new=(directory_ckp+'/iter%04d_trainmse%.3f_bp%.2f.hdf5'  
                            % (iter00,training_mean_squared_error_x100,
                                      training_bad_pixel_ratio) )
        """ 
        Save bad pixel & mean squared error
        """        
        print(save_path_file_new)
        f1 = open(txt_name, 'a')
        f1.write('.'+save_path_file_new+'\n')
        f1.close()              
        t1=time.time()        
        
        ''' save model weights if it get better results than previous one...'''
        if(training_bad_pixel_ratio < best_bad_pixel):
            best_bad_pixel = training_bad_pixel_ratio
            model.save(save_path_file_new)
            print("saved!!!")
            

