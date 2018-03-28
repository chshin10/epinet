# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop

from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate


def layer1_multistream(input_dim1,input_dim2,input_dim3,filt_num):    
    seq = Sequential()
    
#    seq.add(Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
    for i in range(3):
        seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
        seq.add(Activation('relu', name='S1_relu1%d' %(i))) 
        seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) )) 
        seq.add(BatchNormalization(axis=-1, name='S0_BN%d_1' % (i)))
        seq.add(Activation('relu', name='S1_relu2%d' %(i))) 

    seq.add(Reshape((input_dim1-6,input_dim2-6,int(filt_num))))

    return seq  

def layer2_merged(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    seq = Sequential()
    
    for i in range(conv_depth):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
        seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
        seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        seq.add(Activation('relu', name='S2_relu2%d' %(i)))
          
    return seq     

def layer3_last(input_dim1,input_dim2,input_dim3,filt_num):    
    seq = Sequential()
    
    for i in range(1):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(Activation('relu', name='S3_relu1%d' %(i)))
        
    seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last')) # pow(25/23,2)*12*(maybe7?) 43 3

    return seq 

def define_epinet(sz_input,sz_input2,view_n,conv_depth,filt_num,learning_rate):
    
    input_stack_LtoRb = Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_LtoRb')
    input_stack_UtoDb= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_UtoDb')
    input_stack_5cb= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_5cb')
    input_stack_7cb= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_7cb')
    
    mid_LtoRa=layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_LtoRb)
    mid_UtoDa=layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_UtoDb)    
    mid_5ca=layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_5cb)    
    mid_7ca=layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_7cb)   
#
    mid_LRUDa = concatenate([mid_LtoRa,mid_UtoDa,mid_5ca,mid_7ca],  name='mid_LRUDa')
    mid_LRUD3a=layer2_merged(sz_input-6,sz_input2-6,int(4*filt_num),int(4*filt_num),conv_depth)(mid_LRUDa)

    
    outputa=layer3_last(sz_input-18,sz_input2-18,int(4*filt_num),int(4*filt_num))(mid_LRUD3a)

    model_512 = Model(inputs = [input_stack_LtoRb,input_stack_UtoDb,
                               input_stack_5cb,input_stack_7cb], outputs = [outputa])
    opt = RMSprop(lr=learning_rate)
    model_512.compile(optimizer=opt, loss='mae')
    model_512.summary() 
    
    return model_512