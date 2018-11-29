# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:54:01 2018

@author: shinyonsei2
"""

import numpy as np
import imageio

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data
    
def load_LFdata(dir_LFimages):    
    traindata_all=np.zeros((len(dir_LFimages), 512, 512, 9, 9, 3),np.uint8)
    traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)
    
    image_id=0
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread('hci_dataset/'+dir_LFimage+'/input_Cam0%.2d.png' % i)) # load LF images(9x9) 
            except:
                print('hci_dataset/'+dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
            traindata_all[image_id,:,:,i//9,i-9*(i//9),:]=tmp  
            del tmp
        try:            
            tmp  = np.float32(read_pfm('hci_dataset/'+dir_LFimage+'/gt_disp_lowres.pfm')) # load LF disparity map
        except:
            print('hci_dataset/'+dir_LFimage+'/gt_disp_lowres.pfm..does not exist' % i )            
        traindata_label[image_id,:,:]=tmp  
        del tmp
        image_id=image_id+1
    return traindata_all, traindata_label
