# EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images
EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images

Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon and Seon Joo Kim 

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2018 

https://arxiv.org/pdf/1804.02379.pdf


Contact: changhashin@yonsei.ac.kr


# Environments

- Python3.5.2, Anaconda 4.2.0 (64-bit), Tensorflow 1.6.0 - 1.12.0
- `pip install imageio` 



# Train the EPINET
 First, you need to download HCI Light field dataset from http://hci-lightfield.iwr.uni-heidelberg.de/.
 Unzip the LF dataset and move 'additional/, training/, test/, stratified/ ' into the 'hci_dataset/'.
 
 And run `python EPINET_train.py`
 
 - Checkpoint files will be saved in 'epinet_checkpoints/EPINET_train_ckp/iterXXX_XX.hdf5', it could be used for test EPINET model.
 - Training process will be saved 'epinet_output/EPINET_train/train_XX.jpg'. (XX is iteration number). 
 - You might be change the setting 'learning rate','patch_size' and so on to get better result.

# Test the EPINET

Run `python EPINET_plusX_9conv22_save.py`

 - To test your own trained model from `python EPINET_train.py`, you need to modify the line 141-142 like below
`path_weight='epinet_checkpoints/EPINET_train_ckp/iter0097_trainmse2.706_bp12.06.hdf5'`

Last modified date: 11/29/2018
