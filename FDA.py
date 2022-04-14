#!/usr/bin/env python
# coding: utf-8

# # Original Demo
import numpy as np
 
def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    # print(amp_src.shape)
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
    
    # print(a_src.shape)
    _, _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


# In[17]:


import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

im_src = Image.open("demo_images/source.png").convert('RGB')
im_trg = Image.open("demo_images/target.png").convert('RGB')

im_src = im_src.resize( (1024,512), Image.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
# scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src_in_tar.png')

src_in_trg[src_in_trg > 255] = 255
src_in_trg[src_in_trg < 0] = 0
src_in_trg = src_in_trg.astype(np.uint8)
img_src_in_trg = Image.fromarray(src_in_trg)
img_src_in_trg.save('demo_images/src_in_tar.png')
img_src_in_trg.show()


# In[18]:


img_src_in_trg


# # unknown bug code

# In[1]:


# import numpy as np
# from PIL import Image
# import copy
# from utils import FDA_source_to_target_np

# import matplotlib.pyplot as plt


# In[10]:


# def low_freq_mutate_np_center_only( amp_src, amp_trg, L=0.1 ):
#     a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
#     a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

#     _, h, w = a_src.shape
#     b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
#     c_h = np.floor(h/2.0).astype(int)
#     c_w = np.floor(w/2.0).astype(int)

#     h1 = c_h-b
#     h2 = c_h+b+1
#     w1 = c_w-b
#     w2 = c_w+b+1

#     a_src_0 = np.zeros(shape=a_src.shape)
#     a_src_0[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
#     # plt.imshow(a_src_0)
#     # plt.show()
#     a_src_0 = np.fft.ifftshift( a_src_0, axes=(-2, -1) )
#     return a_src_0


# In[11]:


# def FDA_source_to_target_np_center_only( src_img, trg_img, L=0.1 ):
#     # exchange magnitude
#     # input: src_img, trg_img

#     src_img_np = src_img #.cpu().numpy()
#     trg_img_np = trg_img #.cpu().numpy()

#     # get fft of both source and target
#     fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
#     fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

#     # extract amplitude and phase of both ffts
#     amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
#     amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

#     # mutate the amplitude part of source with target
#     amp_src_ = low_freq_mutate_np_center_only( amp_src, amp_trg, L=L )

#     # mutated fft of source
#     fft_src_ = amp_src_ * np.exp( 1j * pha_src )

#     # get the mutated image
#     src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
#     src_in_trg = np.real(src_in_trg)

#     return src_in_trg


# In[12]:


# im_src = Image.open("demo_images/source.png").convert('RGB')
# im_trg = Image.open("demo_images/target.png").convert('RGB')

# im_src = im_src.resize( (1024,512), Image.BICUBIC )
# im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

# im_src = np.asarray(im_src, np.float32)
# im_trg = np.asarray(im_trg, np.float32)

# im_src = im_src.transpose((2, 0, 1))
# im_trg = im_trg.transpose((2, 0, 1))

# im_src.shape


# In[18]:


# im_src = Image.open("demo_images/source.png").convert('RGB')
# im_trg = Image.open("demo_images/target.png").convert('RGB')

# im_src = im_src.resize( (1024,512), Image.BICUBIC )
# im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

# im_src = np.asarray(im_src, np.float32)
# im_trg = np.asarray(im_trg, np.float32)

# im_src = im_src.transpose((2, 0, 1))
# im_trg = im_trg.transpose((2, 0, 1))

# src_in_trg = FDA_source_to_target_np_center_only( im_src, im_trg, L=0.1 )
# # src_in_trg = FDA_source_to_target_np( im_trg, im_src, L=0.01 )


# In[ ]:


# src_in_trg = src_in_trg.transpose((1,2,0))
# src_in_trg[src_in_trg>255] = 255
# src_in_trg[src_in_trg<0] = 0
# src_in_trg = src_in_trg.astype(np.uint8)

# Image.fromarray(src_in_trg)#.save('demo_images/src_in_tar_center_only.png')


# # Read ACDC

# ## (same target)

# In[ ]:


im_trg = Image.open("demo_images/target.png").convert('RGB')
im_trg = im_trg.resize( (1024,512), Image.BICUBIC )
im_trg = np.asarray(im_trg, np.float32)
im_trg = im_trg.transpose((2, 0, 1))


# In[ ]:


Image.fromarray(im_trg.transpose((1,2,0)).astype('uint8'))


# ## (using acdc)

# In[3]:


import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
# from data import CreateTrgDataLoader
import data
from model import CreateModel
import os
from options.test_options import TestOptions
import scipy.io as sio


# In[4]:


from DL_domain_adaptation.acdc import ACDC, get_train_trans

transforms = get_train_trans(
(0.485, 0.456, 0.406), # mean for input normalization -- commented in source code
(0.229, 0.224, 0.225), # std for input normalization -- commented in source code
(512,1024), # target size (rescaling)
(512,1024), # random crop size
0.0, # random color jitter augmentation factor
0.0, # scale factor for random scale augmentation
True) # random horizontal flipping

dataset = ACDC('DL_domain_adaptation/data', split="train", transforms=transforms)
# print(next(iter(dataset)))


# In[20]:


len(dataset)
#train=400, test=500, val=106


# In[21]:


dataset[0][0].shape


# In[23]:


i = 10
# for i in range(dataset):
img = dataset[i][0].clone()
im_src = np.asarray(img, np.float32)
im_src= (im_src*255)[:,:,::-1] #the image is mirrorred..
im_src.shape


# In[24]:


Image.fromarray(im_src.transpose((1,2,0)).astype('uint8'))


# In[25]:


src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
# scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src_in_tar.png')

src_in_trg[src_in_trg > 255] = 255
src_in_trg[src_in_trg < 0] = 0
src_in_trg = src_in_trg.astype(np.uint8)
img_src_in_trg = Image.fromarray(src_in_trg)
img_src_in_trg.save('demo_images/acdc_in_tar.png')
img_src_in_trg.show()


# In[26]:


img_src_in_trg


# In[ ]:


demo_trg = Image.open("demo_images/target.png").convert('RGB')
demo_trg = demo_trg.resize( (1024,512), Image.BICUBIC )
demo_trg = np.asarray(demo_trg, np.float32)
demo_trg = demo_trg.transpose((2, 0, 1))

def to_fda(im_src, im_trg=demo_trg):
    im_src = im_src.clone()
    im_src = np.asarray(im_src, np.float32)
    im_src = (im_src*255)[:,:,::-1] #the image is mirrorred..
    
    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

    src_in_trg = src_in_trg.transpose((1,2,0))
    # scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src_in_tar.png')

    src_in_trg[src_in_trg > 255] = 255
    src_in_trg[src_in_trg < 0] = 0
    src_in_trg = src_in_trg.astype(np.uint8)
    img_src_in_trg = Image.fromarray(src_in_trg)

    return img_src_in_trg   


# # Args

# # Evaluation

# In[27]:


import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
# from data import CreateTrgDataLoader
import data
from model import CreateModel
import os
from options.test_options import TestOptions
import scipy.io as sio


# In[39]:


import numpy as np
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from data.synthia_dataset import SYNDataSet

data_dir_target = '../data_semseg/cityscapes'
data_list_target = './dataset/cityscapes_list/train.txt'
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}
cs_size_test = {'cityscapes': (1344,576)}

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
mean_img = torch.zeros(1, 1)

num_steps = 150
batch_size = 1
num_workers = 4

def CreateTrgDataLoader():
    target_dataset = cityscapesDataSetLabel( data_dir_target, 
                                                data_list_target, 
                                                crop_size=image_sizes['cityscapes'], 
                                                mean=IMG_MEAN, 
                                                max_iters=num_steps * batch_size, 
                                                set='train' )

    target_dataloader = data.DataLoader( target_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True )

    return target_dataloader


# In[47]:


target_dataset = cityscapesDataSetLabel( data_dir_target, 
                                            data_list_target, 
                                            crop_size=image_sizes['cityscapes'], 
                                            mean=IMG_MEAN, 
                                            max_iters=num_steps * batch_size, 
                                            set='train' )


# In[5]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# In[7]:


train_features, train_labels, _ = next(iter(train_dataloader))


# In[22]:


train_features


# In[ ]:


def main():
    # transforms = get_train_trans(
    # (0.485, 0.456, 0.406), # mean for input normalization
    # (0.229, 0.224, 0.225), # std for input normalization
    # (512,1024), # target size (rescaling)
    # (512,1024), # random crop size
    # 0.0, # random color jitter augmentation factor
    # 0.0, # scale factor for random scale augmentation
    # True) # random horizontal flipping

    # dataset = ACDC('DL_domain_adaptation/data', split="train", transforms=transforms)
    # # print('Total testing data: '+ dataset[0].shape)
    # # print(next(iter(dataset)))

    # targetloader = CreateTrgDataLoader(dataset)

    targetloader = CreateTrgDataLoader(args)
    
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.save + "/model1"):
        os.makedirs(args.save + "/model1")
    if not os.path.exists(args.save + "/model2"):
        os.makedirs(args.save + "/model2")
    if not os.path.exists(args.save + "/model3"):
        os.makedirs(args.save + "/model3")
    if not os.path.exists(args.save + "/multi_model"):
        os.makedirs(args.save + "/multi_model")
    
    if args.restore_opt1 is not None:
        args.restore_from = args.restore_opt1
        model1 = CreateModel(args)
        model1.eval()
        # model1.cuda()

    if args.restore_opt2 is not None:
        args.restore_from = args.restore_opt2
        model2 = CreateModel(args)
        model2.eval()
        # model2.cuda()
        
    if args.restore_opt3 is not None:
        args.restore_from = args.restore_opt3
        model3 = CreateModel(args)
        model3.eval()
        # model3.cuda()

    

    # change the mean for different dataset other than CS
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)

    # ------------------------------------------------- #
    # compute scores and save them
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0:
                print( '%d processed' % index )
            image, _, name = batch                              # 1. get image
            name = name[0].split('/')[-1]
            # create mean image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)             # 2. get mean image
            
            image_dfa = to_fda(image)                          # 2.5 compute image after fda 
            image_dfa = image_dfa.clone() - mean_img                    # 3, image_dfa - mean_img
            image_dfa = Variable(image_dfa).cuda()

            if args.restore_opt1 is not None:
                # forward1
                output1 = model1(image)
                output1 = nn.functional.softmax(output1, dim=1)

                # save pred of model1
                output = nn.functional.interpolate(output1, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask)
                output_nomask.save(  '%s/%s/%s' % (args.save, "model1", name)  )
                output_col.save(  '%s/%s/%s_color.png' % (args.save, "model1", name.split('.')[0])  )
            else:
                output1 = 0

            if args.restore_opt2 is not None:
                # forward2
                output2 = model2(image)
                output2 = nn.functional.softmax(output2, dim=1)

                # save pred of model2
                output = nn.functional.interpolate(output2, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask)    
                output_nomask.save(  '%s/%s/%s' % (args.save, "model2", name)  )
                output_col.save(  '%s/%s/%s_color.png' % (args.save, "model2", name.split('.')[0])  )
            else:
                output2 = 0

            if args.restore_opt3 is not None:
                # forward3
                output3 = model3(image)
                output3 = nn.functional.softmax(output3, dim=1)

                # save pred of model3
                output = nn.functional.interpolate(output3, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask)    
                output_nomask.save(  '%s/%s/%s' % (args.save, "model3", name)  )
                output_col.save(  '%s/%s/%s_color.png' % (args.save, "model3", name.split('.')[0])  )
            else:
                output3 = 0

             # model confidence fusion
            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3

            # save pred of fused model
            output = nn.functional.interpolate(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)

            output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
            output_col = colorize_mask(output_nomask)
            output_nomask = Image.fromarray(output_nomask)    
            output_nomask.save(  '%s/%s/%s' % (args.save, "multi_model", name)  )
            output_col.save(  '%s/%s/%s_color.png' % (args.save, "multi_model", name.split('.')[0])  ) 
            
    # scores computed and saved
    # ------------------------------------------------- #
    if args.restore_opt1 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model1", args.devkit_dir, args.save + "/model1" )
    if args.restore_opt2 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model2", args.devkit_dir, args.save + "/model2" )
    if args.restore_opt3 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model3", args.devkit_dir, args.save + "/model3" ) 
    compute_mIoU( args.gt_dir, args.save + "/multi_model", args.devkit_dir, args.save + "/multi_model" ) 


if __name__ == '__main__':
    main()


# In[ ]:




