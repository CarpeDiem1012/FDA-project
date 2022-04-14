#%%
import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data import CreateTrgDataLoader
import data
from model import CreateModel
import os
from options.test_options import TestOptions
import copy

#%%
# ORIGINAL CODE
# color coding of semantic classes
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def fast_hist(a, b, n):
    k = (a>=0) & (a<n)
    return np.bincount( n*a[k].astype(int)+b[k], minlength=n**2 ).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / ( hist.sum(1)+hist.sum(0)-np.diag(hist) )

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[ input==mapping[ind][0] ] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

# def compute_mIoU( gt_dir, pred_dir, devkit_dir='', restore_from='' ):
#     # if (restore_from is None): restore_from = '_' #newly add
    
#     with open( osp.join(devkit_dir, 'info.json'),'r' ) as fp:
#         info = json.load(fp)
#     num_classes = np.int(info['classes'])
#     print('Num classes', num_classes)

#     name_classes = np.array(info['label'], dtype=np.str)
#     mapping = np.array( info['label2train'],dtype=np.int )
#     hist = np.zeros( (num_classes, num_classes) )

#     image_path_list = osp.join( devkit_dir, 'val.txt')
#     label_path_list = osp.join( devkit_dir, 'label.txt')
#     gt_imgs = open(label_path_list, 'r').read().splitlines()
#     gt_imgs = [osp.join(gt_dir, x) for x in gt_imgs]
#     pred_imgs = open(image_path_list, 'r').read().splitlines()
#     pred_imgs = [osp.join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
#     for ind in range(len(gt_imgs)):
#         pred  = np.array(Image.open(pred_imgs[ind]))
#         label = np.array(Image.open(gt_imgs[ind]))
#         label = label_mapping(label, mapping)
#         if len(label.flatten()) != len(pred.flatten()):
#             print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format( len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind] ))
#             continue
#         hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
#         if ind > 0 and ind % 10 == 0:
#             print("writing mIou")
#             with open(restore_from+'_mIoU.txt', 'a') as f:
#                 f.write( '{:d} / {:d}: {:0.2f}\n'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))) )
#             print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
#     hist2 = np.zeros((19, 19))

#     for i in range(19):
#         hist2[i] = hist[i]/np.sum(hist[i])

#     mIoUs = per_class_iu(hist)
#     for ind_class in range(num_classes):
#         with open(restore_from+'_mIoU.txt', 'a') as f:
#             f.write('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)) + '\n')
#         print('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)))
#     with open(restore_from+'_mIoU.txt', 'a') as f:
#         f.write('===> mIoU19: ' + str(round(np.nanmean(mIoUs)*100,2)) + '\n')
#         f.write('===> mIoU16: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )) + '\n')
#         f.write('===> mIoU13: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )) + '\n')

#     print('===> mIoU19: ' + str(round(   np.nanmean(mIoUs)*100,2   )))
#     print('===> mIoU16: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))
#     print('===> mIoU13: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))

# new miou
def compute_mIoU_new(outputs, dataloader, restore_from='' ):
    with open( osp.join('./dataset/cityscapes_list', 'info.json'),'r' ) as fp:
        info = json.load(fp)
    num_classes = int(info['classes'])
    # print('Num classes', num_classes)

    name_classes = np.array(info['label'], dtype=str)
    mapping = np.array( info['label2train'],dtype=int )
    hist = np.zeros( (num_classes, num_classes) )

    with torch.no_grad():
        for ind, batch in enumerate(dataloader):
            # if ind == 2: break
            if ind % 100 == 0:
                print( '%d processed' % ind )
            _, label, name = batch
            label = label_mapping(label, mapping)
            pred = outputs.transpose(2,0,1)[ind:ind+1,:,:]
            # print('{}, {}'.format('Prediction:', pred.shape))
            # print('{}, {}'.format('Outputs:', outputs.shape))
            # print('{}, {}'.format('Label:', label.shape))
            
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, index= {:d}'.format(len(label.flatten()), len(pred.flatten()), ind))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
            if ind > 0 and ind % 10 == 0:
                print("writing mIou")
                with open(restore_from+'_mIoU.txt', 'a') as f:
                    f.write( '{:d} / {:d}: {:0.2f}\n'.format(ind, len(name), 100*np.mean(per_class_iu(hist))) )
                print('{:d} / {:d}: {:0.2f}'.format(ind, len(name), 100*np.mean(per_class_iu(hist))))
            
    hist2 = np.zeros((19, 19))
    for i in range(19):
        hist2[i] = hist[i]/np.sum(hist[i])

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        with open(restore_from+'_mIoU.txt', 'a') as f:
            f.write('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)) + '\n')
        print('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)))
    with open(restore_from+'_mIoU.txt', 'a') as f:
        f.write('===> mIoU: ' + str(round(np.nanmean(mIoUs)*100,2)) + '\n')

    print('===> mIoU19: ' + str(round(   np.nanmean(mIoUs)*100,2   )))
    print('===> mIoU16: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))
    print('===> mIoU13: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))

#%%
# NEW CODE
from PIL import Image
import scipy.misc
 
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
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

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


#%%
from DL_domain_adaptation.acdc import ACDC, get_train_trans
from torch.utils.data import DataLoader

def main():
    # add acdc
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

    targetloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
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
        model1.cuda()

    if args.restore_opt2 is not None:
        args.restore_from = args.restore_opt2
        model2 = CreateModel(args)
        model2.eval()
        model2.cuda()
        
    if args.restore_opt3 is not None:
        args.restore_from = args.restore_opt3
        model3 = CreateModel(args)
        model3.eval()
        model3.cuda()
    
    # change the mean for different dataset other than CS
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)

    # ------------------------------------------------- #
    # compute scores and save them

    # create a place for predicted_images
    outputall_1 = None
    outputall_2 = None
    outputall_3 = None
    outputall_avg = None
                              
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index==2: break
            if index % 100 == 0:
                print( '%d processed' % index )
            image, _, name = batch                              # 1. get image
            name = name[0].split('/')[-1]
            # create mean image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)             # 2. get mean image
            image = image.clone() - mean_img                    # 3, image - mean_img
            image = Variable(image).cuda()

            if args.restore_opt1 is not None:
                # forward1
                output1 = model1(image)
                output1 = nn.functional.softmax(output1, dim=1)

                # save pred of model1
                output = nn.functional.interpolate(output1, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                
                if outputall_1 is None:
                    outputall_1 = copy.copy(output_nomask)
                else:
                    outputall_1 = np.dstack((outputall_1,output_nomask))
                
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask)
                # save results
                # output_nomask.save(  '%s/%s/%s' % (args.save, "model1", name)  )
                # output_col.save(  '%s/%s/%s_color.png' % (args.save, "model1", name.split('.')[0])  )
            else:
                output1 = 0

            if args.restore_opt2 is not None:
                # forward2
                output2 = model2(image)
                output2 = nn.functional.softmax(output2, dim=1)

                # save pred of model2
                print('{} {}'.format('output_shape', output2.shape))
                output = nn.functional.interpolate(output2, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                
                if outputall_2 is None:
                    outputall_2 = copy.copy(output_nomask)
                else:
                    outputall_2 = np.dstack((outputall_2,output_nomask))
                
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask) 
                # save results                
                # output_nomask.save(  '%s/%s/%s' % (args.save, "model2", name)  )
                # output_col.save(  '%s/%s/%s_color.png' % (args.save, "model2", name.split('.')[0])  )
            else:
                output2 = 0

            if args.restore_opt3 is not None:
                # forward3
                output3 = model3(image)
                output3 = nn.functional.softmax(output3, dim=1)

                # save pred of model3
                output = nn.functional.interpolate(output3, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
                
                if outputall_3 is None:
                    outputall_3 = copy.copy(output_nomask)
                else:
                    outputall_3 = np.dstack((outputall_3,output_nomask))
                
                output_col = colorize_mask(output_nomask)
                output_nomask = Image.fromarray(output_nomask)    
                # save results
                # output_nomask.save(  '%s/%s/%s' % (args.save, "model3", name)  )
                # output_col.save(  '%s/%s/%s_color.png' % (args.save, "model3", name.split('.')[0])  )
            else:
                output3 = 0

             # model confidence fusion
            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3

            # save pred of fused model
            output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
            
            if outputall_avg is None:
                outputall_avg = copy.copy(output_nomask)
            else:
                outputall_avg = np.dstack((outputall_avg,output_nomask))

            output_col = colorize_mask(output_nomask)
            output_nomask = Image.fromarray(output_nomask)
            
            # output_nomask.save(  '%s/%s/%s' % (args.save, "multi_model", name)  )
            # output_col.save(  '%s/%s/%s_color.png' % (args.save, "multi_model", name.split('.')[0])  )
            
    # scores computed and saved
    # ------------------------------------------------- #
    if args.restore_opt1 is not None:
        compute_mIoU_new(outputall_1, targetloader, args.save + "/acdc_model1" )
    #     compute_mIoU( args.gt_dir, args.save + "/model1", args.devkit_dir, args.save + "/model1" )
    if args.restore_opt2 is not None:
        compute_mIoU_new(outputall_2, targetloader, args.save + "/acdc_model2" )
    #     compute_mIoU( args.gt_dir, args.save + "/model2", args.devkit_dir, args.save + "/model2" )
    if args.restore_opt3 is not None:
        compute_mIoU_new(outputall_3, targetloader, args.save + "/acdc_model3" )
    #     compute_mIoU( args.gt_dir, args.save + "/model3", args.devkit_dir, args.save + "/model3" ) 
    compute_mIoU_new(outputall_3, targetloader, args.save + "/acdc_multi_model" )
    # compute_mIoU( args.gt_dir, args.save + "/multi_model", args.devkit_dir, args.save + "/multi_model" ) 


if __name__ == '__main__':
    main()