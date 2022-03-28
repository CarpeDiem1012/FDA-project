import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data import CreateTrgDataLoader
from model import CreateModel
import os
from options.test_options import TestOptions
import scipy.io as sio
from evaluation_multi import compute_mIoU

def main():
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if args.restore_opt1 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model1", args.devkit_dir, args.save + "/model1" )
    if args.restore_opt2 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model2", args.devkit_dir, args.save + "/model2" )
    if args.restore_opt3 is not None:
        compute_mIoU( args.gt_dir, args.save + "/model3", args.devkit_dir, args.save + "/model3" ) 
    compute_mIoU( args.gt_dir, args.save + "/multi_model", args.devkit_dir, args.save + "/multi_model" )
    
if __name__ == '__main__':
    main()