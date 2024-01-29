import cv2
import math
import numpy as np
import argparse
import warnings

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.unified_ppl import Pipeline
from core.dataset import SnuFilm
from core.utils.pytorch_msssim import ssim_matlab

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on xiph')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--data_root', type=str, required=True,
            help='root dir of ucf101')
    parser.add_argument('--batch_size', type=int, default=1,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=1,
            help='number of the worker for data loader')

    #**********************************************************#
    # => args for optical flow model
    parser.add_argument('--flow_model_file', type=str,
            default="./checkpoints/ebme-h/bi-flownet.pkl",
            help='weight of the bi-directional flow model')
    parser.add_argument('--pyr_level', type=int, default=3,
            help='the number of pyramid levels in testing')

    #**********************************************************#
    # => args for frame fusion (synthesis) model
    parser.add_argument('--fusion_model_file', type=str,
            default="./checkpoints/ebme-h/fusionnet.pkl",
            help='weight of the frame fusion model')
    # set `high_synthesis` as True, only when training or loading
    # high-resolution synthesis model.
    parser.add_argument('--high_synthesis', type=bool, default=True,
            help='whether use high-resolution synthesis')


    #**********************************************************#
    # => whether use test augmentation
    parser.add_argument('--test_aug', type=bool, default=True,
            help='whether use test time augmentation')

    args = parser.parse_args()

    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True
    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = args.pyr_level
    bi_flownet_args.load_pretrain = True
    bi_flownet_args.model_file = args.flow_model_file

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = args.high_synthesis
    fusionnet_args.load_pretrain = True
    fusionnet_args.model_file = args.fusion_model_file
    module_cfg_dict = dict(
            bi_flownet = bi_flownet_args,
            fusionnet = fusionnet_args
            )

    ppl = Pipeline(module_cfg_dict)

    path = args.data_root
    for strCategory in ['resized', 'cropped']:
        psnr, ssim = [], []
        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']: 
            for intFrame in range(2, 99, 2):
                npyFirst = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
                npySecond = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
                npyReference = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png', flags=-1)
                if strCategory == 'resized':
                    npyFirst = cv2.resize(src=npyFirst, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npySecond = cv2.resize(src=npySecond, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npyReference = cv2.resize(src=npyReference, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

                elif strCategory == 'cropped':
                    npyFirst = npyFirst[540:-540, 1024:-1024, :]
                    npySecond = npySecond[540:-540, 1024:-1024, :]
                    npyReference = npyReference[540:-540, 1024:-1024, :]

                tenFirst = torch.FloatTensor(np.ascontiguousarray(npyFirst.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                tenSecond = torch.FloatTensor(np.ascontiguousarray(npySecond.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()

                padder = InputPadder(tenFirst.shape)
                tenFirst, tenSecond = padder.pad(tenFirst, tenSecond)
                with torch.no_grad():
                    if args.test_aug:
                        pred1,_ = ppl.inference(tenFirst, tenSecond)
                        tenFirst = torch.flip(tenFirst, dims=[2 ,3])
                        tenSecond = torch.flip(tenSecond, dims=[2 ,3])
                        pred2, _ = ppl.inference(tenFirst, tenSecond)
                        pred2 = torch.flip(pred2, dims=[2 ,3])
                        pred = 0.5 * pred1 + 0.5 * pred2
                pred = padder.unpad(pred)[0]
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

                def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
                print('\r {} frame:{}'.format(strFile, intFrame), end = '')
                
                psnr.append(calculate_psnr(pred, npyReference))
                pred_np = torch.Tensor(pred.transpose(2, 0, 1)).unsqueeze(0)
                gt_np = torch.Tensor(npyReference.transpose(2, 0, 1)).unsqueeze(0)
                # print(pred_np.shape,gt_np.shape)
                ssim.append(float(ssim_matlab(pred_np, gt_np)))
                # print(psnr,ssim)
        if strCategory == 'resized':
            print('******2k*****')
        print(sum(psnr) / len(psnr))
        print(sum(ssim) / len(ssim))



