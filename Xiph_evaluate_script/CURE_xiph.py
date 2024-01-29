import os
import sys
import cv2
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

import time
from models.CURE import *


import torch.nn.functional as F



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

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_worker', '-nw', default=5, type=int, help='number of workers to load data by dataloader')
    parser.add_argument('--eval_size', '-ebs', default=4096, type=int, help='eval batch size')
    parser.add_argument('--checkpoint_dir', '-dir', default='CURE', type=str,
                        help='dir or checkpoint')
    parser.add_argument('--dataset_basedir', '-dbdir', default='./', type=str,
                        help='dir or checkpoint')
    return parser.parse_args()
def test():
    args = args_parser()
    model = CURE(batch_size=args.eval_size)
    if '.pth.tar' in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif '.pth.tar' not in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir+'.pth.tar'

    data = torch.load(checkpoint_dir)
    if 'state_dict' in data.keys():
       model.load_state_dict(data['state_dict'])
    else:
       model.load_state_dict(data)
    model.cuda()
    model.eval()
    path = args.dataset_basedir
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
                time_frame = 0.5
                tenFirst = imNorm(tenFirst)
                tenSecond = imNorm(tenSecond)
                pred, _ = pred_frame(args, tenFirst, tenFirst, model, None, time=time_frame)
                # print(gt.shape,pred.shape)
                pred = padder.unpad(pred)[0]
                # print(torch.max(pred), torch.min(pred)) 1,0
                # print(pred.shape)
                pred = (pred.cpu().numpy().transpose(1,2,0)*255).round().astype(np.uint8)
                
                def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
                
                psnr.append(calculate_psnr(pred, npyReference))
                # print(psnr)
                pred_tensor = torch.Tensor(pred.transpose(2, 0, 1)).unsqueeze(0).cuda()
                gt_tensor = torch.Tensor(npyReference.transpose(2, 0, 1)).unsqueeze(0).cuda()
                ssim.append(compare_ssim(pred_tensor,gt_tensor).cpu())
                del tenFirst, gt_tensor, tenSecond, pred
                torch.cuda.empty_cache()
                # print(ssim)
                # print(sum(psnr) / len(psnr))
                # print(sum(ssim) / len(ssim))
        if strCategory=='resized':
            print('****2k*****')
        print(sum(psnr) / len(psnr))
        print(sum(ssim) / len(ssim))

if __name__ == '__main__':
    args = args_parser()
    print(args)
    with torch.no_grad():
        test()


