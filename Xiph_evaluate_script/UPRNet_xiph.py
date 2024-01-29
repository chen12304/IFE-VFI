import cv2
import math
import numpy as np
import argparse
import warnings
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.pipeline import Pipeline
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 256):
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


def evaluate(ppl, test_data_path):
    path = test_data_path
    for strCategory in ['resized', 'cropped']:
        psnr_list = []
        ssim_list = []
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
                
                
                pred, _  = ppl.inference(tenFirst, tenSecond,
                        pyr_level=PYR_LEVEL,
                        nr_lvl_skipped=NR_LVL_SKIPPED)
                pred = padder.unpad(pred)[0]
                def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                psnr = calculate_psnr(pred, npyReference)
                pred_tensor = torch.Tensor(pred.transpose(2, 0, 1)).unsqueeze(0)
                gt_tensor = torch.Tensor(npyReference.transpose(2, 0, 1)).unsqueeze(0)
                ssim = ssim_matlab(pred_tensor, gt_tensor)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                print('\r {} frame:{} psnr:{} ssim:{}'.format(strFile, intFrame, psnr, ssim), end = '')
        print(sum(ssim_list)/len(ssim_list))
        print(sum(psnr_list)/len(psnr_list))







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on 4k1000fps' +\
            'dataset for 8x multiple interpolation')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--test_data_path', type=str, required=True,
            help='the path of 4k10000fps benchmark')


    #**********************************************************#
    # => args for model
    parser.add_argument('--pyr_level', type=int, default=7,
            help='the number of pyramid levels of UPR-Net in testing')
    parser.add_argument('--nr_lvl_skipped', type=int, default=2,
            help='the number of skipped high-resolution pyramid levels '\
                    'of UPR-Net in testing')
    ## test base version of UPR-Net by default
    parser.add_argument('--model_size', type=str, default="LARGE",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--model_file', type=str,
            default="./checkpoints/upr-llarge.pkl",
            help='weight of UPR-Net')

    ## test large version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="large",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-large.pkl",
    #         help='weight of UPR-Net')

    ## test LARGE version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="LARGE",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-llarge.pkl",
    #         help='weight of UPR-Net')


    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True

    #**********************************************************#
    # => init the pipeline and start to benchmark
    args = parser.parse_args()

    model_cfg_dict = dict(
            load_pretrain = True,
            model_size = args.model_size,
            model_file = args.model_file
            )
    ppl = Pipeline(model_cfg_dict)

    # resolution-aware parameter for inference
    PYR_LEVEL = args.pyr_level
    NR_LVL_SKIPPED = args.nr_lvl_skipped

    print("benchmarking on 4K1000FPS...")
    evaluate(ppl, args.test_data_path)
