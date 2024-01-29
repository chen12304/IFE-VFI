from models import make_model, model_profile
from utils.config import make_config
import torch
import argparse
from validate.metrics import calculate_ssim
from datas.utils import imread_rgb
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os 
torch.set_grad_enabled(False)
import cv2

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

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='.yaml config file path')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--path',type=str)
    args = parser.parse_args()
    cfg_file = args.config
    dev_id = args.gpu_id
    torch.cuda.set_device(dev_id)

    cfg = make_config(cfg_file, launch_experiment=False)
    
    print(model_profile(cfg.model))
    
    model = make_model(cfg.model)
    model.cuda()
    model.eval()

    path = args.path
for strCategory in ['resized', 'cropped']:
    fltPsnr, fltSsim = [], []
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
            pred = model(tenFirst, tenSecond)['final'][0]
            pred = padder.unpad(pred)
            pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
                
            psnr = calculate_psnr(pred, npyReference)
            # print(psnr)
            pred_tensor = torch.Tensor(pred.transpose(2, 0, 1)).unsqueeze(0)
            gt_tensor = torch.Tensor(npyReference.transpose(2, 0, 1)).unsqueeze(0)
            ssim = calculate_ssim(pred_tensor, gt_tensor)
            # print(ssim)
            fltPsnr.append(psnr)
            fltSsim.append(ssim)
    if strCategory == 'resized':
        print('***   2K   ***')
    else:
        print('***   4K   ***')
    print(sum(fltPsnr)/len(fltPsnr))
    print(sum(fltSsim)/len(fltSsim))


            
    
    



    


    

    
    
        
    