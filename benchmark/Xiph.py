import os
import sys
import cv2
import skimage
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
from benchmark.utils.pytorch_msssim import ssim_matlab
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--number', default=299,type=int)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
down_scale = 1
if args.model == 'ours':
    
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )

model = Model(-1)
model.load_model(epoch=args.number)
model.eval()
model.device()


print(f'=========================Starting testing=========================')
print(f'Dataset: Xiph   Model: {model.name}   TTA: {TTA}')
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

            # npyEstimate = padder.unpad(model.hr_inference(tenFirst, tenSecond, TTA=TTA, down_scale=down_scale, fast_TTA=False)[0].clamp(0.0, 1.0).cpu())[0]
            npyEstimate = padder.unpad(model.inference(tenFirst, tenSecond, TTA=TTA, fast_TTA=True)[0].clamp(0.0, 1.0).cpu())
            # print(npyEstimate.shape)
            npyEstimate = (npyEstimate.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            # print(npyEstimate.shape)　(1080, 2048, 3)  
            #　print(npyReference.shape)

            def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
            
            psnr = calculate_psnr(npyEstimate, npyReference)
            # psnr = skimage.metrics.peak_signal_noise_ratio(image_true=npyReference, image_test=npyEstimate, data_range=255)
            pred_tensor = torch.Tensor(npyEstimate.transpose(2, 0, 1)).unsqueeze(0)
            gt_tensor = torch.Tensor(npyReference.transpose(2, 0, 1)).unsqueeze(0)
            ssim = ssim_matlab(pred_tensor, gt_tensor)
            # ssim = skimage.metrics.structural_similarity(im1=npyReference, im2=npyEstimate, data_range=255, multichannel=True, channel_axis=2)
            fltPsnr.append(psnr)
            fltSsim.append(ssim)
            print('\r {} frame:{} psnr:{} ssim:{}'.format(strFile, intFrame, psnr, ssim), end = '')

    if strCategory == 'resized':
        print('\n---2K---')
    else:
        print('\n---4K---')
    print('Avg psnr:', np.mean(fltPsnr))
    print('Avg ssim:', np.mean(fltSsim))