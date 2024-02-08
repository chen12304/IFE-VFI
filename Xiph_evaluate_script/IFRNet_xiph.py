import os
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np

import skimage

from utils import read
from metric import calculate_psnr, calculate_ssim
# from models.IFRNet import Model
from models.IFRNet_L import Model
# from models.IFRNet_S import Model
from PIL import Image
import cv2
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Model()
# model.load_state_dict(torch.load('checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))
model.load_state_dict(torch.load('checkpoints/IFRNet/IFRNet_L_Vimeo90K.pth'))
# model.load_state_dict(torch.load('checkpoints/IFRNet_small/IFRNet_S_Vimeo90K.pth'))
model.eval()
model.cuda()

divisor = 20
scale_factor = 0.8

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=divisor):
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

# Replace the 'path' with your Xiph dataset absolute path.
path = '/data/xiph/netflix'
i=0
savepath = 'xiph/'
os.makedirs(savepath, exist_ok=True)
for strCategory in ['resized', 'cropped']:
    psnr_list = []
    ssim_list = []
    for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']: 
        for intFrame in range(2, 99, 2):
            I0 = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
            I1 = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
            I2 = cv2.imread(filename=path + '/' + strFile + '-' + str(intFrame).zfill(3) + '.png', flags=-1)            
            if strCategory == 'resized':
                I0 = cv2.resize(src=I0, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                np_ref = cv2.resize(src=I1, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                I2 = cv2.resize(src=I2, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            elif strCategory == 'cropped':
                I0 = I0[540:-540, 1024:-1024, :]
                np_ref = I1[540:-540, 1024:-1024, :]
                I2 = I2[540:-540, 1024:-1024, :]

            I0 = torch.FloatTensor(np.ascontiguousarray(I0.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
            I2 = torch.FloatTensor(np.ascontiguousarray(I2.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
            padder = InputPadder(I0.shape)
            I0, I2 = padder.pad(I0, I2)
            embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
            # print(I1.shape, I0.shape)
            I1_pred = model.inference(I0, I2, embt, scale_factor=scale_factor)
            # print(I1_pred.shape)
            I1_pred = padder.unpad(I1_pred)[0]
            # print(I1_pred.shape)
            # print(I1.shape, I1_pred.shape)

            # psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
            I1_pred = (I1_pred.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            def calculate_psnr(image1, image2, data_range=255.0):
                    mse = np.mean((image1 - image2) ** 2)
                    psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
                    return psnr_value
                
            psnr = calculate_psnr(I1_pred, np_ref)
            pred_tensor = torch.Tensor(I1_pred.transpose(2, 0, 1)).unsqueeze(0)
            gt_tensor = torch.Tensor(np_ref.transpose(2, 0, 1)).unsqueeze(0)

            ssim = calculate_ssim(pred_tensor, gt_tensor).detach().cpu().numpy()
            print('\r {} frame:{} psnr:{} ssim:{}'.format(strFile, intFrame, psnr, ssim), end = '')
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
    if strCategory == 'resized':
        print('\n---2K---')
    else:
        print('\n---4K---')
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))

