## script on Xiph
Some previous scripts compute PSNR and SSIM before quantization, some compute after quantization. We re-evaluate on it for fair comparison. PSNR and SSIM are computed after quantization.
* DQBC
  add the DQBC_xiph.py to DQBC-master/ and run the command below:
  'bash
  python DQBC_xiph.py --config configs/test.yaml --path /data/cky/xiph/netflix --gpu_id 1
  '
*
