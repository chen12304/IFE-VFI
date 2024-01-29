## script on Xiph
Some previous scripts compute PSNR and SSIM before quantization, some compute after quantization. We re-evaluate on it for fair comparison. PSNR and SSIM are computed after quantization.

**DQBC**
Add the DQBC_xiph.py to DQBC-master/ and run the command below:

```bash
python DQBC_xiph.py --config configs/test.yaml --path /data/yourpath/xiph/netflix --gpu_id 1
```

**CURE**
Add the CURE_xiph.py to CURE-main/ and run the command below:

```bash
python CURE_xiph.py -dbdir /data/yourpath/xiph/netflix
```

**IFRNet**
Add IFRNet_xiph.py to IFRNet-main/benchmarks/ and run the command below:

```bash
python benchmarks/IFRNet_xiph.py
```

**EBME**
Add EBME_xiph.py to EBME-master/tools/ and run the command below:

```bash
python -m tools.EBME_xiph --data_root /data/yourpath/xiph/netflix
```

**EMA-VFI**
Add Xiph.py to EMA-VFI-main/benchmark/ and run the command below:
```bash
python benchmark/Xiph.py --path /data/yourpath/xiph/netflix
```

**UPRNet**
Add UPRNet_xiph.py to UPR-Net-master/tools/ and run the command below:
```bash
python3 -m tools.UPRNet_xiph --test_data_path /data/yourpath/xiph/netflix
```
