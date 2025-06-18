# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from collections import OrderedDict
import sys
import os
from basicsr.archs.fsanet_arch import FSANET
sys.path.append(os.path.abspath('.'))

from basicsr.archs.hfmn_arch import HFMN
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr, scandir

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt

def forward_x8(model, x, scale, device):
    """
    几何自集成算法 (forward_x8) 的实现。
    对输入图像进行 8 种几何变换，通过模型推理后融合结果。
    """
    # 获取输入图像的批次大小、通道数、高度和宽度
    b, c, h, w = x.size()
    
    # 初始化输出张量，用于存储融合后的结果
    output = torch.zeros(b,c,h*scale,w*scale).to(device)
    # 定义旋转和翻转操作
    def rotate90(x):
        return x.transpose(2, 3).flip(2)
    
    def rotate180(x):
        return x.flip(2).flip(3)
    
    def rotate270(x):
        return x.transpose(2, 3).flip(3)
    
    def flip_horizontal(x):
        return x.flip(3)
    
    def flip_vertical(x):
        return x.flip(2)
    # 原始图像
    output += model(x)
    
    # 水平翻转
    output += flip_horizontal(model(flip_horizontal(x)))
    
    # 垂直翻转
    output += flip_vertical(model(flip_vertical(x)))
    
    # 旋转 90 度
    output += rotate270(model(rotate90(x)))
    
    # 旋转 180 度
    output += rotate180(model(rotate180(x)))
    
    # 旋转 270 度
    output += rotate90(model(rotate270(x)))
    
    # 水平 + 垂直翻转
    output += flip_horizontal(flip_vertical(model(flip_vertical(flip_horizontal(x)))))
    
    # 旋转 90 + 水平翻转
    output += rotate270(flip_horizontal(model(flip_horizontal(rotate90(x)))))
    
    # 平均 8 种结果
    output /= 8.0
    
    return output

def forward_x12(model, x, scale, device):
    b, c, h, w = x.shape
    output = torch.zeros(b, c, h * scale, w * scale).to(device)

    # 定义旋转和翻转操作
    def rotate90(x): return x.transpose(2, 3).flip(2)
    def rotate180(x): return x.flip(2).flip(3)
    def rotate270(x): return x.transpose(2, 3).flip(3)
    def flip_horizontal(x): return x.flip(3)
    def flip_vertical(x): return x.flip(2)

    # 原始图像
    output += model(x)
    
    # 水平翻转
    output += flip_horizontal(model(flip_horizontal(x)))
    
    # 垂直翻转
    output += flip_vertical(model(flip_vertical(x)))
    
    # 旋转 90 度
    output += rotate270(model(rotate90(x)))
    
    # 旋转 180 度
    output += rotate180(model(rotate180(x)))
    
    # 旋转 270 度
    output += rotate90(model(rotate270(x)))
    
    # 水平 + 垂直翻转
    output += flip_horizontal(flip_vertical(model(flip_vertical(flip_horizontal(x)))))
    
    # 旋转 90° + 水平翻转
    output += rotate270(flip_horizontal(model(flip_horizontal(rotate90(x)))))

    # 新增变换
    # 旋转 90° + 垂直翻转
    output += rotate270(flip_vertical(model(flip_vertical(rotate90(x)))))

    # 旋转 270° + 垂直翻转
    output += rotate90(flip_vertical(model(flip_vertical(rotate270(x)))))

    # 旋转 180° + 水平翻转
    output += rotate180(flip_horizontal(model(flip_horizontal(rotate180(x)))))

    # 旋转 180° + 垂直翻转
    output += rotate180(flip_vertical(model(flip_vertical(rotate180(x)))))

    # 计算 12 种变换的平均值
    output /= 12.0
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set5/HR', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/HFMN/Set5', help='output folder')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/HFMN_x4sr.pth')
    args = parser.parse_args()
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        imgname = os.path.splitext(os.path.basename(path))[0]
        # print('Testing', idx, imgname)
        # read image
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            output = forward_x12(model,img_lq,args.scale,device)
            # output = model(img_lq)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = calculate_psnr(output, img_gt, crop_border=args.scale)
            ssim = calculate_ssim(output, img_gt, crop_border=args.scale)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = calculate_psnr(output_y, img_gt_y, crop_border=args.scale, test_y_channel=True)
                ssim_y = calculate_ssim(output_y, img_gt_y, crop_border=args.scale, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(args.output, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


def define_model(args):
    model = FSANET(
        upscale=args.scale,
        in_chans=3,
        img_size=64,
        window_size=2,
        img_range=1.,
        depths=[2, 2, 2, 2],
        embed_dim=32,
        num_heads=8,
        mlp_ratio=2,
        resi_connection='1conv')


    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model



if __name__ == '__main__':
    main()
