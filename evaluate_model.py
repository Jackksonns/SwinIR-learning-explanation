import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import glob
from collections import OrderedDict
import json
from datetime import datetime
import matplotlib.pyplot as plt
from models.network_swinir import SwinIR
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim

def bgr2ycbcr(img, only_y=True):
    """Convert BGR image to YCbCr color space.
    
    Args:
        img (numpy.ndarray): Input image in BGR format.
        only_y (bool): If True, only return Y channel.
    
    Returns:
        numpy.ndarray: YCbCr image or Y channel only.
    """
    img = img.astype(np.float32) / 255.
    if only_y:
        return np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                             [65.481, -37.797, 112.0]]) + [16, 128, 128]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', 
                        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('--test_lr_folder', type=str, required=True, help='input low-quality test image folder')
    parser.add_argument('--test_hr_folder', type=str, required=True, help='input high-quality test image folder')
    parser.add_argument('--save_folder', type=str, default='evaluation_results', help='folder to save evaluation results')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    return parser.parse_args()

def define_model(args):
    # 根据任务类型定义模型
    if args.task == 'classical_sr':
        model = SwinIR(upscale=args.scale, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    elif args.task == 'lightweight_sr':
        model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    elif args.task == 'real_sr':
        model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    elif args.task == 'gray_dn':
        model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    elif args.task == 'color_dn':
        model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    elif args.task == 'jpeg':
        model = SwinIR(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    return model

def test_tile(img_lq, model, args, window_size):
    """测试时使用tile策略"""
    if args.tile is None:
        # 测试整张图片
        with torch.no_grad():
            output = model(img_lq)
    else:
        # 测试时使用tile策略
        b, c, h, w = img_lq.size()
        tile = args.tile
        tile_overlap = args.tile_overlap

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*args.scale, w*args.scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(out_patch)
                W[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(out_patch_mask)
        output = E.div_(W)

    return output

def evaluate(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = define_model(args)
    checkpoint = torch.load(args.model_path)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    # 获取测试图片列表
    lr_images = sorted([f for f in os.listdir(args.test_lr_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    hr_images = sorted([f for f in os.listdir(args.test_hr_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # 创建保存目录
    os.makedirs(args.save_folder, exist_ok=True)
    
    # 评估结果
    results = {
        'metrics': {
            'psnr': [],
            'ssim': [],
            'psnr_y': [],
            'ssim_y': []
        },
        'images': {}
    }
    
    # 处理每张图片
    for lr_name, hr_name in tqdm(zip(lr_images, hr_images), total=len(lr_images)):
        # 读取图片
        lr_path = os.path.join(args.test_lr_folder, lr_name)
        hr_path = os.path.join(args.test_hr_folder, hr_name)
        
        img_lr = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        
        # 转换为tensor
        img_lr = torch.from_numpy(img_lr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(img_lr)
        
        # 转换回numpy
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output = output.transpose(1, 2, 0)
        
        # 如果输出和目标尺寸不匹配，调整输出尺寸
        if output.shape != img_hr.shape:
            output = cv2.resize(output, (img_hr.shape[1], img_hr.shape[0]))
        
        # 计算评估指标
        psnr = float(calculate_psnr(output, img_hr, crop_border=0))
        ssim = float(calculate_ssim(output, img_hr, crop_border=0))
        
        # 转换为Y通道
        output_y = bgr2ycbcr(output)
        img_hr_y = bgr2ycbcr(img_hr)
        psnr_y = float(calculate_psnr(output_y, img_hr_y, crop_border=0))
        ssim_y = float(calculate_ssim(output_y, img_hr_y, crop_border=0))
        
        # 保存结果
        results['metrics']['psnr'].append(psnr)
        results['metrics']['ssim'].append(ssim)
        results['metrics']['psnr_y'].append(psnr_y)
        results['metrics']['ssim_y'].append(ssim_y)
        
        # 保存图片结果
        results['images'][lr_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'psnr_y': psnr_y,
            'ssim_y': ssim_y
        }
        
        # 保存超分辨率结果
        save_path = os.path.join(args.save_folder, lr_name)
        cv2.imwrite(save_path, output)
        
        print(f"Image: {lr_name} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}; PSNR_Y: {psnr_y:.2f} dB; SSIM_Y: {ssim_y:.4f}")
    
    # 计算平均值
    avg_psnr = sum(results['metrics']['psnr']) / len(results['metrics']['psnr'])
    avg_ssim = sum(results['metrics']['ssim']) / len(results['metrics']['ssim'])
    avg_psnr_y = sum(results['metrics']['psnr_y']) / len(results['metrics']['psnr_y'])
    avg_ssim_y = sum(results['metrics']['ssim_y']) / len(results['metrics']['ssim_y'])
    
    # 添加平均值到结果
    results['average'] = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'psnr_y': avg_psnr_y,
        'ssim_y': avg_ssim_y
    }
    
    # 保存评估结果
    with open(os.path.join(args.save_folder, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n评估完成！")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 PSNR_Y: {avg_psnr_y:.2f} dB")
    print(f"平均 SSIM_Y: {avg_ssim_y:.4f}")

if __name__ == '__main__':
    args = parse_args()
    evaluate(args) 