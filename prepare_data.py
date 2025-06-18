import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import shutil

def create_folder(folder):
    """创建文件夹，如果已存在则清空"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def process_sr_images(gt_folder, lq_folder, scale):
    """处理超分辨率任务的图像
    Args:
        gt_folder: 高质量图像文件夹
        lq_folder: 低质量图像文件夹
        scale: 下采样倍数
    """
    print(f"处理超分辨率任务，下采样倍数: {scale}")
    create_folder(lq_folder)
    
    for img_name in tqdm(os.listdir(gt_folder)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取GT图像
            gt_path = os.path.join(gt_folder, img_name)
            img = cv2.imread(gt_path)
            if img is None:
                print(f"无法读取图像: {gt_path}")
                continue
                
            # 获取原始尺寸
            h, w = img.shape[:2]
            
            # 下采样生成LQ图像
            lq_img = cv2.resize(img, (w//scale, h//scale), 
                              interpolation=cv2.INTER_CUBIC)
            
            # 保存LQ图像
            lq_path = os.path.join(lq_folder, img_name)
            cv2.imwrite(lq_path, lq_img)

def process_denoise_images(gt_folder, lq_folder, noise_level):
    """处理去噪任务的图像
    Args:
        gt_folder: 高质量图像文件夹
        lq_folder: 低质量图像文件夹
        noise_level: 噪声级别
    """
    print(f"处理去噪任务，噪声级别: {noise_level}")
    create_folder(lq_folder)
    
    for img_name in tqdm(os.listdir(gt_folder)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取GT图像
            gt_path = os.path.join(gt_folder, img_name)
            img = cv2.imread(gt_path)
            if img is None:
                print(f"无法读取图像: {gt_path}")
                continue
                
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
            lq_img = cv2.add(img, noise)
            
            # 保存LQ图像
            lq_path = os.path.join(lq_folder, img_name)
            cv2.imwrite(lq_path, lq_img)

def process_jpeg_images(gt_folder, lq_folder, quality):
    """处理JPEG压缩伪影去除任务的图像
    Args:
        gt_folder: 高质量图像文件夹
        lq_folder: 低质量图像文件夹
        quality: JPEG压缩质量
    """
    print(f"处理JPEG压缩伪影去除任务，压缩质量: {quality}")
    create_folder(lq_folder)
    
    for img_name in tqdm(os.listdir(gt_folder)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取GT图像
            gt_path = os.path.join(gt_folder, img_name)
            img = cv2.imread(gt_path)
            if img is None:
                print(f"无法读取图像: {gt_path}")
                continue
                
            # 保存为JPEG格式（模拟压缩）
            lq_path = os.path.join(lq_folder, img_name)
            cv2.imwrite(lq_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

def check_image_pairs(gt_folder, lq_folder):
    """检查GT和LQ图像对是否匹配"""
    print("检查图像对...")
    gt_images = set(os.listdir(gt_folder))
    lq_images = set(os.listdir(lq_folder))
    
    # 检查文件数量
    if len(gt_images) != len(lq_images):
        print(f"警告: GT图像数量({len(gt_images)})与LQ图像数量({len(lq_images)})不匹配")
    
    # 检查文件名匹配
    missing_in_lq = gt_images - lq_images
    missing_in_gt = lq_images - gt_images
    
    if missing_in_lq:
        print(f"警告: 以下图像在LQ文件夹中缺失: {missing_in_lq}")
    if missing_in_gt:
        print(f"警告: 以下图像在GT文件夹中缺失: {missing_in_gt}")

def main():
    parser = argparse.ArgumentParser(description='准备SwinIR训练数据')
    parser.add_argument('--task', type=str, required=True, 
                      choices=['sr', 'denoise', 'jpeg'],
                      help='任务类型: sr(超分辨率), denoise(去噪), jpeg(压缩伪影去除)')
    parser.add_argument('--gt_folder', type=str, required=True,
                      help='高质量图像文件夹路径')
    parser.add_argument('--lq_folder', type=str, required=True,
                      help='低质量图像文件夹路径')
    parser.add_argument('--scale', type=int, default=2,
                      help='超分辨率下采样倍数 (默认: 2)')
    parser.add_argument('--noise', type=int, default=15,
                      help='噪声级别 (默认: 15)')
    parser.add_argument('--quality', type=int, default=40,
                      help='JPEG压缩质量 (默认: 40)')
    
    args = parser.parse_args()
    
    # 确保输入文件夹存在
    if not os.path.exists(args.gt_folder):
        raise ValueError(f"GT文件夹不存在: {args.gt_folder}")
    
    # 根据任务类型处理图像
    if args.task == 'sr':
        process_sr_images(args.gt_folder, args.lq_folder, args.scale)
    elif args.task == 'denoise':
        process_denoise_images(args.gt_folder, args.lq_folder, args.noise)
    elif args.task == 'jpeg':
        process_jpeg_images(args.gt_folder, args.lq_folder, args.quality)
    
    # 检查处理结果
    check_image_pairs(args.gt_folder, args.lq_folder)
    print("数据处理完成！")

if __name__ == '__main__':
    main() 