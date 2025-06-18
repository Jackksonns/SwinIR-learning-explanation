import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random
from models.network_swinir import SwinIR
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from utils.dataset import SRDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--folder_gt', type=str, default='datasets/train/GT', help='input ground-truth training image folder')
    parser.add_argument('--folder_lq', type=str, default='datasets/train/LQ', help='input low-quality training image folder')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--save_dir', type=str, default='experiments/training', help='save directory')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--log_freq', type=int, default=100, help='log frequency')
    return parser.parse_args()

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = SwinIR(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    # 002 lightweight image sr
    elif args.task == 'lightweight_sr':
        model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    # 005 color image denoising
    elif args.task == 'color_dn':
        model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    # 006 JPEG compression artifact reduction
    elif args.task == 'jpeg_car':
        model = SwinIR(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    return model

def train(args):
    # 设置随机种子
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义模型
    model = define_model(args)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 加载预训练模型（如果有）
    if args.model_path is not None:
        print(f'Loading pretrained model from {args.model_path}')
        model.load_state_dict(torch.load(args.model_path)['params'] if 'params' in torch.load(args.model_path).keys() else torch.load(args.model_path))
    
    # 设置tensorboard
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # 创建数据集和数据加载器
    train_dataset = SRDataset(
        lq_folder=args.folder_lq,
        gt_folder=args.folder_gt,
        patch_size=args.training_patch_size,
        scale=args.scale,
        task=args.task,
        noise_level=args.noise,
        jpeg_quality=args.jpeg
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{args.epochs}]", ncols=100)
        for batch_idx, (lq, gt) in progress_bar:
            lq, gt = lq.to(device), gt.to(device)
            
            # 前向传播
            output = model(lq)
            loss = criterion(output, gt)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录日志
            if batch_idx % args.log_freq == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
            progress_bar.set_postfix({'batch_loss': loss.item()})
        
        # 更新学习率
        scheduler.step()
        
        # 保存模型
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss / len(train_loader),
            }, os.path.join(args.save_dir, 'models', f'model_epoch_{epoch+1}.pth'))
        
        # 输出每个epoch的loss
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.6f}')

if __name__ == '__main__':
    args = parse_args()
    train(args) 