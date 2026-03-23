#!/usr/bin/env python3
"""
Multi-GPU U-Net Training Script for MRI Reconstruction
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
import glob
import os
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import updated_dataloader

# Import model classes
class doubleConv2D(nn.Module):
    def __init__(self,input_ch, output_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,out_channels=output_ch,kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_ch,out_channels=output_ch,kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self,input_channels, output_channels,kernel_size=2,stride=2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size,stride=stride),
            doubleConv2D(input_ch=input_channels,output_ch=output_channels)
        )
    
    def forward(self,x):
        return self.downsample(x)

class UNET_encoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.first_double_conv = doubleConv2D(1,64)
        self.downsample1 = DownSample(64,128)
        self.downsample2 = DownSample(128,256)
        self.downsample3 = DownSample(256,512)
        self.downsample4 = DownSample(512,1024)

    def forward(self,x):
        f1 = self.first_double_conv(x)
        f2 = self.downsample1(f1)
        f3 = self.downsample2(f2)
        f4 = self.downsample3(f3)
        bottleneck = self.downsample4(f4)
        return bottleneck, [f4, f3 ,f2,f1]

class UNET_decoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.tcv1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.double_conv1 = doubleConv2D(input_ch=1024,output_ch=512)
        self.tcv2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.double_conv2 = doubleConv2D(input_ch=512,output_ch=256)
        self.tcv3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.double_conv3 = doubleConv2D(input_ch=256,output_ch=128)
        self.tcv4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.double_conv4 = doubleConv2D(input_ch=128,output_ch=64)

    def forward(self,bottleneck,skip_conns_list):
        upsample_1 = self.tcv1(bottleneck)
        concat_skip_1 = torch.concat([upsample_1,skip_conns_list[0]],dim=1)
        double_conv1_op = self.double_conv1(concat_skip_1)

        upsample_2 = self.tcv2(double_conv1_op)
        concat_skip_2 = torch.concat([upsample_2,skip_conns_list[1]],dim=1)
        double_conv2_op = self.double_conv2(concat_skip_2)

        upsample_3 = self.tcv3(double_conv2_op)
        concat_skip_3 = torch.concat([upsample_3,skip_conns_list[2]],dim=1)
        double_conv3_op = self.double_conv3(concat_skip_3)

        upsample_4 = self.tcv4(double_conv3_op)
        concat_skip_4 = torch.concat([upsample_4,skip_conns_list[3]],dim=1)
        double_conv4_op = self.double_conv4(concat_skip_4)

        return double_conv4_op

class UNET_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNET_encoder()
        self.decoder = UNET_decoder()
        self.one_Cross_one_conv = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1)

    def forward(self,x):
        bottleneck, skip_conns_list = self.encoder(x)
        final_conv_op = self.decoder(bottleneck,skip_conns_list)
        final_image = self.one_Cross_one_conv(final_conv_op)
        return final_image

class SpecialLossFunc(nn.Module):
    def __init__(self,alpha=0.86):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    def forward(self,real,pred):
        l1_loss = self.l1_loss(real,pred)
        ssim = self.ssim(real,pred)
        ssim_loss = 1-ssim
        hybrid_loss =  (1-self.alpha)*l1_loss + self.alpha*ssim_loss
        return hybrid_loss

def main():
    # ==========================================
    # 1. SETUP & HYPERPARAMETERS
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50 
    batch_size = 12  # Increased for dual-GPU
    learning_rate = 2e-4  # Scaled up for larger batch size
    patience = 5  # For early stopping

    train_dir = "/home/biswamitra/health/knee_data/train/deconstructed_train/"
    val_dir = "/home/biswamitra/health/knee_data/val/deconstructed_val/"
    saved_model_path = "/home/biswamitra/health/knee_data/EDA/saved_model/unet_mri_model_ispace_variable_mask.pth"

    # ==========================================
    # 2. DATALOADERS
    # ==========================================
    train_files = sorted(glob.glob(train_dir + "*.npy"))
    val_files = sorted(glob.glob(val_dir + "*.npy"))

    print(f"Found {len(train_files)} training files and {len(val_files)} validation files.")

    class MaskFunc:
        def __init__(self, center_fractions , accelerations):
            self.equispaced = EquiSpacedMaskFunc(center_fractions=center_fractions,accelerations=accelerations)
            self.randommask = RandomMaskFunc(center_fractions=center_fractions,accelerations=accelerations)

        def __call__(self,shape, seed=None, half_scan_percentage=None):
            if random.random()>0.5:
                return self.equispaced(shape, seed, half_scan_percentage)
            else:
                return self.randommask(shape, seed, half_scan_percentage)
            
    fractions = [0.08,0.04,0.02]
    accels = [4,8,16]

    hybrid_masker = MaskFunc(center_fractions=fractions,accelerations=accels)

    train_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=train_files, mask_func=hybrid_masker,
        input_req=[1,1,1,1,1], output_req=[1,1,1,1], methods_flags=[0,0]
    )

    val_data = updated_dataloader.Custom_FMRI_DataLoader_nil(
        data_paths=val_files, mask_func=hybrid_masker,
        input_req=[1,1,1,1,1], output_req=[1,1,1,1], methods_flags=[0,0]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)

    # ==========================================
    # 3. MODEL, LOSS, & OPTIMIZER
    # ==========================================
    model = UNET_final()

    # Multi-GPU Setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel!")
        model = nn.DataParallel(model)

    model = model.to(device)

    class SpecialLossFunc(nn.Module):
        def __init__(self, alpha=0.86):
            super().__init__()
            self.alpha = alpha
            self.l1_loss = nn.L1Loss()
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0) 

        def forward(self, preds, targets):
            l1_loss = self.l1_loss(preds, targets)
            ssim_val = self.ssim(preds, targets)
            ssim_loss = 1.0 - ssim_val
            hybrid_loss = (1.0 - self.alpha) * l1_loss + self.alpha * ssim_loss
            return hybrid_loss
        
    loss_func = SpecialLossFunc().to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ==========================================
    # 4. TRAINING & VALIDATION LOOP
    # ==========================================
    trigger_times = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() 
        running_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") 
        
        for batch in train_loop:
            input_tensor = batch['masked_rss_combined'].unsqueeze(1).to(device, dtype=torch.float32)
            target_tensor = batch['full_rss_combined'].unsqueeze(1).to(device, dtype=torch.float32)

            # Dynamic Normalization
            batch_max = target_tensor.max()
            if batch_max > 0:
                input_tensor = input_tensor / batch_max
                target_tensor = target_tensor / batch_max

            optimizer.zero_grad()
            
            predictions = model(input_tensor)
            loss = loss_func(predictions, target_tensor)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        running_val_psnr = 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for batch in val_loop:
                input_tensor = batch['masked_rss_combined'].unsqueeze(1).to(device, dtype=torch.float32)
                target_tensor = batch['full_rss_combined'].unsqueeze(1).to(device, dtype=torch.float32)

                # Dynamic Normalization
                batch_max = target_tensor.max()
                if batch_max > 0:
                    input_tensor = input_tensor / batch_max
                    target_tensor = target_tensor / batch_max

                predictions = model(input_tensor)
                val_loss = loss_func(predictions, target_tensor)
                running_val_loss += val_loss.item()
                
                psnr_val = psnr_module(predictions, target_tensor)
                running_val_psnr += psnr_val.item()
                
                val_loop.set_postfix(val_loss=val_loss.item(), psnr=psnr_val.item())

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_psnr = running_val_psnr / len(val_loader)

        print(f"\nEpoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB")

        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            
            # Safe saving for Multi-GPU
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), saved_model_path)
            else:
                torch.save(model.state_dict(), saved_model_path)
                
            print(f"--> Best model saved! (Val Loss: {best_val_loss:.4f})\n")
        else:
            trigger_times += 1
            print(f"--> No improvement. Early stopping trigger: {trigger_times} / {patience}\n")
            
            if trigger_times >= patience:
                print(f"Early stopping triggered! Training halted at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    main()
