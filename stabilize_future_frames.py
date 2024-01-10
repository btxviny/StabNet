import numpy as np
import cv2
import argparse
from time import time
import os
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

device = 'cuda'
batch_size = 1
grid_h,grid_w = 15,15
H,W = height,width = 360,640


def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using StabNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def get_warp(net_out,img):
    '''
    Inputs:
        net_out: torch.Size([batch_size,grid_h +1 ,grid_w +1,2])
        img: image to warp
    '''
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1, grid_h + 1),
                                    torch.linspace(-1,1, grid_h + 1),
                                    indexing='ij')
    src_grid = torch.stack([grid_x,grid_y],dim = -1).unsqueeze(0).repeat(batch_size,1,1,1).to(device)
    new_grid = src_grid + 1 * net_out
    grid_upscaled = F.interpolate(new_grid.permute(0,-1,1,2),size = (height,width), mode = 'bilinear',align_corners= True)
    warped = F.grid_sample(img, grid_upscaled.permute(0,2,3,1),align_corners=False,padding_mode='zeros')
    return warped

def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    out.release()

class StabNet(nn.Module):
    def __init__(self,trainable_layers = 10):
        super(StabNet, self).__init__()
        # Load the pre-trained ResNet model
        vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1')
        # Extract conv1 pretrained weights for RGB input
        rgb_weights = vgg19.features[0].weight.clone() #torch.Size([64, 3, 3, 3])
        # Calculate the average across the RGB channels
        tiled_rgb_weights = rgb_weights.repeat(1,5,1,1) 
        # Change size of the first layer from 3 to 9 channels
        vgg19.features[0] = nn.Conv2d(15,64, kernel_size=3, stride=1, padding=1, bias=False)
        # set new weights
        vgg19.features[0].weight = nn.Parameter(tiled_rgb_weights)
        # Determine the total number of layers in the model
        total_layers = sum(1 for _ in vgg19.parameters())
        # Freeze the layers except the last 10
        for idx, param in enumerate(vgg19.parameters()):
            if idx > total_layers - trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Remove the last layer of ResNet
        self.encoder = nn.Sequential(*list(vgg19.children())[0][:-1])
        self.regressor = nn.Sequential(nn.Linear(512,2048),
                                       nn.ReLU(),
                                       nn.Linear(2048,1024),
                                       nn.ReLU(),
                                       nn.Linear(1024,512),
                                       nn.ReLU(),
                                       nn.Linear(512, ((grid_h + 1) * (grid_w + 1) * 2)))
        #self.regressor[-1].bias.data.fill_(0)
        total_resnet_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_regressor_params = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
        print("Total Trainable encoder Parameters: ", total_resnet_params)
        print("Total Trainable regressor Parameters: ", total_regressor_params)
        print("Total Trainable parameters:",total_regressor_params + total_resnet_params)
    
    def forward(self, x_tensor):
        x_batch_size = x_tensor.size()[0]
        x = x_tensor[:, :3, :, :]

        # summary 1, dismiss now
        x_tensor = self.encoder(x_tensor)
        x_tensor = torch.mean(x_tensor, dim=[2, 3])
        x = self.regressor(x_tensor)
        x = x.view(x_batch_size,grid_h + 1,grid_w + 1,2)
        return x
    
def stabilize(in_path,out_path):
    
    if not os.path.exists(in_path):
        print(f"The input file '{in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()

    #Load frames and stardardize
    cap = cv2.VideoCapture(in_path)
    mean = np.array([0.485, 0.456, 0.406],dtype = np.float32) 
    std = np.array([0.229, 0.224, 0.225],dtype = np.float32)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret: break
        img = cv2.resize(img, (W,H))
        img = (img / 255.0).astype(np.float32)
        img = (img - mean)/std
        frames.append(img)
    frames = np.array(frames,dtype = np.float32)
    frame_count,_,_,_ = frames.shape
    
    # stabilize video
    frames_tensor = torch.from_numpy(frames).permute(0,3,1,2).float().to('cpu')
    stable_frames_tensor = frames_tensor.clone()
    SKIP = 16
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    def get_batch(idx):
        batch = torch.zeros((5,3,H,W)).float()
        for i,j in enumerate(range(idx - SKIP, idx + SKIP + 1, SKIP//2)):
                batch[i,...] = frames_tensor[j,...]
        batch = batch.view(1,-1,H,W)
        return batch.to(device)

    for frame_idx in range(SKIP,frame_count - SKIP):
        batch = get_batch(frame_idx)
        with torch.no_grad():
            transform = stabnet(batch)
            warped = get_warp(transform, frames_tensor[frame_idx: frame_idx + 1,...].cuda())
        stable_frames_tensor[frame_idx] = warped
        img = warped.permute(0,2,3,1)[0,...].cpu().detach().numpy()
        img *= std
        img += mean
        img = np.clip(img * 255.0,0,255).astype(np.uint8)
        cv2.imshow('window', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    #undo standardization
    stable_frames = np.clip(((stable_frames_tensor.permute(0,2,3,1).numpy() * std) + mean) * 255,0,255).astype(np.uint8)
    frames = np.clip(((frames_tensor.permute(0,2,3,1).numpy() * std) + mean) * 255,0,255).astype(np.uint8)
    save_video(stable_frames,out_path)


if __name__ == '__main__':
    args = parse_args()
    ckpt_dir = './ckpts/with_future_frames/'
    stabnet = StabNet().to(device).eval()
    ckpts = os.listdir(ckpt_dir)
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: datetime.datetime.strptime(x.split('_')[2].split('.')[0], "%H-%M-%S"), reverse=True)
        latest = os.path.join(ckpt_dir, ckpts[0])
        state = torch.load(latest)
        stabnet.load_state_dict(state['model'])
        print('Loaded StabNet',latest)
    stabilize(args.in_path, args.out_path)