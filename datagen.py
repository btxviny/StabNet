import numpy as np
import cv2
import os 
import random
import torch

class Datagen:
    def __init__(self,shape = (256,256),txt_path = './trainlist.txt'):
        self.shape = shape
        with open(txt_path,'r') as f:
            self.trainlist = f.read().splitlines()
    def preprocess(self,img,gray = True):
        h,w = self.shape
        if gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(w,h))
        img = img / 255.0
        return img
    def __call__(self):
        self.trainlist = random.sample(self.trainlist, len(self.trainlist))
        for sample in self.trainlist:
            s_path = sample.split(',')[0]
            u_path = sample.split(',')[1]
            flow_path = sample.split(',')[2]
            feature_path = sample.split(',')[3]
            idx = int(sample.split(',')[4])
            s_cap = cv2.VideoCapture(s_path)
            u_cap = cv2.VideoCapture(u_path)
            seq1 = []
            seq2 = []
            s_cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 33)
            for i in range(5,-1,-1):
                pos = 2 ** i + 1
                s_cap.set(cv2.CAP_PROP_POS_FRAMES, idx - pos)
                _,temp1 = s_cap.read()
                temp1 = self.preprocess(temp1) # -33
                temp1 = random_translation(temp1)
                seq2.append(temp1)
                _,temp2 = s_cap.read()
                temp2 = self.preprocess(temp2) # -32
                temp2 = random_translation(temp2)
                seq1.append(temp2)
            seq1 = np.array(seq1,dtype=np.uint8)
            seq2 = np.array(seq2,dtype=np.uint8)
            _,Igt = s_cap.read()
            Igt = self.preprocess(Igt, gray= False)
            u_cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            _,It_prev = u_cap.read()
            It_prev = self.preprocess(It_prev, gray= False)
            _,It_curr = u_cap.read()
            It_curr = self.preprocess(It_curr, gray= False)
            flow = np.load(flow_path,mmap_mode='r')
            flo = torch.from_numpy(flow[idx - 1,...].copy()).permute(-1,0,1).float()
            features = np.load(feature_path,mmap_mode='r')
            features = torch.from_numpy(features[idx,...].copy()).float()
            seq1 = np.flip(seq1,axis = 0)
            seq1 = torch.from_numpy(seq1.copy()).float()
            seq2 = np.flip(seq2,axis = 0)
            seq2 = torch.from_numpy(seq2.copy()).float()
            Igt = torch.from_numpy(Igt).permute(-1,0,1).float()
            It_prev = torch.from_numpy(It_prev).permute(-1,0,1).float()
            It_curr = torch.from_numpy(It_curr).permute(-1,0,1).float()
            
            yield seq1, seq2, It_curr, Igt, It_prev, flo, features

def random_translation(img):
    img = np.array(img)
    (h,w) = img.shape
    dx = np.random.randint(-w//8,w//8)
    dy = np.random.randint(-h//8,h//8)
    mat = np.array([[1,0,dx],[0,1,dy]],dtype=np.float32)
    img = cv2.warpAffine(img, mat, (w,h))
    return img