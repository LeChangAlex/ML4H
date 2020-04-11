from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAE import VAE
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image
# from dmsa_model_v0 import *

import wandb

import argparse
import math
import os

import torch

from torch import optim
from torch.nn import functional as F
print(torch.__version__)

from torchvision import transforms

print("-")
from PIL import Image
print("-")
from tqdm import tqdm
print("-")
import lpips


print("-")
# from model import *
print("-")
# from dataset import FairDataset
print("-")
import numpy as np
# from models2 import VariationalAutoEncoderLite
import cv2
from skimage import exposure






def make_image(tensor):

    tmp = tensor.detach().permute(0, 2, 3, 1).to('cpu').numpy()

    im = tmp - np.amin(tmp)
    im = im / np.amax(im)
    im = im * 255


    return  im

class FairDataset(Dataset):
    # def __init__(self, path, transform, resolution=256):

    # # def __init__(self,csv_file,root_dir,transform=None):
    #     self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
    #     self.root_dir = path 
    #     self.transform = transform
    
    # def __len__(self):
    #     return (len(self.annotations))

    # def __getitem__(self,index):
    #     volume_name = os.path.join(self.root_dir,
    #     self.annotations.iloc[index,0])
    #     np_volume = np.load(volume_name)
    #     volume = Image.fromarray(np_volume)
    #     # annotations = self.annotations.iloc[index,0].as_matrix()
    #     # annotations = annotations.astype('float').reshape(-1,2)
    #     sample = volume#[np.newaxis, ...]

    #     if self.transform:
    #         sample = self.transform(sample)
        
    #     return sample
    def __init__(self, path, transform, reg, resolution=512, split="train", run=0, intensity=1, size=10, nodule_mask=0):


        self.nodule_mask = abs(nodule_mask)
        self.metadata = pd.read_csv("../mri_gan_cancer/data/preproc_chest_metadata.csv")
        if split == "train":
            self.metadata = self.metadata[self.metadata["train"] == 1]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["train"] == 0]
        else:
            raise Exception("Invalid data split")



        

        data_mean = 0.175
        data_std = 0.17

        adjusted_intensity = intensity / data_std
        adjusted_max = (1 - data_mean) / data_std
        
        self.data = np.load("../mri_gan_cancer/data/chest_data.npy")
        self.min_val = np.amin(self.data)
        self.max_val = np.amax(self.data)
        # print("mean:", np.mean(self.data.flatten()))
        # print("std:", np.std(self.data.flatten()))
        # print("max:", np.amax(self.data))



        self.masks = np.zeros_like(self.data)

        self.positions = pd.read_csv("../mri_gan_cancer/data/nodule_positions.csv")["run_{}".format(run)]

        for i in range(self.data.shape[0]):
            positions = [int(n) for n in self.positions[i].split(",")]

            self.data[i] = self.insert_nodule(self.data[i], adjusted_intensity, size, positions)
            self.masks[i, positions[1] - self.nodule_mask: positions[1] + self.nodule_mask, positions[0] - self.nodule_mask: positions[0] + self.nodule_mask] = 1

            # print("nodules inserted")
        self.data[self.data > adjusted_max] = adjusted_max
        # print("clipped at", adjusted_max)
        # self.data = self.data[self.metadata["npy_idx"]]
        cv2.imwrite("test/diseased_{}.png".format(run), (self.data[0] * data_std + data_mean) * 255) 
        cv2.imwrite("test/diseased_{}_mask.png".format(run), (self.masks[0] * 255) )

        self.transform = transform
        self.reg = reg

    
    def insert_nodule(self, im, intensity, sigma, position):
        
        x, y = np.meshgrid(np.linspace(-25, 25, 50), np.linspace(-25, 25, 50))
        d = np.sqrt(x * x + y * y)
        nodule = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

        nodule_x, nodule_y = position[0], position[1]

        im[nodule_y - 25: nodule_y + 25, nodule_x - 25: nodule_x + 25] += nodule * intensity



        return im



    def __len__(self):
        if self.reg:
            return self.metadata["patient_n"].unique().shape[0]
        return self.metadata.shape[0]


    def __getitem__(self,index):

        if not self.reg:

            npy_idx = self.metadata["npy_idx"].iloc[index] - 1
            im = self.data[int(npy_idx)]
        else:
            patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


            # print(patient_rows, index)
            npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

            im = self.data[npy_idx - 1]
            

        volume = Image.fromarray(im)
        # annotations = self.annotations.iloc[index,0].as_matrix()
        # annotations = annotations.astype('float').reshape(-1,2)
        sample = volume#[np.newaxis, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_nodule_mask(self, index):

        if not self.reg:

            npy_idx = self.metadata["npy_idx"].iloc[index] - 1
            mask = self.masks[int(npy_idx)]
        else:
            patient_rows = self.metadata[self.metadata["patient_n"] == self.metadata["patient_n"].unique()[index]]


            # print(patient_rows, index)
            npy_idx = random.sample(list(patient_rows["npy_idx"]), 1)[0]

            mask = self.masks[npy_idx - 1]
        return mask


class MyDataParallel(torch.nn.DataParallel):
  def __init__(self, model, device_ids):
      super(MyDataParallel, self).__init__(model, device_ids)

  def __getattr__(self, name):
      try:
          return super(MyDataParallel, self).__getattr__(name)
      except AttributeError:
          return getattr(self.module, name)

if __name__ == '__main__':
    device = 'cuda'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--nodule_intensity', type=float, default=1)
    parser.add_argument('--nodule_size', type=float, default=10)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--mask_val', type=str, default="zero")

    parser.add_argument('--feature_extractor', type=str, default="d")
    parser.add_argument('--latent_space', type=str, default="w")    
    parser.add_argument('--w_plus', action='store_true')

    parser.add_argument('files', metavar='FILES', nargs='+')

    
    args = parser.parse_args()

    ckpt_run = args.ckpt.split("/")[-1]

    data_mean = 0.175
    data_std = 0.17
    min_v = - data_mean / data_std
    max_v = (1 - data_mean) / data_std


    
    transform = transforms.Compose([
            # transforms.Resize(resize),
           # transforms.CenterCrop(resize),
           transforms.ToTensor()
           # transforms.Normalize([0.5, 0.5, 0.5],
                               # [0.5, 0.5, 0.5])
        ])
    print("loading data")
    test_data = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=args.run, intensity=args.nodule_intensity, size=args.nodule_size, nodule_mask=args.mask)
    print("loaded data")
    imgs = []
    
    # for imgfile in args.files:
    #     # img = transform(Image.open(imgfile).convert('RGB'))

    #     img = transform(Image.fromarray(np.load(imgfile)))
    #     imgs.append(img)
    for i in range(len(test_data)):
        imgs.append(test_data[i])


    imgs = torch.stack(imgs, 0).to(device).to(torch.float) # tmp
    print(imgs.shape)
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    model_module = import_module('vae_model.model')
    
    model = VAE(
        model_module.reconstruction_log_prob,
        model_module.prior_network,
        model_module.generative_network
    ).cuda()
    checkpoint = torch.load("vae_model/last_vae_256.tar")
    model.load_state_dict(checkpoint['model_state_dict'])


    test_latents = torch.zeros(95, 256)

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), gpu_ids=range(args.n_gpu))



    print("loaded models")


    

    p_losses = torch.zeros(len(imgs))
    mse_losses = torch.zeros(len(imgs))
    for batch_start in range(0, len(imgs), args.batch_size):

        batch_imgs = imgs[batch_start: batch_start + args.batch_size]

        with torch.no_grad():
            latents = model.make_latent_distributions(batch_imgs).mean

        optimizer = optim.Adam([latents], lr=args.lr)
        
        pbar = tqdm(range(args.step))
        # latent_path = []

        query = batch_imgs.clone()
        # get nodule masks
        if args.mask:
            masks = []
            mask_means = []

            for i in range(batch_start, batch_start + len(batch_imgs)):
                m = torch.from_numpy(test_data.get_nodule_mask(i)).float().unsqueeze(0).unsqueeze(0)
                masks.append(m)

            masks = torch.cat(masks).to(device)


            masks = 1 - masks





            if args.mask_val == "zero":
                batch_imgs[masks==0] = test_data.min_val
            elif args.mask_val == "dataset_mean":
                batch_imgs = batch_imgs * masks

            elif args.mask_val == "patch_mean":
                patch_means = batch_imgs[masks == 0].reshape(len(batch_imgs), -1).mean(-1).unsqueeze(1)


                mean_mask = (torch.ones_like(batch_imgs[masks==0]).reshape(len(batch_imgs), -1) * patch_means).reshape(batch_imgs[masks==0].shape)
                batch_imgs[masks==0] = mean_mask


        for i in pbar:

            img_gen = nn.parallel.data_parallel(model.generative_network, latents, device_ids=range(4))[:, 0].unsqueeze(1)

            if args.mask:
                if args.mask_val == "zero":
                    img_gen[masks==0] = test_data.min_val
                elif args.mask_val == "dataset_mean":
                    img_gen = img_gen * masks

                elif args.mask_val == "patch_mean":

                    img_gen[masks==0] = mean_mask
                        


            p_loss = percept(img_gen, batch_imgs).reshape(batch_imgs.shape[0], -1).mean(-1)
            

            mse_loss = F.l1_loss(img_gen, batch_imgs, reduction="none")
            mse_loss = mse_loss.reshape(mse_loss.shape[0], -1).mean(-1)


            loss = p_loss.mean() + args.mse * mse_loss.mean()


                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # if (i + 1) % 1 == 0:
                # latent_path.append(latent_in.detach().clone())
                
            # pbar.set_description((f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
            #                      f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'))
            pbar.set_description((f'perceptual: {p_loss.mean().item():.4f};'
                                 f' mse: {mse_loss.mean().item():.4f}'))


        p_losses[batch_start: batch_start + args.batch_size] = p_loss.reshape(-1).detach()
        mse_losses[batch_start: batch_start + args.batch_size] = mse_loss.reshape(-1).detach()

        

        # optimize W
        with torch.no_grad():
            img_gen = nn.parallel.data_parallel(model.generative_network, latents, device_ids=range(4))[:, 0].unsqueeze(1)

        test_latents[batch_start: batch_start + args.batch_size] = latents.squeeze(2).squeeze(2).detach()


        filename = os.path.splitext(os.path.basename(args.files[0]))[0] + '.pt'


        print(torch.max(batch_imgs))
        print(torch.max(img_gen))


        img_gen = torch.clamp(img_gen, min_v, max_v)

        img_ar = make_image(batch_imgs)
        gen_ar = make_image(img_gen)
        query_ar = make_image(query)

        print(np.amin(img_ar), np.amax(img_ar))
        print(np.amin(gen_ar), np.amax(gen_ar))
        
        for n in range(10):

            # result_file[input_name] = {'img': img_gen[i], 'latent': latent_in[i]} 
            img_name = "gen_imgs/test_diseased_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(args.nodule_intensity, args.nodule_size, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse, args.mask, args.mask_val, batch_start + n)
            # img_name = "gen_imgs/diseased_{}_{}_{}_{}_{}_{}_{}_project.png".format(args.run, args.nodule_intensity, args.feature_extractor, batch_start + n, args.latent_space, args.mask)

            # print(img_ar.shape, img_ar.dtype, np.amax(img_ar), np.amin(img_ar))
            # found = exposure.rescale_intensity(img_ar[i])
            # query = exposure.rescale_intensity(img_ar[-1])

            cv2.imwrite(img_name, np.concatenate((query_ar[n], img_ar[n], gen_ar[n]), axis=1))


    result_file = {
        "latents": test_latents,
        "p_losses": p_losses,
        "mse_losses": mse_losses
    }
    

        
    torch.save(result_file,
        "latents/test_diseased_{}_{}_{}_{}_{}_{}_{}_{}_{}_losses".format(args.nodule_intensity, args.nodule_size, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse, args.mask, args.mask_val))