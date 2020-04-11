from io import BytesIO
import pandas as pd
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import random
from tqdm import tqdm
import cv2
import torch

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
        self.metadata = pd.read_csv("../data/preproc_chest_metadata.csv")
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
        
        self.data = np.load("../data/chest_data.npy")
        self.min_val = np.amin(self.data)
        self.max_val = np.amax(self.data)
        # print("mean:", np.mean(self.data.flatten()))
        # print("std:", np.std(self.data.flatten()))
        # print("max:", np.amax(self.data))



        self.masks = np.zeros_like(self.data)

        self.positions = pd.read_csv("../data/nodule_positions.csv")["run_{}".format(run)]

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


    # def __init__(self, path, transform, resolution=256):
    #     self.env = lmdb.open(
    #         path,
    #         max_readers=32,
    #         readonly=True,
    #         lock=False,
    #         readahead=False,
    #         meminit=False,
    #     )

    #     # if not self.env:
    #     #     raise IOError('Cannot open lmdb dataset', path)

    #     # with self.env.begin(write=False) as txn:
    #     #     self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    #     self.resolution = resolution
    #     self.transform = transform

    #     self.data = 

    # def __len__(self):
    #     return 360

    # def __getitem__(self, index):
    #     # with self.env.begin(write=False) as txn:
    #     #     key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
    #     #     img_bytes = txn.get(key)

    #     # buffer = BytesIO(img_bytes)


    #     img = Image.open(buffer)
    #     img = self.transform(img)

    #     return img
