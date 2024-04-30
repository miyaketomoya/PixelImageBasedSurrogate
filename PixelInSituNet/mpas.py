import os
import pandas as pd
import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse

class ImageDataset(Dataset):
  def __init__(self, root,dvp,dsp,data_len = 0,train = True, transform = None):
    self.root = root
    self.train = train
    self.transform = transform
    self.data_len = data_len
    self.file_paths = self._get_file_paths(root)
    self.img_params = self._get_img_params(root)
    self.dvp = dvp
    self.dsp = dsp
    #新しく,vparams,sparamsなどの次元数をしていするようにする

  def _get_file_paths(self, root):
    if self.train:
      file_paths = (pd.read_csv(os.path.join(root,"train","filenames.txt"),sep = " ",header = None))
    else:
      file_paths = (pd.read_csv(os.path.join(root,"test","filenames.txt"),sep = " ",header = None))
    return file_paths

  def _get_img_params(self,root):
    if self.train:
      img_params = (pd.read_csv(os.path.join(root,"train","params.csv"),sep = ",",header = None , skiprows = 1).values)
    else:
      img_params = (pd.read_csv(os.path.join(root,"test","params.csv"),sep = ",",header = None , skiprows = 1).values)
    return img_params

  def __getitem__(self,index):
    if self.train:
      img_name = os.path.join(self.root,"train","img",self.file_paths.iloc[index][0])
    else:
      img_name = os.path.join(self.root,"test","img",self.file_paths.iloc[index][0])
      
    image = io.imread(img_name)
    vparams = self.img_params[index][1:1+self.dvp] #x,y,z
    sparams = self.img_params[index][1+self.dvp:1+self.dvp+self.dsp] #V,T
    sample = {"image":image, "vparams":vparams, "sparams":sparams}
    if self.transform:
      sample = self.transform(sample)
    return sample
  
  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.file_paths)
  
class ImageDatasetPredict(Dataset):
  def __init__(self, root,dvp,dsp, data_len = 0, transform = None):
    self.root = root
    self.transform = transform
    self.data_len = data_len
    self.file_paths = self._get_file_paths(root)
    self.img_params = self._get_img_params(root)
    self.dvp = dvp
    self.dsp = dsp

  def _get_file_paths(self, root):
    file_paths = (pd.read_csv(os.path.join(root,"filenames.txt"),sep = " ",header = None))
    return file_paths

  def _get_img_params(self,root):
    img_params = (pd.read_csv(os.path.join(root,"params.csv"),sep = ",",header = None , skiprows = 1).values)
    return img_params

  def __getitem__(self,index):
    img_name = os.path.join(self.root,"img",self.file_paths.iloc[index][0])
    image = io.imread(img_name)
    filenames = self.file_paths.iloc[index][0]
    vparams = self.img_params[index][1:1+self.dvp] #x,y,z
    sparams = self.img_params[index][1+self.dvp:1+self.dvp+self.dsp] #V,T

    sample = {"image":image, "vparams":vparams, "sparams":sparams}
    if self.transform:
      sample = self.transform(sample)
    sample["filename"] = filenames
    return sample

  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.img_params)




class Resize(object):
  # Resize(256) ← 数字1つ　or Resize((256,128)) ← taple ()のどちらかで宣言
  def __init__(self,size):
    assert isinstance(size,(int,tuple))
    self.size = size

  def __call__(self,sample):
    image = sample["image"]
    h,w = image.shape[:2]
    if isinstance(self.size, int):
      if h > w:
        new_h, new_w = self.size * h / w, self.size
      else:
        new_h, new_w = self.size, self.size * w / h
    else:
      new_h, new_w = self.size

    new_h, new_w = int(new_h), int(new_w)

    #半分ぐらい調べても出てこないパラメータ mode?
    image = transform.resize(
    image, (new_h, new_w), order=1, mode="reflect",
    preserve_range=True, anti_aliasing=True).astype(np.float32)

    return {"image": image,
            "vparams":sample["vparams"],
            "sparams":sample["sparams"]
            }
    
class Normalize(object):
  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]

    image = (image.astype(np.float32) - 127.5) / 127.5

    # sparams min [1.]
    #         max [4.]
    sparams = (sparams - np.array([0.0225], dtype=np.float32)) / np.array([0.0125], dtype=np.float32)
    #→今回のデータだとVとTの正規化をするのか検討　実験データはパラメータ1つだったため少し違う


    return {"image": image,
            "vparams": sample["vparams"],
            "sparams": sparams}
    
class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"].astype(np.float32)
    vparams = sample["vparams"].astype(np.float32)
    sparams = sample["sparams"].astype(np.float32)

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {"image": torch.from_numpy(image),
            "vparams": torch.from_numpy(vparams),
            "sparams": torch.from_numpy(sparams),
            }
    
    
