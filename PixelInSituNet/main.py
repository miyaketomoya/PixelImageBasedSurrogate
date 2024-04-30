# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# main file for training

from __future__ import absolute_import, division, print_function

import os
import argparse
import math
import time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../")
from mpas import *

from PixelInSituNet.model.generator import Generator
from PixelInSituNet.model.discriminator import Discriminator
from PixelInSituNet.model.discriminator1 import Discriminator1
from PixelInSituNet.module.vgg19 import VGG19

# parse arguments
def parse_args():
  parser = argparse.ArgumentParser(description="InSituNet")

  parser.add_argument("--no-cuda", action="store_true", default=False,
                      help="disables CUDA training")
  parser.add_argument("--data-parallel", action="store_true", default=False,
                      help="enable data parallelism")
  parser.add_argument("--seed", type=int, default=1,
                      help="random seed (default: 1)")

  # parser.add_argument("--root", required=True, type=str,
  #                     help="root of the dataset")
  
  parser.add_argument("--root-dataset", required=True, type=str,
                      help="root of the dataset")
  
  parser.add_argument("--root-store", required=True, type=str,
                      help="root of the dataset")
  
  parser.add_argument("--resume", type=str, default="",
                      help="path to the latest checkpoint (default: none)")

  parser.add_argument("--dsp", type=int, default=3,
                      help="dimensions of the simulation parameters (default: 3)")
  parser.add_argument("--dvo", type=int, default=3,
                      help="dimensions of the visualization operations (default: 3)")
  parser.add_argument("--dvp", type=int, default=3,
                      help="dimensions of the view parameters (default: 3)")
  parser.add_argument("--dspe", type=int, default=512,
                      help="dimensions of the simulation parameters' encode (default: 512)")
  parser.add_argument("--dvoe", type=int, default=512,
                      help="dimensions of the visualization operations' encode (default: 512)")
  parser.add_argument("--dvpe", type=int, default=512,
                      help="dimensions of the view parameters' encode (default: 512)")
  parser.add_argument("--ch", type=int, default=64,
                      help="channel multiplier (default: 64)")
  parser.add_argument("--pixelshuffle",action="store_true", default=False,
                      help="use pixelshuffle or not")
  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")
  parser.add_argument("--mse-loss", action="store_true", default=False,
                      help="enable mse loss")
  parser.add_argument("--newDis", action="store_true", default=False,
                      help="use new discriminator")
  parser.add_argument("--in-re", type=int, default=64,
                      help="in_resolutions")
  parser.add_argument("--perc-loss", type=str, default="relu1_2",
                      help="layer that perceptual loss is computed on (default: relu1_2)")
  parser.add_argument("--gan-loss", type=str, default="none",
                      help="gan loss (default: none)")
  parser.add_argument("--gan-loss-weight", type=float, default=0.,
                      help="weight of the gan loss (default: 0.)")

  parser.add_argument("--lr", type=float, default=1e-3,
                      help="learning rate (default: 1e-3)")
  parser.add_argument("--d-lr", type=float, default=1e-3,
                      help="learning rate of the discriminator (default: 1e-3)")
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 of Adam (default: 0.9)")
  parser.add_argument("--beta2", type=float, default=0.999,
                      help="beta2 of Adam (default: 0.999)")
  parser.add_argument("--batch-size", type=int, default=50,
                      help="batch size for training (default: 50)")
  parser.add_argument("--start-epoch", type=int, default=0,
                      help="start epoch number (default: 0)")
  parser.add_argument("--epochs", type=int, default=10,
                      help="number of epochs to train (default: 10)")

  parser.add_argument("--log-every", type=int, default=10,
                      help="log training status every given number of batches (default: 10)")
  parser.add_argument("--check-every", type=int, default=20,
                      help="save checkpoint every given number of epochs (default: 20)")

  return parser.parse_args()

# the main function
def main(args):
  # log hyperparameters
  print(args)

  # select device
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda:0" if args.cuda else "cpu")
  print(device)

  # set random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # make dir
  os.makedirs(os.path.join(args.root_store,"result"), exist_ok=True)
  os.makedirs(os.path.join(args.root_store,"tmp"), exist_ok=True)
  os.makedirs(os.path.join(args.root_store,"logger"), exist_ok=True)
  writer = SummaryWriter(os.path.join(args.root_store,"logger"))

  end_flag = False

  # data loader
  train_dataset = ImageDataset(
      root=args.root_dataset,
      train=True,
      dvp = args.dvp,
      dsp = args.dsp,
      transform=transforms.Compose([Normalize(), ToTensor()])
      )

  test_dataset = ImageDataset(
      root=args.root_dataset,
      train=False,
      dvp = args.dvp,
      dsp = args.dsp,
      data_len=6,
      transform=transforms.Compose([Normalize(), ToTensor()])
      )

  kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=6,
                           shuffle=True, **kwargs)

  # model
  def weights_init(m):
    if isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m

  g_model = Generator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                      dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                      ch=args.ch , pixelshuffle=args.pixelshuffle)
  g_model.apply(weights_init)
  # if args.sn:
  #   g_model = add_sn(g_model)

  if args.data_parallel and torch.cuda.device_count() > 1:
    g_model = nn.DataParallel(g_model)
  g_model.to(device)

  if args.gan_loss != "none":
    if args.newDis:
      d_model = Discriminator1(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                            dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                            ch=args.ch, in_re = args.in_re)
    else:
      d_model = Discriminator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                            dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                            ch=args.ch)


    d_model.apply(weights_init)
    if args.sn:
      d_model = add_sn(d_model)

    if args.data_parallel and torch.cuda.device_count() > 1:
      d_model = nn.DataParallel(d_model)
    d_model.to(device)

  # loss
  if args.perc_loss != "none":
    norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
    norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
    vgg = VGG19(args.perc_loss).eval()
    if args.data_parallel and torch.cuda.device_count() > 1:
      vgg = nn.DataParallel(vgg)
    vgg.to(device)

  mse_criterion = nn.MSELoss()
  bce_criterion = nn.BCELoss()
  train_losses, test_losses = [], []
  d_losses, g_losses = [], []

  # optimizer
  g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))
  if args.gan_loss != "none":
    d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                             betas=(args.beta1, args.beta2))

  # load checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint {}".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint["epoch"]
      g_model.load_state_dict(checkpoint["g_model_state_dict"])
      g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
      if args.gan_loss != "none":
        d_model.load_state_dict(checkpoint["d_model_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        d_losses = checkpoint["d_losses"]
        g_losses = checkpoint["g_losses"]
      train_losses = checkpoint["train_losses"]
      test_losses = checkpoint["test_losses"]
      print("=> loaded checkpoint {} (epoch {})"
          .format(args.resume, checkpoint["epoch"]))
      
  #original
  real_target = torch.full((args.batch_size, ),1., device = device).view(-1,1)
  fake_target = torch.full((args.batch_size, ),0., device = device).view(-1,1)

  # main loop
  for epoch in tqdm(range(args.start_epoch, args.epochs)):
    # training...
    g_model.train()
    if args.gan_loss != "none":
      d_model.train()
    train_loss = 0.
    perc_mean_loss = 0.
    g_loss_only = 0.
    train_mse_loss = 0.
    d_loss_mean = 0.
    d_loss_mean_fake = 0.
    d_loss_mean_real = 0.
    
    
    time_for = 0
    time_back = 0
    for i, sample in enumerate(train_loader):
      image = sample["image"].to(device)
      sparams = sample["sparams"].to(device)
      #vops = sample["vops"].to(device)
      vparams = sample["vparams"].to(device)
      g_optimizer.zero_grad()
      fake_image = g_model(sparams, vparams)

      loss = 0.

      # gan loss
      if args.gan_loss != "none":
        # update discriminator
        d_optimizer.zero_grad()
        decision = d_model(sparams, vparams, image)
        
        if args.gan_loss == "vanilla":
          d_loss_real = torch.mean(F.softplus(-decision))
          #d_loss_real = bce_criterion(decision, real_target)
        elif args.gan_loss == "hinge":
          d_loss_real = torch.mean(F.relu(1. - decision))

        fake_decision = d_model(sparams, vparams, fake_image.detach())

        if args.gan_loss == "vanilla":
          d_loss_fake = torch.mean(F.softplus(fake_decision))
          #d_loss_fake = bce_criterion(fake_decision, fake_target)
        elif args.gan_loss == "hinge":
          d_loss_fake = torch.mean(F.relu(1. + fake_decision))

        d_loss = d_loss_real + d_loss_fake
        
        
        d_loss.backward()

        d_optimizer.step()
          

        # loss of generator
        g_optimizer.zero_grad()
        
        time_sta = time.time()
        fake_decision = d_model(sparams, vparams, fake_image)
        time_end = time.time()
        tim = time_end- time_sta
        time_for += tim
      
        if args.gan_loss == "vanilla":
          g_loss = args.gan_loss_weight * torch.mean(F.softplus(-fake_decision))
          #g_loss = args.gan_loss_weight * bce_criterion(fake_decision, real_target)
        elif args.gan_loss == "hinge":
          g_loss = -args.gan_loss_weight * torch.mean(fake_decision)
        loss += g_loss

      # mse loss
      
      mse_loss = mse_criterion(image, fake_image)
      if args.mse_loss:
        loss += mse_loss

      # perceptual loss
      if args.perc_loss != "none":
        # normalize
        image = ((image + 1.) * .5 - norm_mean) / norm_std
        fake_image = ((fake_image + 1.) * .5 - norm_mean) / norm_std

        features = vgg(image)
        fake_features = vgg(fake_image)

        perc_loss = mse_criterion(features, fake_features)
        loss += perc_loss
      
      loss.backward()
      
      
      time_sta = time.time()
      g_optimizer.step()
      time_end = time.time()
      tim1 = time_end- time_sta
      time_back += tim1
      
      train_loss      += loss.item() * len(sparams)
      perc_mean_loss  += perc_loss.item() * len(sparams)
      g_loss_only     += g_loss.item() * len(sparams)
      train_mse_loss  += mse_loss.item() * len(sparams)
      d_loss_mean     += d_loss.item() *len(sparams)
      d_loss_mean_fake += d_loss_fake.item() * len(sparams)
      d_loss_mean_real += d_loss_real.item() * len(sparams)

      # #はじめに１度だけランダムパラメータを生成
      # if epoch == args.start_epoch and i == 0:
      #     # sparamsの形状を取得
      #     sparams_shape = sparams.shape
      #     # vparamsの形状を取得
      #     vparams_shape = vparams.shape

      #     # 0から1の乱数で同じ形状の変数を生成
      #     random_sparams = torch.rand(*sparams_shape, device=device)
      #     random_vparams = torch.rand(*vparams_shape, device=device)
                    
      #     print("random_sparams:",random_sparams)
      #     print("random_vparans:",random_vparams)
        

      # log training status
      if i % args.log_every == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
          epoch, i * len(sparams), len(train_loader.dataset),
          100. * i / len(train_loader),
          loss.item()))
        if args.gan_loss != "none":
          print("DLoss: {:.6f}, GLoss: {:.6f}".format(
            d_loss.item(), g_loss.item()))
          d_losses.append(d_loss.item())
          g_losses.append(g_loss.item())
        train_losses.append(loss.item())
    

    print("====> Epoch: {} Average loss: {:.4f}".format(
      epoch, train_loss / len(train_loader.dataset)))
    
    print("backwoard_time",time_back)
    print("forward_time",time_for)

    # testing...
    g_model.eval()
    if args.gan_loss != "none":
      d_model.eval()
    test_loss = 0.
    with torch.no_grad():
      for i, sample in enumerate(test_loader):
        if i > 1:
          continue
        image = sample["image"].to(device)
        sparams = sample["sparams"].to(device)
        #vops = sample["vops"].to(device)
        vparams = sample["vparams"].to(device)
        fake_image = g_model(sparams, vparams)
        test_loss += mse_criterion(image, fake_image).item() * len(sparams)
        
        # print("")
        # #fake_image,true paramsの時のdiscriminator 
        # fake_true_eval = d_model(sparams,vparams,fake_image)
        # #print("fake_image, true_params:",torch.mean(fake_true_eval))
        # print("fake_image, true_params:",torch.mean(torch.sigmoid(fake_true_eval)))
        
        # #fake_image, false paramsの時のdisicriminator
        # fake_fake_eval = d_model(random_sparams,random_vparams,fake_image)
        # #print("fake_image, false_params:",torch.mean(fake_fake_eval))
        # print("fake_image, false_params:",torch.mean(torch.sigmoid(fake_fake_eval)))
        
        # #true_image, true paramsの時のdiscriminator
        # true_true_eval = d_model(sparams,vparams,image)
        # #print("true_image, true_params:",torch.mean(true_true_eval))
        # print("true_image, true_params:",torch.mean(torch.sigmoid(true_true_eval)))
        
        # #true_image, false paramsの時のdiscriminatio
        # true_fake_eval = d_model(random_sparams,random_vparams,image)
        # #print("true_image, false_params:",torch.mean(true_fake_eval))
        # print("true_image, false_params_sigmoid:",torch.mean(torch.sigmoid(true_fake_eval)))
        
        if i == 0:
          n = min(len(sparams), 8)
          comparison = torch.cat(
              [image[:n], fake_image.view(len(sparams), 3, 512, 512)[:n]])
          save_image(((comparison.cpu() + 1.) * .5),
                     os.path.join(args.root_store,"tmp",
                                  str(args.mse_loss)+"_"+args.perc_loss+"_"+str(args.gan_loss)+"_"+str(epoch)+".png"),nrow=n)
          print("=====> Saved image")

    test_losses.append(test_loss / len(test_loader.dataset))
    print("====> Epoch: {} Test set loss: {:.4f}".format(
      epoch, test_losses[-1]))
    
    
    writer.add_scalar("loss/generator/all_loss",train_loss / len(train_loader.dataset),epoch) 
    writer.add_scalar("loss/generator/feature_comparator_loss",perc_mean_loss / len(train_loader.dataset),epoch)  
    writer.add_scalar("loss/generator/generator_only_loss",g_loss_only / len(train_loader.dataset),epoch)
    writer.add_scalar("loss/generator/mse_loss",train_mse_loss / len(train_loader.dataset),epoch)
    writer.add_scalar("loss/generator/test",test_loss / len(test_loader.dataset),epoch)
    writer.add_scalar("loss/discriminator/all_loss",d_loss_mean / len(train_loader.dataset),epoch)
    writer.add_scalar("loss/discriminator/real_loss",d_loss_mean_real / len(train_loader.dataset),epoch)
    writer.add_scalar("loss/discriminator/fake_loss",d_loss_mean_fake / len(train_loader.dataset),epoch)
    
    
    if perc_mean_loss/len(train_loader.dataset) <= 0.007:
      end_flag = True
    
    # saving...
    if epoch % args.check_every == 0 or end_flag:
      print("=> saving checkpoint at epoch {}".format(epoch))
      if args.gan_loss != "none":
        torch.save({"epoch": epoch + 1,
                    "g_model_state_dict": g_model.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_model_state_dict": d_model.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "d_losses": d_losses,
                    "g_losses": g_losses,
                    "train_losses": train_losses,
                    "test_losses": test_losses},
                   os.path.join(args.root_store,"result", "model_" + str(args.mse_loss) + "_" + \
                                args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                                str(epoch) + ".pth.tar"))
      else:
        torch.save({"epoch": epoch + 1,
                    "g_model_state_dict": g_model.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses},
                   os.path.join(args.root_store,"result" ,"model_" + str(args.mse_loss) + "_" + \
                                args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                                str(epoch) + ".pth.tar"))

      torch.save(g_model.state_dict(),
                 os.path.join(args.root_store,"result", "model_" + str(args.mse_loss) + "_" + \
                              args.perc_loss + "_" + str(args.gan_loss) + "_" + \
                              str(epoch) + ".pth"))
      
    if end_flag:
      exit(print("training over")) 

if __name__ == "__main__":
  main(parse_args())
