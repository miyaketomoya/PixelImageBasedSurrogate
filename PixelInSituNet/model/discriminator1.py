# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from PixelInSituNet.module.resblock import FirstBlockDiscriminator, BasicBlockDiscriminator
from PixelInSituNet.module.self_attention import SelfAttention

class Discriminator1(nn.Module):
  def __init__(self, dsp=1, dvo=3, dvp=3, dspe=512, dvoe=512, dvpe=512, ch=64, in_re = 64):
    super(Discriminator1, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.dvo, self.dvoe = dvo, dvoe
    self.dvp, self.dvpe = dvp, dvpe
    self.ch = ch
    self.in_re = in_re

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # visualization operations subnet
    """
    self.vops_subnet = nn.Sequential(
      nn.Linear(dvo, dvoe), nn.ReLU(),
      nn.Linear(dvoe, dvoe), nn.ReLU()
    )
    """

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe + dvpe, 1024),nn.ReLU(),
      nn.Linear(1024,self.in_re*self.in_re),nn.ReLU(),
    
    )

    # image classification subnet
    self.img_subnet1 = nn.Sequential(
      FirstBlockDiscriminator(3, ch, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3,
                              stride=1, padding=1),
      # SelfAttention(ch * 2),
      
      #resol 64用
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3,
                              stride=1, padding=1),
    )

    self.img_subnet2 = nn.Sequential(
        # resol 64用
        BasicBlockDiscriminator(ch * 4 +1 , ch * 8, kernel_size=3,
                              stride=1, padding=1),
        #ここまで
        
        # #resol 128用
        # BasicBlockDiscriminator(ch * 2 + 1, ch * 4, kernel_size=3,
        #                       stride=1, padding=1),
        # BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3,
        #                        stride=1, padding=1),
        # #ここまで
        
        BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3,
                              stride=1, padding=1),
        BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3,
                              stride=1, padding=1),
      
        #512用
        BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1),
        BasicBlockDiscriminator(ch * 16, ch, kernel_size=3,
                              stride=1, padding=1),
        nn.ReLU(),
        BasicBlockDiscriminator(ch, 1, kernel_size=3,
                              stride=1, padding=1),
    )


  def forward(self, sp, vp, x):
    sp = self.sparams_subnet(sp)
    #vo = self.vops_subnet(vo)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp, vp), 1)
    mp = self.mparams_subnet(mp)
    mp = mp.view(mp.size(0),1,self.in_re,self.in_re)

    x = self.img_subnet1(x)
    out = torch.cat([x,mp],dim=1)
    out = self.img_subnet2(out)
    out = out.squeeze()
    return out
