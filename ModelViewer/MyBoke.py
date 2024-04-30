from bokeh.io import curdoc,save
from bokeh.layouts import column
from bokeh.models import Slider,Range1d,Label
from bokeh.plotting import figure
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from PixelInSituNet.model.generator import Generator

def toXYZ(tht, phi, rp):
    x = rp * math.sin(math.radians(tht)) * math.sin(math.radians(phi))
    y = rp * math.cos(math.radians(tht))
    z = rp * math.sin(math.radians(tht)) * math.cos(math.radians(phi))
    return x, y, z

def parse_args():
  parser = argparse.ArgumentParser(description="Your description here")
  parser.add_argument("--ModelName", required=True, type=str,
                      help="modelPath")
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

  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")
  
  parser.add_argument("--resolution", type=int, default=512,
                      help="resolution of image")
  parser.add_argument("--pixelshuffle",action="store_true", default=False,
                      help="use pixelshuffle or not")
  # parser.add_argument("--newDis", action="store_true", default=False,
  #                     help="use new discriminator")
  # parser.add_argument("--in-re", type=int, default=64,
  #                     help="in_resolutions")

  return parser.parse_args()

def model_set(args,device):
  
  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m
  #g_model = Generator(ch=64)
  g_model = Generator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                      dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                      ch=args.ch,pixelshuffle=args.pixelshuffle)
  if args.sn:
    g_model = add_sn(g_model)
      
  order_dict = torch.load(args.ModelName,map_location=torch.device(device))
  g_model.load_state_dict(order_dict)
  g_model.eval()
  return g_model

#def generate_imageTuple(g_model,device,tht,phi,r,v,t):
def generate_imageTuple(args,g_model,tht,phi,rp,v):
  sparams = torch.tensor([[(v-2.225)/1.225]],dtype=torch.float32)
  #vops = torch.tensor([[0.]],dtype=torch.float32)
  x,y,z = toXYZ(tht,phi,rp)
  vparams = torch.tensor([[x,y,z]],dtype=torch.float32)
  print(sparams,vparams)
  fake_image = g_model(sparams, vparams)
  # テンソルをNumPy配列に変換し、0から255の範囲にスケーリング
  image_np = fake_image.detach().cpu().numpy()
  # テンソルの値が [0, 1] の範囲にあることを確認（必要に応じて調整）
  image_np = np.clip(image_np, -1, 1)
  # 0から255の範囲にスケーリング
  image_np = (image_np * 127.5 + 127.5).astype(np.uint8)
  # チャンネル次元を最後に移動 (256, 256, 3) 形状にする
  image_np = np.transpose(image_np[0], (1, 2, 0))
  # RGBA形式の画像を格納する配列を作成
  img_rgba = np.zeros((args.resolution, args.resolution, 4), dtype=np.uint8)
  img_rgba[..., :3] = image_np
  img_rgba[..., 3] = 255
  return img_rgba.view(dtype=np.uint32).reshape((args.resolution, args.resolution)),x,y,z


args = parse_args()
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
g_model = model_set(args,device)


v = 3
p = figure(x_range=Range1d(start=0, end=args.resolution), y_range=Range1d(start=0, end=args.resolution), tools="zoom_in,zoom_out,reset")
p.xaxis.visible = False
p.yaxis.visible = False

# 初期画像データを設定
initial_img,x,y,z = generate_imageTuple(args,g_model,23,-63,1.,v)  # 初期値を適当に設定
r = p.image_rgba(image=[initial_img], x=0, y=0, dw=args.resolution, dh=args.resolution)

#label = Label(x=10, y=240, text=f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v/100:.2f}, t: {t/100:.2f}", text_font_size="10pt", text_color="black")
label = Label(x=10, y=500, text=f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v/100:.2f}", text_font_size="10pt", text_color="white")
p.add_layout(label)

# Create a slider
slider1 = Slider(start=1, end=3.5, value=3, step=.1, title="simulation parameter（V）")
#slider2 = Slider(start=3, end=7, value=3, step=.1, title="simulation parameter（T）")
slider2 = Slider(start=-180., end=180., value=23., step=1., title="θ")
slider3 = Slider(start=-180., end=180., value=-63., step=1., title="φ")
# Define a callback function for the sliders

def update_data(attrname, old, new):
    # Get current slider values
    v = slider1.value
    #t = slider2.value
    tht = slider2.value
    phi = slider3.value
    # Generate new image based on current slider values
    img,x,y,z = generate_imageTuple(args,g_model,tht,phi,1.,v)
    # 画像データソースを更新
    r.data_source.data["image"] = [img]
    #label.text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v/100:.3f}, t: {t/100:.3f}"
    label.text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v/100:.3f}"

# コールバック関数をスライダーに接続
for slider in [slider1, slider2, slider3]:
    #slider.on_change("value", update_data)
    slider.on_change("value_throttled", update_data)
    
# Arrange plots and widgets in layouts
layout = column(slider1, slider2, slider3, p)

# Add the layout to the current document
curdoc().add_root(layout)

#bokeh serve --show MyBoke.py --args --ModelName [path/to/model.pth] --dsp 1 --dvo 0 --dvoe 0 --pixelshuffle