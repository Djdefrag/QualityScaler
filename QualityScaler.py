import multiprocessing
import os.path
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from customtkinter import (CTk, 
                           CTkButton, 
                           CTkEntry, 
                           CTkFont, 
                           CTkImage,
                           CTkLabel, 
                           CTkOptionMenu, 
                           CTkScrollableFrame,
                           filedialog, 
                           set_appearance_mode,
                           set_default_color_theme)
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image


app_name  = "QualityScaler"
version   = "2.5"

githubme    = "https://github.com/Djdefrag/QualityScaler"
telegramme  = "https://linktr.ee/j3ngystudio"


# RRDB models
RRDB_vram_multiplier   = 0.9
BSRGAN_models_list     = [ 'BSRGANx4', 'BSRGANx2' ]
RealESRGAN_models_list = [ 'RealESRGANx4' ]

# SRVGGNetCompact
SRVGGNetCompact_vram_multiplier = 1.8
SRVGGNetCompact_models_list     = [ 'RealESR_Gx4', 'RealSRx4_Anime' ]

# SAFM models
# SAFM_vram_multiplier = 2
# SAFM_models_list     = ['SAFMN_x4', 'SAFMN_L_x4', 'SAFMN_L_Real_x4', 'SAFMN_L_Real_x4_v2']

AI_models_list = BSRGAN_models_list + RealESRGAN_models_list + SRVGGNetCompact_models_list

image_extension_list  = [ '.jpg', '.png', '.bmp', '.tiff' ]
video_extension_list  = [ '.mp4', '.avi', '.webm' ]
interpolation_list    = [ 'Yes', 'No' ]
AI_modes_list         = [ "Half precision", "Full precision" ]

log_file_path  = f"{app_name}.log"
temp_dir       = f"{app_name}_temp"
audio_path     = f"{app_name}_temp{os.sep}audio.mp3"
frame_sequence = f"{app_name}_temp{os.sep}frame_%01d.jpg"

device_list_names    = []
device_list          = []
gpus_found           = torch_directml.device_count()
downscale_algorithm  = cv2.INTER_AREA
upscale_algorithm    = cv2.INTER_CUBIC

offset_y_options = 0.1125
row0_y           = 0.6
row1_y           = row0_y + offset_y_options
row2_y           = row1_y + offset_y_options
row3_y           = row2_y + offset_y_options

app_name_color = "#DA70D6"
dark_color     = "#080808"

if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")



# AI models -------------------

def prepare_model(selected_AI_model, scaling_factor, backend, half_precision):
    update_process_status(f"Preparing AI model")

    model_path = find_by_relative_path(f"AI{os.sep}{selected_AI_model}.pth")
    
    with torch.no_grad():

        if selected_AI_model in BSRGAN_models_list:
            model            = BSRGAN_Net(in_nc = 3, 
                                          out_nc = 3, 
                                          nf = 64, 
                                          nb = 23, 
                                          gc = 32, 
                                          sf = scaling_factor)
            pretrained_model = torch.load(model_path)

        elif selected_AI_model in RealESRGAN_models_list:
            model            = RealESRGAN_Net(num_in_ch = 3, 
                                              num_out_ch = 3, 
                                              num_feat = 64, 
                                              num_block = 23, 
                                              num_grow_ch = 32, 
                                              scale = 4)
            pretrained_model = torch.load(model_path)['params_ema']

        elif selected_AI_model in SRVGGNetCompact_models_list:
            num_conv         = 32 if 'RealESR_Gx4' in selected_AI_model else 16
            model            = SRVGGNetCompact(num_in_ch = 3, 
                                               num_out_ch = 3, 
                                               num_feat = 64, 
                                               num_conv = num_conv, 
                                               upscale = 4, 
                                               act_type = 'prelu')
            pretrained_model = torch.load(model_path, map_location = torch.device('cpu'))['params']

        '''
        elif selected_AI_model in SAFM_models_list:
            if 'SAFMN_x4' in selected_AI_model:
                dim = 36
                n_blocks = 8
            else:
                dim = 128
                n_blocks = 16                

            model            = SAFMN(dim = dim, n_blocks = n_blocks, ffn_scale = 2.0, upscaling_factor = 4)
            pretrained_model = torch.load(model_path)['params']
            model.load_state_dict(pretrained_model, strict = True)
        '''
            
        model.load_state_dict(pretrained_model, strict = True)
        model.eval()
        if half_precision: model = model.half()
        model = model.to(backend, non_blocking = True)

    return model

## BSRGAN Architecture

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB_BSRGAN(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB_BSRGAN, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class BSRGAN_Net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(BSRGAN_Net, self).__init__()
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB_BSRGAN(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1    = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        if sf == 4: self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv    = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        fea   = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea   = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4: fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

# RealESRGAN

class ResidualDenseBlock(nn.Module):
    def default_init_weights(module_list, scale=1, bias_fill=0):
        if not isinstance(module_list, list): module_list = [module_list]
        
        for module in module_list:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                module.weight.data *= scale
                if module.bias is not None:
                    module.bias.data.fill_(bias_fill)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                if module.bias is not None:
                    module.bias.data.fill_(bias_fill)

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        self.default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB_RealESRGAN(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB_RealESRGAN, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class RealESRGAN_Net(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RealESRGAN_Net, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body       = nn.Sequential(*[RRDB_RealESRGAN(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out  = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# SRVGGNetCompact

class SRVGGNetCompact(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

## SAFM Architecture

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x
        
class SAFMN(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

# AI inference -------------------

def get_image_mode(image):
    if len(image.shape) == 2: return 'Grayscale'  # Immagine in scala di grigi
    elif image.shape[2] == 3: return 'RGB'        # RGB
    elif image.shape[2] == 4: return 'RGBA'       # RGBA
    else:                     return 'Unknown'
    
def preprocess_image(image, half_precision, backend):
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
    if half_precision: image = image.unsqueeze(0).half().to(backend, non_blocking=True)
    else:              image = image.unsqueeze(0).to(backend, non_blocking=True)
    
    return image

def process_image_with_model(AI_model, image):
    output_image = AI_model(image)
    output_image = output_image.squeeze().float().clamp(0, 1).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    
    return output_image

def postprocess_output(output_image, max_range):
    output = (output_image * max_range).round().astype(np.uint16 if max_range == 65535 else np.uint8)
    return output

def AI_enhance(AI_model, image, backend, half_precision):
    with torch.no_grad():
        image = image.astype(np.float32)

        max_range = 65535 if np.max(image) > 256 else 255
        image /= max_range
        img_mode = get_image_mode(image)

        if img_mode == "RGB":
            image        = preprocess_image(image, half_precision, backend)
            output_image = process_image_with_model(AI_model, image)
            output_image = postprocess_output(output_image, max_range)
            return output_image

        elif img_mode == 'Grayscale':
            image        = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image        = preprocess_image(image, half_precision, backend)
            output_image = process_image_with_model(AI_model, image)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
            output_image = postprocess_output(output_image, max_range)
            return output_image
        
        elif img_mode == 'RGBA':
            alpha = image[:, :, 3]
            image = image[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)

            # Image
            image        = preprocess_image(image, half_precision, backend)
            output_image = process_image_with_model(AI_model, image)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGRA)

            # Alpha
            alpha        = preprocess_image(alpha, half_precision, backend)
            output_alpha = process_image_with_model(AI_model, alpha)
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_RGB2GRAY)

            # Fusion Image + Alpha
            output_image[:, :, 3] = output_alpha
            output_image = postprocess_output(output_image, max_range)
            return output_image



# Classes and utils -------------------

class Gpu:
    def __init__(self, index, name):
        self.name   = name
        self.index  = index

class ScrollableImagesTextFrame(CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.label_list  = []
        self.button_list = []
        self.file_list   = []

    def get_selected_file_list(self): 
        return self.file_list

    def add_clean_button(self):
        label = CTkLabel(self, text = "")
        button = CTkButton(self, 
                            font  = bold11,
                            text  = "CLEAN", 
                            fg_color   = "#282828",
                            text_color = "#E0E0E0",
                            image    = clear_icon,
                            compound = "left",
                            width    = 85, 
                            height   = 27,
                            corner_radius = 25)
        button.configure(command=lambda: self.clean_all_items())
        button.grid(row = len(self.button_list), column=1, pady=(0, 10), padx = 5)
        self.label_list.append(label)
        self.button_list.append(button)

    def add_item(self, text_to_show, file_element, image = None):
        label = CTkLabel(self, 
                        text  = text_to_show,
                        font  = bold11,
                        image = image, 
                        #fg_color   = "#282828",
                        text_color = "#E0E0E0",
                        compound = "left", 
                        padx     = 10,
                        pady     = 5,
                        corner_radius = 25,
                        anchor   = "center")
                        
        label.grid(row  = len(self.label_list), column = 0, 
                   pady = (3, 3), padx = (3, 3), sticky = "w")
        self.label_list.append(label)
        self.file_list.append(file_element)    

    def clean_all_items(self):
        self.label_list  = []
        self.button_list = []
        self.file_list   = []
        place_up_background()
        place_loadFile_section()

for index in range(gpus_found): 
    gpu = Gpu(index = index, name = torch_directml.device_name(index))
    device_list.append(gpu)
    device_list_names.append(gpu.name)

supported_file_extensions = [
                            '.jpg', '.jpeg', '.JPG', '.JPEG',
                            '.png', '.PNG',
                            '.webp', '.WEBP',
                            '.bmp', '.BMP',
                            '.tif', '.tiff', '.TIF', '.TIFF',
                            '.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt', '.3gp', 
                            '.mpg', '.mpeg'
                            ]

supported_video_extensions  = [
                                '.mp4', '.MP4',
                                '.webm', '.WEBM',
                                '.mkv', '.MKV',
                                '.flv', '.FLV',
                                '.gif', '.GIF',
                                '.m4v', ',M4V',
                                '.avi', '.AVI',
                                '.mov', '.MOV',
                                '.qt', '.3gp', 
                                '.mpg', '.mpeg'
                            ]



#  Slice functions -------------------

def split_image_into_tiles(image, 
                           file_number, 
                           num_tiles_x, 
                           num_tiles_y, 
                           should_print):
    if should_print:
        update_process_status(f"{file_number}. Tiling image in {num_tiles_x * num_tiles_y}")

    img_height, img_width, _ = image.shape

    tile_width  = img_width // num_tiles_x
    tile_height = img_height // num_tiles_y

    tiles = []

    for y in range(num_tiles_y):
        y_start = y * tile_height
        y_end   = (y + 1) * tile_height

        for x in range(num_tiles_x):
            x_start = x * tile_width
            x_end   = (x + 1) * tile_width
            tile    = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)

    return tiles

def combine_tiles_into_image(tiles, 
                             file_number,
                             image_target_height, 
                             image_target_width,
                             num_tiles_x, 
                             num_tiles_y,
                             should_print):

    if should_print:
        update_process_status(f"{file_number}. Reconstructing image by tiles")

    tiled_image = np.zeros((image_target_height, image_target_width, 4), dtype = np.uint8)

    for i, tile in enumerate(tiles):
        tile_height, tile_width, _ = tile.shape
        row     = i // num_tiles_x
        col     = i % num_tiles_x
        y_start = row * tile_height
        y_end   = y_start + tile_height
        x_start = col * tile_width
        x_end   = x_start + tile_width
        tiled_image[y_start:y_end, x_start:x_end] = add_alpha_channel(tile)

    return tiled_image

def file_need_tiles(image, tiles_resolution):
    height, width, _ = image.shape

    tile_size = tiles_resolution

    num_tiles_horizontal = (width + tile_size - 1) // tile_size
    num_tiles_vertical = (height + tile_size - 1) // tile_size

    total_tiles = num_tiles_horizontal * num_tiles_vertical

    if total_tiles <= 1:
        return False, 0, 0
    else:
        return True, num_tiles_horizontal, num_tiles_vertical

def add_alpha_channel(tile):
    if tile.shape[2] == 3:  # Check if the tile does not have an alpha channel
        alpha_channel = np.full((tile.shape[0], tile.shape[1], 1), 255, dtype=np.uint8)
        tile = np.concatenate((tile, alpha_channel), axis=2)
    return tile

def fix_tile_shape(tile, tile_upscaled, scaling_factor):
    tile_height, tile_width, _ = tile.shape
    target_tile_height = tile_height * scaling_factor
    target_tile_width  = tile_width  * scaling_factor

    tile_upscaled = cv2.resize(tile_upscaled, (target_tile_width, target_tile_height))

    return tile_upscaled



# Utils functions ------------------------

def opengithub(): webbrowser.open(githubme, new=1)

def opentelegram(): webbrowser.open(telegramme, new=1)

def image_write(file_path, file_data): cv2.imwrite(file_path, file_data)

def image_read(file_path, flags = cv2.IMREAD_UNCHANGED): 
    return cv2.imread(file_path, flags)

# Copies metadata from original image to new image
# This also works for videos but few metadata tags are supported
# Level is one of "disabled", "basic", or "extensive":
#  - disabled = do not copy metadata, return immediately
#  - basic = copy informational metadata only
#  - extensive = copy all metadata, including embedded thumbnails and color profiles
def copy_metadata(original_file_path, new_file_path, level = "basic"):
    cmd = ['exiftool', '-fast', '-TagsFromFile', original_file_path, '-overwrite_original', '-all:all']
    
    if level == "disabled": return
    elif level == "basic": cmd.extend([new_file_path])
    elif level == "extensive": cmd.extend(['-unsafe', '-largetags', new_file_path])
    else: raise ValueError("Invalid level argument")

    try:
        print(f"Copying metadata to {new_file_path} (level={level})")
        subprocess.run(cmd, check=True)
    except Exception as ex:
        print(f"Error while copying metadata to {new_file_path}: {ex}")

def prepare_output_image_filename(image_path, 
                                  selected_AI_model, 
                                  resize_factor, 
                                  selected_image_extension, 
                                  selected_interpolation):
    
    result_path, _    = os.path.splitext(image_path)
    resize_percentage = str(int(resize_factor * 100)) + "%"

    if selected_interpolation:
        to_append = f"_{selected_AI_model}_{resize_percentage}_interpolated{selected_image_extension}"
    else:
        to_append = f"_{selected_AI_model}_{resize_percentage}{selected_image_extension}"

    result_path += to_append

    return result_path

def prepare_output_video_filename(video_path, 
                                  selected_AI_model, 
                                  resize_factor, 
                                  selected_video_extension,
                                  selected_interpolation):
    
    result_path, _    = os.path.splitext(video_path)
    resize_percentage = str(int(resize_factor * 100)) + "%"

    if selected_interpolation:
        to_append = f"_{selected_AI_model}_{resize_percentage}_interpolated{selected_video_extension}"
    else:
        to_append = f"_{selected_AI_model}_{resize_percentage}{selected_video_extension}"

    result_path += to_append

    return result_path

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir, mode=0o777)

def remove_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)

def write_in_log_file(text_to_insert):
    with open(log_file_path,'w') as log_file: 
        os.chmod(log_file_path, 0o777)
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    with open(log_file_path,'r') as log_file: 
        os.chmod(log_file_path, 0o777)
        step = log_file.readline()
    log_file.close()
    return step

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def remove_temp_files():
    remove_dir(temp_dir)
    remove_file(log_file_path)

def stop_thread(): 
    stop = 1 + "x"

def resize_image(image, resize_factor):
    old_height, old_width, _ = image.shape
    new_width  = int(old_width * resize_factor)
    new_height = int(old_height * resize_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation = downscale_algorithm)
    return resized_image       

def resize_frame(frame, new_width, new_height):
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation = downscale_algorithm)
    return resized_image 

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def extract_video_frames_and_audio(video_path, file_number):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_file_clip = VideoFileClip(video_path)
    
    # Extract audio
    try: 
        update_process_status(f"{file_number}. Extracting video audio")
        video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
    except:
        pass

    # Extract frames
    update_process_status(f"{file_number}. Extracting video frames")
    video_frames_list = video_file_clip.write_images_sequence(frame_sequence, verbose = False, logger = None, fps = frame_rate)
    
    return video_frames_list

def video_reconstruction_by_frames(input_video_path,
                                   file_number, 
                                   frames_upscaled_list, 
                                   selected_AI_model, 
                                   resize_factor, 
                                   cpu_number,
                                   selected_video_extension, 
                                   selected_interpolation):
    
    update_process_status(f"{file_number}. Processing upscaled video")
    
    # Find original video FPS
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Choose the appropriate codec
    if selected_video_extension   == '.mp4':  codec = 'libx264'
    elif selected_video_extension == '.avi':  codec = 'png'
    elif selected_video_extension == '.webm': codec = 'libvpx'

    upscaled_video_path = prepare_output_video_filename(input_video_path, 
                                                        selected_AI_model, 
                                                        resize_factor, 
                                                        selected_video_extension,
                                                        selected_interpolation)

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os.path.exists(audio_path) and selected_video_extension != '.webm':
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            audio   = audio_path,
                            codec   = codec,
                            verbose = False,
                            logger  = None,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                             fps     = frame_rate,
                             codec   = codec,
                             verbose = False,
                             logger  = None,
                             threads = cpu_number)  
        
def interpolate_images(starting_image, 
                       upscaled_image, 
                       image_target_height, 
                       image_target_width):
    
    starting_image     = add_alpha_channel(cv2.resize(starting_image, (image_target_width, image_target_height), interpolation = upscale_algorithm))
    upscaled_image     = add_alpha_channel(upscaled_image)
    interpolated_image = cv2.addWeighted(upscaled_image, 0.5, starting_image, 0.5, 0)

    return interpolated_image

def get_final_image_shape(image_to_upscale, scaling_factor):
    # Calculate final image shape
    image_to_upscale_height, image_to_upscale_width, _ = image_to_upscale.shape
    target_height = image_to_upscale_height * scaling_factor
    target_width  = image_to_upscale_width  * scaling_factor
    
    return target_height, target_width

def get_resized_frame_shape(first_frame, resize_factor):
    height, width, _ = first_frame.shape
    resized_width  = int(width * resize_factor)
    resized_height = int(height * resize_factor)

    return resized_width, resized_height

def get_final_frame_shape(resized_width, resized_height, scaling_factor):
    # Calculate final frame shape
    target_height = resized_height * scaling_factor
    target_width  = resized_width  * scaling_factor
    
    return target_height, target_width

def calculate_time_to_complete_video(start_timer, 
                                     end_timer, 
                                     how_many_frames, 
                                     index_frame):
    
    seconds_for_frame = round(end_timer - start_timer, 2)
    frames_left       = how_many_frames - (index_frame + 1)
    seconds_left      = seconds_for_frame * frames_left

    hours_left   = seconds_left // 3600
    minutes_left = (seconds_left % 3600) // 60
    seconds_left = round((seconds_left % 3600) % 60)

    time_left = ""

    if int(hours_left) > 0: 
        time_left = f"{int(hours_left):02d}h:"
    
    if int(minutes_left) > 0:
        time_left = f"{time_left}{int(minutes_left):02d}m:"

    if seconds_left > 0:
        time_left = f"{time_left}{seconds_left:02d}s"

    return time_left        



# Core functions ------------------------

def stop_upscale_process():
    global process_upscale_orchestrator
    process_upscale_orchestrator.terminate()
    process_upscale_orchestrator.join()

def check_upscale_steps():
    time.sleep(2)
    try:
        while True:
            step = read_log_file()

            info_message.set(step)

            if "All files completed! :)" in step or "Upscaling stopped" in step:
                stop_upscale_process()
                remove_temp_files()
                stop_thread()
            elif "Error during upscale process :(" in step:
                info_message.set('Error during upscale process :(')
                remove_temp_files()
                stop_thread()

            time.sleep(2)
    except:
        place_upscale_button()
        
def update_process_status(actual_process_phase):
    print(f"{actual_process_phase}")
    write_in_log_file(actual_process_phase) 

def stop_button_command():
    stop_upscale_process()
    write_in_log_file("Upscaling stopped") 

def upscale_button_command(): 
    global selected_file_list
    global selected_AI_model
    global selected_interpolation
    global half_precision
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    global process_upscale_orchestrator

    remove_file(app_name + ".log")
    
    if user_input_checks():
        info_message.set("Loading")
        write_in_log_file("Loading")

        print("=" * 50)
        print("> Starting upscale:")
        print(f"  Files to upscale: {len(selected_file_list)}")
        print(f"  Selected AI model: {selected_AI_model}")
        print(f"  AI half precision: {half_precision}")
        print(f"  Interpolation: {selected_interpolation}")
        print(f"  Selected GPU: {torch_directml.device_name(selected_AI_device)}")
        print(f"  Selected image output extension: {selected_image_extension}")
        print(f"  Selected video output extension: {selected_video_extension}")
        print(f"  Tiles resolution for selected GPU VRAM: {tiles_resolution}x{tiles_resolution}px")
        print(f"  Resize factor: {int(resize_factor * 100)}%")
        print(f"  Cpu number: {cpu_number}")
        print("=" * 50)

        backend = torch.device(torch_directml.device(selected_AI_device))

        place_stop_button()

        process_upscale_orchestrator = multiprocessing.Process(
                                            target = upscale_orchestrator,
                                            args   = (selected_file_list,
                                                     selected_AI_model,
                                                     backend, 
                                                     selected_image_extension,
                                                     tiles_resolution,
                                                     resize_factor,
                                                     cpu_number,
                                                     half_precision,
                                                     selected_video_extension,
                                                     selected_interpolation))
        process_upscale_orchestrator.start()

        thread_wait = threading.Thread(target = check_upscale_steps, 
                                       daemon = True)
        thread_wait.start()

def upscale_orchestrator(selected_file_list,
                         selected_AI_model,
                         backend, 
                         selected_image_extension,
                         tiles_resolution,
                         resize_factor,
                         cpu_number,
                         half_precision,
                         selected_video_extension,
                         selected_interpolation):
    
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.set_num_threads(1)

    scaling_factor = 2 if 'x2' in selected_AI_model else 4

    try:
        AI_model = prepare_model(selected_AI_model, scaling_factor, backend, half_precision)

        for file_number, file_path in enumerate(selected_file_list, 0):
            file_number = file_number + 1
            update_process_status(f"Upscaling {file_number}/{len(selected_file_list)}")

            if check_if_file_is_video(file_path):
                upscale_video(file_path, 
                            file_number,
                            AI_model, 
                            selected_AI_model, 
                            scaling_factor,
                            backend, 
                            selected_image_extension, 
                            tiles_resolution, 
                            resize_factor, 
                            cpu_number, 
                            half_precision, 
                            selected_video_extension,
                            selected_interpolation)
            else:
                upscale_image(file_path, 
                            file_number,
                            AI_model, 
                            selected_AI_model, 
                            scaling_factor,
                            backend, 
                            selected_image_extension, 
                            tiles_resolution, 
                            resize_factor, 
                            half_precision, 
                            selected_interpolation)

        update_process_status(f"All files completed! :)")

    except Exception as exception:
        update_process_status(f"Error during upscale process :(\n\n {str(exception)}")
        show_error(exception)

# Images

def upscale_image(image_path, 
                  file_number,
                  AI_model, 
                  selected_AI_model, 
                  scaling_factor,
                  backend, 
                  selected_image_extension, 
                  tiles_resolution, 
                  resize_factor, 
                  half_precision,
                  selected_interpolation):
    
    starting_image    = image_read(image_path)
    result_image_path = prepare_output_image_filename(image_path, 
                                                      selected_AI_model, 
                                                      resize_factor, 
                                                      selected_image_extension,
                                                      selected_interpolation)
                                                      
    if resize_factor != 1: image_to_upscale = resize_image(starting_image, resize_factor) 
    else:                  image_to_upscale = starting_image

    image_target_height, image_target_width = get_final_image_shape(image_to_upscale, scaling_factor)
    need_tiles, num_tiles_x, num_tiles_y    = file_need_tiles(image_to_upscale, tiles_resolution)

    # Complete image
    if need_tiles == False:
        update_process_status(f"{file_number}. Upscaling image")
        image_upscaled = AI_enhance(AI_model, image_to_upscale, backend, half_precision)

    # Tilled image
    else:
        tiles_list     = split_image_into_tiles(image_to_upscale, file_number, num_tiles_x, num_tiles_y, True)
        how_many_tiles = len(tiles_list)

        for tile_index, tile in enumerate(tiles_list, 0):
            update_process_status(f"{file_number}. Upscaling tiles {tile_index}/{how_many_tiles}")       
            tile_upscaled = AI_enhance(AI_model, tile, backend, half_precision)
            tile_upscaled = fix_tile_shape(tile, tile_upscaled, scaling_factor)
            tiles_list[tile_index] = tile_upscaled

        image_upscaled = combine_tiles_into_image(tiles_list, file_number, image_target_height, image_target_width, num_tiles_x, num_tiles_y, True)
    
    if selected_interpolation:
        image_upscaled = interpolate_images(starting_image, image_upscaled, image_target_height, image_target_width)
        image_write(result_image_path, image_upscaled)
    else: 
        image_write(result_image_path, image_upscaled)

# Videos

def upscale_video(video_path, 
                  file_number,
                  AI_model, 
                  selected_AI_model, 
                  scaling_factor,
                  backend, 
                  selected_image_extension, 
                  tiles_resolution,
                  resize_factor, 
                  cpu_number, 
                  half_precision, 
                  selected_video_extension,
                  selected_interpolation):
    
    create_temp_dir(temp_dir)

    frame_list_paths           = extract_video_frames_and_audio(video_path, file_number)
    how_many_frames            = len(frame_list_paths)          
    frames_upscaled_paths_list = [] 

    update_process_status(f"{file_number}. Upscaling video")
    first_frame                                = image_read(frame_list_paths[0])  
    frame_resized_width, frame_resized_height  = get_resized_frame_shape(first_frame, resize_factor)
    frame_target_height, frame_target_width    = get_final_frame_shape(frame_resized_width, frame_resized_height, scaling_factor)
    need_tiles, num_tiles_x, num_tiles_y       = file_need_tiles(first_frame, tiles_resolution)

    for index_frame, frame_path in enumerate(frame_list_paths, 0):

        start_timer = timer()

        starting_frame    = image_read(frame_path)
        result_frame_path = prepare_output_image_filename(frame_path, selected_AI_model, resize_factor, selected_image_extension, selected_interpolation)

        if resize_factor != 1: frame_to_upscale = resize_frame(starting_frame, frame_resized_width, frame_resized_height)
        else:                  frame_to_upscale = starting_frame

        # Complete frame w/out tilling
        if need_tiles == False:
            frame_upscaled = AI_enhance(AI_model, frame_to_upscale, backend, half_precision)

        # Tilled frame
        else:
            tiles_list = split_image_into_tiles(frame_to_upscale, file_number, num_tiles_x, num_tiles_y, False)
            for tile_index, tile in enumerate(tiles_list, 0):
                tile_upscaled          = AI_enhance(AI_model, tile, backend, half_precision)
                tile_upscaled          = fix_tile_shape(tile, tile_upscaled, scaling_factor)
                tiles_list[tile_index] = tile_upscaled

            frame_upscaled = combine_tiles_into_image(tiles_list, file_number, frame_target_height, frame_target_width, num_tiles_x, num_tiles_y, False)

        # Interpolation
        if selected_interpolation:
            frame_upscaled = interpolate_images(starting_frame, frame_upscaled, frame_target_height, frame_target_width)
            image_write(result_frame_path, frame_upscaled)
        else: 
            image_write(result_frame_path, frame_upscaled)
        
        frames_upscaled_paths_list.append(result_frame_path)

        # Update process status every 4 frames
        if index_frame != 0 and (index_frame + 1) % 4 == 0: 
            end_timer        = timer()    
            percent_complete = (index_frame + 1) / how_many_frames * 100 
            time_left        = calculate_time_to_complete_video(start_timer, end_timer, how_many_frames, index_frame)
        
            update_process_status(f"{file_number}. Upscaling video {percent_complete:.2f}% ({time_left})")

    video_reconstruction_by_frames(video_path, 
                                   file_number,
                                   frames_upscaled_paths_list, 
                                   selected_AI_model, 
                                   resize_factor, 
                                   cpu_number, 
                                   selected_video_extension,
                                   selected_interpolation)



# GUI utils function ---------------------------

def user_input_checks():
    global selected_file_list
    global selected_AI_model
    global half_precision
    global selected_AI_device 
    global selected_image_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    is_ready = True

    # Selected files 
    try: selected_file_list = scrollable_frame_file_list.get_selected_file_list()
    except:
        info_message.set("No file selected. Please select a file")
        is_ready = False

    if len(selected_file_list) <= 0:
        info_message.set("No file selected. Please select a file")
        is_ready = False


    # File resize factor 
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_message.set("Resize % must be a numeric value")
        is_ready = False

    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_message.set("Resize % must be a value > 0")
        is_ready = False

    
    # Tiles resolution 
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_message.set("VRAM/RAM value must be a numeric value")
        is_ready = False 

    if tiles_resolution > 0: 
        if   selected_AI_model in BSRGAN_models_list:          vram_multiplier = RRDB_vram_multiplier
        elif selected_AI_model in RealESRGAN_models_list:      vram_multiplier = RRDB_vram_multiplier
        elif selected_AI_model in SRVGGNetCompact_models_list: vram_multiplier = SRVGGNetCompact_vram_multiplier

        #elif selected_AI_model in SAFM_models_list:           vram_multiplier = SAFM_vram_multiplier

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))

        if half_precision == True: 
            tiles_resolution = int(selected_vram * 100)

            #if selected_AI_model in SAFM_models_list: 
            #    info_message.set("SAFM not compatible with Half precision")
            #    is_ready = False

        elif half_precision == False: 
            tiles_resolution = int(selected_vram * 100 * 0.5)
        
    else:
        info_message.set("VRAM/RAM value must be > 0")
        is_ready = False


    # Cpu number 
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_message.set("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        info_message.set("Cpu number value must be > 0")
        is_ready = False
    else: cpu_number = int(cpu_number)


    return is_ready

def extract_image_info(image_file):
    image_name = str(image_file.split("/")[-1])

    image  = image_read(image_file, cv2.IMREAD_UNCHANGED)
    width  = int(image.shape[1])
    height = int(image.shape[0])

    image_label = ( "IMAGE" + " | " + image_name + " | " + str(width) + "x" + str(height) )

    ctkimage = CTkImage(Image.open(image_file), size = (25, 25))

    return image_label, ctkimage

def extract_video_info(video_file):
    cap          = cv2.VideoCapture(video_file)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    video_name   = str(video_file.split("/")[-1])
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        image_write("temp.jpg", frame)
        break
    cap.release()

    video_label = ( "VIDEO" + " | " + video_name + " | " + str(width) + "x" 
                   + str(height) + " | " + str(minutes) + 'm:' 
                   + str(round(seconds)) + "s | " + str(num_frames) 
                   + "frames | " + str(round(frame_rate)) + "fps" )

    ctkimage = CTkImage(Image.open("temp.jpg"), size = (25, 25))
    
    return video_label, ctkimage

def check_if_file_is_video(file):
    for video_extension in supported_video_extensions:
        if video_extension in file:
            return True

def check_supported_selected_files(uploaded_file_list):
    supported_files_list = []

    for file in uploaded_file_list:
        for supported_extension in supported_file_extensions:
            if supported_extension in file:
                supported_files_list.append(file)

    return supported_files_list

def open_files_action():
    info_message.set("Selecting files...")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        place_up_background()

        global scrollable_frame_file_list
        scrollable_frame_file_list = ScrollableImagesTextFrame(master = window, fg_color = dark_color, bg_color = dark_color)
        scrollable_frame_file_list.place(relx = 0.5, rely = 0.25, relwidth = 1.0, relheight = 0.475, anchor = tk.CENTER)
        
        scrollable_frame_file_list.add_clean_button()

        for index in range(supported_files_counter):

            actual_file = supported_files_list[index]

            if check_if_file_is_video(actual_file):
                # video
                video_label, ctkimage = extract_video_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = video_label, image = ctkimage, file_element = actual_file)
                remove_file("temp.jpg")
                
            else:
                # image
                image_label, ctkimage = extract_image_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = image_label, image = ctkimage, file_element = actual_file)
    
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")

def show_error(exception):
    import tkinter as tk
    tk.messagebox.showerror(title   = 'Error', 
                            message = 'Upscale failed caused by:\n\n' +
                                        str(exception) + '\n\n' +
                                        'Please report the error on Github.com or Telegram group' +
                                        '\n\nThank you :)')



# GUI select from menus functions ---------------------------

def select_AI_from_menu(new_value: str):
    global selected_AI_model    
    selected_AI_model = new_value

def select_AI_mode_from_menu(new_value: str):
    global half_precision

    if new_value == "Full precision": half_precision = False
    elif new_value == "Half precision": half_precision = True

def select_AI_device_from_menu(new_value: str):
    global selected_AI_device    

    for device in device_list:
        if device.name == new_value:
            selected_AI_device = device.index

def select_image_extension_from_menu(new_value: str):
    global selected_image_extension    
    selected_image_extension = new_value

def select_video_extension_from_menu(new_value: str):
    global selected_video_extension   
    selected_video_extension = new_value

def select_interpolation_from_menu(new_value: str):
    global selected_interpolation
    if new_value == 'Yes':
        selected_interpolation = True
    elif new_value == 'No':
        selected_interpolation = False



# GUI info functions ---------------------------

def open_info_AI_model():
    info = """This widget allows to choose between different AIs.

[ RRDB (2020) ]
Complex and heavy AIs, high-quality upscale
 • BSRGANx4 | upscale by 4
 • BSRGANx2 | upscale by 2
 • RealESRGANx4 | upscale by 4
_________________________________________

[ SRVGGNetCompact (2022) ]
Fast and lightweight AIs, good-quality upscale
 • RealESR_Gx4 | upscale by 4
 • RealSRx4_Anime | upscale by 4 
""" 
    
    tk.messagebox.showinfo(title = 'AI model', message = info)

def open_info_device():
    info = """This widget allows you to select the GPU for AI processing. \n 
 • Keep in mind that the more powerful your GPU is, the faster the upscaling will be
 • For optimal results, it's essential to regularly update your GPU drivers"""

    tk.messagebox.showinfo(title='GPU', message=info)

def open_info_file_extension():
    info = """This widget allows to choose the extension of upscaled image/frame:\n
 • png | very good quality | supports transparent images
 • jpg | good quality | very fast
 • bmp | highest quality | slow
 • tiff | highest quality | very slow"""

    tk.messagebox.showinfo(title = 'Image output', message = info)

def open_info_resize():
    info = """This widget allows to choose the resolution input to the AI.\n
For example for a 100x100px image:
 • Input resolution 50% => input to AI 50x50px
 • Input resolution 100% => input to AI 100x100px
 • Input resolution 200% => input to AI 200x200px """

    tk.messagebox.showinfo(title = 'Input resolution %', message = info)

def open_info_vram_limiter():
    info = """This widget allows to set a limit on the gpu's VRAM memory usage.\n
 • For a gpu with 4 GB of Vram you must select 4
 • For a gpu with 6 GB of Vram you must select 6
 • For a gpu with 8 GB of Vram you must select 8
 • For integrated gpus (Intel-HD series | Vega 3,5,7) 
   with no dedicated memory, you must select 2 \n
Selecting a value greater than the actual amount of gpu VRAM may result in upscale failure """

    tk.messagebox.showinfo(title = 'GPU Vram (GB)', message = info)
    
def open_info_cpu():
    info = """This widget allows you to choose how many cpus to devote to the app.\n
Where possible the app will use the number of cpus selected."""

    tk.messagebox.showinfo(title = 'Cpu number', message = info)

def open_info_AI_precision():
    info = """This widget allows you to choose the AI upscaling mode.

 • Full precision (>=8GB Vram recommended)
    • compatible with all GPUs 
    • uses 50% more GPU memory than Half precision mode
    • is 30-70% faster than Half precision mode
  
 • Half precision
    • some old GPUs are not compatible with this mode
    • uses 50% less GPU memory than Full precision mode
    • is 30-70% slower than Full precision mode"""

    tk.messagebox.showinfo(title = 'AI precision', message = info)

def open_info_video_extension():
    info = """This widget allows you to choose the video output.

 • .mp4  | produces good quality and well compressed video
 • .avi  | produces the highest quality video
 • .webm | produces low quality but light video"""

    tk.messagebox.showinfo(title = 'Video output', message = info)    

def open_info_interpolation():
    info = """This widget allows you to choose interpolating 
the upscaled image/frame with the original image/frame.

[ INTERPOLATION ]
 • Interpolation is intended as, the fusion of the original 
   image with the one produced by the AI
 • Allows to increase the quality of the final result, 
   especially when using the tilling/merging function.
 • Allows to increase the quality of the final result at low 
   "Input resolution %" values (e.g. <50%)."""

    tk.messagebox.showinfo(title = 'Video output', message = info) 



# GUI place functions ---------------------------
        
def place_up_background():
    up_background = CTkLabel(master  = window, 
                            text    = "",
                            fg_color = dark_color,
                            font     = bold12,
                            anchor   = "w")
    
    up_background.place(relx = 0.5, 
                        rely = 0.0, 
                        relwidth = 1.0,  
                        relheight = 1.0,  
                        anchor = tk.CENTER)

def place_github_button():
    git_button = CTkButton(master      = window, 
                            width      = 30,
                            height     = 30,
                            fg_color   = "black",
                            text       = "", 
                            font       = bold11,
                            image      = logo_git,
                            command    = opengithub)
    
    git_button.place(relx = 0.045, rely = 0.87, anchor = tk.CENTER)

def place_telegram_button():
    telegram_button = CTkButton(master     = window, 
                                width      = 30,
                                height     = 30,
                                fg_color   = "black",
                                text       = "", 
                                font       = bold11,
                                image      = logo_telegram,
                                command    = opentelegram)
    telegram_button.place(relx = 0.045, rely = 0.93, anchor = tk.CENTER)
 
def place_stop_button(): 
    stop_button = CTkButton(master   = window, 
                            width      = 140,
                            height     = 30,
                            fg_color   = "#282828",
                            text_color = "#E0E0E0",
                            text       = "STOP", 
                            font       = bold11,
                            image      = stop_icon,
                            command    = stop_button_command)
    stop_button.place(relx = 0.79, rely = row3_y, anchor = tk.CENTER)

def place_loadFile_section():

    text_drop = """ - SUPPORTED FILES -

IMAGES - jpg png tif bmp webp
VIDEOS - mp4 webm mkv flv gif avi mov mpg qt 3gp"""

    input_file_text = CTkLabel(master    = window, 
                                text     = text_drop,
                                fg_color = dark_color,
                                bg_color = dark_color,
                                width   = 300,
                                height  = 150,
                                font    = bold12,
                                anchor  = "center")
    
    input_file_button = CTkButton(master = window, 
                                width    = 140,
                                height   = 30,
                                text     = "SELECT FILES", 
                                font     = bold11,
                                border_spacing = 0,
                                command        = open_files_action)

    input_file_text.place(relx = 0.5, rely = 0.22,  anchor = tk.CENTER)
    input_file_button.place(relx = 0.5, rely = 0.385, anchor = tk.CENTER)

def place_app_name():
    app_name_label = CTkLabel(master     = window, 
                              text       = app_name + " " + version,
                              text_color = app_name_color,
                              font       = bold20,
                              anchor     = "w")
    
    app_name_label.place(relx = 0.21, rely = 0.56, anchor = tk.CENTER)

def place_AI_menu():
    AI_menu_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "AI model",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_AI_model)

    AI_menu = CTkOptionMenu(master  = window, 
                            values  = AI_models_list,
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            anchor     = "center",
                            command    = select_AI_from_menu,
                            dropdown_font = bold11,
                            dropdown_fg_color = "#000000")

    AI_menu_button.place(relx = 0.21, rely = row1_y - 0.05, anchor = tk.CENTER)
    AI_menu.place(relx = 0.21, rely = row1_y, anchor = tk.CENTER)

def place_AI_mode_menu():
    AI_mode_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "AI precision",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_AI_precision)

    AI_mode_menu = CTkOptionMenu(master    = window, 
                                values     = AI_modes_list,
                                width      = 140,
                                font       = bold11,
                                height     = 30,
                                fg_color   = "#000000",
                                anchor     = "center",
                                dynamic_resizing = False,
                                command          = select_AI_mode_from_menu,
                                dropdown_font    = bold11,
                                dropdown_fg_color = "#000000")
    
    AI_mode_button.place(relx = 0.21, rely = row2_y - 0.05, anchor = tk.CENTER)
    AI_mode_menu.place(relx = 0.21, rely = row2_y, anchor = tk.CENTER)

def place_interpolation_menu():
    interpolation_button = CTkButton(master    = window, 
                                    fg_color   = "black",
                                    text_color = "#ffbf00",
                                    text       = "Interpolation",
                                    height     = 23,
                                    width      = 125,
                                    font       = bold11,
                                    corner_radius = 25,
                                    anchor     = "center",
                                    command    = open_info_interpolation)

    interpolation_menu = CTkOptionMenu(master      = window, 
                                        values     = interpolation_list,
                                        width      = 140,
                                        font       = bold10,
                                        height     = 30,
                                        fg_color   = "#000000",
                                        anchor     = "center",
                                        dynamic_resizing = False,
                                        command    = select_interpolation_from_menu,
                                        dropdown_font     = bold11,
                                        dropdown_fg_color = "#000000")
    
    interpolation_button.place(relx = 0.21, rely = row3_y - 0.05, anchor = tk.CENTER)
    interpolation_menu.place(relx = 0.21, rely  = row3_y, anchor = tk.CENTER)

def place_image_extension_menu():
    file_extension_button = CTkButton(master   = window, 
                                    fg_color   = "black",
                                    text_color = "#ffbf00",
                                    text       = "Image output",
                                    height     = 23,
                                    width      = 125,
                                    font       = bold11,
                                    corner_radius = 25,
                                    anchor     = "center",
                                    command    = open_info_file_extension)

    file_extension_menu = CTkOptionMenu(master     = window, 
                                        values     = image_extension_list,
                                        width      = 140,
                                        font       = bold11,
                                        height     = 30,
                                        fg_color   = "#000000",
                                        anchor     = "center",
                                        command    = select_image_extension_from_menu,
                                        dropdown_font = bold11,
                                        dropdown_fg_color = "#000000")
    
    file_extension_button.place(relx = 0.5, rely = row0_y - 0.05, anchor = tk.CENTER)
    file_extension_menu.place(relx = 0.5, rely = row0_y, anchor = tk.CENTER)

def place_video_extension_menu():
    video_extension_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "Video output",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_video_extension)

    video_extension_menu = CTkOptionMenu(master  = window, 
                                    values     = video_extension_list,
                                    width      = 140,
                                    font       = bold11,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    anchor     = "center",
                                    dynamic_resizing = False,
                                    command    = select_video_extension_from_menu,
                                    dropdown_font = bold11,
                                    dropdown_fg_color = "#000000")
    
    video_extension_button.place(relx = 0.5, rely = row1_y - 0.05, anchor = tk.CENTER)
    video_extension_menu.place(relx = 0.5, rely = row1_y, anchor = tk.CENTER)

def place_gpu_menu():
    AI_device_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "GPU",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_device)

    AI_device_menu = CTkOptionMenu(master  = window, 
                                    values   = device_list_names,
                                    width      = 140,
                                    font       = bold9,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    anchor     = "center",
                                    dynamic_resizing = False,
                                    command    = select_AI_device_from_menu,
                                    dropdown_font = bold11,
                                    dropdown_fg_color = "#000000")
    
    AI_device_button.place(relx = 0.5, rely = row2_y - 0.05, anchor = tk.CENTER)
    AI_device_menu.place(relx = 0.5, rely  = row2_y, anchor = tk.CENTER)

def place_vram_textbox():
    vram_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "GPU Vram (GB)",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_vram_limiter)

    vram_textbox = CTkEntry(master      = window, 
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            textvariable = selected_VRAM_limiter)
    
    vram_button.place(relx = 0.5, rely = row3_y - 0.05, anchor = tk.CENTER)
    vram_textbox.place(relx = 0.5, rely  = row3_y, anchor = tk.CENTER)

def place_input_resolution_textbox():
    resize_factor_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "Input resolution (%)",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_resize)

    resize_factor_textbox = CTkEntry(master    = window, 
                                    width      = 140,
                                    font       = bold11,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    textvariable = selected_resize_factor)
    
    resize_factor_button.place(relx = 0.790, rely = row0_y - 0.05, anchor = tk.CENTER)
    resize_factor_textbox.place(relx = 0.790, rely = row0_y, anchor = tk.CENTER)

def place_cpu_textbox():
    cpu_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "CPU number",
                              height   = 23,
                              width    = 125,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_cpu)

    cpu_textbox = CTkEntry(master    = window, 
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            textvariable = selected_cpu_number)

    cpu_button.place(relx = 0.79, rely = row1_y - 0.05, anchor = tk.CENTER)
    cpu_textbox.place(relx = 0.79, rely  = row1_y, anchor = tk.CENTER)

def place_message_label():
    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold10,
                            fg_color     = "#ffbf00",
                            text_color   = "#000000",
                            anchor       = "center",
                            corner_radius = 25)
    message_label.place(relx = 0.79, rely = row2_y, anchor = tk.CENTER)

def place_upscale_button(): 
    upscale_button = CTkButton(master    = window, 
                                width      = 140,
                                height     = 30,
                                fg_color   = "#282828",
                                text_color = "#E0E0E0",
                                text       = "UPSCALE", 
                                font       = bold11,
                                image      = play_icon,
                                command    = upscale_button_command)
    upscale_button.place(relx = 0.79, rely = row3_y, anchor = tk.CENTER)
   


class App():
    def __init__(self, window):
        window.title('')
        width        = 675
        height       = 600
        window.geometry("675x600")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os.sep + "logo.ico"))

        place_up_background()

        place_app_name()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_AI_mode_menu()
        place_interpolation_menu()

        place_image_extension_menu()
        place_video_extension_menu()
        place_gpu_menu()
        place_vram_textbox()
        
        place_input_resolution_textbox()
        place_cpu_textbox()
        place_message_label()
        place_upscale_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    global selected_file_list
    global selected_AI_model
    global half_precision
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global selected_interpolation
    global tiles_resolution
    global resize_factor
    global cpu_number

    selected_file_list = []

    if   AI_modes_list[0] == "Half precision": half_precision = True
    elif AI_modes_list[0] == "Full precision": half_precision = False

    if interpolation_list[0]   == "No": selected_interpolation = False
    elif interpolation_list[0] == "Yes": selected_interpolation = True

    selected_AI_device       = 0
    selected_AI_model        = AI_models_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    info_message            = tk.StringVar()
    selected_resize_factor  = tk.StringVar()
    selected_VRAM_limiter   = tk.StringVar()
    selected_cpu_number     = tk.StringVar()

    info_message.set("Hi :)")
    selected_resize_factor.set("50")
    selected_VRAM_limiter.set("8")
    selected_cpu_number.set(str(int(os.cpu_count()/2)))

    bold8  = CTkFont(family = "Segoe UI", size = 8, weight = "bold")
    bold9  = CTkFont(family = "Segoe UI", size = 9, weight = "bold")
    bold10 = CTkFont(family = "Segoe UI", size = 10, weight = "bold")
    bold11 = CTkFont(family = "Segoe UI", size = 11, weight = "bold")
    bold12 = CTkFont(family = "Segoe UI", size = 12, weight = "bold")
    bold18 = CTkFont(family = "Segoe UI", size = 18, weight = "bold")
    bold19 = CTkFont(family = "Segoe UI", size = 19, weight = "bold")
    bold20 = CTkFont(family = "Segoe UI", size = 20, weight = "bold")
    bold21 = CTkFont(family = "Segoe UI", size = 21, weight = "bold")

    logo_git      = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "github_logo.png")),    size=(15, 15))
    logo_telegram = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "telegram_logo.png")),  size=(15, 15))
    stop_icon     = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "stop_icon.png")),      size=(15, 15))
    play_icon     = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "upscale_icon.png")),   size=(15, 15))
    clear_icon    = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "clear_icon.png")),     size=(15, 15))

    app = App(window)
    window.update()
    window.mainloop()