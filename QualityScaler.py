import ctypes
import functools
import itertools
import multiprocessing
import os.path
import platform
import shutil
import sys
import threading
import time
import tkinter
import tkinter as tk
import webbrowser
from math import sqrt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from customtkinter import (CTk, CTkButton, CTkEntry, CTkFont, CTkImage,
                           CTkLabel, CTkOptionMenu, CTkScrollableFrame,
                           filedialog, set_appearance_mode,
                           set_default_color_theme)
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from win32mica import MICAMODE, ApplyMica

# CHANGELOG
# NEW
# - It is now possible to upscale images and videos in one shot
# - The message box is now more "conversational," having added many more messages
# - Now the app will save the upscaled files by adding the chosen resolution % tag. This allows you to try different % resolutions without the produced file overwriting the previous one. 
#   - For example, an image with BSRGANx4 and Resolution 70%:
#   - QualityScaler 1.14 => image_BSRGANx4.png
#   - QualityScaler 2.0 => image_BSRGANx4_70%.png
# - New GUI based on the splendid work of @customtkinter:
#   - it is now possible to select files via a "Select files" button instead of the Drag&Drop function that presented several problems
#   - this new library allows for much more organized and optimized code
#   - the new interface is fully resizable and so adaptable to any screen 
#   - the new interface also allows more space to add new widgets (in the future) to customize the upscale operations 
# BUGFIX & IMPROVEMENTS
# - A comprehensive restructuring of the code and many micro-optimizations:
#   - more than 50% of the code has been restructured to be faster and more readable
# - Updated all dependencies
# FOR DEVS
# - With the new GUI based on @customtkinter, it is easier to run the app via python script and should give less headaches than the old GUI which needed more 3/4 different libraries
# - Many more logs have been added in console (for those who use the app via Python code) 

# I want to sincerely thank the people who support and will support this work (financially and otherwise). 
# Thank you for allowing me to learn how to develop these projects and at the same time be able to help my parents financially. ❤



app_name  = "QualityScaler"
version   = "2.0"

githubme             = "https://github.com/Djdefrag/QualityScaler"
itchme               = "https://jangystudio.itch.io/qualityscaler"

AI_models_list       = [ 'BSRGANx4', 'BSRGANx2', 'RealSR_JPEGx4' ]
file_extension_list  = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
device_list_names    = []
device_list          = []
half_precision       = True
vram_multiplier      = 1
multiplier_num_tiles = 4
windows_subversion   = int(platform.version().split('.')[2])
resize_algorithm     = cv2.INTER_AREA

offset_y_options = 0.1125
option_y_1       = 0.705
option_y_2       = option_y_1 + offset_y_options
option_y_3       = option_y_2 + offset_y_options
option_y_4       = option_y_1
option_y_5       = option_y_4 + offset_y_options
option_y_6       = option_y_5 + offset_y_options

transparent_color = "#080808"

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

for index in range(torch_directml.device_count()): 
    gpu_name = torch_directml.device_name(index)

    gpu = Gpu(index = index, name = gpu_name)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

supported_file_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG',
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
                            '.qt', '.3gp', '.mpg', '.mpeg']

supported_video_extensions  = ['.mp4', '.MP4',
                                '.webm', '.WEBM',
                                '.mkv', '.MKV',
                                '.flv', '.FLV',
                                '.gif', '.GIF',
                                '.m4v', ',M4V',
                                '.avi', '.AVI',
                                '.mov', '.MOV',
                                '.qt',
                                '.3gp', '.mpg', '.mpeg']

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")



#  Slice functions -------------------

def split_image(image_path, 
                rows, cols, 
                should_cleanup, 
                output_dir = None):
    
    im = Image.open(image_path)
    im_width, im_height = im.size
    row_width  = int(im_width / cols)
    row_height = int(im_height / rows)
    name, ext  = os.path.splitext(image_path)
    name       = os.path.basename(name)

    if output_dir != None:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
    else:
        output_dir = "./"

    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            box = (j * row_width, i * row_height, j * row_width +
                   row_width, i * row_height + row_height)
            outp = im.crop(box)
            outp_path = name + "_" + str(n) + ext
            outp_path = os.path.join(output_dir, outp_path)
            outp.save(outp_path)
            n += 1

    if should_cleanup: os.remove(image_path)

def reverse_split(paths_to_merge, 
                  rows, 
                  cols, 
                  image_path, 
                  should_cleanup):
        
    images_to_merge = [Image.open(p) for p in paths_to_merge]
    image1     = images_to_merge[0]
    new_width  = image1.size[0] * cols
    new_height = image1.size[1] * rows
    new_image  = Image.new(image1.mode, (new_width, new_height))

    for i in range(0, rows):
        for j in range(0, cols):
            image = images_to_merge[i * cols + j]
            new_image.paste(image, (j * image.size[0], i * image.size[1]))
    new_image.save(image_path)

    if should_cleanup:
        for p in paths_to_merge:
            os.remove(p)

def get_tiles_paths_after_split(original_image, rows, cols):
    number_of_tiles = rows * cols

    tiles_paths = []
    for index in range(number_of_tiles):
        tile_path      = os.path.splitext(original_image)[0]
        tile_extension = os.path.splitext(original_image)[1]

        tile_path = tile_path + "_" + str(index) + tile_extension
        tiles_paths.append(tile_path)

    return tiles_paths

def video_need_tiles(frame, tiles_resolution):
    img_tmp             = image_read(frame)
    image_pixels        = (img_tmp.shape[1] * img_tmp.shape[0])
    tile_pixels         = (tiles_resolution * tiles_resolution)

    n_tiles = image_pixels/tile_pixels

    if n_tiles <= 1:
        return False, 0
    else:
        if (n_tiles % 2) != 0: n_tiles += 1
        n_tiles = round(sqrt(n_tiles * multiplier_num_tiles))

        return True, n_tiles

def image_need_tiles(image, tiles_resolution):
    img_tmp             = image_read(image)
    image_pixels        = (img_tmp.shape[1] * img_tmp.shape[0])
    tile_pixels         = (tiles_resolution * tiles_resolution)

    n_tiles = image_pixels/tile_pixels

    if n_tiles <= 1: 
        return False, 0
    else:
        if (n_tiles % 2) != 0: n_tiles += 1
        n_tiles = round(sqrt(n_tiles * multiplier_num_tiles))

        return True, n_tiles

def split_frames_list_in_tiles(frame_list, n_tiles, cpu_number):
    list_of_tiles_list = []   # list of list of tiles, to rejoin
    tiles_to_upscale   = []   # list of all tiles to upscale
    
    frame_directory_path = os.path.dirname(os.path.abspath(frame_list[0]))

    with ThreadPool(cpu_number) as pool:
        pool.starmap(split_image, zip(frame_list, 
                                  itertools.repeat(n_tiles), 
                                  itertools.repeat(n_tiles), 
                                  itertools.repeat(False),
                                  itertools.repeat(frame_directory_path)))

    for frame in frame_list:    
        tiles_list = get_tiles_paths_after_split(frame, n_tiles, n_tiles)
        list_of_tiles_list.append(tiles_list)
        for tile in tiles_list: tiles_to_upscale.append(tile)

    return tiles_to_upscale, list_of_tiles_list

def reverse_split_multiple_frames(list_of_tiles_list, 
                                  frames_upscaled_list, 
                                  num_tiles, 
                                  cpu_number):
    
    with ThreadPool(cpu_number) as pool:
        pool.starmap(reverse_split, zip(list_of_tiles_list, 
                                    itertools.repeat(num_tiles), 
                                    itertools.repeat(num_tiles), 
                                    frames_upscaled_list,
                                    itertools.repeat(False)))



# Utils functions ------------------------

def opengithub(): webbrowser.open(githubme, new=1)

def openitch(): webbrowser.open(itchme, new=1)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir, mode=0o777)

def remove_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)

def write_in_log_file(text_to_insert):
    log_file_name = app_name + ".log"
    with open(log_file_name,'w') as log_file: 
        os.chmod(log_file_name, 0o777)
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name = app_name + ".log"
    with open(log_file_name,'r') as log_file: 
        os.chmod(log_file_name, 0o777)
        step = log_file.readline()
    log_file.close()
    return step

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def prepare_output_image_filename(image_path, selected_AI_model, resize_factor, selected_output_file_extension):
    # remove extension
    result_path = os.path.splitext(image_path)[0]

    resize_percentage = str(int(resize_factor * 100)) + "%"
    to_append = "_"  + selected_AI_model + "_" + resize_percentage + selected_output_file_extension

    if "_resized" in result_path: 
        result_path = result_path.replace("_resized", "") 
        result_path = result_path + to_append
    else:
        result_path = result_path + to_append

    return result_path

def prepare_output_video_filename(video_path, selected_AI_model, resize_factor):
    result_video_path = os.path.splitext(video_path)[0] # remove extension

    resize_percentage = str(int(resize_factor * 100)) + "%"
    to_append = "_"  + selected_AI_model + "_" + resize_percentage + ".mp4"
    result_video_path = result_video_path + to_append

    return result_video_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def image_write(path, image_data):
    _, file_extension = os.path.splitext(path)
    r, buff = cv2.imencode(file_extension, image_data)
    buff.tofile(path)

def image_read(image_to_prepare, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(image_to_prepare, dtype=np.uint8), flags)

def resize_image(image_path, resize_factor, selected_output_file_extension):
    new_image_path = (os.path.splitext(image_path)[0] 
                      + "_resized" 
                      + selected_output_file_extension)

    old_image  = image_read(image_path, cv2.IMREAD_UNCHANGED)
    new_width  = int(old_image.shape[1] * resize_factor)
    new_height = int(old_image.shape[0] * resize_factor)

    resized_image = cv2.resize(old_image, (new_width, new_height), interpolation = resize_algorithm)    
    image_write(new_image_path, resized_image)
    return new_image_path

def extract_frames_from_video(video_path):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    img_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(img_sequence, logger = 'bar', fps = frame_rate)
    
    # extract audio
    try: video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except Exception as e: pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, frames_upscaled_list, 
                                   selected_AI_model, resize_factor, cpu_number):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    upscaled_video_path = prepare_output_video_filename(input_video_path, selected_AI_model, resize_factor)
    audio_file = app_name + "_temp" + os.sep + "audio.mp3"

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os.path.exists(audio_file):
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            audio   = audio_file,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                             fps     = frame_rate,
                             threads = cpu_number)       

def resize_frame(image_path, new_width, new_height, target_file_extension):
    new_image_path = image_path.replace('.jpg', "" + target_file_extension)
    
    old_image = cv2.imread(image_path.strip(), cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = resize_algorithm)    
    image_write(new_image_path, resized_image)

def resize_frame_list(image_list, resize_factor, target_file_extension, cpu_number):
    downscaled_images = []

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)
    
    with ThreadPool(cpu_number) as pool:
        pool.starmap(resize_frame, zip(image_list, 
                                    itertools.repeat(new_width), 
                                    itertools.repeat(new_height), 
                                    itertools.repeat(target_file_extension)))

    for image in image_list:
        resized_image_path = image.replace('.jpg', "" + target_file_extension)
        downscaled_images.append(resized_image_path)

    return downscaled_images

def is_Windows11():
    if windows_subversion >= 22000: return True

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def show_error(exception):
    import tkinter as tk
    tk.messagebox.showerror(title   = 'Error', 
                            message = 'Upscale failed caused by:\n\n' +
                                        str(exception) + '\n\n' +
                                        'Please report the error on Github.com or Itch.io.' +
                                        '\n\nThank you :)')



# ------------------ AI ------------------

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4: self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4: fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def AI_enhance(model, img, backend, half_precision):
    img = img.astype(np.float32)

    if np.max(img) > 256: max_range = 65535 # 16 bit images
    else: max_range = 255

    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- process image (without the alpha channel) ------------------- #
    
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    if half_precision: img = img.unsqueeze(0).half().to(backend, non_blocking = True)
    else: img = img.unsqueeze(0).to(backend, non_blocking = True)

    output = model(img)
    
    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    if img_mode == 'L':  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #
    
    if img_mode == 'RGBA':
        alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
        if half_precision: alpha = alpha.unsqueeze(0).half().to(backend, non_blocking = True)
        else: alpha = alpha.unsqueeze(0).to(backend, non_blocking = True)

        output_alpha = model(alpha) ## model

        output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535: output = (output_img * 65535.0).round().astype(np.uint16) # 16-bit image
    else: output = (output_img * 255.0).round().astype(np.uint8)

    return output

def prepare_model(AI_model, device, half_precision, upscale_factor):
    backend    = torch.device(torch_directml.device(device))
    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")

    model = RRDBNet(in_nc = 3, out_nc = 3, nf = 64, nb = 23, gc = 32, sf = upscale_factor)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters(): v.requires_grad = False
        
    if half_precision: model = model.half()
    model = model.to(backend, non_blocking = True)

    return model



# Core functions ------------------------

def stop_thread():
    # to stop a thread execution
    stop = 1 + "x"

def check_upscale_steps(not_used1, not_used2):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "All files completed" in step:
                info_message.set(step)
                stop_thread()
            elif "Error while upscaling" in step:
                info_message.set("Error while upscaling :(")
                stop_thread()
            elif "Stopped upscaling" in step:
                info_message.set("Stopped upscaling")
                stop_thread()
            else:
                info_message.set(step)
            time.sleep(2)
    except:
        place_upscale_button()

def update_process_status(actual_process_phase):
    print("> " + actual_process_phase)
    write_in_log_file(actual_process_phase) 

def stop_button_command():
    global process_upscale_orchestrator
    process_upscale_orchestrator.terminate()
    process_upscale_orchestrator.join()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped upscaling") 

def upscale_button_function(): 
    global selected_file_list
    global selected_AI_model
    global selected_AI_device 
    global selected_output_file_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    global process_upscale_orchestrator

    remove_file(app_name + ".log")
    
    if user_input_checks():
        update_process_status('Loading...')
        info_message.set("Loading...")

        print("=================================================")
        print("> Starting upscale:")
        print("   Files to upscale: " + str(len(selected_file_list)))
        print("   AI model: "         + str(selected_AI_model))
        print("   AI device: "        + str(selected_AI_device))
        print("   Output extension: " + str(selected_output_file_extension))
        print("   GPU vram: "         + str(int(tiles_resolution/100)) + "GB" + " => " + "tiles resolution: " + str(tiles_resolution) + "x" + str(tiles_resolution) + "px")
        print("   Resize factor: "    + str(int(resize_factor*100)) + "%")
        print("   Cpu number: "       + str(cpu_number))
        print("=================================================")

        place_stop_button()

        process_upscale_orchestrator = multiprocessing.Process(
                                            target = upscale_orchestrator,
                                            args   = (selected_file_list,
                                                    selected_AI_model,
                                                    selected_AI_device, 
                                                    selected_output_file_extension,
                                                    tiles_resolution,
                                                    resize_factor,
                                                    cpu_number,
                                                    half_precision))
        process_upscale_orchestrator.start()

        thread_wait = threading.Thread(
                                target = check_upscale_steps,
                                args   = (1, 2), 
                                daemon = True)
        thread_wait.start()

def upscale_image(image_path,
                  AI_model, 
                  selected_AI_model,
                  upscale_factor,
                  selected_AI_device, 
                  selected_output_file_extension, 
                  tiles_resolution,
                  resize_factor,
                  cpu_number,
                  half_precision):
    
    # if image need resize before AI work
    if resize_factor != 1:
        image_path = resize_image(image_path, 
                                   resize_factor, 
                                   selected_output_file_extension)

    result_path = prepare_output_image_filename(image_path, selected_AI_model, resize_factor, selected_output_file_extension)
    upscale_image_and_save(image_path, 
                            AI_model, 
                            result_path, 
                            tiles_resolution,
                            upscale_factor, 
                            selected_AI_device, 
                            half_precision)
    
    # if the image was sized before the AI work
    if resize_factor != 1: remove_file(image_path)

def upscale_image_and_save(image, 
                           AI_model, 
                           result_path, 
                           tiles_resolution, 
                           upscale_factor, 
                           selected_AI_device, 
                           half_precision):

    backend = torch.device(torch_directml.device(selected_AI_device))
    need_tiles, n_tiles = image_need_tiles(image, tiles_resolution)

    if need_tiles:
        split_image(image_path     = image, 
                    rows           = n_tiles, 
                    cols           = n_tiles, 
                    should_cleanup = False, 
                    output_dir     = os.path.dirname(os.path.abspath(image)))

        tiles_list = get_tiles_paths_after_split(image, n_tiles, n_tiles)

        with torch.no_grad():
            for tile in tiles_list:
                tile_adapted  = image_read(tile, cv2.IMREAD_UNCHANGED)
                tile_upscaled = AI_enhance(AI_model, tile_adapted, backend, half_precision)
                image_write(tile, tile_upscaled)

        reverse_split(paths_to_merge = tiles_list, 
                      rows           = n_tiles, 
                      cols           = n_tiles, 
                      image_path     = result_path, 
                      should_cleanup = False)

        delete_list_of_files(tiles_list)
    else:
        with torch.no_grad():
            img_adapted  = image_read(image, cv2.IMREAD_UNCHANGED)
            img_upscaled = AI_enhance(AI_model, img_adapted, backend, half_precision)
            image_write(result_path, img_upscaled)

def upscale_video(video_path,
                 AI_model,
                 selected_AI_model,
                 upscale_factor,
                 selected_AI_device,
                 selected_output_file_extension,
                 tiles_resolution,
                 resize_factor,
                 cpu_number,
                 half_precision):
    
    remove_file("temp.png")

    create_temp_dir(app_name + "_temp")
    
    update_process_status("Extracting video frames")
    frame_list = extract_frames_from_video(video_path)
    
    if resize_factor != 1:
        update_process_status("Resizing video frames")
        frame_list  = resize_frame_list(frame_list, 
                                        resize_factor, 
                                        selected_output_file_extension, 
                                        cpu_number)

    upscale_video_and_save(video_path, frame_list, AI_model,  
                           tiles_resolution, selected_AI_model, selected_AI_device, 
                           resize_factor, selected_output_file_extension, 
                           half_precision, cpu_number)
    
    remove_dir(app_name + "_temp")

def upscale_video_and_save(video_path, frame_list, AI_model,  
                           tiles_resolution, selected_AI_model, selected_AI_device,
                           resize_factor, selected_output_file_extension, 
                           half_precision, cpu_number):
    
    update_process_status("Upscaling video")
    frames_upscaled_list = []
    need_tiles, n_tiles  = video_need_tiles(frame_list[0], tiles_resolution)

    # Prepare upscaled frames file paths
    for frame in frame_list:
        result_path = prepare_output_image_filename(frame, selected_AI_model, resize_factor, selected_output_file_extension)
        frames_upscaled_list.append(result_path) 

    if need_tiles:
        update_process_status("Tiling frames...")
        tiles_to_upscale, list_of_tiles_list = split_frames_list_in_tiles(frame_list, n_tiles, cpu_number)
        how_many_tiles = len(tiles_to_upscale)
        
        for index in range(how_many_tiles):
            upscale_tiles(tiles_to_upscale[index], 
                          AI_model, 
                          selected_AI_device, 
                          half_precision)
            if (index % 4) == 0: update_process_status("Upscaled tiles " + str( index + 1 ) + "/" + str(how_many_tiles))

        update_process_status("Reconstructing frames by tiles...")
        reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list, n_tiles, cpu_number)

    else:
        how_many_frames = len(frame_list)

        for index in range(how_many_frames):
            upscale_single_frame(frame_list[index], 
                                    AI_model, 
                                    frames_upscaled_list[index], 
                                    selected_AI_device, 
                                    half_precision)
            if (index % 4) == 0: update_process_status("Upscaled frames " + str( index + 1 ) + "/" + str(how_many_frames))
    
    # Reconstruct the video with upscaled frames
    update_process_status("Processing upscaled video")
    video_reconstruction_by_frames(video_path, frames_upscaled_list, selected_AI_model, 
                                   resize_factor, cpu_number)

def upscale_tiles(tile, AI_model, selected_AI_device, half_precision):
    backend = torch.device(torch_directml.device(selected_AI_device))

    with torch.no_grad():
        tile_adapted  = image_read(tile, cv2.IMREAD_UNCHANGED)
        tile_upscaled = AI_enhance(AI_model, tile_adapted, backend, half_precision)
        image_write(tile, tile_upscaled)

def upscale_single_frame(frame, AI_model, result_path, selected_AI_device, half_precision):
    backend = torch.device(torch_directml.device(selected_AI_device))

    with torch.no_grad():
        img_adapted  = image_read(frame, cv2.IMREAD_UNCHANGED)
        img_upscaled = AI_enhance(AI_model, img_adapted, backend, half_precision)
        image_write(result_path, img_upscaled)

def upscale_orchestrator(selected_file_list,
                         selected_AI_model,
                         selected_AI_device, 
                         selected_output_file_extension,
                         tiles_resolution,
                         resize_factor,
                         cpu_number,
                         half_precision):
    
    start = timer()

    update_process_status("Preparing AI model")
    if   "x2" in selected_AI_model: upscale_factor = 2
    elif "x4" in selected_AI_model: upscale_factor = 4 

    try:
        AI_model = prepare_model(selected_AI_model, selected_AI_device, half_precision, upscale_factor)

        how_many_files = len(selected_file_list)
        for index in range(how_many_files):
            update_process_status("Upscaling " + str(index + 1) + "/" + str(how_many_files))
            file_element = selected_file_list[index]
            if check_if_file_is_video(file_element):
                upscale_video(file_element, AI_model, 
                              selected_AI_model, upscale_factor, 
                              selected_AI_device, selected_output_file_extension,
                              tiles_resolution, resize_factor, 
                              cpu_number, half_precision)
            else:
                upscale_image(file_element, AI_model, 
                              selected_AI_model, upscale_factor, 
                              selected_AI_device, selected_output_file_extension, 
                              tiles_resolution, resize_factor, 
                              cpu_number, half_precision)

        update_process_status("All files completed (" + str(round(timer() - start)) + " sec.)")

    except Exception as exception:
        write_in_log_file('Error while upscaling' + '\n\n' + str(exception)) 
        show_error(exception)



# GUI utils function ---------------------------

def user_input_checks():
    global selected_file_list
    global selected_AI_model
    global selected_AI_device 
    global selected_output_file_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    is_ready = True

    # files -------------------------------------------------
    try: selected_file_list = scrollable_frame_file_list.get_selected_file_list()
    except:
        info_message.set("No file selected. Please select a file")
        is_ready = False

    if len(selected_file_list) <= 0:
        info_message.set("No file selected. Please select a file")
        is_ready = False

    # resize factor -------------------------------------------------
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_message.set("Resize % must be a numeric value")
        is_ready = False

    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_message.set("Resize % must be a value > 0")
        is_ready = False

    
    # vram limiter -------------------------------------------------
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_message.set("VRAM/RAM value must be a numeric value")
        is_ready = False 

    if tiles_resolution > 0: 
        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))
        tiles_resolution = 100 * selected_vram
    else:
        info_message.set("VRAM/RAM value must be > 0")
        is_ready = False


    # cpu number -------------------------------------------------
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

    uploaded_files_list = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        place_up_background()

        global scrollable_frame_file_list
        scrollable_frame_file_list = ScrollableImagesTextFrame(master = window, 
                                                               fg_color = transparent_color, 
                                                               bg_color = transparent_color)
        scrollable_frame_file_list.place(relx = 0.5, 
                                         rely = 0.25, 
                                         relwidth = 1.0, 
                                         relheight = 0.475, 
                                         anchor = tkinter.CENTER)
        
        scrollable_frame_file_list.add_clean_button()

        for index in range(supported_files_counter):
            actual_file = supported_files_list[index]
            if check_if_file_is_video(actual_file):
                # video
                video_label, ctkimage = extract_video_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = video_label, 
                                                    image = ctkimage,
                                                    file_element = actual_file)
            else:
                # image
                image_label, ctkimage = extract_image_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = image_label, 
                                                    image = ctkimage,
                                                    file_element = actual_file)
    
        info_message.set("Ready")
    else: 
        info_message.set("Not supported files :(")


# GUI select from menus functions ---------------------------

def select_AI_from_menu(new_value: str):
    global selected_AI_model    
    selected_AI_model = new_value

def select_AI_device_from_menu(new_value: str):
    global selected_AI_device    

    for device in device_list:
        if device.name == new_value:
            selected_AI_device = device.index

def select_output_file_extension_from_menu(new_value: str):
    global selected_output_file_extension    
    selected_output_file_extension = new_value



# GUI info functions ---------------------------

def open_info_ai_model():
    info = """This widget allows you to choose between different AIs: \n
- BSRGANx2 | high upscale quality | slow | enlarge by 2.
- BSRGANx4 | high upscale quality | slow | enlarge by 4.
- RealSR_JPEG | good upscale quality | slow | enlarge by 4. \n
Try them all and find the one that meets your needs :)""" 
    
    tk.messagebox.showinfo(title = 'AI model', message = info)
    
def open_info_device():
    info = """This widget allows you to choose the gpu on which to run your chosen AI. \n 
Keep in mind that the more powerful your gpu is, 
the faster the upscale will be. \n
If the list is empty it means the app couldn't find 
a compatible gpu, try updating your video card driver :)"""

    tk.messagebox.showinfo(title = 'AI device', message = info)

def open_info_file_extension():
    info = """This widget allows you to choose the extension of the file generated by AI.\n
- png | very good quality | supports transparent images
- jpg | good quality | very fast
- jp2 (jpg2000) | very good quality | not very popular
- bmp | highest quality | slow
- tiff | highest quality | very slow"""

    tk.messagebox.showinfo(title = 'AI output extension', message = info)

def open_info_resize():
    info = """This widget allows you to choose the percentage of the resolution input to the AI.\n
For example for a 100x100px image:
- Input resolution 50% => input to AI 50x50px
- Input resolution 100% => input to AI 100x100px
- Input resolution 200% => input to AI 200x200px """

    tk.messagebox.showinfo(title = 'Input resolution %', message = info)

def open_info_vram_limiter():
    info = """This widget allows you to set a limit on the gpu's VRAM memory usage. \n
- For a gpu with 4 GB of Vram you must select 4
- For a gpu with 6 GB of Vram you must select 6
- For a gpu with 8 GB of Vram you must select 8
- For integrated gpus (Intel-HD series | Vega 3,5,7) 
  that do not have dedicated memory, you must select 2 \n
Selecting a value greater than the actual amount of gpu VRAM may result in upscale failure. """

    tk.messagebox.showinfo(title = 'VRAM limiter GB', message = info)
    
def open_info_cpu():
    info = """This widget allows you to choose how much cpu to devote to the app.\n
Where possible the app will use the number of processors you select, for example:
- Extracting frames from videos
- Resizing frames from videos
- Recostructing final video"""

    tk.messagebox.showinfo(title = 'Cpu number', message = info)



# GUI place functions ---------------------------
        
def place_up_background():
    up_background = CTkLabel(master  = window, 
                            text    = "",
                            fg_color = transparent_color,
                            font     = bold12,
                            anchor   = "w")
    
    up_background.place(relx = 0.5, 
                        rely = 0.0, 
                        relwidth = 1.0,  
                        relheight = 1.0,  
                        anchor = tkinter.CENTER)

def place_app_name():
    app_name_label = CTkLabel(master     = window, 
                              text       = app_name + " " + version,
                              text_color = "#DA70D6",
                              font       = bold20,
                              anchor     = "w")
    
    app_name_label.place(relx = 0.5, rely = 0.56, anchor = tkinter.CENTER)

def place_itch_button(): 
    itch_button = CTkButton(master     = window, 
                            width      = 30,
                            height     = 30,
                            fg_color   = "black",
                            text       = "", 
                            font       = bold11,
                            image      = logo_itch,
                            command    = openitch)
    itch_button.place(relx = 0.045, rely = 0.55, anchor = tkinter.CENTER)

def place_github_button():
    git_button = CTkButton(master      = window, 
                            width      = 30,
                            height     = 30,
                            fg_color   = "black",
                            text       = "", 
                            font       = bold11,
                            image      = logo_git,
                            command    = opengithub)
    git_button.place(relx = 0.045, rely = 0.61, anchor = tkinter.CENTER)

def place_upscale_button(): 
    upscale_button = CTkButton(master    = window, 
                                width      = 140,
                                height     = 30,
                                fg_color   = "#282828",
                                text_color = "#E0E0E0",
                                text       = "UPSCALE", 
                                font       = bold11,
                                image      = play_icon,
                                command    = upscale_button_function)
    upscale_button.place(relx = 0.8, rely = option_y_6, anchor = tkinter.CENTER)
    
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
    stop_button.place(relx = 0.8, rely = option_y_6, anchor = tkinter.CENTER)

def place_AI_menu():
    AI_menu_label = CTkLabel(master  = window, 
                            text    = "AI model",
                            width   = 140,
                            height  = 30,
                            font    = bold12,
                            anchor  = "w")

    AI_menu = CTkOptionMenu(master  = window, 
                            values  = AI_models_list,
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            anchor     = "center",
                            command    = select_AI_from_menu)

    AI_menu_info_button = CTkButton(master   = window, 
                                    width      = 16,
                                    height     = 16,
                                    fg_color     = "#ffbf00",
                                    text_color   = "#000000",
                                    text       = "info", 
                                    font       = bold9,
                                    border_spacing = 0,
                                    corner_radius  = 25,
                                    command        = open_info_ai_model)
    
    AI_menu_label.place(relx = 0.20, rely = option_y_1 - 0.05, anchor = tkinter.CENTER)
    AI_menu.place(relx = 0.20, rely = option_y_1, anchor = tkinter.CENTER)
    AI_menu_info_button.place(relx = 0.235, rely = option_y_1 - 0.05, anchor = tkinter.CENTER)

def place_AI_device_menu():
    AI_device_menu_label = CTkLabel(master  = window, 
                                    text    = "AI device",
                                    width   = 140,
                                    height  = 30,
                                    font    = bold12,
                                    anchor  = "w")

    AI_device_menu = CTkOptionMenu(master  = window, 
                                    values   = device_list_names,
                                    width      = 140,
                                    font       = bold9,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    anchor     = "center",
                                    dynamic_resizing = False,
                                    command    = select_AI_device_from_menu)
    
    AI_device_menu_info_button = CTkButton(master   = window, 
                                            width      = 16,
                                            height     = 16,
                                            fg_color     = "#ffbf00",
                                            text_color   = "#000000",
                                            text       = "info", 
                                            font       = bold9,
                                            border_spacing = 0,
                                            corner_radius  = 25,
                                            command        = open_info_device)
    
    AI_device_menu_label.place(relx = 0.20, rely = option_y_2 - 0.05, anchor = tkinter.CENTER)
    AI_device_menu.place(relx = 0.20, rely = option_y_2, anchor = tkinter.CENTER)
    AI_device_menu_info_button.place(relx = 0.235, rely = option_y_2 - 0.05, anchor = tkinter.CENTER)

def place_file_extension_menu():
    file_extension_menu_label = CTkLabel(master  = window, 
                                        text    = "AI output",
                                        width   = 140,
                                        height  = 30,
                                        font    = bold12,
                                        anchor  = "w")

    file_extension_menu = CTkOptionMenu(master  = window, 
                                        values     = file_extension_list,
                                        width      = 140,
                                        font       = bold11,
                                        height     = 30,
                                        fg_color   = "#000000",
                                        anchor     = "center",
                                        command    = select_output_file_extension_from_menu)
    
    file_extension_menu_info_button = CTkButton(master   = window, 
                                                width      = 16,
                                                height     = 16,
                                                fg_color     = "#ffbf00",
                                                text_color   = "#000000",
                                                text       = "info", 
                                                font       = bold9,
                                                border_spacing = 0,
                                                corner_radius  = 25,
                                                command        = open_info_file_extension)

    file_extension_menu_label.place(relx = 0.20, rely = option_y_3 - 0.05, anchor = tkinter.CENTER)
    file_extension_menu.place(relx = 0.20, rely = option_y_3, anchor = tkinter.CENTER)
    file_extension_menu_info_button.place(relx = 0.235, rely = option_y_3 - 0.05, anchor = tkinter.CENTER)

def place_resize_factor_textbox():
    resize_factor_label = CTkLabel(master   = window, 
                                    text    = "Resolution%",
                                    width   = 140,
                                    height  = 30,
                                    font    = bold12,
                                    anchor  = "w")

    resize_factor_textbox = CTkEntry(master    = window, 
                                    width      = 140,
                                    font       = bold11,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    textvariable = selected_resize_factor)
    
    resize_factor_info_button = CTkButton(master   = window, 
                                        width      = 16,
                                        height     = 16,
                                        fg_color     = "#ffbf00",
                                        text_color   = "#000000",
                                        text       = "info", 
                                        font       = bold9,
                                        border_spacing = 0,
                                        corner_radius  = 25,
                                        command        = open_info_resize)
    
    resize_factor_label.place(relx = 0.5, rely = option_y_4 - 0.05, anchor = tkinter.CENTER)
    resize_factor_textbox.place(relx = 0.5, rely  = option_y_4, anchor = tkinter.CENTER)
    resize_factor_info_button.place(relx = 0.535, rely = option_y_4 - 0.05, anchor = tkinter.CENTER)

def place_vram_textbox():
    vram_label = CTkLabel(master  = window, 
                        text    = "GPU Vram",
                        width   = 140,
                        height  = 30,
                        font    = bold12,
                        anchor  = "w")

    vram_textbox = CTkEntry(master      = window, 
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            textvariable = selected_VRAM_limiter)
    
    vram_info_button = CTkButton(master   = window, 
                                width      = 16,
                                height     = 16,
                                fg_color     = "#ffbf00",
                                text_color   = "#000000",
                                text       = "info", 
                                font       = bold9,
                                border_spacing = 0,
                                corner_radius  = 25,
                                command        = open_info_vram_limiter)

    vram_label.place(relx = 0.5, rely = option_y_5 - 0.05, anchor = tkinter.CENTER)
    vram_textbox.place(relx = 0.5, rely  = option_y_5, anchor = tkinter.CENTER)
    vram_info_button.place(relx = 0.535, rely = option_y_5 - 0.05, anchor = tkinter.CENTER)

def place_cpu_textbox():
    cpu_label = CTkLabel(master  = window, 
                        text    = "CPU",
                        width   = 140,
                        height  = 30,
                        font    = bold12,
                        anchor  = "w")

    cpu_textbox = CTkEntry(master    = window, 
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            textvariable = selected_cpu_number)
    
    cpu_info_button = CTkButton(master   = window, 
                                width      = 16,
                                height     = 16,
                                fg_color     = "#ffbf00",
                                text_color   = "#000000",
                                text       = "info", 
                                font       = bold9,
                                border_spacing = 0,
                                corner_radius  = 25,
                                command        = open_info_cpu)

    cpu_label.place(relx = 0.5, rely = option_y_6 - 0.05, anchor = tkinter.CENTER)
    cpu_textbox.place(relx = 0.5, rely  = option_y_6, anchor = tkinter.CENTER)
    cpu_info_button.place(relx = 0.535, rely = option_y_6 - 0.05, anchor = tkinter.CENTER)

def place_loadFile_section():

    text_drop = """SUPPORTED FILES

IMAGE - jpg png tif bmp webp
 VIDEO - mp4 webm mkv flv gif
         avi mov mpg qt 3gp"""

    input_file_text = CTkLabel(master    = window, 
                                text     = text_drop,
                                fg_color = transparent_color,
                                bg_color = transparent_color,
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

    input_file_text.place(relx = 0.5, rely = 0.22,  anchor = tkinter.CENTER)
    input_file_button.place(relx = 0.5, rely = 0.4, anchor = tkinter.CENTER)

def place_message_label():
    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold10,
                            fg_color     = "#ffbf00",
                            text_color   = "#000000",
                            anchor       = "center",
                            corner_radius = 25)
    message_label.place(relx = 0.8, rely = 0.56, anchor = tkinter.CENTER)

def apply_windows_transparency_effect(window_root):
    window_root.wm_attributes("-transparent", transparent_color)
    hwnd = ctypes.windll.user32.GetParent(window_root.winfo_id())
    ApplyMica(hwnd, MICAMODE.DARK )



class App():
    def __init__(self, window):
        window.title('')
        width        = 650
        height       = 600
        window.geometry("650x600")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os.sep + "logo.ico"))

        place_up_background()
        place_app_name()
        place_itch_button()
        place_github_button()

        place_AI_menu()
        place_AI_device_menu()
        place_file_extension_menu()

        place_resize_factor_textbox()
        place_vram_textbox()
        place_cpu_textbox()

        place_message_label()
        place_upscale_button()

        place_loadFile_section()

        if is_Windows11(): apply_windows_transparency_effect(window)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    global selected_file_list

    global selected_AI_model
    global selected_AI_device 
    global selected_output_file_extension

    global tiles_resolution
    global resize_factor
    global cpu_number


    selected_file_list = []
    selected_AI_model  = AI_models_list[0]
    selected_output_file_extension = file_extension_list[0]
    selected_AI_device = 0

    info_message = tk.StringVar()
    selected_AI  = tk.StringVar()
    selected_resize_factor  = tk.StringVar()
    selected_VRAM_limiter   = tk.StringVar()
    selected_backend        = tk.StringVar()
    selected_file_extension = tk.StringVar()
    selected_cpu_number     = tk.StringVar()

    info_message.set("Hi :)")

    selected_resize_factor.set("70")
    selected_VRAM_limiter.set("8")
    selected_cpu_number.set("4")

    bold8  = CTkFont(family = "Segoe UI", size = 8, weight = "bold")
    bold9  = CTkFont(family = "Segoe UI", size = 9, weight = "bold")
    bold10 = CTkFont(family = "Segoe UI", size = 10, weight = "bold")
    bold11 = CTkFont(family = "Segoe UI", size = 11, weight = "bold")
    bold12 = CTkFont(family = "Segoe UI", size = 12, weight = "bold")
    bold20 = CTkFont(family = "Segoe UI", size = 20, weight = "bold")
    bold21 = CTkFont(family = "Segoe UI", size = 21, weight = "bold")


    global stop_icon
    global clear_icon
    global play_icon
    global logo_itch
    global logo_git
    logo_git   = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "github_logo.png")), size=(15, 15))
    logo_itch  = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "itch_logo.png")),  size=(13, 13))
    stop_icon  = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "stop_icon.png")), size=(15, 15))
    play_icon  = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "upscale_icon.png")), size=(15, 15))
    clear_icon = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "clear_icon.png")), size=(15, 15))

    app = App(window)
    window.update()
    window.mainloop()