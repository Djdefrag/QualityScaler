
import ctypes
import functools
import itertools
import multiprocessing
import os
import os.path
import platform
import shutil
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
import webbrowser
from collections import Counter
from math import sqrt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import numpy as np
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from win32mica import MICAMODE, ApplyMica

import sv_ttk

version  = "1.12"

# NEW
# The app is now a single, portable .exe:
#   - no more confusing directory with lots of files
# Completely rewrote the image splitting function 
#   - (when the whole image does not fit into gpu memory) (again)
#   - this should reduce the vertical and horizontal lines that 
#   - were encountered when this function was used
# For video, frame splitting/joining functions also benefits from multithreading optimization
#   - comparing with the previous algorithms 
#   - tiling frames is at least x2 faster 
#   - joining frames together after upscale is at least x4 faster
#   - (can be even faster depending on the number of cpu selected)

# GUI
# Updated info widget texts

# BUGFIX/IMPROVEMENTS
# General bugfixes & improvements
# Updated dependencies
# Optimized AI models, will now use fewer resources

global app_name
app_name = "QualityScaler"

models_array          = [ 'BSRGANx4', 'BSRGANx2', 'RealSR_JPEGx4' ]
AI_model              = models_array[0]
half_precision        = True

image_path            = "none"
device                = 0
input_video_path      = ""
target_file_extension = ".png"
file_extension_list   = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
single_image          = False
multiple_images       = False
video_file            = False
multi_img_list        = []
video_frames_list     = []
frames_upscaled_list  = []
vram_multiplier       = 1
default_vram_limiter  = 8
multiplier_num_tiles  = 2
cpu_number            = 4
resize_algorithm      = cv2.INTER_AREA
windows_subversion    = int(platform.version().split('.')[2])
compatible_gpus       = torch_directml.device_count()

device_list_names     = []
device_list           = []

class Gpu:
    def __init__(self, name, index):
        self.name = name
        self.index = index

for index in range(compatible_gpus): 
    gpu = Gpu(name = torch_directml.device_name(index), index = index)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")

githubme           = "https://github.com/Djdefrag/QualityScaler"
itchme             = "https://jangystudio.itch.io/qualityscaler"

default_font          = 'Segoe UI'
background_color      = "#181818"
text_color            = "#DCDCDC"
selected_button_color = "#ffbf00"
window_width          = 1300
window_height         = 850
left_bar_width        = 410
left_bar_height       = window_height
drag_drop_width       = window_width - left_bar_width
drag_drop_height      = window_height
show_image_width      = drag_drop_width * 0.8
show_image_height     = drag_drop_width * 0.6
image_text_width      = drag_drop_width * 0.8
support_button_height = 95 
button1_y             = 170
button2_y             = button1_y + 90
button3_y             = button2_y + 90
button4_y             = button3_y + 90
button5_y             = button4_y + 90
button6_y             = button5_y + 90


supported_file_list     = ['.jpg', '.jpeg', '.JPG', '.JPEG',
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

supported_video_list    = ['.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt',
                            '.3gp', '.mpg', '.mpeg']

not_supported_file_list = ['.txt', '.exe', '.xls', '.xlsx', '.pdf',
                           '.odt', '.html', '.htm', '.doc', '.docx',
                           '.ods', '.ppt', '.pptx', '.aiff', '.aif',
                           '.au', '.bat', '.java', '.class',
                           '.csv', '.cvs', '.dbf', '.dif', '.eps',
                           '.fm3', '.psd', '.psp', '.qxd',
                           '.ra', '.rtf', '.sit', '.tar', '.zip',
                           '.7zip', '.wav', '.mp3', '.rar', '.aac',
                           '.adt', '.adts', '.bin', '.dll', '.dot',
                           '.eml', '.iso', '.jar', '.py',
                           '.m4a', '.msi', '.ini', '.pps', '.potx',
                           '.ppam', '.ppsx', '.pptm', '.pst', '.pub',
                           '.sys', '.tmp', '.xlt', '.avif']

ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
font_scale = round(1/scaleFactor, 1)


# ------------------- Split functions -------------------


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
                  rows, cols, 
                  image_path, 
                  should_cleanup):
    
    if len(paths_to_merge) == 0:
        print("No images to merge!")
        return
    for index, path in enumerate(paths_to_merge):
        path_number = int(path.split("_")[-1].split(".")[0])
        if path_number != index:
            print("Warning: Image " + path + " has a number that does not match its index!")
            print("Please rename it first to match the rest of the images.")
            return
        
    images_to_merge = [Image.open(p) for p in paths_to_merge]
    image1    = images_to_merge[0]
    new_width  = image1.size[0] * cols
    new_height = image1.size[1] * rows
    new_image = Image.new(image1.mode, (new_width, new_height))

    for i in range(0, rows):
        for j in range(0, cols):
            image = images_to_merge[i * cols + j]
            new_image.paste(image, (j * image.size[0], i * image.size[1]))
    new_image.save(image_path)

    if should_cleanup:
        for p in paths_to_merge:
            os.remove(p)

def determine_bg_color(im):
    im_width, im_height = im.size
    rgb_im = im.convert('RGBA')
    all_colors = []
    areas = [[(0, 0), (im_width, im_height / 10)],
             [(0, 0), (im_width / 10, im_height)],
             [(im_width * 9 / 10, 0), (im_width, im_height)],
             [(0, im_height * 9 / 10), (im_width, im_height)]]
    for area in areas:
        start = area[0]
        end = area[1]
        for x in range(int(start[0]), int(end[0])):
            for y in range(int(start[1]), int(end[1])):
                pix = rgb_im.getpixel((x, y))
                all_colors.append(pix)
    return Counter(all_colors).most_common(1)[0][0]

def get_tiles_paths_after_split(original_image, rows, cols):
    number_of_tiles = rows * cols

    tiles_paths = []
    for index in range(number_of_tiles):
        tile_path      = os.path.splitext(original_image)[0]
        tile_extension = os.path.splitext(original_image)[1]

        tile_path = tile_path + "_" + str(index) + tile_extension
        tiles_paths.append(tile_path)

    return tiles_paths

def check_number_of_tiles(num_tiles, multiplier_num_tiles):
    num_tiles = round(num_tiles)
    if (num_tiles % 2) != 0: num_tiles += 1
    num_tiles = round(sqrt(num_tiles * multiplier_num_tiles))

    return num_tiles

def video_need_tiles(frame, tiles_resolution):
    img_tmp             = image_read(frame)
    image_resolution    = max(img_tmp.shape[1], img_tmp.shape[0])
    needed_tiles        = image_resolution/tiles_resolution

    if needed_tiles <= 1:
        return False
    else:
        return True

def split_frames_list_in_tiles(frame_list, tiles_resolution, cpu_number):
    list_of_tiles_list = [] # list of list of tiles, to rejoin
    tiles_to_upscale   = [] # list of all tiles to upscale
    
    img_tmp          = image_read(frame_list[0])
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    num_tiles = check_number_of_tiles(num_tiles, multiplier_num_tiles)
    frame_directory_path = os.path.dirname(os.path.abspath(frame_list[0]))

    with ThreadPool(cpu_number) as pool:
        pool.starmap(split_image, zip(frame_list, 
                                  itertools.repeat(num_tiles), 
                                  itertools.repeat(num_tiles), 
                                  itertools.repeat(False),
                                  itertools.repeat(frame_directory_path)))

    for frame in frame_list:    
        tiles_list = get_tiles_paths_after_split(frame, num_tiles, num_tiles)

        list_of_tiles_list.append(tiles_list)

        for tile in tiles_list:
            tiles_to_upscale.append(tile)

    return tiles_to_upscale, list_of_tiles_list, num_tiles

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



# ------------------ / Split functions ------------------

# ------------------------ Utils ------------------------

def remove_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def adapt_image_to_show(image_to_prepare):
    old_image     = image_read(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    if actual_width >= actual_height:
        max_val = actual_width
        max_photo_resolution = show_image_width
    else:
        max_val = actual_height
        max_photo_resolution = show_image_height

    if max_val >= max_photo_resolution:
        downscale_factor = max_val/max_photo_resolution
        new_width        = round(old_image.shape[1]/downscale_factor)
        new_height       = round(old_image.shape[0]/downscale_factor)
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = resize_algorithm)
        image_write("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = resize_algorithm)
        image_write("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, target_file_extension):
    result_path = (img.replace("_resized" + target_file_extension, "").replace(target_file_extension, "") 
                    + "_"  + AI_model + target_file_extension)
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def write_in_log_file(text_to_insert):
    log_file_name   = find_by_relative_path(app_name + ".log")
    with open(log_file_name,'w') as log_file: log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = find_by_relative_path(app_name + ".log")
    with open(log_file_name,'r') as log_file: step = log_file.readline()
    log_file.close()
    return step


# IMAGE

def image_write(path, image_data):
    _, file_extension = os.path.splitext(path)
    r, buff = cv2.imencode(file_extension, image_data)
    buff.tofile(path)

def image_read(image_to_prepare, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(image_to_prepare, dtype=np.uint8), flags)

def resize_image(image_path, resize_factor, target_file_extension):
    new_image_path = (os.path.splitext(image_path)[0] + "_resized" + target_file_extension).strip()

    old_image  = image_read(image_path.strip(), cv2.IMREAD_UNCHANGED)
    new_width  = int(old_image.shape[1] * resize_factor)
    new_height = int(old_image.shape[0] * resize_factor)

    resized_image = cv2.resize(old_image, 
                               (new_width, new_height), 
                                interpolation = resize_algorithm)    
    image_write(new_image_path, resized_image)

def resize_image_list(image_list, resize_factor, target_file_extension):
    files_to_delete   = []
    downscaled_images = []
    how_much_images = len(image_list)

    index = 1
    for image in image_list:
        resized_image_path = (os.path.splitext(image)[0] + "_resized" + target_file_extension).strip()
        
        resize_image(image.strip(), resize_factor, target_file_extension)
        write_in_log_file("Resizing image " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)
        files_to_delete.append(resized_image_path)

        index += 1

    return downscaled_images, files_to_delete


#VIDEO

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
    try:
        video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except Exception as e:
        pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    for video_type in supported_video_list: video_name = video_name.replace(video_type, "")
    upscaled_video_path = (only_path + video_name + "_" + AI_model + ".mp4")
    cap.release()

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



# ----------------------- /Utils ------------------------


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


# ------------------ /AI ------------------


# ----------------------- Core ------------------------

def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(2)
    except:
        place_upscale_button()

def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale video completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(2)
    except:
        place_upscale_button()


def prepare_model(AI_model, device, half_precision):
    backend = torch.device(torch_directml.device(device))

    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")

    if "x2" in AI_model: upscale_factor = 2
    elif "x4" in AI_model: upscale_factor = 4

    model = RRDBNet(in_nc = 3, out_nc = 3, nf = 64, nb = 23, gc = 32, sf = upscale_factor)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters(): v.requires_grad = False
        
    if half_precision: model = model.half()
    model = model.to(backend)

    return model

def enhance(model, img, backend, half_precision):
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
    img = img.unsqueeze(0).to(backend)
    if half_precision: img = img.half()

    output = model(img) ## model
    
    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    if img_mode == 'L':  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #
    
    if img_mode == 'RGBA':
        alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
        alpha = alpha.unsqueeze(0).to(backend)
        if half_precision: alpha = alpha.half()

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

    return output, img_mode



def upscale_tiles(tile, 
                model, 
                device,
                half_precision):

    backend = torch.device(torch_directml.device(device))

    with torch.no_grad():
        tile_adapted     = image_read(tile, cv2.IMREAD_UNCHANGED)
        tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
        image_write(tile, tile_upscaled)

def upscale_single_frame(frame, 
                         model, 
                         result_path,
                         device,
                         half_precision):

    backend = torch.device(torch_directml.device(device))

    with torch.no_grad():
        img_adapted     = image_read(frame, cv2.IMREAD_UNCHANGED)
        img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
        image_write(result_path, img_upscaled)

def process_upscale_video_frames(input_video_path, 
                                 AI_model, 
                                 resize_factor, 
                                 device,
                                 tiles_resolution, 
                                 target_file_extension, 
                                 cpu_number,
                                 half_precision):
    try:
        start = timer()

        create_temp_dir(app_name + "_temp")
      
        write_in_log_file('Extracting video frames...')
        frame_list = extract_frames_from_video(input_video_path)
        
        if resize_factor != 1:
            write_in_log_file('Resizing video frames...')
            frame_list  = resize_frame_list(frame_list, 
                                            resize_factor, 
                                            target_file_extension, 
                                            cpu_number)
            
        model= prepare_model(AI_model, device, half_precision)   

        write_in_log_file('Upscaling...')
        frames_upscaled_list = []
        need_tiles           = video_need_tiles(frame_list[0], tiles_resolution)

        # prepare upscaled frames file paths
        for frame in frame_list:
            result_path = prepare_output_filename(frame, AI_model, target_file_extension)
            frames_upscaled_list.append(result_path) 

        if need_tiles:
            write_in_log_file('Tiling frames...')
            tiles_to_upscale, list_of_tiles_list, num_tiles = split_frames_list_in_tiles(frame_list, 
                                                                                         tiles_resolution,
                                                                                         cpu_number)
            
            done_tiles     = 0
            how_many_tiles = len(tiles_to_upscale)
            
            for tile in tiles_to_upscale:
                upscale_tiles(tile, model, device, half_precision)
                done_tiles += 1
                write_in_log_file("Upscaled tiles " + str(done_tiles) + "/" + str(how_many_tiles))

            write_in_log_file("Reconstructing frames by tiles...")
            reverse_split_multiple_frames(list_of_tiles_list, 
                                          frames_upscaled_list, 
                                          num_tiles, 
                                          cpu_number)

        else:
            how_many_images  = len(frame_list)
            done_images      = 0

            for frame in frame_list:
                upscale_single_frame(frame, 
                                     model, 
                                     prepare_output_filename(frame, AI_model, target_file_extension), 
                                     device, 
                                     half_precision)

                done_images += 1
                write_in_log_file("Upscaled frame " + str(done_images) + "/" + str(how_many_images))

        write_in_log_file("Processing upscaled video...")
        video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number)
        write_in_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")

        remove_dir(app_name + "_temp")
        remove_file("temp.png")

    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')



def upscale_image_and_save(image, 
                           model, 
                           result_path, 
                           tiles_resolution,
                           upscale_factor,
                           device, 
                           half_precision):

    backend = torch.device(torch_directml.device(device))

    original_image          = image_read(image)
    original_image_width    = original_image.shape[1]
    original_image_height   = original_image.shape[0]

    image_resolution = max(original_image_width, original_image_height)
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        # not using tiles
        with torch.no_grad():
            img_adapted     = image_read(image, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
            image_write(result_path, img_upscaled)
    else:
        # using tiles
        num_tiles = check_number_of_tiles(num_tiles, multiplier_num_tiles)
        image_directory_path = os.path.dirname(os.path.abspath(image))

        split_image(image_path = image, 
                    rows       = num_tiles, 
                    cols       = num_tiles, 
                    should_cleanup = False, 
                    output_dir     = image_directory_path)

        tiles_list = get_tiles_paths_after_split(image, num_tiles, num_tiles)
        
        with torch.no_grad():
            for tile in tiles_list:
                tile_adapted     = image_read(tile, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
                image_write(tile, tile_upscaled)

        reverse_split(paths_to_merge = tiles_list, 
                      rows = num_tiles, 
                      cols = num_tiles, 
                      image_path     = result_path, 
                      should_cleanup = False)

        delete_list_of_files(tiles_list)

def process_upscale_multiple_images(image_list, 
                                    AI_model, 
                                    resize_factor, 
                                    device, 
                                    tiles_resolution, 
                                    target_file_extension, 
                                    half_precision):
    
    try:
        start = timer()

        if "x2" in AI_model: upscale_factor = 2
        elif "x4" in AI_model: upscale_factor = 4        

        write_in_log_file('Resizing images...')
        image_list, files_to_delete = resize_image_list(image_list, resize_factor, target_file_extension)

        how_many_images = len(image_list)
        done_images     = 0

        write_in_log_file('Upscaling...')
        model = prepare_model(AI_model, device, half_precision)
        for img in image_list:
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            upscale_image_and_save(img, model, result_path, tiles_resolution, 
                                    upscale_factor, device, half_precision)
            done_images += 1
            write_in_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                
        write_in_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete)
        remove_file("temp.png")

    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')



# ----------------------- /Core ------------------------

# ---------------------- GUI related ----------------------

def opengithub(): webbrowser.open(githubme, new=1)

def openitch(): webbrowser.open(itchme, new=1)

def open_info_ai_model():
    info = """This widget allows you to choose between different AIs: \n
- BSRGANx2 | high upscale quality | slow | enlarge by 2.
- BSRGANx4 | high upscale quality | slow | enlarge by 4.
- RealSR_JPEG | good upscale quality | slow | enlarge by 4. \n
Try them all and find the one that meets your needs :)""" 
    
    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title   = 'AI model', message = info )
    info_window.destroy()
    
def open_info_backend():
    info = """This widget allows you to choose the gpu on which to run your chosen AI. \n 
Keep in mind that the more powerful your gpu is, 
the faster the upscale will be. \n
If the list is empty it means the app couldn't find 
a compatible gpu, try updating your video card driver :)"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'AI device', message = info)
    info_window.destroy()

def open_info_file_extension():
    info = """This widget allows you to choose the extension of the file generated by AI.\n
- png | very good quality | supports transparent images
- jpg | good quality | very fast
- jpg2 (jpg2000) | very good quality | not very popular
- bmp | highest quality | slow
- tiff | highest quality | very slow"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'AI output extension', message = info)
    info_window.destroy()

def open_info_resize():
    info = """This widget allows you to choose the percentage of the resolution input to the AI.\n
For example for a 100x100px image:
- Input resolution 50% => input to AI 50x50px
- Input resolution 100% => input to AI 100x100px
- Input resolution 200% => input to AI 200x200px """

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title   = 'Input resolution %', message = info)
    info_window.destroy()

def open_info_vram_limiter():
    info = """This widget allows you to set a limit on the gpu's VRAM memory usage. \n
- For a gpu with 4 GB of Vram you must select 4
- For a gpu with 6 GB of Vram you must select 6
- For a gpu with 8 GB of Vram you must select 8
- For integrated gpus (Intel-HD series | Vega 3,5,7) 
  that do not have dedicated memory, you must select 2 \n
Selecting a value greater than the actual amount of gpu VRAM may result in upscale failure. """

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title   = 'VRAM limiter GB', message = info)
    info_window.destroy()
    
def open_info_cpu():
    info = """This widget allows you to choose how much cpu to devote to the app.\n
Where possible the app will use the number of processors you select, for example:
- Extracting frames from videos
- Resizing frames from videos
- Recostructing final video"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title   = 'Cpu number', message = info)
    info_window.destroy() 

def user_input_checks():
    global tiles_resolution
    global resize_factor
    global cpu_number

    is_ready = True

    if compatible_gpus == 0:
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Sorry, your gpu is not compatible with QualityScaler :(')
        is_ready = False

    # resize factor
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_string.set("Resize % must be a numeric value")
        is_ready = False

    #if resize_factor > 0 and resize_factor <= 100: resize_factor = resize_factor/100
    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_string.set("Resize % must be a value > 0")
        is_ready = False
    
    # vram limiter
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_string.set("VRAM/RAM value must be a numeric value")
        is_ready = False 

    if tiles_resolution > 0: tiles_resolution = 100 * (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))    
    else:
        info_string.set("VRAM/RAM value must be > 0")
        is_ready = False

    # cpu number
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_string.set("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        info_string.set("Cpu number value must be > 0")
        is_ready = False
    elif cpu_number == 1: cpu_number = 1
    else: cpu_number = int(cpu_number)

    return is_ready

def upscale_button_command():
    global image_path
    global multiple_images
    global process_upscale
    global thread_wait
    global video_frames_list
    global video_file
    global frames_upscaled_list
    global input_video_path
    global device
    global tiles_resolution
    global target_file_extension
    global cpu_number
    global half_precision

    remove_file(app_name + ".log")

    info_string.set("Loading...")
    write_in_log_file("Loading...")

    is_ready = user_input_checks()

    if is_ready:
        if video_file:
            place_stop_button()

            process_upscale = multiprocessing.Process(target = process_upscale_video_frames,
                                                    args   = (input_video_path, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                cpu_number,
                                                                half_precision))
            process_upscale.start()

            thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                        args   = (1, 2), 
                                        daemon = True)
            thread_wait.start()

        elif multiple_images or single_image:
            place_stop_button()
            
            process_upscale = multiprocessing.Process(target = process_upscale_multiple_images,
                                                        args   = (multi_img_list, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                half_precision))
            process_upscale.start()

            thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                            args   = (1, 2), daemon = True)
            thread_wait.start()

        elif "none" in image_path:
            info_string.set("No file selected")
  
def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    process_upscale.join()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped upscaling") 

def drop_event_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list: image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() 

    return image_list

def file_drop_event(event):
    global image_path
    global multiple_images
    global multi_img_list
    global video_file
    global single_image
    global input_video_path

    supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number = count_files_dropped(event)
    all_supported, single_image, multiple_images, video_file, more_than_one_video = check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number)

    if video_file:
        # video section
        if not all_supported:
            info_string.set("Some files are not supported")
            return
        elif all_supported:
            if multiple_images:
                info_string.set("Only one video supported")
                return
            elif not multiple_images:
                if not more_than_one_video:
                    input_video_path = str(event.data).replace("{", "").replace("}", "")
                    show_video_in_GUI(input_video_path)

                    # reset variable
                    image_path = "none"
                    multi_img_list = []

                elif more_than_one_video:
                    info_string.set("Only one video supported")
                    return
    else:
        # image section
        if not all_supported:
            if multiple_images:
                info_string.set("Some files are not supported")
                return
            elif single_image:
                info_string.set("This file is not supported")
                return
        elif all_supported:
            if multiple_images:
                image_list_dropped = drop_event_to_image_list(event)
                show_list_images_in_GUI(image_list_dropped)
                multi_img_list = image_list_dropped
                # reset variable
                image_path = "none"
                video_frames_list = []

            elif single_image:
                image_list_dropped = drop_event_to_image_list(event)
                show_image_in_GUI(image_list_dropped[0])
                multi_img_list = image_list_dropped

                # reset variable
                image_path = "none"
                video_frames_list = []

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, 
                        supported_video_dropped_number):
    all_supported  = True
    single_image    = False
    multiple_images = False
    video_file    = False
    more_than_one_video = False

    if not_supported_file_dropped_number > 0:
        all_supported = False

    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_image = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_images = True

    if supported_video_dropped_number == 1:
        video_file = True
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_file = True
        more_than_one_video = True

    return all_supported, single_image, multiple_images, video_file, more_than_one_video

def count_files_dropped(event):
    supported_file_dropped_number = 0
    not_supported_file_dropped_number = 0
    supported_video_dropped_number = 0

    # count compatible images files
    for file_type in supported_file_list:
        supported_file_dropped_number = supported_file_dropped_number + \
            str(event.data).count(file_type)

    # count compatible video files
    for file_type in supported_video_list:
        supported_video_dropped_number = supported_video_dropped_number + \
            str(event.data).count(file_type)

    # count not supported files
    for file_type in not_supported_file_list:
        not_supported_file_dropped_number = not_supported_file_dropped_number + \
            str(event.data).count(file_type)

    return supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number

def clear_input_variables():
    global image_path
    global multi_img_list
    global video_frames_list
    global single_image
    global multiple_images
    global video_file

    # reset variable
    image_path        = "none"
    multi_img_list    = []
    video_frames_list = []
    single_image       = False
    multiple_images    = False
    video_file       = False

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = file_drop_event,
                          relief = "flat",
                          background = background_color,
                          foreground = text_color)
    drag_drop.place(x = left_bar_width + 50, y=0,
                    width = drag_drop_width, height = drag_drop_height)


def show_video_in_GUI(video_path):
    clear_app_background()
    
    fist_frame   = "temp.jpg"
    cap          = cv2.VideoCapture(video_path)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    path_as_list = video_path.split("/")
    video_name   = str(path_as_list[-1])
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        image_write(fist_frame, frame)
        break
    cap.release()

    resized_image_to_show = adapt_image_to_show(fist_frame)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(image_read(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(image_read(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                y = image_y_center,
                width  = resized_image_to_show_width,
                height = resized_image_to_show_height)

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = ( video_name + "\n" + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps" ),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 85,
                            width  = image_text_width,
                            height = 40)

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()          

    os.remove(fist_frame)
              
def show_list_images_in_GUI(image_list):
    clear_app_background()

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                       y = 125,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

    final_string = "\n"
    counter_img = 0
    for elem in image_list:
        counter_img += 1
        if counter_img <= 8:
            img     = image_read(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) + ".  " + img_name + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    images_list_label = ttk.Label(root,
                            text    = final_string,
                            ondrop  = file_drop_event,
                            font    = bold12,
                            anchor  = "n",
                            relief  = "flat",
                            justify = "left",
                            background = background_color,
                            foreground = "#D3D3D3",
                            wraplength = list_width)

    images_list_label.place(x = left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2 -25,
                               width  = list_width,
                               height = list_height)

    images_counter = ttk.Entry(root, 
                                foreground = text_color,
                                ondrop     = file_drop_event,
                                font       = bold12, 
                                justify    = 'center')
    images_counter.insert(0, str(len(image_list)) + ' images')
    images_counter.configure(state='disabled')
    images_counter.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                        y  = drag_drop_height/2 + 225,
                        width  = 200,
                        height = 42)

def show_image_in_GUI(original_image):
    clear_app_background()
    original_image = original_image.replace('{', '').replace('}', '')
    resized_image_to_show = adapt_image_to_show(original_image)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(image_read(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(image_read(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                      y = image_y_center,
                      width  = resized_image_to_show_width,
                      height = resized_image_to_show_height)

    img_name     = str(original_image.split("/")[-1])
    width        = round(image_read(original_image).shape[1])
    height       = round(image_read(original_image).shape[0])

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = (img_name + " | [" + str(width) + "x" + str(height) + "]"),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 70,
                            width  = image_text_width,
                            height = 40)

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

    text_drop = (" DROP FILES HERE \n\n"
                + " ⥥ \n\n"
                + " IMAGE   - jpg png tif bmp webp \n\n"
                + " IMAGE LIST   - jpg png tif bmp webp \n\n"
                + " VIDEO   - mp4 webm mkv flv gif avi mov mpg qt 3gp \n\n")

    drag_drop = ttk.Notebook(root, ondrop  = file_drop_event)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.75)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.75)/2

    drag_drop.place(x = x_center, 
                    y = y_center, 
                    width  = drag_drop_width * 0.75, 
                    height = drag_drop_height * 0.75)

    drag_drop_text = ttk.Label(root,
                            text    = text_drop,
                            ondrop  = file_drop_event,
                            font    = bold12,
                            anchor  = "center",
                            relief  = 'flat',
                            justify = "center",
                            foreground = text_color)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.5)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.5)/2
    
    drag_drop_text.place(x = x_center, 
                         y = y_center, 
                         width  = drag_drop_width * 0.50, 
                         height = drag_drop_height * 0.50)

def combobox_AI_selection(event):
    global AI_model
    selected = str(selected_AI.get())
    AI_model = selected
    combo_box_AI.set('')
    combo_box_AI.set(selected)

def combobox_backend_selection(event):
    global device

    selected_option = str(selected_backend.get())
    combo_box_backend.set('')
    combo_box_backend.set(selected_option)

    for obj in device_list: 
        if obj.name == selected_option: 
            device = obj.index

def combobox_extension_selection(event):
    global target_file_extension
    selected = str(selected_file_extension.get()).strip()
    target_file_extension = selected
    combobox_file_extension.set('')
    combobox_file_extension.set(selected)


def place_AI_combobox():
    Ai_container = ttk.Notebook(root)
    Ai_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button1_y - 17, 
                        width  = 370,
                        height = 75)

    Ai_label = ttk.Label(root, 
                    font       = bold11, 
                    foreground = text_color, 
                    justify    = 'left', 
                    relief     = 'flat', 
                    text       = " AI model ")
    Ai_label.place(x = 90,
                    y = button1_y - 2,
                    width  = 130,
                    height = 42)

    global combo_box_AI
    combo_box_AI = ttk.Combobox(root, 
                        textvariable = selected_AI, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = models_array,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = bold10)
    combo_box_AI.place(x = 10 + left_bar_width/2, 
                        y = button1_y, 
                        width  = 200, 
                        height = 40)
    combo_box_AI.bind('<<ComboboxSelected>>', combobox_AI_selection)
    combo_box_AI.set(AI_model)

    Ai_combobox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    Ai_combobox_info_button.place(x = 50,
                                y = button1_y + 6,
                                width  = 30,
                                height = 30)
    Ai_combobox_info_button["command"] = lambda: open_info_ai_model()

def place_backend_combobox():
    backend_container = ttk.Notebook(root)
    backend_container.place(x = 45 + left_bar_width/2 - 370/2, 
                            y = button2_y - 17, 
                            width  = 370,
                            height = 75)

    backend_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " AI device ")
    backend_label.place(x = 90,
                        y = button2_y - 2,
                        width  = 155,
                        height = 42)

    global combo_box_backend
    combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = device_list_names,
                            state        = 'readonly',
                            takefocus    = False,
                            font         = bold10)
    combo_box_backend.place(x = 10 + left_bar_width/2, 
                            y = button2_y, 
                            width  = 200, 
                            height = 40)
    combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    combo_box_backend.set(device_list_names[0])

    backend_combobox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    backend_combobox_info_button.place(x = 50,
                                    y = button2_y + 6,
                                    width  = 30,
                                    height = 30)
    backend_combobox_info_button["command"] = lambda: open_info_backend()

def place_file_extension_combobox():
    file_extension_container = ttk.Notebook(root)
    file_extension_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button3_y - 17, 
                        width  = 370,
                        height = 75)

    file_extension_label = ttk.Label(root, 
                        font       = bold11, 
                        foreground = text_color, 
                        justify    = 'left', 
                        relief     = 'flat', 
                        text       = " AI output extension ")
    file_extension_label.place(x = 90,
                            y = button3_y - 2,
                            width  = 155,
                            height = 42)

    global combobox_file_extension
    combobox_file_extension = ttk.Combobox(root, 
                        textvariable = selected_file_extension, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = file_extension_list,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = bold11)
    combobox_file_extension.place(x = 65 + left_bar_width/2, 
                        y = button3_y, 
                        width  = 145, 
                        height = 40)
    combobox_file_extension.bind('<<ComboboxSelected>>', combobox_extension_selection)
    combobox_file_extension.set(target_file_extension)

    file_extension_combobox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    file_extension_combobox_info_button.place(x = 50,
                                    y = button3_y + 6,
                                    width  = 30,
                                    height = 30)
    file_extension_combobox_info_button["command"] = lambda: open_info_file_extension()


def place_resize_factor_spinbox():
    resize_factor_container = ttk.Notebook(root)
    resize_factor_container.place(x = 45 + left_bar_width/2 - 370/2, 
                               y = button4_y - 17, 
                               width  = 370,
                               height = 75)

    global spinbox_resize_factor
    spinbox_resize_factor = ttk.Spinbox(root,  
                                        from_        = 1, 
                                        to           = 100, 
                                        increment    = 1,
                                        textvariable = selected_resize_factor, 
                                        justify      = 'center',
                                        foreground   = text_color,
                                        takefocus    = False,
                                        font         = bold12)
    spinbox_resize_factor.place(x = 65 + left_bar_width/2, 
                                y = button4_y, 
                                width  = 145, 
                                height = 40)
    spinbox_resize_factor.insert(0, '70')

    resize_factor_label = ttk.Label(root, 
                                    font       = bold11, 
                                    foreground = text_color, 
                                    justify    = 'left', 
                                    relief     = 'flat', 
                                    text       = " Input resolution | % ")
    resize_factor_label.place(x = 90,
                            y = button4_y - 2,
                            width  = 155,
                            height = 42)
    
    resize_spinbox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    resize_spinbox_info_button.place(x = 50,
                                    y = button4_y + 6,
                                    width  = 30,
                                    height = 30)
    resize_spinbox_info_button["command"] = lambda: open_info_resize()

def place_VRAM_spinbox():
    vram_container = ttk.Notebook(root)
    vram_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button5_y - 17, 
                        width  = 370,
                        height = 75)

    global spinbox_VRAM
    spinbox_VRAM = ttk.Spinbox(root,  
                                from_     = 1, 
                                to        = 100, 
                                increment = 1,
                                textvariable = selected_VRAM_limiter, 
                                justify      = 'center',
                                foreground   = text_color,
                                takefocus    = False,
                                font         = bold12)
    spinbox_VRAM.place(x = 65 + left_bar_width/2, 
                        y = button5_y, 
                        width  = 145, 
                        height = 40)
    spinbox_VRAM.insert(0, str(default_vram_limiter))

    vram_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " Vram limiter | GB ")
    vram_label.place(x = 90,
                    y = button5_y - 2,
                    width  = 155,
                    height = 42)
    
    vram_spinbox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    vram_spinbox_info_button.place(x = 50,
                                    y = button5_y + 6,
                                    width  = 30,
                                    height = 30)
    vram_spinbox_info_button["command"] = lambda: open_info_vram_limiter()

def place_cpu_number_spinbox():
    cpu_number_container = ttk.Notebook(root)
    cpu_number_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button6_y - 17, 
                        width  = 370,
                        height = 75)

    global spinbox_cpus
    spinbox_cpus = ttk.Spinbox(root,  
                                from_     = 1, 
                                to        = 100, 
                                increment = 1,
                                textvariable = selected_cpu_number, 
                                justify      = 'center',
                                foreground   = text_color,
                                takefocus    = False,
                                font         = bold12)
    spinbox_cpus.place(x = 65 + left_bar_width/2, 
                        y = button6_y, 
                        width  = 145, 
                        height = 40)
    spinbox_cpus.insert(0, str(cpu_number))

    cpus_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " Cpu number ")
    cpus_label.place(x = 90,
                    y = button6_y - 2,
                    width  = 155,
                    height = 42)
    
    cpu_spinbox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    cpu_spinbox_info_button.place(x = 50,
                                    y = button6_y + 6,
                                    width  = 30,
                                    height = 30)
    cpu_spinbox_info_button["command"] = lambda: open_info_cpu()


def place_app_title():
    Title = ttk.Label(root, 
                      font       = bold21,
                      foreground = "#DA70D6", 
                      background = background_color,
                      anchor     = 'w', 
                      text       = app_name)
    Title.place(x = 75,
                y = 40,
                width  = 300,
                height = 55)


    version_button = ttk.Button(root,
                               image = logo_itch,
                               padding = '0 0 0 0',
                               text    = " " + version,
                               compound = 'left',
                               style    = 'Bold.TButton')
    version_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 30,
                        width  = 125,
                        height = 35)
    version_button["command"] = lambda: openitch()

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    github_button = ttk.Button(root,
                               image = logo_git,
                               padding = '0 0 0 0',
                               text    = ' Github',
                               compound = 'left',
                               style    = 'Bold.TButton')
    github_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 75,
                        width  = 125,
                        height = 35)
    github_button["command"] = lambda: opengithub()

def place_message_box():
    message_label = ttk.Label(root,
                            font       = bold11,
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            background = background_color,
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 45 + left_bar_width/2 - left_bar_width/2,
                        y = window_height - 120,
                        width  = left_bar_width,
                        height = 30)

def place_upscale_button():    
    button_Style = ttk.Style()
    button_Style.configure("Bold.TButton",  font = bold11, foreground = text_color)

    Upscale_button = ttk.Button(root, 
                                text  = ' UPSCALE',
                                image = play_icon,
                                compound = tk.LEFT,
                                style = "Bold.TButton")

    Upscale_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                         y      = left_bar_height - 80,
                         width  = 280,
                         height = 47)
    Upscale_button["command"] = lambda: upscale_button_command()

def place_stop_button():
    Stop_button = ttk.Button(root, 
                                text  = '  STOP UPSCALE ',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                      y      = left_bar_height - 80,
                      width  = 280,
                      height = 47)

    Stop_button["command"] = lambda: stop_button_command()

def place_background():
    background = ttk.Label(root, background = background_color, relief = 'flat')
    background.place(x = 0, 
                     y = 0, 
                     width  = window_width,
                     height = window_height)


# ---------------------- /GUI related ----------------------


def apply_windows_dark_bar(window_root):
    window_root.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute          = ctypes.windll.dwmapi.DwmSetWindowAttribute
    get_parent                    = ctypes.windll.user32.GetParent
    hwnd                          = get_parent(window_root.winfo_id())
    rendering_policy              = DWMWA_USE_IMMERSIVE_DARK_MODE
    value                         = 2
    value                         = ctypes.c_int(value)
    set_window_attribute(hwnd, rendering_policy, ctypes.byref(value), ctypes.sizeof(value))    

    #Changes the window size
    window_root.geometry(str(window_root.winfo_width()+1) + "x" + str(window_root.winfo_height()+1))
    #Returns to original size
    window_root.geometry(str(window_root.winfo_width()-1) + "x" + str(window_root.winfo_height()-1))

def apply_windows_transparency_effect(window_root):
    window_root.wm_attributes("-transparent", background_color)
    hwnd = ctypes.windll.user32.GetParent(window_root.winfo_id())
    ApplyMica(hwnd, MICAMODE.DARK )


class App:
    def __init__(self, root):
        sv_ttk.use_dark_theme()
        
        root.title('')
        width        = window_width
        height       = window_height
        screenwidth  = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        root.iconphoto(False, PhotoImage(file = find_by_relative_path("Assets"  + os.sep + "logo.png")))

        if windows_subversion >= 22000: apply_windows_transparency_effect(root) # Windows 11
        apply_windows_dark_bar(root)

        place_background()                                  # Background
        place_app_title()                                   # App title
        place_AI_combobox()                                 # AI models widget
        place_resize_factor_spinbox()                       # Upscale factor widget
        place_VRAM_spinbox()                                # VRAM widget
        place_backend_combobox()                            # Backend widget
        place_file_extension_combobox()
        place_cpu_number_spinbox()
        place_message_box()                                 # Message box
        place_upscale_button()                              # Upscale button
        place_drag_drop_widget()                            # Drag&Drop widget
        
if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_AI = tk.StringVar()
    selected_resize_factor  = tk.StringVar()
    selected_VRAM_limiter   = tk.StringVar()
    selected_backend        = tk.StringVar()
    selected_file_extension = tk.StringVar()
    selected_cpu_number     = tk.StringVar()

    bold10 = tkFont.Font(family = default_font, size   = round(10 * font_scale), weight = 'bold')
    bold11 = tkFont.Font(family = default_font, size   = round(11 * font_scale), weight = 'bold')
    bold12 = tkFont.Font(family = default_font, size   = round(12 * font_scale), weight = 'bold')
    bold13 = tkFont.Font(family = default_font, size   = round(13 * font_scale), weight = 'bold')
    bold14 = tkFont.Font(family = default_font, size   = round(14 * font_scale), weight = 'bold')
    bold15 = tkFont.Font(family = default_font, size   = round(15 * font_scale), weight = 'bold')
    bold20 = tkFont.Font(family = default_font, size   = round(20 * font_scale), weight = 'bold')
    bold21 = tkFont.Font(family = default_font, size   = round(21 * font_scale), weight = 'bold')

    global stop_icon
    global clear_icon
    global play_icon
    global logo_itch
    global logo_git
    logo_git      = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "github_logo.png"))
    logo_itch     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "itch_logo.png"))
    stop_icon     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "stop_icon.png"))
    play_icon     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "upscale_icon.png"))
    clear_icon    = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "clear_icon.png"))

    app = App(root)
    root.update()
    root.mainloop()
