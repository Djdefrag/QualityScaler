import ctypes
import functools
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
from math import ceil, floor, sqrt
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import numpy as np
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io import AudioFileClip
from moviepy.video.io import ImageSequenceClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from win32mica import MICAMODE, ApplyMica

import sv_ttk

pay      = True
version  = "v. 5.1"

if not pay:
    version = version + ".f"

global app_name
app_name = "QualityScaler"

image_path            = "no file"
AI_model              = "BSRGANx4"
device                = "dml"
input_video_path      = ""
windows_subversion    = int(platform.version().split('.')[2])
tiles_resolution      = 800
single_file           = False
multiple_files        = False
video_files           = False
multi_img_list        = []
video_frames_list     = []
video_frames_upscaled_list = []
blur_pre_resize       = True
target_file_extension = ".png"

paypalme           = "https://www.paypal.com/paypalme/jjstd/5"
githubme           = "https://github.com/Djdefrag/QualityScaler"
itchme             = "https://jangystudio.itch.io/qualityscaler"

default_font          = 'Segoe UI'
background_color      = "#181818"
window_width          = 1300
window_height         = 750
left_bar_width        = 410
left_bar_height       = window_height
drag_drop_width       = window_width - left_bar_width
drag_drop_height      = window_height
show_image_width      = drag_drop_width * 0.8
show_image_height     = drag_drop_width * 0.6
image_text_width      = drag_drop_width * 0.8
support_button_height = 95 
button_1_y            = 215
button_2_y            = 315
button_3_y            = 415
button_4_y            = 515
text_color            = "#DCDCDC"
selected_button_color = "#ffbf00"

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


# ---------------------- /Dimensions ----------------------

# ---------------------- Functions ----------------------

# ------------------- Slice functions -------------------


class Tile(object):
    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self):
        return self.position[0]

    @property
    def column(self):
        return self.position[1]

    @property
    def basename(self):
        return get_basename(self.filename)

    def generate_filename(
        self, directory=os.getcwd(), prefix="tile", format="png", path=True
    ):
        filename = prefix + "_{col:02d}_{row:02d}.{ext}".format(
            col=self.column, row=self.row, ext=format.lower().replace("jpeg", "jpg")
        )
        if not path:
            return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format="png"):
        if not filename:
            filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return "<Tile #{} - {}>".format(
                self.number, os.path.basename(self.filename)
            )
        return "<Tile #{}>".format(self.number)

def get_basename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def calc_columns_rows(n):
    """
    Calculate the number of columns and rows required to divide an image
    into ``n`` parts.

    Return a tuple of integers in the format (num_columns, num_rows)
    """
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)

def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def join(tiles, width=0, height=0):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @param ``width`` - Optional, width of combined image.
    @param ``height`` - Optional, height of combined image.
    @return ``Image`` instance.
    """
    # Don't calculate size if width and height are provided
    # this allows an application that knows what the
    # combined size should be to construct an image when
    # pieces are missing.

    if width > 0 and height > 0:
        im = Image.new("RGBA", (width, height), None)
    else:
        im = Image.new("RGBA", get_combined_size(tiles), None)
    columns, rows = calc_columns_rows(len(tiles))
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            # do nothing, blank out the image
            continue
    return im

def validate_image(image, number_tiles):
    """Basic sanity checks prior to performing a split."""
    TILE_LIMIT = 99 * 99

    try:
        number_tiles = int(number_tiles)
    except BaseException:
        raise ValueError("number_tiles could not be cast to integer.")

    if number_tiles > TILE_LIMIT or number_tiles < 2:
        raise ValueError(
            "Number of tiles must be between 2 and {} (you \
                          asked for {}).".format(
                TILE_LIMIT, number_tiles
            )
        )

def validate_image_col_row(image, col, row):
    """Basic checks for columns and rows values"""
    SPLIT_LIMIT = 99

    try:
        col = int(col)
        row = int(row)
    except BaseException:
        raise ValueError("columns and rows values could not be cast to integer.")

    if col < 1 or row < 1 or col > SPLIT_LIMIT or row > SPLIT_LIMIT:
        raise ValueError(
            f"Number of columns and rows must be between 1 and"
            f"{SPLIT_LIMIT} (you asked for rows: {row} and col: {col})."
        )
    if col == 1 and row == 1:
        raise ValueError("There is nothing to divide. You asked for the entire image.")

def img_cutter(filename, number_tiles=None, col=None, row=None, save=True, DecompressionBombWarning=True):
    if DecompressionBombWarning is False:
        Image.MAX_IMAGE_PIXELS = None

    im = Image.open(filename)
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if number_tiles:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    if save:
        save_tiles(tiles, prefix=get_basename(filename), directory=os.path.dirname(filename))
    return tuple(tiles)

def save_tiles(tiles, prefix="", directory=os.getcwd(), format="png"):
    for tile in tiles:
        tile.save(
            filename=tile.generate_filename(
                prefix=prefix, directory=directory, format=format
            ),
            format=format,
        )
    return tuple(tiles)


# ------------------ / Slice functions ------------------

# ------------------------ Utils ------------------------

def openpaypal():
    webbrowser.open(paypalme, new=1)

def opengithub():
    webbrowser.open(githubme, new=1)

def openitch():
    webbrowser.open(itchme, new=1)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir):
        shutil.rmtree(name_dir)

    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def image_to_uint(path, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def save_image(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def uint_to_tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    #return F.interpolate(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0), 256)

    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def tensor_to_uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def reunion_image(tiles):
    image = join(tiles)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def delete_tiles(tiles):
    for tile in tiles:
        os.remove(tile.filename)

def adapt_image_to_show(image_to_prepare):
    old_image     = cv2.imread(image_to_prepare)
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
                                   interpolation = cv2.INTER_NEAREST )
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_NEAREST)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, target_file_extension):
    result_path = (img.replace("_resized" + target_file_extension, "").replace(target_file_extension, "") 
                    + "_"  + AI_model  
                    + target_file_extension)
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def adapt_image_for_deeplearning(img, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'dml' in device:
        backend = torch.device('dml')

    img = image_to_uint(img, n_channels=3)
    img = uint_to_tensor4(img)
    img = img.to(backend, non_blocking = True)
    return img


# IMAGE

def resize_image(image_path, resize_factor):
    new_image_path = image_path.replace(target_file_extension, 
                                        "_resized" + target_file_extension)

    old_image = Image.open(image_path)

    if blur_pre_resize:
        old_image = old_image.filter(ImageFilter.GaussianBlur(radius = 0.75))

    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)

    max_height_or_width = int(max(new_height, new_width))

    resized_image = old_image.resize((new_width, new_height), 
                                        resample = Image.BILINEAR)

    if not pay:
        font = ImageFont.truetype(find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "arial.ttf"), 
                                    int(max_height_or_width/40))
        img_text = ImageDraw.Draw(resized_image)
        img_text.text((20, 20), 
                        "Upscaled with QualityScaler"
                        + "\nDownload it on"
                        + "\nhttps://github.com/Djdefrag/QualityScaler" , 
                        font = font, 
                        fill = (250, 250, 250))
                                    
    resized_image.save(new_image_path)

def resize_image_list(image_list, resize_factor):
    files_to_delete   = []
    downscaled_images = []
    how_much_images = len(image_list)

    index = 1
    for image in image_list:
        resized_image_path = image.replace(target_file_extension, 
                                            "_resized" + target_file_extension)
        
        resize_image(image, resize_factor)
        write_in_log_file("Resizing image " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)
        files_to_delete.append(resized_image_path)

        index += 1

    return downscaled_images, files_to_delete


#VIDEO

def resize_frame(image_path, new_width, new_height, max_height_or_width):
    new_image_path = image_path.replace(target_file_extension, 
                                        "_resized" + target_file_extension)

    old_image = Image.open(image_path)

    if blur_pre_resize:
        old_image = old_image.filter(ImageFilter.GaussianBlur(radius = 0.75))

    resized_image = old_image.resize((new_width, new_height), 
                                        resample = Image.BILINEAR)

    if not pay:
        font = ImageFont.truetype(find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "arial.ttf"), 
                                    int(max_height_or_width/40))
        img_text = ImageDraw.Draw(resized_image)
        img_text.text((20, 20), 
                        "Upscaled with QualityScaler"
                        + "\nDownload it on"
                        + "\nhttps://github.com/Djdefrag/QualityScaler" , 
                        font = font, 
                        fill = (250, 250, 250))
                                    
    resized_image.save(new_image_path)

def resize_frame_list(image_list, resize_factor):
    downscaled_images = []
    how_much_images = len(image_list)    

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)

    max_height_or_width = int(max(new_height, new_width))
    
    index = 1

    for image in image_list:
        resized_image_path = image.replace(target_file_extension, 
                                            "_resized" + target_file_extension)
        
        resize_frame(image, new_width, new_height, max_height_or_width)
        write_in_log_file("Resizing frame " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)

        index += 1

    return downscaled_images

def extract_frames_from_video(video_path):
    create_temp_dir(app_name + "_temp")
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extr_frame   = 0
    video_frames_list = []
    
    # extract video frames
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        extr_frame += 1
        result_path = app_name + "_temp" + os.sep + "frame_" + str(extr_frame) + target_file_extension
        
        cv2.imwrite(result_path, frame)
        video_frames_list.append(result_path)

        write_in_log_file("Extracted frames " + str(extr_frame) + "/" + str(total_frames))
    cap.release()

    # extract audio from video
    try:
        video = VideoFileClip.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except:
        pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model):
    # 1) get original video informations
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    cap.release()

    # 2) remove any file extension from original video path string
    for video_type in supported_video_list:
        video_name = video_name.replace(video_type, "")

    # 4) create upscaled video with upscaled frames
    try:
        temp_video_path = app_name + "_temp" + os.sep + "tempVideo.mp4"
        clip = ImageSequenceClip.ImageSequenceClip(video_frames_upscaled_list, 
                                                                    fps=frame_rate)
        clip.write_videofile(temp_video_path)

        # 5) add audio in upscaled video
        upscaled_video_path = (only_path + video_name + "_" +
                                AI_model + ".mp4")

        video = VideoFileClip.VideoFileClip(temp_video_path)
        audio = AudioFileClip.AudioFileClip(app_name + "_temp" + os.sep + "audio.mp3")

        new_audioclip = CompositeAudioClip([audio])
        video.audio = new_audioclip
    except:
        pass
    
    video.write_videofile(upscaled_video_path)




def convert_image_list(image_list, target_file_extension):
    converted_images = []
    for image in image_list:
        image = image.strip()
        converted_img = convert_image_and_save(image, target_file_extension)
        converted_images.append(converted_img)

    return converted_images

def convert_image_and_save(image_to_prepare, target_file_extension):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    new_image_path = image_to_prepare

    for file_type in supported_file_list:
        new_image_path = new_image_path.replace(file_type, target_file_extension)

    cv2.imwrite(new_image_path, cv2.imread(image_to_prepare))
    return new_image_path

def write_in_log_file(text_to_insert):
    log_file_name   = app_name + ".log"
    with open(log_file_name,'w') as log_file:
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = app_name + ".log"
    with open(log_file_name,'r') as log_file:
        step = log_file.readline()
    log_file.close()
    return step


# ----------------------- /Utils ------------------------


# ------------------ Neural Net related ------------------


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
        if self.sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(
                fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


# ------------------ /Neural Net related ------------------

# ----------------------- Core ------------------------


def file_drop_event(event):
    global image_path
    global multiple_files
    global multi_img_list
    global video_files
    global single_file
    global input_video_path

    supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number = count_files_dropped(event)
    all_supported, single_file, multiple_files, video_files, more_than_one_video = check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number)

    if video_files:
        # video section
        if not all_supported:
            info_string.set("Some files are not supported")
            return
        elif all_supported:
            if multiple_files:
                info_string.set("Only one video supported")
                return
            elif not multiple_files:
                if not more_than_one_video:
                    input_video_path = str(event.data).replace("{", "").replace("}", "")
                    
                    show_video_info_with_drag_drop(input_video_path)

                    # reset variable
                    image_path = "no file"
                    multi_img_list = []

                elif more_than_one_video:
                    info_string.set("Only one video supported")
                    return
    else:
        # image section
        if not all_supported:
            if multiple_files:
                info_string.set("Some files are not supported")
                return
            elif single_file:
                info_string.set("This file is not supported")
                return
        elif all_supported:
            if multiple_files:
                image_list_dropped = drop_event_to_image_list(event)

                show_list_images_in_GUI(image_list_dropped)
                
                multi_img_list = image_list_dropped

                place_clean_button()

                # reset variable
                image_path = "no file"
                video_frames_list = []

            elif single_file:
                image_list_dropped = drop_event_to_image_list(event)

                # convert images to target file extension
                show_single_image_inGUI = threading.Thread(target = show_image_in_GUI,
                                                         args=(str(image_list_dropped[0]), 1),
                                                         daemon=True)
                show_single_image_inGUI.start()

                multi_img_list = image_list_dropped

                # reset variable
                image_path = "no file"
                video_frames_list = []

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number):
    all_supported  = True
    single_file    = False
    multiple_files = False
    video_files    = False
    more_than_one_video = False

    if not_supported_file_dropped_number > 0:
        all_supported = False

    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_file = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_files = True

    if supported_video_dropped_number == 1:
        video_files = True
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_files = True
        more_than_one_video = True

    return all_supported, single_file, multiple_files, video_files, more_than_one_video

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

def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(1)
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
            time.sleep(1)
    except:
        place_upscale_button()

def drop_event_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list:
        image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() 

    return image_list

def upscale_image_and_save(img, AI_model, model, result_path, device, tiles_resolution):
    img_tmp          = cv2.imread(img)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        img_adapted  = adapt_image_for_deeplearning(img, device)
        with torch.no_grad():
            img_upscaled_tensor = model(img_adapted)
            img_upscaled = tensor_to_uint(img_upscaled_tensor)
        save_image(img_upscaled, result_path)
    else:
        num_tiles = round(num_tiles)
        if (num_tiles % 2) != 0:
            num_tiles += 1
        num_tiles = round(num_tiles * 3)

        tiles = img_cutter(img, num_tiles)

        if "x2" in AI_model: upscale_factor = 2
        elif "x4" in AI_model: upscale_factor = 4

        with torch.no_grad():
            for tile in tiles:
                tile_adapted  = adapt_image_for_deeplearning(tile.filename, device)
                tile_upscaled = tensor_to_uint(model(tile_adapted))
                save_image(tile_upscaled, tile.filename)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0] * upscale_factor, 
                                tile.coords[1] * upscale_factor)

        # 4) then reconstruct the image by tiles upscaled
        image_upscaled = reunion_image(tiles)

        # 5) remove tiles file
        delete_tiles(tiles)

        # 6) save reconstructed image
        cv2.imwrite(result_path, image_upscaled)

def optimize_torch():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

def prepare_AI_model(AI_model, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'dml' in device:
        backend = torch.device('dml')

    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")

    if "x2" in AI_model: upscale_factor = 2
    elif "x4" in AI_model: upscale_factor = 4

    model = RRDBNet(in_nc = 3, out_nc = 3, nf = 64, 
                    nb = 23, gc = 32, sf = upscale_factor)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False
    
    model = model.to(backend, non_blocking = True)

    return model

def process_upscale_multiple_images_qualityscaler(image_list, AI_model, resize_factor, device, tiles_resolution, target_file_extension):
    try:
        start = timer()

        optimize_torch()

        write_in_log_file('Resizing images')
        image_list = convert_image_list(image_list, target_file_extension)
        image_list, files_to_delete = resize_image_list(image_list, resize_factor)

        how_many_images = len(image_list)
        done_images     = 0

        write_in_log_file('Upscaling')
        for img in image_list:
            model = prepare_AI_model(AI_model, device)
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            upscale_image_and_save(img, AI_model, model, 
                                    result_path, device,    
                                    tiles_resolution)

            done_images += 1
            write_in_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                
        write_in_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete) 
    except:
        write_in_log_file('Error while upscaling')

def process_upscale_video_frames_qualityscaler(input_video_path, AI_model, resize_factor, device, tiles_resolution, target_file_extension):
        start = timer()
                        
        optimize_torch()

        write_in_log_file('Extracting video frames')
        image_list = extract_frames_from_video(input_video_path)
        
        write_in_log_file('Resizing video frames')
        image_list  = resize_frame_list(image_list, resize_factor)

        write_in_log_file('Upscaling')
        how_many_images = len(image_list)
        done_images     = 0
        
        video_frames_upscaled_list = []

        model = prepare_AI_model(AI_model, device)

        for img in image_list:
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            video_frames_upscaled_list.append(result_path)
            upscale_image_and_save(img, AI_model, model, 
                                    result_path, device,    
                                    tiles_resolution)
            done_images += 1
            write_in_log_file("Upscaled frame " + str(done_images) + "/" + str(how_many_images))

        write_in_log_file("Processing upscaled video")
        
        video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model)

        write_in_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")

        create_temp_dir(app_name + "_temp")

    

# ----------------------- /Core ------------------------

# ---------------------- GUI related ----------------------


def upscale_button_command():
    global image_path
    global multiple_files
    global process_upscale
    global thread_wait
    global resize_factor
    global video_frames_list
    global video_files
    global video_frames_upscaled_list
    global input_video_path
    global device

    try:
        resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_string.set("Resize input must be a numeric value")
        return

    if resize_factor > 0 and resize_factor <= 100:
        resize_factor = resize_factor/100
        pass    
    else:
        info_string.set("Resize must be in range 1 - 100")
        return

    info_string.set("...")

    if video_files:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_video_frames_qualityscaler,
                                                  args   = (input_video_path, 
                                                            AI_model, 
                                                            resize_factor, 
                                                            device,
                                                            tiles_resolution,
                                                            target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                       args   = (1, 2), 
                                       daemon = True)
        thread_wait.start()

    elif multiple_files:
        place_stop_button()
        
        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_qualityscaler,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             resize_factor, 
                                                             device,
                                                             tiles_resolution,
                                                             target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                        args   = (1, 2), daemon = True)
        thread_wait.start()

    elif single_file:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_qualityscaler,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             resize_factor, 
                                                             device,
                                                             tiles_resolution,
                                                             target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                       args   = (1, 2), daemon = True)
        thread_wait.start()

    elif "no file" in image_path:
        info_string.set("No file selected")
  
def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped upscaling") 

def clear_input_variables():
    global image_path
    global multi_img_list
    global video_frames_list
    global single_file
    global multiple_files
    global video_files

    # reset variable
    image_path        = "no file"
    multi_img_list    = []
    video_frames_list = []
    single_file       = False
    multiple_files    = False
    video_files       = False
    multi_img_list    = []
    video_frames_list = []

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = file_drop_event,
                          relief = "flat",
                          background = background_color,
                          foreground = text_color)
    drag_drop.place(x = left_bar_width + 50, y=0,
                    width = drag_drop_width, height = drag_drop_height)

def show_video_info_with_drag_drop(video_path):
    global image
    
    fist_frame = "temp.jpg"
    
    clear_app_background()

    # 1) get video informations
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
    
    # 2) get first frame of the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(fist_frame, frame)
        break
    cap.release()

    # 3) resize the frame to fit the UI
    image_to_show_resized = adapt_image_to_show(fist_frame)

    # 4) show the resized image in the UI
    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     image   = image,
                                     ondrop  = file_drop_event,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2 - 15,
                               width  = show_image_width,
                               height = show_image_height)

    # 5) remove the temp first frame
    os.remove(fist_frame)

    # 6) create string video description
    file_description = ( video_name + "\n" + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps")

    video_info_width = drag_drop_width * 0.8

    video_info_space = ttk.Label(root,
                                 text    = file_description,
                                 ondrop  = file_drop_event,
                                 font    = (default_font, round(11 * font_scale), "bold"),
                                 anchor  = "center",
                                 relief  = "flat",
                                 justify = "center",
                                 background = background_color,
                                 foreground = "#D3D3D3",
                                 wraplength = video_info_width * 0.95)
    video_info_space.place(x = 30 + left_bar_width + drag_drop_width/2 - video_info_width/2,
                           y = drag_drop_height - 85,
                           width  = video_info_width,
                           height = 65)

    # 7) show clear button
    place_clean_button()

def show_list_images_in_GUI(image_list_prepared):
    clear_app_background()

    final_string = "\n"
    counter_img = 0

    for elem in image_list_prepared:
        counter_img += 1
        if counter_img <= 8:
            # add first 8 files in list
            img     = cv2.imread(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) 
                            + ".  " 
                            + img_name 
                            + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    images_list = ttk.Label(root,
                            text    = final_string,
                            ondrop  = file_drop_event,
                            font    = (default_font, round(12 * font_scale)),
                            anchor  = "n",
                            relief  = "flat",
                            justify = "left",
                            background = background_color,
                            foreground = "#D3D3D3",
                            wraplength = list_width)

    images_list.place(x = 30 + left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2 -25,
                               width  = list_width,
                               height = list_height)

    images_counter = ttk.Entry(root, 
                                foreground = text_color,
                                ondrop  = file_drop_event,
                                font    = (default_font, round(12 * font_scale), "bold"), 
                                justify = 'center')

    images_counter.insert(0, str(len(image_list_prepared)) + ' images')

    images_counter.configure(state='disabled')

    images_counter.place(x = left_bar_width + drag_drop_width/2 + 125,
                        y = drag_drop_height/2 + 250,
                        width  = 250,
                        height = 42)

def show_image_in_GUI(image_to_show, _ ):
    global image

    image_to_show = image_to_show.replace('{', '').replace('}', '')

    # 1) resize image to fit the UI
    image_to_show_resized = adapt_image_to_show(image_to_show)

    # 2) clean the background
    clear_app_background()

    # 3) show the resized image in the UI
    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     text="",
                                     image   = image,
                                     ondrop  = file_drop_event,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2,
                               width  = show_image_width,
                               height = show_image_height)

    # 4) show the image file information in the UI
    path_as_list = image_to_show.split("/")
    img_name     = str(path_as_list[-1])
    img          = cv2.imread(image_to_show)
    width        = round(img.shape[1])
    height       = round(img.shape[0])

    single_image_path = (img_name + " | [" + str(width) + "x" + str(height) + "]")
    single_image_info = ttk.Label(root,
                                  font =(default_font, round(11 * font_scale), "bold"),
                                  text = single_image_path,
                                  relief  = "flat",
                                  justify = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    single_image_info.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 70,
                            width  = image_text_width,
                            height = 40)

    # 5) delete the resized temp image
    if "temp.png" in image_to_show_resized:
        os.remove(image_to_show_resized)

    # 6) show clear button
    place_clean_button()

def place_upscale_button():
    global play_icon
    play_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "upscale_icon.png"))
    
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft, foreground = text_color)

    Upscale_button = ttk.Button(root, 
                                text  = '  UPSCALE',
                                image = play_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Upscale_button.place(x      = 50 + left_bar_width/2 - 310/2,  
                         y      = left_bar_height - 110,
                         width  = 310,
                         height = 45)
    Upscale_button["command"] = lambda: upscale_button_command()

def place_stop_button():
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    
    global stop_icon
    stop_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "stop_icon.png"))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft)

    Stop_button = ttk.Button(root, 
                                text  = '  STOP UPSCALE ',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 50 + left_bar_width/2 - 310/2,  
                      y      = left_bar_height - 110,
                      width  = 310,
                      height = 45)

    Stop_button["command"] = lambda: stop_button_command()

def combobox_AI_selection(event):
    global AI_model

    selected = str(selected_AI.get())

    AI_model = selected

    Combo_box_AI.set('')
    Combo_box_AI.set(selected)

def combobox_backend_selection(event):
    global device

    selected = str(selected_backend.get())
    if 'gpu' in selected:
        device = "dml"
    elif 'cpu' in selected:
        device = "cpu"

    Combo_box_backend.set('')
    Combo_box_backend.set(selected)

def combobox_VRAM_selection(event):
    global tiles_resolution

    selected = str(selected_VRAM.get())

    if 'Minimal(2GB)' == selected:
        tiles_resolution = 200
    if 'Medium(4GB)' == selected:
        tiles_resolution = 400
    if 'Normal(6GB)' == selected:
        tiles_resolution = 600
    if 'High(8GB)' == selected:
        tiles_resolution = 800
    if 'Ultra(12GB)' == selected:
        tiles_resolution = 1000
    if 'Max(>16GB)' == selected:
        tiles_resolution = 1200

    Combo_box_VRAM.set('')
    Combo_box_VRAM.set(selected)




def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

    ft = tkFont.Font(family = default_font,
                        size   = round(12 * font_scale),
                        weight = "bold")

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
                            font    = ft,
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

def place_backend_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify',    'center')

    global Combo_box_backend
    Combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = ['GPU', 'CPU'],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_backend.place(x = 50 + left_bar_width/2 - 285/2, 
                            y = button_3_y, 
                            width  = 290, 
                            height = 42)
    Combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    Combo_box_backend.set('GPU')

def place_resize_factor_entrybox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    global Entry_box_resize_factor
    Entry_box_resize_factor = ttk.Entry(root, 
                                        textvariable = selected_resize_factor, 
                                        justify      = 'center',
                                        foreground   = text_color,
                                        takefocus    = False,
                                        font         = ft)
    Entry_box_resize_factor.place(x = 50 + left_bar_width/2 - 285/2, 
                                    y = button_2_y, 
                                    width  = 290 * 0.8, 
                                    height = 42)
    Entry_box_resize_factor.insert(0, '70')

    Label_percentage = ttk.Label(root,
                                 text       = "%",
                                 justify    = "center",
                                 font       = tkFont.Font(family = default_font,
                                            size   = round(13 * font_scale),
                                            weight = "bold"),
                                 foreground = text_color)
    Label_percentage.place(x = left_bar_width/2 + 160, 
                            y = button_2_y + 2, 
                            width  = 30, 
                            height = 42)
    
def place_AI_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    models_array = [ 'BSRGANx2', 'BSRGANx4', 'RealSR_JPEGx4' ]

    global Combo_box_AI
    Combo_box_AI = ttk.Combobox(root, 
                        textvariable = selected_AI, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = models_array,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = ft)
    Combo_box_AI.place(x = 50 + left_bar_width/2 - 285/2,  
                       y = button_1_y, 
                       width  = 290, 
                       height = 42)
    Combo_box_AI.bind('<<ComboboxSelected>>', combobox_AI_selection)
    Combo_box_AI.set(AI_model)

def place_VRAM_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    global Combo_box_VRAM
    Combo_box_VRAM = ttk.Combobox(root, 
                            textvariable = selected_VRAM, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = ['Minimal(2GB)', 'Medium(4GB)', 
                                            'Normal(6GB)', 'High(8GB)', 
                                            'Ultra(12GB)', 'Max(>16GB)' ],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_VRAM.place(x = 50 + left_bar_width/2 - 285/2, 
                         y = button_4_y, 
                         width  = 290, 
                         height = 42)
    Combo_box_VRAM.bind('<<ComboboxSelected>>', combobox_VRAM_selection)

    if tiles_resolution == 200:
        Combo_box_VRAM.set('Minimal(2GB)')
    if tiles_resolution == 400:
        Combo_box_VRAM.set('Medium(4GB)')
    if tiles_resolution == 600:
        Combo_box_VRAM.set('Normal(6GB)')
    if tiles_resolution == 800:
        Combo_box_VRAM.set('High(8GB)')
    if tiles_resolution == 1000:
        Combo_box_VRAM.set('Ultra(12GB)')
    if tiles_resolution == 1200:
        Combo_box_VRAM.set('Max(>16GB)')

def place_clean_button():
    global clear_icon
    clear_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "clear_icon.png"))

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = 50 + left_bar_width + drag_drop_width/2 - 175/2,
                       y = 25,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_background():
    Background = ttk.Label(root, background = background_color, relief = 'flat')
    Background.place(x = 0, 
                     y = 0, 
                     width  = window_width,
                     height = window_height)

def place_left_bar():
    Left_bar = ttk.Notebook(root)
    if windows_subversion >= 22000:  # Windows 11
        Left_bar.place(x = 50, 
                       y = 0, 
                       width  = left_bar_width,
                       height = 700)
    else:                            # Windows 10
        Left_bar.place(x = 50, 
                       y = 22, 
                       width  = left_bar_width,
                       height = left_bar_height * 0.92)

def place_app_title():
    Title = ttk.Label(root, 
                      font = (default_font, round(20 * font_scale), "bold"),
                      foreground = "#DA70D6", 
                      anchor     = 'w', 
                      text       = app_name)
    Title.place(x = 50 + 35,
                y = 70,
                width  = 300,
                height = 55)

    global logo_itch
    logo_itch = PhotoImage(file = find_by_relative_path( "Assets" 
                                                        + os.sep 
                                                        + "itch_logo.png"))

    version_button = ttk.Button(root,
                               image = logo_itch,
                               padding = '0 0 0 0',
                               text    = " " + version,
                               compound = 'left',
                               style    = 'Bold.TButton')
    version_button.place(x = (left_bar_width + 50) - (125 + 25),
                        y = 40,
                        width  = 125,
                        height = 35)
    version_button["command"] = lambda: openitch()


def place_github_button():
    global logo_git
    logo_git = PhotoImage(file = find_by_relative_path( "Assets" 
                                                        + os.sep 
                                                        + "github_logo.png"))

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    github_button = ttk.Button(root,
                               image = logo_git,
                               padding = '0 0 0 0',
                               text    = ' Github',
                               compound = 'left',
                               style    = 'Bold.TButton')
    github_button.place(x = (left_bar_width + 50) - (125 + 25),
                        y = 85,
                        width  = 125,
                        height = 35)
    github_button["command"] = lambda: opengithub()

def place_paypal_button():
    global logo_paypal
    logo_paypal = PhotoImage(file=find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "paypal_logo.png"))

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    paypal_button = ttk.Button(root,
                                image = logo_paypal,
                                padding = '0 0 0 0',
                                text = ' Paypal',
                                compound = 'left',
                                style    = 'Bold.TButton')
    paypal_button.place(x = (left_bar_width + 50) - (125 + 25),
                            y = 130,
                            width  = 125,
                            height = 35)
    paypal_button["command"] = lambda: openpaypal()

def place_AI_models_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = text_color, 
                                   justify    = 'left', 
                                   relief     = 'flat', 
                                   text       = " ◪  AI model ")
    IA_selection_title.place(x = left_bar_width/2 - 115,
                             y = button_1_y - 45,
                             width  = 200,
                             height = 40)

def place_resize_factor_title():
    Upscale_fact_selection_title = ttk.Label(root, 
                                            font = (default_font, round(12 * font_scale), "bold"), 
                                            foreground = text_color, 
                                            justify    = 'left', 
                                            relief     = 'flat', 
                                            text       = " ⤮  Resize before upscaling ")
    Upscale_fact_selection_title.place(x = left_bar_width/2 - 115,
                                        y = button_2_y - 45,
                                        width  = 300,
                                        height = 40)

def place_backend_title():
    Upscale_backend_selection_title = ttk.Label(root, 
                                                font = (default_font, round(12 * font_scale), "bold"), 
                                                foreground = text_color, 
                                                justify    = 'left', 
                                        	    relief     = 'flat', 
                                                text       = " ⍚  AI backend ")
    Upscale_backend_selection_title.place(x = left_bar_width/2 - 115,
                                          y = button_3_y - 45,
                                          width  = 200,
                                          height = 40)

def place_VRAM_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = text_color, 
                                   justify    = 'left', 
                                   relief     = 'flat', 
                                   text       = " ⋈  Vram/Ram ")
    IA_selection_title.place(x = left_bar_width/2 - 115,
                             y = button_4_y - 45,
                             width  = 200,
                             height = 40)

def place_message_box():
    message_label = ttk.Label(root,
                            font = (default_font, round(11 * font_scale), "bold"),
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 50 + left_bar_width/2 - (left_bar_width * 0.75)/2,
                        y = 600,
                        width  = left_bar_width * 0.75,
                        height = 30)

    info_string.set("...")


# ---------------------- /GUI related ----------------------

# ---------------------- /Functions ----------------------


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
        root.iconphoto(False, PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "logo.png")))

        if windows_subversion >= 22000: # Windows 11
            apply_windows_transparency_effect(root)
        apply_windows_dark_bar(root)

        place_background()               # Background
        place_left_bar()                 # Left bar background
        place_app_title()                # App title
        place_github_button()
        place_paypal_button()
        place_AI_models_title()          # AI models title
        place_AI_combobox()              # AI models widget
        place_resize_factor_title()      # Upscale factor title
        place_resize_factor_entrybox()   # Upscale factor widget
        place_backend_title()            # Backend title
        place_backend_combobox()         # Backend widget
        place_VRAM_title()               # VRAM title
        place_VRAM_combobox()            # VRAM widget
        place_message_box()              # Message box
        place_upscale_button()           # Upscale button
        place_drag_drop_widget()         # Drag&Drop widget
        
if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_AI = tk.StringVar()
    selected_resize_factor = tk.StringVar()
    selected_backend = tk.StringVar()
    selected_VRAM    = tk.StringVar()

    app = App(root)
    root.update()
    root.mainloop()
