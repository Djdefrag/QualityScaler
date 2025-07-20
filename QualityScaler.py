
# Standard library imports
import sys
from functools  import cache
from time       import sleep
from webbrowser import open as open_browser
from subprocess import run  as subprocess_run
from shutil     import rmtree as remove_directory
from timeit     import default_timer as timer

from typing    import Callable
from threading import Thread
from itertools import repeat
from multiprocessing import ( 
    Process, 
    Event          as multiprocessing_Event,
    Pool           as multiprocessing_Pool,
    Queue          as multiprocessing_Queue,
    Manager        as multiprocessing_Manager,
    freeze_support as multiprocessing_freeze_support
)

from json import (
    load  as json_load, 
    dumps as json_dumps
)

from os import (
    sep        as os_separator,
    devnull    as os_devnull,
    makedirs   as os_makedirs,
    listdir    as os_listdir,
    remove     as os_remove,
    fdopen     as os_fdopen,
    open       as os_open,
    O_WRONLY,
    O_CREAT
)

from os.path import (
    basename   as os_path_basename,
    dirname    as os_path_dirname,
    abspath    as os_path_abspath,
    join       as os_path_join,
    exists     as os_path_exists,
    splitext   as os_path_splitext,
    expanduser as os_path_expanduser
)

# Third-party library imports
from natsort import natsorted
from psutil  import virtual_memory as psutil_virtual_memory
from onnxruntime import InferenceSession as onnxruntime_InferenceSession
from onnxruntime import set_default_logger_severity as onnxruntime_set_default_logger_severity
onnxruntime_set_default_logger_severity(0)

from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_GRAY2RGB,
    COLOR_BGR2RGBA,
    COLOR_RGB2GRAY,
    IMREAD_UNCHANGED,
    INTER_AREA,
    INTER_CUBIC,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    addWeighted  as opencv_addWeighted,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from numpy import (
    ascontiguousarray as numpy_ascontiguousarray,
    frombuffer        as numpy_frombuffer,
    concatenate       as numpy_concatenate, 
    transpose         as numpy_transpose,
    full              as numpy_full, 
    expand_dims       as numpy_expand_dims,
    squeeze           as numpy_squeeze,
    clip              as numpy_clip,
    mean              as numpy_mean,
    repeat            as numpy_repeat,
    array_split       as numpy_array_split,
    zeros             as numpy_zeros, 
    max               as numpy_max, 
    ndarray           as numpy_ndarray,
    float32,
    uint8
)

# GUI imports
from tkinter import StringVar
from tkinter import DISABLED
from customtkinter import (
    CTk,
    CTkFrame,
    CTkButton,
    CTkEntry,
    CTkFont,
    CTkImage,
    CTkLabel,
    CTkOptionMenu,
    CTkScrollableFrame,
    CTkToplevel,
    filedialog,
    set_appearance_mode,
    set_default_color_theme
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)


app_name   = "QualityScaler"
version    = "4.4"
githubme   = "https://github.com/Djdefrag/QualityScaler/releases"
telegramme = "https://linktr.ee/j3ngystudio"

app_name_color          = "#DA70D6"
background_color        = "#000000"
widget_background_color = "#181818"
text_color              = "#B8B8B8"

VRAM_model_usage = {
    'RealESR_Gx4':     2.5,
    'RealESR_Animex4': 2.5,
    'BSRGANx4':        0.75,
    'RealESRGANx4':    0.75,
    'BSRGANx2':        0.8,
    'IRCNN_Mx1':       4,
    'IRCNN_Lx1':       4,
}

MENU_LIST_SEPARATOR         = [ "----" ]
SRVGGNetCompact_models_list = [ "RealESR_Gx4", "RealESR_Animex4" ]
BSRGAN_models_list          = [ "BSRGANx4", "BSRGANx2", "RealESRGANx4" ]
IRCNN_models_list           = [ "IRCNN_Mx1", "IRCNN_Lx1" ]

AI_models_list         = ( SRVGGNetCompact_models_list + MENU_LIST_SEPARATOR + BSRGAN_models_list + MENU_LIST_SEPARATOR + IRCNN_models_list )
AI_multithreading_list = [ "OFF", "2 threads", "4 threads", "6 threads", "8 threads"]
blending_list          = [ "OFF", "Low", "Medium", "High" ]
gpus_list              = [ "Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4" ]
keep_frames_list       = [ "OFF", "ON" ]
image_extension_list   = [ ".png", ".jpg", ".bmp", ".tiff" ]
video_extension_list   = [ ".mp4", ".mkv", ".avi", ".mov" ]
video_codec_list = [ 
    "x264",       "x265",       MENU_LIST_SEPARATOR[0],
    "h264_nvenc", "hevc_nvenc", MENU_LIST_SEPARATOR[0],
    "h264_amf",   "hevc_amf",   MENU_LIST_SEPARATOR[0],
    "h264_qsv",   "hevc_qsv",
    ]

OUTPUT_PATH_CODED    = "Same path as input files"
DOCUMENT_PATH        = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_{version}_UserPreference.json")
FFMPEG_EXE_PATH      = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH    = find_by_relative_path(f"Assets{os_separator}exiftool.exe")

FRAMES_TO_SAVE_BATCH = 16

COMPLETED_STATUS = "Completed"
ERROR_STATUS     = "Error"
STOP_STATUS      = "Stop"


offset_y_options = 0.0825
row1  = 0.125
row2  = row1 + offset_y_options
row3  = row2 + offset_y_options
row4  = row3 + offset_y_options
row5  = row4 + offset_y_options
row6  = row5 + offset_y_options
row7  = row6 + offset_y_options
row8  = row7 + offset_y_options
row9  = row8 + offset_y_options
row10 = row9 + offset_y_options

column_offset = 0.2
column_info1  = 0.625
column_info2  = 0.858
column_1      = 0.66
column_2      = column_1 + column_offset
column_1_5    = column_info1 + 0.08
column_1_4    = column_1_5 - 0.0127
column_3      = column_info2 + 0.08
column_2_9    = column_3 - 0.0127
column_3_5    = column_2 + 0.0355

little_textbox_width = 74
little_menu_width = 98



supported_file_extensions = [
    '.heic', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png',
    '.PNG', '.webp', '.WEBP', '.bmp', '.BMP', '.tif',
    '.tiff', '.TIF', '.TIFF', '.mp4', '.MP4', '.webm',
    '.WEBM', '.mkv', '.MKV', '.flv', '.FLV', '.gif',
    '.GIF', '.m4v', ',M4V', '.avi', '.AVI', '.mov',
    '.MOV', '.qt', '.3gp', '.mpg', '.mpeg', ".vob"
]

supported_video_extensions = [
    '.mp4', '.MP4', '.webm', '.WEBM', '.mkv', '.MKV',
    '.flv', '.FLV', '.gif', '.GIF', '.m4v', ',M4V',
    '.avi', '.AVI', '.mov', '.MOV', '.qt', '.3gp',
    '.mpg', '.mpeg', ".vob"
]




# AI -------------------

class AI_upscale:

    # CLASS INIT FUNCTIONS

    def __init__(
            self, 
            AI_model_name: str, 
            directml_gpu: str, 
            input_resize_factor: int,
            output_resize_factor: int,
            max_resolution: int
            ):
        
        # Passed variables
        self.AI_model_name        = AI_model_name
        self.directml_gpu         = directml_gpu
        self.input_resize_factor  = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution       = max_resolution

        # Calculated variables
        self.AI_model_path    = find_by_relative_path(f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        self.upscale_factor   = self._get_upscale_factor()
        self.inferenceSession = self._load_inferenceSession()

    def _get_upscale_factor(self) -> int:
        if   "x1" in self.AI_model_name: return 1
        elif "x2" in self.AI_model_name: return 2
        elif "x4" in self.AI_model_name: return 4

    def _load_inferenceSession(self) -> None:
        
        providers = ['DmlExecutionProvider']

        match self.directml_gpu:
            case 'Auto':  provider_options = [{"performance_preference": "high_performance"}]
            case 'GPU 1': provider_options = [{"device_id": "0"}]
            case 'GPU 2': provider_options = [{"device_id": "1"}]
            case 'GPU 3': provider_options = [{"device_id": "2"}]
            case 'GPU 4': provider_options = [{"device_id": "3"}]

        inference_session = onnxruntime_InferenceSession(
            path_or_bytes    = self.AI_model_path, 
            providers        = providers,
            provider_options = provider_options,
        )

        return inference_session



    # INTERNAL CLASS FUNCTIONS

    def get_image_mode(self, image: numpy_ndarray) -> str:
        shape = image.shape
        if len(shape) == 2:  # Grayscale: 2D array (rows, cols)
            return "Grayscale"
        elif len(shape) == 3 and shape[2] == 3:  # RGB: 3D array with 3 channels
            return "RGB"
        elif len(shape) == 3 and shape[2] == 4:  # RGBA: 3D array with 4 channels
            return "RGBA"

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width  = image.shape[1]

        return height, width 

    def calculate_target_resolution(self, image: numpy_ndarray) -> tuple:
        height, width = self.get_image_resolution(image)
        target_height = height * self.upscale_factor
        target_width  = width  * self.upscale_factor

        return target_height, target_width

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        
        old_height, old_width = self.get_image_resolution(image)

        new_width  = int(old_width * self.input_resize_factor)
        new_height = int(old_height * self.input_resize_factor)

        new_width  = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.input_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation = INTER_CUBIC)
        elif self.input_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation = INTER_AREA)
        else:
            return image

    def resize_with_output_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        
        old_height, old_width = self.get_image_resolution(image)

        new_width  = int(old_width * self.output_resize_factor)
        new_height = int(old_height * self.output_resize_factor)

        new_width  = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.output_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation = INTER_CUBIC)
        elif self.output_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation = INTER_AREA)
        else:
            return image



    # VIDEO CLASS FUNCTIONS

    def calculate_optimal_multithreads_number(self, video_frame_path: str, selected_AI_multithreading: int) -> int:
        resized_video_frame   = self.resize_with_input_factor(image_read(video_frame_path))
        height, width         = self.get_image_resolution(resized_video_frame)
        image_pixels          = height * width
        max_supported_pixels  = self.max_resolution * self.max_resolution
        frames_simultaneously = max_supported_pixels // image_pixels 
        print(f"Frames supported simultaneously by GPU: {frames_simultaneously}")

        threads_number = min(frames_simultaneously, selected_AI_multithreading)
        if threads_number <= 0: threads_number = 1
        print(f"Selected threads number: {threads_number}")

        return threads_number

    # TILLING FUNCTIONS

    def image_need_tilling(self, image: numpy_ndarray) -> bool:
        height, width = self.get_image_resolution(image)
        image_pixels  = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        if image_pixels > max_supported_pixels:
            return True
        else:
            return False

    def add_alpha_channel(self, image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full((image.shape[0], image.shape[1], 1), 255, dtype = uint8)
            image = numpy_concatenate((image, alpha), axis = 2)
        return image

    def calculate_tiles_number(self, image: numpy_ndarray) -> tuple:
        
        height, width = self.get_image_resolution(image)

        tiles_x = (width  + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution

        return tiles_x, tiles_y
    
    def split_image_into_tiles(self, image: numpy_ndarray, tiles_x: int, tiles_y: int) -> list[numpy_ndarray]:

        img_height, img_width = self.get_image_resolution(image)

        tile_width  = img_width // tiles_x
        tile_height = img_height // tiles_y

        tiles = []

        for y in range(tiles_y):
            y_start = y * tile_height
            y_end   = (y + 1) * tile_height

            for x in range(tiles_x):
                x_start = x * tile_width
                x_end   = (x + 1) * tile_width
                tile    = image[y_start:y_end, x_start:x_end]
                tiles.append(tile)

        return tiles

    def combine_tiles_into_image(self, image: numpy_ndarray, tiles: list[numpy_ndarray], t_height: int, t_width: int, num_tiles_x: int) -> numpy_ndarray:

        match self.get_image_mode(image):
            case "Grayscale": tiled_image = numpy_zeros((t_height, t_width, 3), dtype = uint8)
            case "RGB":       tiled_image = numpy_zeros((t_height, t_width, 3), dtype = uint8)
            case "RGBA":      tiled_image = numpy_zeros((t_height, t_width, 4), dtype = uint8)

        for tile_index in range(len(tiles)):
            actual_tile = tiles[tile_index]

            tile_height, tile_width = self.get_image_resolution(actual_tile)

            row     = tile_index // num_tiles_x
            col     = tile_index % num_tiles_x
            y_start = row * tile_height
            y_end   = y_start + tile_height
            x_start = col * tile_width
            x_end   = x_start + tile_width

            match self.get_image_mode(image):
                case "Grayscale": tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGB":       tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGBA":      tiled_image[y_start:y_end, x_start:x_end] = self.add_alpha_channel(actual_tile)

        return tiled_image



    # AI CLASS FUNCTIONS

    def normalize_image(self, image: numpy_ndarray) -> tuple:
        range = 255
        if numpy_max(image) > 256: range = 65535
        normalized_image = image / range

        return normalized_image, range
    
    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_transpose(image, (2, 0, 1))
        image = numpy_expand_dims(image, axis=0)

        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:

        # IO BINDING
        #io_binding = self.inferenceSession.io_binding()
        #io_binding.bind_cpu_input(self.inferenceSession.get_inputs()[0].name, image.astype(float16))
        #io_binding.bind_output(self.inferenceSession.get_outputs()[0].name)
        #self.inferenceSession.run_with_iobinding(io_binding)
        #onnx_output = io_binding.copy_outputs_to_cpu()[0]

        onnx_input  = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]

        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))

        return onnx_output

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:    
        match max_range:
            case 255:   return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)



    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        image        = image.astype(float32)
        image_mode   = self.get_image_mode(image)
        image, range = self.normalize_image(image)

        match image_mode:
            case "RGB":
                image = self.preprocess_image(image)
                onnx_output  = self.onnxruntime_inference(image)
                onnx_output  = self.postprocess_output(onnx_output)
                output_image = self.de_normalize_image(onnx_output, range)

                return output_image
            
            case "RGBA":
                alpha = image[:, :, 3]
                image = image[:, :, :3]
                image = opencv_cvtColor(image, COLOR_BGR2RGB)

                image = image.astype(float32)
                alpha = alpha.astype(float32)

                # Image
                image = self.preprocess_image(image)
                onnx_output_image = self.onnxruntime_inference(image)
                onnx_output_image = self.postprocess_output(onnx_output_image)
                onnx_output_image = opencv_cvtColor(onnx_output_image, COLOR_BGR2RGBA)

                # Alpha
                alpha = numpy_expand_dims(alpha, axis=-1)
                alpha = numpy_repeat(alpha, 3, axis=-1)
                alpha = self.preprocess_image(alpha)
                onnx_output_alpha = self.onnxruntime_inference(alpha)
                onnx_output_alpha = self.postprocess_output(onnx_output_alpha)
                onnx_output_alpha = opencv_cvtColor(onnx_output_alpha, COLOR_RGB2GRAY)

                # Fusion Image + Alpha
                onnx_output_image[:, :, 3] = onnx_output_alpha
                output_image = self.de_normalize_image(onnx_output_image, range)

                return output_image
            
            case "Grayscale":
                image = opencv_cvtColor(image, COLOR_GRAY2RGB)
                
                image = self.preprocess_image(image)
                onnx_output  = self.onnxruntime_inference(image)
                onnx_output  = self.postprocess_output(onnx_output)
                output_image = opencv_cvtColor(onnx_output, COLOR_RGB2GRAY)
                output_image = self.de_normalize_image(onnx_output, range)

                return output_image

    def AI_upscale_with_tilling(self, image: numpy_ndarray) -> numpy_ndarray:
        t_height, t_width = self.calculate_target_resolution(image)
        tiles_x, tiles_y  = self.calculate_tiles_number(image)
        tiles_list        = self.split_image_into_tiles(image, tiles_x, tiles_y)
        tiles_list        = [self.AI_upscale(tile) for tile in tiles_list]

        return self.combine_tiles_into_image(image, tiles_list, t_height, t_width, tiles_x)


    # PUBLIC FUNCTION

    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:
        resized_image = self.resize_with_input_factor(image)

        success = False
        while not success:
            try:
                if self.image_need_tilling(resized_image):
                    upscaled_image = self.AI_upscale_with_tilling(resized_image)
                else:
                    upscaled_image = self.AI_upscale(resized_image)
                
                success = True
            
            except Exception as e:
                print(f"error upscaling : {e}")
                sleep(0.25)
        
        return self.resize_with_output_factor(upscaled_image)




# GUI utils ---------------------------

class MessageBox(CTkToplevel):

    def __init__(
            self,
            messageType: str,
            title: str,
            subtitle: str,
            default_value: str,
            option_list: list,
            ) -> None:

        super().__init__()

        self._running: bool = False

        self._messageType = messageType
        self._title       = title        
        self._subtitle    = subtitle
        self._default_value = default_value
        self._option_list   = option_list
        self._ctkwidgets_index = 0

        self.title('')
        self.lift()                          # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()                       # make other windows not clickable

    def _ok_event(
            self, 
            event = None
            ) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(
            self
            ) -> None:
        self.grab_release()
        self.destroy()

    def createEmptyLabel(self) -> CTkLabel:
        return CTkLabel(
            master   = self,
            fg_color = "transparent",
            width    = 500,
            height   = 17,
            text     = ''
        )

    def placeInfoMessageTitleSubtitle(self) -> None:

        spacingLabel1 = self.createEmptyLabel()
        spacingLabel2 = self.createEmptyLabel()

        if self._messageType == "info":
            title_subtitle_text_color = "#3399FF"
        elif self._messageType == "error":
            title_subtitle_text_color = "#FF3131"

        titleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold22,
            text       = self._title
            )
        
        if self._default_value != None:
            defaultLabel = CTkLabel(
                master     = self,
                width      = 500,
                anchor     = 'w',
                justify    = "left",
                fg_color   = "transparent",
                text_color = "#3399FF",
                font       = bold17,
                text       = f"Default: {self._default_value}"
                )
        
        subtitleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold14,
            text       = self._subtitle
            )
        
        spacingLabel1.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        titleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        if self._default_value != None:
            self._ctkwidgets_index += 1
            defaultLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        subtitleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        spacingLabel2.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOptionsText(self) -> None:
        
        for option_text in self._option_list:
            optionLabel = CTkLabel(
                master        = self,
                width         = 600,
                height        = 45,
                anchor        = 'w',
                justify       = "left",
                text_color    = text_color,
                fg_color      = "#282828",
                bg_color      = "transparent",
                font          = bold13,
                text          = option_text,
                corner_radius = 10,
            )
            
            self._ctkwidgets_index += 1
            optionLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 4, sticky = "ew")

        spacingLabel3 = self.createEmptyLabel()

        self._ctkwidgets_index += 1
        spacingLabel3.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOkButton(
            self
            ) -> None:
        
        ok_button = CTkButton(
            master  = self,
            command = self._ok_event,
            text    = 'OK',
            width   = 125,
            font         = bold11,
            border_width = 1,
            fg_color     = "#282828",
            text_color   = "#E0E0E0",
            border_color = "#0096FF"
        )
        
        self._ctkwidgets_index += 1
        ok_button.grid(row = self._ctkwidgets_index, column = 1, columnspan = 1, padx = (10, 20), pady = (10, 20), sticky = "e")

    def _create_widgets(
            self
            ) -> None:

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.placeInfoMessageTitleSubtitle()
        self.placeInfoMessageOptionsText()
        self.placeInfoMessageOkButton()

class FileWidget(CTkScrollableFrame):

    def __init__(
            self, 
            master,
            selected_file_list,
            upscale_factor       = 1,
            input_resize_factor  = 0,
            output_resize_factor = 0,
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list            = selected_file_list
        self.upscale_factor       = upscale_factor
        self.input_resize_factor  = input_resize_factor
        self.output_resize_factor = output_resize_factor

        self.index_row = 1
        self.ui_components = []
        self._create_widgets()

    def _destroy_(self) -> None:
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def _create_widgets(self) -> None:
        self.add_clean_button()
        for file_path in self.file_list:
            file_name_label, file_info_label = self.add_file_information(file_path)
            self.ui_components.append(file_name_label)
            self.ui_components.append(file_info_label)

    def add_file_information(self, file_path) -> tuple:
        infos, icon = self.extract_file_info(file_path)

        # File name
        file_name_label = CTkLabel(
            self, 
            text       = os_path_basename(file_path),
            font       = bold14,
            text_color = text_color,
            compound   = "left", 
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )      
        file_name_label.grid(
            row    = self.index_row, 
            column = 0,
            pady   = (0, 2),
            padx   = (3, 3),
            sticky = "w"
        )

        # File infos and icon
        file_info_label = CTkLabel(
            self, 
            text       = infos,
            image      = icon, 
            font       = bold12,
            text_color = text_color,
            compound   = "left", 
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )      
        file_info_label.grid(
            row    = self.index_row + 1, 
            column = 0,
            pady   = (0, 15),
            padx   = (3, 3),
            sticky = "w"
        )

        self.index_row += 2

        return file_name_label, file_info_label

    def add_clean_button(self) -> None:

        button = CTkButton(
            master        = self, 
            command       = self._destroy_,
            text          = "CLEAN",
            image         = clear_icon,
            width         = 90, 
            height        = 28,
            font          = bold11,
            border_width  = 1,
            corner_radius = 1,
            fg_color      = "#282828",
            text_color    = "#E0E0E0",
            border_color  = "#0096FF"
        )
        
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        



    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 60

        if check_if_file_is_video(file_path):
            video_cap   = opencv_VideoCapture(file_path)
            _, frame    = video_cap.read()
            source_icon = opencv_cvtColor(frame, COLOR_BGR2RGB)
            video_cap.release()
        else:
            source_icon = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)

        ratio       = min(max_size / source_icon.shape[0], max_size / source_icon.shape[1])
        new_width   = int(source_icon.shape[1] * ratio)
        new_height  = int(source_icon.shape[0] * ratio)
        source_icon = opencv_resize(source_icon,(new_width, new_height))
        ctk_icon    = CTkImage(pillow_image_fromarray(source_icon, mode="RGB"), size = (new_width, new_height))

        return ctk_icon
        
    def extract_file_info(self, file_path) -> tuple:
        
        if check_if_file_is_video(file_path):
            cap          = opencv_VideoCapture(file_path)
            width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
            height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
            num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
            frame_rate   = cap.get(CAP_PROP_FPS)
            duration     = num_frames/frame_rate
            minutes      = int(duration/60)
            seconds      = duration % 60
            cap.release()

            file_icon  = self.extract_file_icon(file_path)
            file_infos = f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} \n"
            
            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0 :
                input_resized_height = int(height * (self.input_resize_factor/100))
                input_resized_width  = int(width * (self.input_resize_factor/100))

                upscaled_height = int(input_resized_height * self.upscale_factor)
                upscaled_width  = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(upscaled_height * (self.output_resize_factor/100))
                output_resized_width  = int(upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Video output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        else:
            height, width = get_image_resolution(image_read(file_path))
            file_icon     = self.extract_file_icon(file_path)

            file_infos = f"{width}x{height}\n"
            
            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0 :
                input_resized_height = int(height * (self.input_resize_factor/100))
                input_resized_width  = int(width * (self.input_resize_factor/100))

                upscaled_height = int(input_resized_height * self.upscale_factor)
                upscaled_width  = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(upscaled_height * (self.output_resize_factor/100))
                output_resized_width  = int(upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Image output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        return file_infos, file_icon


    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        self.index_row = 1
        for ui_component in self.ui_components: ui_component.grid_forget()
    
    def get_selected_file_list(self) -> list: 
        return self.file_list 

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_input_resize_factor(self, input_resize_factor) -> None:
        self.input_resize_factor = input_resize_factor

    def set_output_resize_factor(self, output_resize_factor) -> None:
        self.output_resize_factor = output_resize_factor
 


def get_values_for_file_widget() -> tuple:
    # Upscale factor
    upscale_factor = get_upscale_factor()

    # Input resolution %
    try:
        input_resize_factor = int(float(str(selected_input_resize_factor.get())))
    except:
        input_resize_factor = 0

    # Output resolution %
    try:
        output_resize_factor = int(float(str(selected_output_resize_factor.get())))
    except:
        output_resize_factor = 0

    return upscale_factor, input_resize_factor, output_resize_factor

def update_file_widget(a, b, c) -> None:
    try:
        global file_widget
        file_widget
    except:
        return
        
    upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

    file_widget.clean_file_list()
    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_resize_factor)
    file_widget.set_output_resize_factor(output_resize_factor)
    file_widget._create_widgets()

def create_option_background():
    return CTkFrame(
        master   = window,
        bg_color = background_color,
        fg_color = widget_background_color,
        height   = 46,
        corner_radius = 10
    )

def create_info_button(
        command: Callable, 
        text:    str, 
        width:   int = 200
        ) -> CTkFrame:
    
    frame = CTkFrame(master = window, fg_color = widget_background_color, height = 25)

    button = CTkButton(
        master        = frame,
        command       = command,
        font          = bold12,
        text          = "?",
        border_color  = "#0096FF",
        border_width  = 1,
        fg_color      = widget_background_color,
        hover_color   = background_color,
        width         = 23,
        height        = 15,
        corner_radius = 1
    )
    button.grid(row=0, column=0, padx=(0, 7), pady=2, sticky="w")

    label = CTkLabel(
        master     = frame,
        text       = text,
        width      = width,
        height     = 22,
        fg_color   = "transparent",
        bg_color   = widget_background_color,
        text_color = text_color,
        font       = bold13,
        anchor     = "w"
    )
    label.grid(row=0, column=1, sticky="w")

    frame.grid_propagate(False)
    frame.grid_columnconfigure(1, weight=1)

    return frame

def create_option_menu(
        command:       Callable, 
        values:        list,
        default_value: str,
        border_color:  str = "#404040", 
        border_width:  int = 1,
        width:         int = 159,
        height:        int = 26
    ) -> CTkFrame:

    total_width  = (width + 2 * border_width)
    total_height = (height + 2 * border_width)
    
    frame = CTkFrame(
        master        = window,
        fg_color      = border_color,
        width         = total_width,
        height        = total_height,
        border_width  = 0,
        corner_radius = 1,
    )
    
    option_menu = CTkOptionMenu(
        master             = frame, 
        command            = command,
        values             = values,
        width              = width,
        height             = height,
        corner_radius      = 0,
        dropdown_font      = bold12,
        font               = bold11,
        anchor             = "center",
        text_color         = text_color,
        fg_color           = background_color,
        button_color       = background_color,
        button_hover_color = background_color,
        dropdown_fg_color  = background_color
    )
    
    option_menu.place(x = (total_width - width) / 2, y = (total_height - height) / 2)
    option_menu.set(default_value)
    return frame

def create_text_box(
        textvariable: StringVar, 
        width:        int,
        height:       int = 26
    ) -> CTkEntry:
    
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        corner_radius = 1,
        width         = width,
        height        = height,
        font          = bold11,
        justify       = "center",
        text_color    = text_color,
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
    )

def create_text_box_output_path(
        textvariable: StringVar,
        height:       int = 26
    ) -> CTkEntry:
    
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        corner_radius = 1,
        width         = 250,
        height        = height,
        font          = bold11,
        justify       = "center",
        text_color    = text_color,
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
        state         = DISABLED
    )

def create_active_button(
        command: Callable,
        text: str,
        icon: CTkImage = None,
        width: int = 140,
        height: int = 30,
        border_color: str = "#0096FF"
        ) -> CTkButton:
    
    return CTkButton(
        master        = window, 
        command       = command,
        text          = text,
        image         = icon,
        width         = width,
        height        = height,
        font          = bold11,
        border_width  = 1,
        corner_radius = 1,
        fg_color      = "#282828",
        text_color    = "#E0E0E0",
        border_color  = border_color
    )




# File Utils functions ------------------------

def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir): remove_directory(name_dir)
    if not os_path_exists(name_dir): os_makedirs(name_dir, mode=0o777)

def image_read(file_path: str) -> numpy_ndarray: 
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_ascontiguousarray(numpy_frombuffer(file.read(), uint8)), IMREAD_UNCHANGED)

def image_write(file_path: str, file_data: numpy_ndarray, file_extension: str = ".jpg") -> None: 
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def copy_file_metadata(original_file_path: str, upscaled_file_path: str) -> None:
    print(f"[EXIFTOOL] exporting original file tags")
    exiftool_cmd = [
        EXIFTOOL_EXE_PATH, 
        '-fast', 
        '-TagsFromFile', 
        original_file_path, 
        '-overwrite_original', 
        '-all:all',
        '-unsafe',
        '-largetags', 
        upscaled_file_path
    ]
    
    try: 
        subprocess_run(exiftool_cmd, check = True, shell = "False")
    except:
        pass

def prepare_output_image_filename(
        image_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        input_resize_factor: int, 
        output_resize_factor: int,
        selected_image_extension: str,
        selected_blending_factor: float
        ) -> str:
        
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(image_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(image_path)
        file_path_no_extension, _ = os_path_splitext(file_name)
        output_path = f"{selected_output_path}{os_separator}{file_path_no_extension}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    # Selected image extension
    to_append += f"{selected_image_extension}"
        
    output_path += to_append

    return output_path

def prepare_output_video_frame_filename(
        frame_path: str, 
        selected_AI_model: str, 
        input_resize_factor: int, 
        output_resize_factor: int,
        selected_blending_factor: float
        ) -> str:
            
    file_path_no_extension, _ = os_path_splitext(frame_path)
    output_path = file_path_no_extension

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    # Selected image extension
    to_append += f".jpg"
        
    output_path += to_append

    return output_path

def prepare_output_video_filename(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        input_resize_factor: int, 
        output_resize_factor: int,
        selected_video_extension: str,
        selected_blending_factor: float
        ) -> str:

    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        file_path_no_extension, _ = os_path_splitext(file_name)
        output_path = f"{selected_output_path}{os_separator}{file_path_no_extension}"
    
    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    # Selected video extension
    to_append += f"{selected_video_extension}"
        
    output_path += to_append

    return output_path

def prepare_output_video_directory_name(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        input_resize_factor: int, 
        output_resize_factor: int,
        selected_blending_factor: float
        ) -> str:
    
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        file_path_no_extension, _ = os_path_splitext(file_name)
        output_path = f"{selected_output_path}{os_separator}{file_path_no_extension}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    output_path += to_append

    return output_path




# Image/video Utils functions ------------------------

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate
   
def get_image_resolution(image: numpy_ndarray) -> tuple:
    height = image.shape[0]
    width  = image.shape[1]

    return height, width 

def video_encoding(
        process_status_q: multiprocessing_Queue,
        video_path: str,
        video_output_path: str,
        upscaled_frame_paths: list[str], 
        selected_video_codec: str, 
        ) -> None:


    if   "x264" in selected_video_codec: codec = "libx264"
    elif "x265" in selected_video_codec: codec = "libx265"
    else: codec = selected_video_codec
    
    txt_path      = f"{os_path_splitext(video_output_path)[0]}.txt"
    no_audio_path = f"{os_path_splitext(video_output_path)[0]}_no_audio{os_path_splitext(video_output_path)[1]}"
    video_fps     = str(get_video_fps(video_path))

    # Cleaning files from previous encoding
    if os_path_exists(no_audio_path): os_remove(no_audio_path)
    if os_path_exists(txt_path):      os_remove(txt_path)

    # Create a file .txt with all upscaled video frames paths || this file is essential
    with os_fdopen(os_open(txt_path, O_WRONLY | O_CREAT, 0o777), 'w', encoding="utf-8") as txt:
        for frame_path in upscaled_frame_paths:
            txt.write(f"file '{frame_path}' \n")

    # Create the upscaled video without audio
    print(f"[FFMPEG] ENCODING ({codec})")
    try: 
        encoding_command = [
            FFMPEG_EXE_PATH,
            "-y",
            "-loglevel",    "error",
            "-f",           "concat",
            "-safe",        "0",
            "-r",           video_fps,
            "-i",           txt_path,
            "-c:v",         codec,
            "-vf",          "scale=in_range=full:out_range=limited,format=yuv420p",
            "-color_range", "tv",
            "-b:v",         "12000k",
            no_audio_path
        ]
        subprocess_run(encoding_command, check = True, shell = "False")
        if os_path_exists(txt_path): os_remove(txt_path)
    except:
        write_process_status(
            process_status_q, 
            f"{ERROR_STATUS}An error occurred during video encoding. \n Have you selected a codec compatible with your GPU? If the issue persists, try selecting 'x264'."
        )

    # Copy the audio from original video
    print("[FFMPEG] AUDIO PASSTHROUGH")
    audio_passthrough_command = [
        FFMPEG_EXE_PATH,
        "-y",
        "-loglevel", "error",
        "-i",        video_path,
        "-i",        no_audio_path,
        "-c:v",      "copy",
        "-map",      "1:v:0",
        "-map",      "0:a?",
        "-c:a",      "copy",
        video_output_path
    ]
    try: 
        subprocess_run(audio_passthrough_command, check = True, shell = "False")
        if os_path_exists(no_audio_path): os_remove(no_audio_path)
    except:
        pass
    
def check_video_upscaling_resume(
        target_directory: str, 
        selected_AI_model: str
        ) -> bool:
    
    if os_path_exists(target_directory):
        directory_files      = os_listdir(target_directory)
        upscaled_frames_path = [file for file in directory_files if selected_AI_model in file]

        if len(upscaled_frames_path) > 1:
            return True
        else:
            return False
    else:
        return False

def get_video_frames_for_upscaling_resume(
        target_directory: str,
        selected_AI_model: str,
        ) -> list[str]:
    
    # Only file names
    directory_files      = os_listdir(target_directory)
    original_frames_path = [file for file in directory_files if file.endswith('.jpg')]
    original_frames_path = [file for file in original_frames_path if selected_AI_model not in file]

    # Adding the complete path to file
    original_frames_path = natsorted([os_path_join(target_directory, file) for file in original_frames_path])

    return original_frames_path

def calculate_time_to_complete_video(
        time_for_frame: float,
        remaining_frames: int,
        ) -> str:
    
    remaining_time = time_for_frame * remaining_frames

    hours_left   = remaining_time // 3600
    minutes_left = (remaining_time % 3600) // 60
    seconds_left = round((remaining_time % 3600) % 60)

    time_left = ""

    if int(hours_left) > 0: 
        time_left = f"{int(hours_left):02d}h"
    
    if int(minutes_left) > 0: 
        time_left = f"{time_left}{int(minutes_left):02d}m"

    if seconds_left > 0: 
        time_left = f"{time_left}{seconds_left:02d}s"

    return time_left        

def blend_images_and_save(
        target_path: str,
        starting_image: numpy_ndarray,
        upscaled_image: numpy_ndarray,
        starting_image_importance: float,
        file_extension: str = ".jpg"
        ) -> None:
    
    def add_alpha_channel(image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full((image.shape[0], image.shape[1], 1), 255, dtype = uint8)
            image = numpy_concatenate((image, alpha), axis = 2)
        return image
    
    def get_image_mode(image: numpy_ndarray) -> str:
        shape = image.shape
        if len(shape) == 2:                     return "Grayscale"
        elif len(shape) == 3 and shape[2] == 3: return "RGB"
        elif len(shape) == 3 and shape[2] == 4: return "RGBA"

    upscaled_image_importance       = 1 - starting_image_importance
    starting_height, starting_width = get_image_resolution(starting_image)
    target_height, target_width     = get_image_resolution(upscaled_image)

    starting_resolution = starting_height + starting_width
    target_resolution   = target_height + target_width

    if starting_resolution > target_resolution:
        starting_image = opencv_resize(starting_image,(target_width, target_height), INTER_AREA)
    else:
        starting_image = opencv_resize(starting_image,(target_width, target_height))

    try: 
        if get_image_mode(starting_image) == "RGBA":
            starting_image = add_alpha_channel(starting_image)
            upscaled_image = add_alpha_channel(upscaled_image)

        interpolated_image = opencv_addWeighted(starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, 0)
        image_write(target_path, interpolated_image, file_extension)
    
    except:
        image_write(target_path, upscaled_image, file_extension)




# Core functions ------------------------

def check_upscale_steps() -> None:
    sleep(1)

    while True:
        actual_step = process_status_q.get()
        print(f"[{app_name}] check_upscale_steps - {actual_step}")

        if actual_step == COMPLETED_STATUS:
            info_message.set(f"All files completed! :)")
            stop_upscale_process()
            place_upscale_button()
            break

        elif actual_step == STOP_STATUS:
            info_message.set(f"Upscaling stopped")
            stop_upscale_process()
            place_upscale_button()
            break

        elif ERROR_STATUS in actual_step:
            info_message.set(f"Error while upscaling :(")
            error_to_show = actual_step.replace(ERROR_STATUS, "")
            show_error_message(error_to_show.strip())
            place_upscale_button()
            break

        else:
            info_message.set(actual_step)

        sleep(1)
        
def write_process_status(
        process_status_q: multiprocessing_Queue, 
        step: str
        ) -> None:
    
    while not process_status_q.empty(): process_status_q.get()
    process_status_q.put(f"{step}")

def stop_upscale_process() -> None:
    global process_upscale_orchestrator

    print(f"[{app_name}] stop_upscale_process - setting upscale process stop event")
    event_stop_upscale_process.set()
    print(f"[{app_name}] stop_upscale_process - upscale process stop event setted")

    sleep(1)

    try:
        process_upscale_orchestrator
    except:
        pass
    else:
        print(f"[{app_name}] stop_upscale_process - waiting for upscale orchestrator to terminate")
        process_upscale_orchestrator.kill()
        print(f"[{app_name}] stop_upscale_process - upscale orchestrator terminated")

def stop_button_command() -> None:
    stop_upscale_process()
    write_process_status(process_status_q, f"{STOP_STATUS}") 

# ORCHESTRATOR

def upscale_button_command() -> None: 
    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_keep_frames
    global selected_AI_multithreading
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor

    global process_upscale_orchestrator
    
    if user_input_checks():
        info_message.set("Loading")

        print("=" * 50)
        print("> Starting upscale:")
        print(f"  Files to upscale: {len(selected_file_list)}")
        print(f"  Output path: {(selected_output_path.get())}")
        print(f"  Selected AI model: {selected_AI_model}")
        print(f"  Selected GPU: {selected_gpu}")
        print(f"  AI multithreading: {selected_AI_multithreading}")
        print(f"  Blending factor: {selected_blending_factor}")
        print(f"  Selected image output extension: {selected_image_extension}")
        print(f"  Selected video output extension: {selected_video_extension}")
        print(f"  Selected video output codec: {selected_video_codec}")
        print(f"  Tiles resolution for selected GPU VRAM: {tiles_resolution}x{tiles_resolution}px")
        print(f"  Input resize factor: {int(input_resize_factor * 100)}%")
        print(f"  Output resize factor: {int(output_resize_factor * 100)}%")
        print(f"  Save frames: {selected_keep_frames}")
        print("=" * 50)

        place_stop_button()

        event_stop_upscale_process.clear()
        while not process_status_q.empty():        process_status_q.get_nowait()
        while not video_frames_and_info_q.empty(): video_frames_and_info_q.get_nowait()

        process_upscale_orchestrator = Process(
            target = upscale_orchestrator,
            args = (
                process_status_q,
                video_frames_and_info_q,
                event_stop_upscale_process,
                selected_file_list, 
                selected_output_path.get(),
                selected_AI_model, 
                selected_AI_multithreading,
                input_resize_factor, 
                output_resize_factor,
                selected_gpu,
                tiles_resolution, 
                selected_blending_factor,
                selected_keep_frames,
                selected_image_extension,
                selected_video_extension,
                selected_video_codec,
            )
        )
        process_upscale_orchestrator.start()

        Thread(target = check_upscale_steps).start()

def upscale_orchestrator(
        process_status_q:           multiprocessing_Queue,
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event,

        selected_file_list: list,
        selected_output_path: str,
        selected_AI_model: str,
        selected_AI_multithreading: int,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_gpu: str,
        tiles_resolution: int,
        selected_blending_factor: float,
        selected_keep_frames: bool,
        selected_image_extension: str,
        selected_video_extension: str,
        selected_video_codec: str,
        ) -> None:

    try:
        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(
                    process_status_q            = process_status_q,
                    video_frames_and_info_q     = video_frames_and_info_q,
                    event_stop_upscale_process  = event_stop_upscale_process,
                    video_path                  = file_path, 
                    file_number                 = file_number,
                    selected_output_path        = selected_output_path, 
                    selected_AI_model           = selected_AI_model, 
                    selected_blending_factor    = selected_blending_factor,
                    selected_AI_multithreading  = selected_AI_multithreading,
                    selected_gpu                = selected_gpu,
                    input_resize_factor         = input_resize_factor,
                    output_resize_factor        = output_resize_factor,
                    tiles_resolution            = tiles_resolution,
                    selected_video_extension    = selected_video_extension,
                    selected_video_codec        = selected_video_codec,
                    selected_keep_frames        = selected_keep_frames,
                )
            else:
                upscale_image(
                    process_status_q         = process_status_q,
                    image_path               = file_path, 
                    file_number              = file_number,
                    selected_output_path     = selected_output_path,
                    AI_instance              = AI_upscale(selected_AI_model, selected_gpu, input_resize_factor, output_resize_factor, tiles_resolution) ,
                    selected_AI_model        = selected_AI_model,
                    selected_image_extension = selected_image_extension, 
                    input_resize_factor      = input_resize_factor, 
                    output_resize_factor     = output_resize_factor,
                    selected_blending_factor = selected_blending_factor
                )

        write_process_status(process_status_q, f"{COMPLETED_STATUS}")

    except Exception as exception:
        error_message = str(exception)
        write_process_status(process_status_q, f"{ERROR_STATUS} {error_message}")
 
# IMAGES

def upscale_image(
        process_status_q: multiprocessing_Queue,
        image_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_instance: AI_upscale,
        selected_AI_model: str,
        selected_image_extension: str,
        input_resize_factor: int, 
        output_resize_factor: int,
        selected_blending_factor: float
        ) -> None:
    
    starting_image = image_read(image_path)
    upscaled_image_path = prepare_output_image_filename(image_path, selected_output_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_image_extension, selected_blending_factor)

    write_process_status(process_status_q, f"{file_number}. Upscaling image")
    upscaled_image = AI_instance.AI_orchestration(starting_image)

    if selected_blending_factor > 0:
        blend_images_and_save(
            upscaled_image_path, 
            starting_image, 
            upscaled_image, 
            selected_blending_factor, 
            selected_image_extension
            )
    else:
        image_write(upscaled_image_path, upscaled_image, selected_image_extension)

    copy_file_metadata(image_path, upscaled_image_path)

# VIDEOS

# Function executed as process

def manage_extracted_frames_save_on_disk(
        video_frames_and_info_q:          multiprocessing_Queue,
        event_stop_upscale_process:       multiprocessing_Event,
        event_stop_extracted_save_thread: multiprocessing_Event,
        ):

    while True:
        if event_stop_upscale_process.is_set():
            print(f"[Extracted save thread] terminating by upscale stop event")
            break

        if event_stop_extracted_save_thread.is_set() and video_frames_and_info_q.empty():
            print(f"[Extracted save thread] terminating correctly")
            break
        
        while not video_frames_and_info_q.empty():
            sleep(0.01)
            
            # Get extracted frame and path from queue
            queue_item           = video_frames_and_info_q.get_nowait()
            extracted_frame_path = queue_item["extracted_frame_path"]
            extracted_frame      = queue_item["extracted_frame"]

            # Save extracted frame on disk
            image_write(
                file_path = extracted_frame_path, 
                file_data = extracted_frame, 
                file_extension = ".jpg"
            )

def upscale_video_frames_async(
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event,
        threads_number:             int,
        selected_AI_model:          str,
        selected_gpu:               str,
        input_resize_factor:        int,
        output_resize_factor:       int,
        tiles_resolution:           int,
        frame_chunk:                list[tuple[str, str]],
    ) -> None:

    AI_instance = AI_upscale(selected_AI_model, selected_gpu, input_resize_factor, output_resize_factor, tiles_resolution)

    for input_path, output_path in frame_chunk:

        start_timer = timer()

        if event_stop_upscale_process.is_set():
            print("[Upscale process] Terminating early due to stop event")
            break
        
        # Upscale frame
        starting_frame  = image_read(input_path)
        upscaled_frame  = AI_instance.AI_orchestration(starting_frame)
        
        # Calculate processing time
        end_timer       = timer()
        processing_time = (end_timer - start_timer)/threads_number

        # Add things in queue
        video_frames_and_info_q.put(
            {
                "starting_frame":      starting_frame,
                "upscaled_frame":      upscaled_frame,
                "upscaled_frame_path": output_path,
                "processing_time":     processing_time
            }
        )

    if event_stop_upscale_process.is_set():
        print("[Upscale process] Terminated")
    else:
        print("[Upscale process] Upscale process finished the job")

# -------------------------------------

def upscale_video(
        process_status_q:           multiprocessing_Queue,
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event,
        video_path:                 str, 
        file_number:                int,
        selected_output_path:       str,
        selected_AI_model:          str,
        selected_blending_factor:   float,
        selected_AI_multithreading: int,
        selected_gpu:               str,
        input_resize_factor:        int,
        output_resize_factor:       int,
        tiles_resolution:           int, 
        selected_video_extension:   str,
        selected_video_codec:       str,
        selected_keep_frames:       bool,
        ) -> None:
    

    # Internal functions

    def update_video_extraction_process_status(
            process_status_q:                 multiprocessing_Queue, 
            file_number:                      int,
            total_frames_counter:             int,
            already_extracted_frames_counter: int,
            average_extraction_time:          float
        ) -> None:

        remaining_frames = total_frames_counter - already_extracted_frames_counter 
        remaining_time   = calculate_time_to_complete_video(average_extraction_time, remaining_frames)
        if remaining_time != "":
            percent_complete = (already_extracted_frames_counter / total_frames_counter) * 100 
            write_process_status(process_status_q, f"{file_number}. Extracting frames {percent_complete:.2f}% ({remaining_time})")

    def extract_video_frames(
            process_status_q:           multiprocessing_Queue,
            video_frames_and_info_q:    multiprocessing_Queue,
            event_stop_upscale_process: multiprocessing_Event,
            file_number:                int,
            target_directory:           str,
            video_path:                 str, 
        ) -> list[str]:

        event_stop_extracted_save_thread = multiprocessing_Event()
        save_extracted_frames_process = Process(
            target = manage_extracted_frames_save_on_disk,
            args = (
                video_frames_and_info_q, 
                event_stop_upscale_process,
                event_stop_extracted_save_thread,
                ),
        )
        save_extracted_frames_process.start()

        create_dir(target_directory)

        # Video frame extraction
        video_capture           = opencv_VideoCapture(video_path)
        total_frames_counter    = int(video_capture.get(CAP_PROP_FRAME_COUNT))
        video_frames_paths_list = []
        extraction_times_list   = []

        for frame_number in range(total_frames_counter):
            start_timer = timer()

            # Extract frame
            success, extracted_frame = video_capture.read()
            if not success: break

            # Calculate frame path
            extracted_frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}.jpg"            

            # Put extracted frame and path in queue
            video_frames_and_info_q.put(
                {
                    "extracted_frame_path": extracted_frame_path,
                    "extracted_frame":      extracted_frame,
                }
            )

            # Add frame path in list to return
            video_frames_paths_list.append(extracted_frame_path)

            # Calculate processing time
            end_timer       = timer()
            extraction_time = end_timer - start_timer
            extraction_times_list.append(extraction_time)

            # Update process status if necessary
            if frame_number % FRAMES_TO_SAVE_BATCH == 0:
                average_extraction_time = numpy_mean(extraction_times_list)
                if len(extraction_times_list) >= 100: extraction_times_list = []
                update_video_extraction_process_status(
                    process_status_q                 = process_status_q, 
                    file_number                      = file_number,
                    total_frames_counter             = total_frames_counter,
                    already_extracted_frames_counter = frame_number,
                    average_extraction_time          = average_extraction_time
                )

        video_capture.release()

        event_stop_extracted_save_thread.set()
        save_extracted_frames_process.join()

        return video_frames_paths_list

    def update_video_upscale_process_status(
        process_status_q:        multiprocessing_Queue, 
        file_number:             int,
        upscaled_frame_paths:    list[str],
        average_processing_time: float
        ) -> None:
    
        # Remaining frames
        total_frames_counter            = len(upscaled_frame_paths)
        frames_already_upscaled_counter = len([path for path in upscaled_frame_paths if os_path_exists(path)])
        frames_to_upscale_counter       = len([path for path in upscaled_frame_paths if not os_path_exists(path)])

        remaining_frames = frames_to_upscale_counter
        remaining_time   = calculate_time_to_complete_video(average_processing_time, remaining_frames)
        if remaining_time != "":
            percent_complete = (frames_already_upscaled_counter / total_frames_counter) * 100 
            write_process_status(process_status_q, f"{file_number}. Upscaling video {percent_complete:.2f}% ({remaining_time})")

    def manage_upscaled_frames_save_on_disk(
            process_status_q:                multiprocessing_Queue,
            video_frames_and_info_q:         multiprocessing_Queue,
            event_stop_upscale_process:      multiprocessing_Event,
            event_stop_upscaled_save_thread: multiprocessing_Event,
            file_number:                     int,
            upscaled_frame_paths:            list[str],
            selected_blending_factor:        float,
            ) -> None:

        saved_frames_count    = 0
        processing_times_list = []
        
        while True:
            if event_stop_upscale_process.is_set():
                print(f"[Upscaled save thread] terminating by upscale stop event")
                break

            if event_stop_upscaled_save_thread.is_set() and video_frames_and_info_q.empty():
                print(f"[Upscaled save thread] terminating correctly")
                break

            while not video_frames_and_info_q.empty():
                sleep(0.01)

                # Get upscale infos from queue
                item = video_frames_and_info_q.get_nowait()
                starting_frame      = item["starting_frame"]
                upscaled_frame      = item["upscaled_frame"]
                upscaled_frame_path = item["upscaled_frame_path"]
                processing_time     = item["processing_time"]

                # Save image on disk
                if selected_blending_factor > 0:
                    blend_images_and_save(upscaled_frame_path, starting_frame, upscaled_frame, selected_blending_factor)
                else:
                    image_write(upscaled_frame_path, upscaled_frame)

                # Update process status if necessary
                saved_frames_count += 1
                processing_times_list.append(processing_time)
                if saved_frames_count % FRAMES_TO_SAVE_BATCH == 0: 
                    if processing_times_list:
                        average_processing_time = numpy_mean(processing_times_list)
                        if len(processing_times_list) >= 100: processing_times_list = []
                        update_video_upscale_process_status(process_status_q, file_number, upscaled_frame_paths, average_processing_time)

    def upscale_video_frames(
            process_status_q:           multiprocessing_Queue,
            video_frames_and_info_q:    multiprocessing_Queue,
            event_stop_upscale_process: multiprocessing_Event,

            file_number:              int, 
            selected_AI_model:        str,
            selected_gpu:             str,
            input_resize_factor:      int,
            output_resize_factor:     int,
            tiles_resolution:         int,
            extracted_frames_paths:   list[str],
            upscaled_frame_paths:     list[str],
            threads_number:           int,
            selected_blending_factor: float,
        ) -> None:

        event_stop_upscaled_save_thread = multiprocessing_Event()
        Thread(
            target = manage_upscaled_frames_save_on_disk,
            args = (
                process_status_q, 
                video_frames_and_info_q, 
                event_stop_upscale_process,
                event_stop_upscaled_save_thread,
                file_number, 
                upscaled_frame_paths, 
                selected_blending_factor, 
            ),
        ).start()

        frame_pairs_to_process = [
            (input_path, output_path)
            for input_path, output_path in zip(
                extracted_frames_paths,
                upscaled_frame_paths
                ) if not os_path_exists(output_path)
        ]

        if not frame_pairs_to_process:
            print("[Upscale] Nessun frame da elaborare, tutti già upscalati.")
            event_stop_upscaled_save_thread.set()
            return

        frame_chunks = numpy_array_split(frame_pairs_to_process, threads_number)
        frame_chunks = [list(chunk) for chunk in frame_chunks]

        write_process_status(process_status_q, f"{file_number}. Upscaling video ({threads_number} threads)")

        with multiprocessing_Pool(threads_number) as pool:
            pool.starmap(
                upscale_video_frames_async,
                zip(
                    repeat(video_frames_and_info_q),
                    repeat(event_stop_upscale_process),
                    repeat(threads_number),
                    repeat(selected_AI_model),
                    repeat(selected_gpu),
                    repeat(input_resize_factor),
                    repeat(output_resize_factor),
                    repeat(tiles_resolution),
                    frame_chunks,
                )
            )

        write_process_status(process_status_q, f"{file_number}. Finalizing upscaling")
        event_stop_upscaled_save_thread.set()
        sleep(5)


    # Main function

    # 1.Preparation
    target_directory  = prepare_output_video_directory_name(video_path, selected_output_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_blending_factor)
    video_output_path = prepare_output_video_filename(video_path, selected_output_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_video_extension, selected_blending_factor)
    

    # 2. Resume upscaling OR extract video frames
    video_upscale_continue = check_video_upscaling_resume(target_directory, selected_AI_model)
    if video_upscale_continue:
        write_process_status(process_status_q, f"{file_number}. Resume video upscaling")
        extracted_frames_paths = get_video_frames_for_upscaling_resume(target_directory, selected_AI_model)
    else:
        write_process_status(process_status_q, f"{file_number}. Extracting frames")
        extracted_frames_paths = extract_video_frames(
            process_status_q           = process_status_q,
            video_frames_and_info_q    = video_frames_and_info_q,
            event_stop_upscale_process = event_stop_upscale_process,
            file_number                = file_number, 
            target_directory           = target_directory, 
            video_path                 = video_path
        )

    upscaled_frame_paths = [prepare_output_video_frame_filename(frame_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_blending_factor) for frame_path in extracted_frames_paths]


    # 3. Check if video need tiles OR video multithreading upscale
    AI_instance    = AI_upscale(selected_AI_model, selected_gpu, input_resize_factor, output_resize_factor, tiles_resolution) 
    threads_number = AI_instance.calculate_optimal_multithreads_number(extracted_frames_paths[0], selected_AI_multithreading)
    AI_instance    = None


    # 4. Upscaling video frames
    write_process_status(process_status_q, f"{file_number}. Upscaling video") 
    upscale_video_frames(
        process_status_q           = process_status_q, 
        video_frames_and_info_q    = video_frames_and_info_q,
        event_stop_upscale_process = event_stop_upscale_process,
        file_number                = file_number, 
        selected_AI_model          = selected_AI_model, 
        selected_gpu               = selected_gpu, 
        input_resize_factor        = input_resize_factor, 
        output_resize_factor       = output_resize_factor, 
        tiles_resolution           = tiles_resolution, 
        extracted_frames_paths     = extracted_frames_paths, 
        upscaled_frame_paths       = upscaled_frame_paths, 
        threads_number             = threads_number, 
        selected_blending_factor   = selected_blending_factor,
        )


    # 6. Video encoding
    write_process_status(process_status_q, f"{file_number}. Encoding upscaled video")
    video_encoding(process_status_q, video_path, video_output_path, upscaled_frame_paths, selected_video_codec)
    copy_file_metadata(video_path, video_output_path)


    # 7. Delete frames folder
    if selected_keep_frames == False: 
        if os_path_exists(target_directory): remove_directory(target_directory)



# GUI utils function ---------------------------

def check_if_file_is_video(file: str) -> bool:
    return any(video_extension in file for video_extension in supported_video_extensions)

def user_input_checks() -> bool:
    global selected_file_list
    global selected_AI_model
    global selected_image_extension
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor

    # Selected files 
    try: selected_file_list = file_widget.get_selected_file_list()
    except:
        info_message.set("Please select a file")
        return False

    if len(selected_file_list) <= 0:
        info_message.set("Please select a file")
        return False


    # AI model
    if selected_AI_model == MENU_LIST_SEPARATOR[0]:
        info_message.set("Please select the AI model")
        return False


    # Input resize factor 
    try: input_resize_factor = int(float(str(selected_input_resize_factor.get())))
    except:
        info_message.set("Input resolution % must be a number")
        return False

    if input_resize_factor > 0: input_resize_factor = input_resize_factor/100
    else:
        info_message.set("Input resolution % must be a value > 0")
        return False


    # Output resize factor 
    try: output_resize_factor = int(float(str(selected_output_resize_factor.get())))
    except:
        info_message.set("Output resolution % must be a number")
        return False

    if output_resize_factor > 0: output_resize_factor = output_resize_factor/100
    else:
        info_message.set("Output resolution % must be a value > 0")
        return False

    
    # VRAM limiter
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_message.set("GPU VRAM value must be a number")
        return False

    if tiles_resolution > 0: 
        vram_multiplier = VRAM_model_usage.get(selected_AI_model)

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))
        tiles_resolution = int(selected_vram * 100)
    else:
        info_message.set("GPU VRAM value must be a value > 0")
        return False

    return True

def show_error_message(exception: str) -> None:
    messageBox_title    = "Upscale error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text     = f"\n {str(exception)} \n"

    MessageBox(
        messageType   = "error",
        title         = messageBox_title,
        subtitle      = messageBox_subtitle,
        default_value = None,
        option_list   = [messageBox_text]
    )

def get_upscale_factor() -> int:
    global selected_AI_model
    if MENU_LIST_SEPARATOR[0] in selected_AI_model: upscale_factor = 0
    elif 'x1' in selected_AI_model: upscale_factor = 1
    elif 'x2' in selected_AI_model: upscale_factor = 2
    elif 'x4' in selected_AI_model: upscale_factor = 4

    return upscale_factor

def open_files_action():

    def check_supported_selected_files(uploaded_file_list: list) -> list:
        return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:

        upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

        global file_widget
        file_widget = FileWidget(
            master               = window, 
            selected_file_list   = supported_files_list,
            upscale_factor       = upscale_factor,
            input_resize_factor  = input_resize_factor,
            output_resize_factor = output_resize_factor,
            fg_color             = background_color, 
            bg_color             = background_color
        )
        file_widget.place(relx = 0.0, rely = 0.0, relwidth = 0.5, relheight = 1.0)
        info_message.set("Ready")
    else: 
        info_message.set("Not supported files :(")

def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(OUTPUT_PATH_CODED)
    else:
        selected_output_path.set(asked_selected_output_path)




# GUI select from menus functions ---------------------------

def select_AI_from_menu(selected_option: str) -> None:
    global selected_AI_model    
    selected_AI_model = selected_option
    update_file_widget(1, 2, 3)

def select_AI_multithreading_from_menu(selected_option: str) -> None:
    global selected_AI_multithreading
    if selected_option == "OFF": 
        selected_AI_multithreading = 1
    else: 
        selected_AI_multithreading = int(selected_option.split()[0])

def select_blending_from_menu(selected_option: str) -> None:
    global selected_blending_factor

    match selected_option:
        case "OFF": selected_blending_factor = 0
        case "Low":      selected_blending_factor = 0.3
        case "Medium":   selected_blending_factor = 0.5
        case "High":     selected_blending_factor = 0.7

def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu    
    selected_gpu = selected_option

def select_save_frame_from_menu(selected_option: str):
    global selected_keep_frames
    if   selected_option == "ON":  selected_keep_frames = True
    elif selected_option == "OFF": selected_keep_frames = False

def select_image_extension_from_menu(selected_option: str) -> None:
    global selected_image_extension   
    selected_image_extension = selected_option

def select_video_extension_from_menu(selected_option: str) -> None:
    global selected_video_extension   
    selected_video_extension = selected_option

def select_video_codec_from_menu(selected_option: str) -> None:
    global selected_video_codec
    selected_video_codec = selected_option




# GUI place functions ---------------------------

def place_github_button():
    
    def opengithub() -> None: open_browser(githubme, new=1)

    git_button = CTkButton(
        master        = window,
        command       = opengithub,
        image         = logo_git,
        width         = 32,
        height        = 32,
        border_width  = 1,
        fg_color      = "transparent",
        text_color    = text_color,
        border_color  = "#404040",
        anchor        = "center",
        text          = "", 
        font          = bold11,
        corner_radius = 1
    )
    
    git_button.place(relx = column_2 + 0.1, rely = 0.04, anchor = "center")

def place_telegram_button():

    def opentelegram() -> None: open_browser(telegramme, new=1)

    telegram_button = CTkButton(
        master        = window,
        command       = opentelegram,
        image         = logo_telegram,
        width         = 32,
        height        = 32,
        border_width  = 1,
        fg_color      = "transparent",
        text_color    = text_color,
        border_color  = "#404040",
        anchor        = "center",
        text          = "", 
        font          = bold11,
        corner_radius = 1
    )

    telegram_button.place(relx = column_2 + 0.055, rely = 0.04, anchor = "center")
 
def place_loadFile_section():
    background = CTkFrame(master = window, fg_color = background_color, corner_radius = 1)

    text_drop = (" SUPPORTED FILES \n\n "
               + "IMAGES • jpg png tif bmp webp heic \n " 
               + "VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp ")

    input_file_text = CTkLabel(
        master     = window, 
        text       = text_drop,
        fg_color   = background_color,
        bg_color   = background_color,
        text_color = text_color,
        width      = 300,
        height     = 150,
        font       = bold13,
        anchor     = "center"
    )
    
    input_file_button = CTkButton(
        master       = window,
        command      = open_files_action, 
        text         = "SELECT FILES",
        width        = 140,
        height       = 30,
        font         = bold12,
        border_width  = 1,
        corner_radius = 1,
        fg_color      = "#282828",
        text_color    = "#E0E0E0",
        border_color  = "#0096FF"
    )
    
    background.place(relx = 0.0, rely = 0.0, relwidth = 0.5, relheight = 1.0)
    input_file_text.place(relx = 0.25, rely = 0.4,  anchor = "center")
    input_file_button.place(relx = 0.25, rely = 0.5, anchor = "center")

def place_app_name():
    background = CTkFrame(master = window, fg_color = background_color, corner_radius = 1)
    app_name_label = CTkLabel(
        master     = window, 
        text       = app_name + " " + version,
        fg_color   = background_color, 
        text_color = app_name_color,
        font       = bold20,
        anchor     = "w"
    )
    background.place(relx = 0.5, rely = 0.0, relwidth = 0.5, relheight = 1.0)
    app_name_label.place(relx = column_1 - 0.05, rely = 0.04, anchor = "center")

def place_AI_menu():

    def open_info_AI_model():
        option_list = [
            "\n IRCNN_Mx1 | IRCNN_Lx1 \n"
            "\n • Simple and lightweight AI models\n"
            " • Year: 2017\n"
            " • Function: Denoising\n",

            "\n RealESR_Gx4 | RealESR_Animex4 \n"
            "\n • Fast and lightweight AI models\n"
            " • Year: 2022\n"
            " • Function: Upscaling\n",

            "\n BSRGANx2 | BSRGANx4 | RealESRGANx4 \n"
            "\n • Complex and heavy AI models\n"
            " • Year: 2020\n"
            " • Function: High-quality upscaling\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI model",
            subtitle      = "This widget allows to choose between different AI models for upscaling",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row1
    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")
    
    info_button = create_info_button(open_info_AI_model, "AI model")
    option_menu = create_option_menu(select_AI_from_menu, AI_models_list, default_AI_model)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_AI_blending_menu():

    def open_info_AI_blending():
        option_list = [
            " Blending combines the upscaled image produced by AI with the original image",

            " \n BLENDING OPTIONS\n" +
            "  • [OFF] No blending is applied\n" +
            "  • [Low] The result favors the upscaled image, with a slight touch of the original\n" +
            "  • [Medium] A balanced blend of the original and upscaled images\n" +
            "  • [High] The result favors the original image, with subtle enhancements from the upscaled version\n",

            " \n NOTES\n" +
            "  • Can enhance the quality of the final result\n" +
            "  • Especially effective when using the tiling/merging function (useful for low VRAM)\n" +
            "  • Particularly helpful at low input resolution percentages (<50%)\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI blending", 
            subtitle      = "This widget allows you to choose the blending between the upscaled and original image/frame",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row2

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")
    
    info_button = create_info_button(open_info_AI_blending, "AI blending")
    option_menu = create_option_menu(select_blending_from_menu, blending_list, default_blending)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_AI_multithreading_menu():

    def open_info_AI_multithreading():
        option_list = [
            " This option can enhance video upscaling performance, especially on powerful GPUs.",

            " \n AI MULTITHREADING OPTIONS\n"
            + "  • OFF - Processes one frame at a time.\n"
            + "  • 2 threads - Processes two frames simultaneously.\n"
            + "  • 4 threads - Processes four frames simultaneously.\n"
            + "  • 6 threads - Processes six frames simultaneously.\n"
            + "  • 8 threads - Processes eight frames simultaneously.\n",

            " \n NOTES\n"
            + "  • Higher thread counts increase CPU, GPU, and RAM usage.\n"
            + "  • The GPU may be heavily stressed, potentially reaching high temperatures.\n"
            + "  • Monitor your system's temperature to prevent overheating.\n"
            + "  • If the chosen thread count exceeds GPU capacity, the app automatically selects an optimal value.\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "AI multithreading (EXPERIMENTAL)", 
            subtitle      = "This widget allows to choose how many video frames are upscaled simultaneously",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row3
    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    info_button = create_info_button(open_info_AI_multithreading, "AI multithreading")
    option_menu = create_option_menu(select_AI_multithreading_from_menu, AI_multithreading_list, default_AI_multithreading)

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3_5,   rely = widget_row, anchor = "center")

def place_input_output_resolution_textboxs():

    def open_info_input_resolution():
        option_list = [
            " A high value (>50%) will create high quality photos/videos but will be slower",
            " While a low value (<50%) will create good quality photos/videos but will much faster",

            " \n For example, for a 1080p (1920x1080) image/video\n" + 
            " • Input scale 25% => input to AI 270p (480x270)\n" +
            " • Input scale 50% => input to AI 540p (960x540)\n" + 
            " • Input scale 75% => input to AI 810p (1440x810)\n" + 
            " • Input scale 100% => input to AI 1080p (1920x1080) \n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Input resolution %",
            subtitle      = "This widget allows to choose the resolution input to the AI",
            default_value = None,
            option_list   = option_list
        )

    def open_info_output_resolution():
        option_list = [
            " 100% keeps the exact resolution produced by the AI upscaling",
            " A lower value (<100%) will downscale the AI result to a smaller resolution, saving space and processing time",
            " A higher value (>100%) will further upscale the AI output, increasing size but not adding real details",

            "\n For example, if the AI generates a 4K (3840x2160) image/video\n" +
            " • Output scale 50%  => final output 1920x1080 (downscaled)\n" +
            " • Output scale 100% => final output 3840x2160 (AI native)\n" +
            " • Output scale 200% => final output 7680x4320 (8K, interpolated)\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Output resolution %",
            subtitle      = "This widget allows to choose upscaled files resolution",
            default_value = None,
            option_list   = option_list
        )


    widget_row = row4

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Input scale %
    info_button = create_info_button(open_info_input_resolution, "Input scale %")
    option_menu = create_text_box(selected_input_resize_factor, width = little_textbox_width) 

    info_button.place(relx = column_info1, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_5,   rely = widget_row, anchor = "center")

    # Output scale %
    info_button = create_info_button(open_info_output_resolution, "Output scale %")
    option_menu = create_text_box(selected_output_resize_factor, width = little_textbox_width)  

    info_button.place(relx = column_info2, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3,     rely = widget_row, anchor = "center")

def place_gpu_gpuVRAM_menus():

    def open_info_gpu():
        option_list = [
            "\n It is possible to select up to 4 GPUs for AI processing\n" +
            "  • Auto (the app will select the most powerful GPU)\n" + 
            "  • GPU 1 (GPU 0 in Task manager)\n" + 
            "  • GPU 2 (GPU 1 in Task manager)\n" + 
            "  • GPU 3 (GPU 2 in Task manager)\n" + 
            "  • GPU 4 (GPU 3 in Task manager)\n",

            "\n NOTES\n" +
            "  • Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be\n" +
            "  • For optimal performance, it is essential to regularly update your GPUs drivers\n" +
            "  • Selecting a GPU not present in the PC will cause the app to use the CPU for AI processing\n"
        ]

        MessageBox(
            messageType   = "info",
            title         = "GPU",
            subtitle      = "This widget allows to select the GPU for AI upscale",
            default_value = None,
            option_list   = option_list
        )

    def open_info_vram_limiter():
        option_list = [
            " Make sure to enter the correct value based on the selected GPU's VRAM",
            " Setting a value higher than the available VRAM may cause upscale failure",
            " For integrated GPUs (Intel HD series • Vega 3, 5, 7), select 2 GB to avoid issues",
        ]

        MessageBox(
            messageType   = "info",
            title         = "GPU VRAM (GB)",
            subtitle      = "This widget allows to set a limit on the GPU VRAM memory usage",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row5

    background  = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # GPU
    info_button = create_info_button(open_info_gpu, "GPU")
    option_menu = create_option_menu(select_gpu_from_menu, gpus_list, default_gpu, width = little_menu_width) 

    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row,  anchor = "center")

    # GPU VRAM
    info_button = create_info_button(open_info_vram_limiter, "GPU VRAM (GB)")
    option_menu = create_text_box(selected_VRAM_limiter, width = little_textbox_width)  

    info_button.place(relx = column_info2, rely = widget_row, anchor = "center")
    option_menu.place(relx = column_3,     rely = widget_row, anchor = "center")

def place_image_video_output_menus():

    def open_info_image_output():
        option_list = [
            " \n PNG\n"
            " • Very good quality\n"
            " • Slow and heavy file\n"
            " • Supports transparent images\n"
            " • Lossless compression (no quality loss)\n"
            " • Ideal for graphics, web images, and screenshots\n",

            " \n JPG\n"
            " • Good quality\n"
            " • Fast and lightweight file\n"
            " • Lossy compression (some quality loss)\n"
            " • Ideal for photos and web images\n"
            " • Does not support transparency\n",

            " \n BMP\n"
            " • Highest quality\n"
            " • Slow and heavy file\n"
            " • Uncompressed format (large file size)\n"
            " • Ideal for raw images and high-detail graphics\n"
            " • Does not support transparency\n",

            " \n TIFF\n"
            " • Highest quality\n"
            " • Very slow and heavy file\n"
            " • Supports both lossless and lossy compression\n"
            " • Often used in professional photography and printing\n"
            " • Supports multiple layers and transparency\n",
        ]


        MessageBox(
            messageType   = "info",
            title         = "Image output",
            subtitle      = "This widget allows to choose the extension of upscaled images",
            default_value = None,
            option_list   = option_list
        )

    def open_info_video_extension():
        option_list = [
            " \n MP4\n"
            " • Most widely supported format\n"
            " • Good quality with efficient compression\n"
            " • Fast and lightweight file\n"
            " • Ideal for streaming and general use\n",

            " \n MKV\n"
            " • High-quality format with multiple audio and subtitle tracks support\n"
            " • Larger file size compared to MP4\n"
            " • Supports almost any codec\n"
            " • Ideal for high-quality videos and archiving\n",

            " \n AVI\n"
            " • Older format with high compatibility\n"
            " • Larger file size due to less efficient compression\n"
            " • Supports multiple codecs but lacks modern features\n"
            " • Ideal for older devices and raw video storage\n",

            " \n MOV\n"
            " • High-quality format developed by Apple\n"
            " • Large file size due to less compression\n"
            " • Best suited for editing and high-quality playback\n"
            " • Compatible mainly with macOS and iOS devices\n",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Video output",
            subtitle      = "This widget allows to choose the extension of the upscaled video",
            default_value = None,
            option_list   = option_list
        )

    widget_row = row6

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Image output
    info_button = create_info_button(open_info_image_output, "Image output")
    option_menu = create_option_menu(select_image_extension_from_menu, image_extension_list, default_image_extension, width = little_menu_width)
    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row, anchor = "center")

    # Video output
    info_button = create_info_button(open_info_video_extension, "Video output")
    option_menu = create_option_menu(select_video_extension_from_menu, video_extension_list, default_video_extension, width = little_menu_width)
    info_button.place(relx = column_info2,      rely = widget_row, anchor = "center")
    option_menu.place(relx = column_2_9, rely = widget_row, anchor = "center")

def place_video_codec_keep_frames_menus():

    def open_info_video_codec():
        option_list = [
            " \n SOFTWARE ENCODING (CPU)\n"
            " • x264 | H.264 software encoding\n"
            " • x265 | HEVC (H.265) software encoding\n",

            " \n NVIDIA GPU ENCODING (NVENC - Optimized for NVIDIA GPU)\n"
            " • h264_nvenc | H.264 hardware encoding\n"
            " • hevc_nvenc | HEVC (H.265) hardware encoding\n",

            " \n AMD GPU ENCODING (AMF - Optimized for AMD GPU)\n"
            " • h264_amf | H.264 hardware encoding\n"
            " • hevc_amf | HEVC (H.265) hardware encoding\n",

            " \n INTEL GPU ENCODING (QSV - Optimized for Intel GPU)\n"
            " • h264_qsv | H.264 hardware encoding\n"
            " • hevc_qsv | HEVC (H.265) hardware encoding\n"
        ]


        MessageBox(
            messageType   = "info",
            title         = "Video codec",
            subtitle      = "This widget allows to choose video codec for upscaled video",
            default_value = None,
            option_list   = option_list
        )

    def open_info_keep_frames():
        option_list = [
            "\n ON \n" + 
            " The app does NOT delete the video frames after creating the upscaled video \n",

            "\n OFF \n" + 
            " The app deletes the video frames after creating the upscaled video \n"
        ]

        MessageBox(
            messageType   = "info",
            title         = "Keep video frames",
            subtitle      = "This widget allows to choose to keep video frames",
            default_value = None,
            option_list   = option_list
        )


    widget_row = row7

    background = create_option_background()
    background.place(relx = 0.75, rely = widget_row, relwidth = 0.48, anchor = "center")

    # Video codec
    info_button = create_info_button(open_info_video_codec, "Video codec")
    option_menu = create_option_menu(select_video_codec_from_menu, video_codec_list, default_video_codec, width = little_menu_width)
    info_button.place(relx = column_info1,        rely = widget_row, anchor = "center")
    option_menu.place(relx = column_1_4, rely = widget_row, anchor = "center")

    # Keep frames
    info_button = create_info_button(open_info_keep_frames, "Keep frames")
    option_menu = create_option_menu(select_save_frame_from_menu, keep_frames_list, default_keep_frames, width = little_menu_width)
    info_button.place(relx = column_info2,      rely = widget_row, anchor = "center")
    option_menu.place(relx = column_2_9, rely = widget_row, anchor = "center")

def place_output_path_textbox():

    def open_info_output_path():
        option_list = [
              "\n The default path is defined by the input files."
            + "\n For example: selecting a file from the Download folder,"
            + "\n the app will save upscaled files in the Download folder \n",

            " Otherwise it is possible to select the desired path using the SELECT button",
        ]

        MessageBox(
            messageType   = "info",
            title         = "Output path",
            subtitle      = "This widget allows to choose upscaled files path",
            default_value = None,
            option_list   = option_list
        )

    background    = create_option_background()
    info_button   = create_info_button(open_info_output_path, "Output path")
    option_menu   = create_text_box_output_path(selected_output_path) 
    active_button = create_active_button(command = open_output_path_action, text = "SELECT", width = 60, height = 25)
  
    background.place(   relx = 0.75,                 rely = row10, relwidth = 0.48, anchor = "center")
    info_button.place(  relx = column_info1,         rely = row10 - 0.003,           anchor = "center")
    active_button.place(relx = column_info1 + 0.052, rely = row10,                   anchor = "center")
    option_menu.place(  relx = column_2 - 0.008,     rely = row10,                   anchor = "center")

def place_message_label():
    message_label = CTkLabel(
        master        = window, 
        textvariable  = info_message,
        height        = 24,
        width         = 247,
        font          = bold11,
        fg_color      = "#ffbf00",
        text_color    = "#000000",
        anchor        = "center",
        corner_radius = 1
    )
    message_label.place(relx = 0.85, rely = 0.9495, anchor = "center")

def place_stop_button(): 
    stop_button = create_active_button(
        command      = stop_button_command,
        text         = "STOP",
        icon         = stop_icon,
        width        = 140,
        height       = 30,
        border_color = "#EC1D1D"
    )
    stop_button.place(relx = 0.75 - 0.1, rely = 0.95, anchor = "center")

def place_upscale_button(): 
    upscale_button = create_active_button(
        command = upscale_button_command,
        text    = "UPSCALE",
        icon    = upscale_icon,
        width   = 140,
        height  = 30
    )
    upscale_button.place(relx = 0.75 - 0.1, rely = 0.95, anchor = "center")
   



# Main functions ---------------------------

def on_app_close() -> None:
    window.grab_release()
    window.destroy()

    global selected_AI_model
    global selected_AI_multithreading
    global selected_gpu
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor

    AI_model_to_save        = f"{selected_AI_model}"
    gpu_to_save             = selected_gpu
    image_extension_to_save = selected_image_extension
    video_extension_to_save = selected_video_extension
    video_codec_to_save     = selected_video_codec
    blending_to_save        = {0: "OFF", 0.3: "Low", 0.5: "Medium", 0.7: "High"}.get(selected_blending_factor)

    if selected_keep_frames == True:
        keep_frames_to_save = "ON"
    else:
        keep_frames_to_save = "OFF"

    if selected_AI_multithreading == 1: 
        AI_multithreading_to_save = "OFF"
    else: 
        AI_multithreading_to_save = f"{selected_AI_multithreading} threads"

    user_preference = {
        "default_AI_model":             AI_model_to_save,
        "default_AI_multithreading":    AI_multithreading_to_save,
        "default_gpu":                  gpu_to_save,
        "default_keep_frames":          keep_frames_to_save,
        "default_image_extension":      image_extension_to_save,
        "default_video_extension":      video_extension_to_save,
        "default_video_codec":          video_codec_to_save,
        "default_blending":             blending_to_save,
        "default_output_path":          selected_output_path.get(),
        "default_input_resize_factor":  str(selected_input_resize_factor.get()),
        "default_output_resize_factor": str(selected_output_resize_factor.get()),
        "default_VRAM_limiter":         str(selected_VRAM_limiter.get()),
    }
    user_preference_json = json_dumps(user_preference)
    with open(USER_PREFERENCE_PATH, "w") as preference_file:
        preference_file.write(user_preference_json)

    stop_upscale_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        window.geometry("1000x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_loadFile_section()

        place_app_name()
        place_output_path_textbox()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_AI_multithreading_menu()
        place_AI_blending_menu()
        place_input_output_resolution_textboxs()

        place_gpu_gpuVRAM_menus()
        place_video_codec_keep_frames_menus()

        place_image_video_output_menus()

        place_message_label()
        place_upscale_button()

if __name__ == "__main__":
    if os_path_exists(FFMPEG_EXE_PATH): 
        print(f"[{app_name}] ffmpeg.exe found")
    else:
        print(f"[{app_name}] ffmpeg.exe not found, please install ffmpeg.exe following the guide")

    if os_path_exists(USER_PREFERENCE_PATH):
        print(f"[{app_name}] Preference file exist")
        with open(USER_PREFERENCE_PATH, "r") as json_file:
            json_data = json_load(json_file)
            default_AI_model             = json_data.get("default_AI_model",             AI_models_list[0])
            default_AI_multithreading    = json_data.get("default_AI_multithreading",    AI_multithreading_list[0])
            default_gpu                  = json_data.get("default_gpu",                  gpus_list[0])
            default_keep_frames          = json_data.get("default_keep_frames",          keep_frames_list[1])
            default_image_extension      = json_data.get("default_image_extension",      image_extension_list[0])
            default_video_extension      = json_data.get("default_video_extension",      video_extension_list[0])
            default_video_codec          = json_data.get("default_video_codec",          video_codec_list[0])
            default_blending             = json_data.get("default_blending",             blending_list[1])
            default_output_path          = json_data.get("default_output_path",          OUTPUT_PATH_CODED)
            default_input_resize_factor  = json_data.get("default_input_resize_factor",  str(50))
            default_output_resize_factor = json_data.get("default_output_resize_factor", str(100))
            default_VRAM_limiter         = json_data.get("default_VRAM_limiter",         str(4))

    else:
        print(f"[{app_name}] Preference file does not exist, using default coded value")
        default_AI_model             = AI_models_list[0]
        default_AI_multithreading    = AI_multithreading_list[0]
        default_gpu                  = gpus_list[0]
        default_keep_frames          = keep_frames_list[1]
        default_image_extension      = image_extension_list[0]
        default_video_extension      = video_extension_list[0]
        default_video_codec          = video_codec_list[0]
        default_blending             = blending_list[1]
        default_output_path          = OUTPUT_PATH_CODED
        default_input_resize_factor  = str(50)
        default_output_resize_factor = str(100)
        default_VRAM_limiter         = str(4)

    multiprocessing_freeze_support()
    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    # Leggi la RAM totale in GB
    ram_gb = round(psutil_virtual_memory().total / (1024**3))

    if ram_gb <= 8:    queue_maxsize = 100
    elif ram_gb <= 16: queue_maxsize = 200
    elif ram_gb <= 32: queue_maxsize = 400
    else:              queue_maxsize = 800
    
    # Multiprocessing utilities
    multiprocessing_manager    = multiprocessing_Manager()
    process_status_q           = multiprocessing_manager.Queue(maxsize=1)
    video_frames_and_info_q    = multiprocessing_manager.Queue(maxsize=queue_maxsize)
    event_stop_upscale_process = multiprocessing_manager.Event()

    window = CTk() 
    info_message                  = StringVar()
    selected_output_path          = StringVar()
    selected_input_resize_factor  = StringVar()
    selected_output_resize_factor = StringVar()
    selected_VRAM_limiter         = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_keep_frames
    global selected_AI_multithreading
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global selected_blending_factor
    global tiles_resolution
    global input_resize_factor

    selected_file_list = []

    selected_AI_model        = default_AI_model
    selected_gpu             = default_gpu
    selected_image_extension = default_image_extension
    selected_video_extension = default_video_extension
    selected_video_codec     = default_video_codec

    if default_AI_multithreading == "OFF": 
        selected_AI_multithreading = 1
    else: 
        selected_AI_multithreading = int(default_AI_multithreading.split()[0])
    
    if default_keep_frames == "ON": 
        selected_keep_frames = True
    else:
        selected_keep_frames = False

    selected_blending_factor = {"OFF": 0, "Low": 0.3, "Medium": 0.5, "High": 0.7}.get(default_blending)

    selected_input_resize_factor.set(default_input_resize_factor)
    selected_output_resize_factor.set(default_output_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_output_path.set(default_output_path)

    info_message.set("Hi :)")
    selected_input_resize_factor.trace_add('write', update_file_widget)
    selected_output_resize_factor.trace_add('write', update_file_widget)

    font   = "Segoe UI"    
    bold8  = CTkFont(family = font, size = 8, weight = "bold")
    bold9  = CTkFont(family = font, size = 9, weight = "bold")
    bold10 = CTkFont(family = font, size = 10, weight = "bold")
    bold11 = CTkFont(family = font, size = 11, weight = "bold")
    bold12 = CTkFont(family = font, size = 12, weight = "bold")
    bold13 = CTkFont(family = font, size = 13, weight = "bold")
    bold14 = CTkFont(family = font, size = 14, weight = "bold")
    bold16 = CTkFont(family = font, size = 16, weight = "bold")
    bold17 = CTkFont(family = font, size = 17, weight = "bold")
    bold18 = CTkFont(family = font, size = 18, weight = "bold")
    bold19 = CTkFont(family = font, size = 19, weight = "bold")
    bold20 = CTkFont(family = font, size = 20, weight = "bold")
    bold21 = CTkFont(family = font, size = 21, weight = "bold")
    bold22 = CTkFont(family = font, size = 22, weight = "bold")
    bold23 = CTkFont(family = font, size = 23, weight = "bold")
    bold24 = CTkFont(family = font, size = 24, weight = "bold")

    # Images
    logo_git       = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(22, 22))
    logo_telegram  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(18, 18))
    stop_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon   = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(18, 18))

    app = App(window)
    window.update()
    window.mainloop()