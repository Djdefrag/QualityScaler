
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
from multiprocessing.pool import ThreadPool
from multiprocessing import ( 
    Process, 
    Queue          as multiprocessing_Queue,
    freeze_support as multiprocessing_freeze_support
)

from json import (
    load  as json_load, 
    dumps as json_dumps
)

from os import (
    sep        as os_separator,
    devnull    as os_devnull,
    environ    as os_environ,
    cpu_count  as os_cpu_count,
    makedirs   as os_makedirs,
    listdir    as os_listdir,
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
from natsort          import natsorted
from moviepy.editor   import VideoFileClip 
from moviepy.video.io import ImageSequenceClip 
from onnxruntime      import InferenceSession as onnxruntime_inferenceSession

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
    INTER_LINEAR,
    INTER_AREA,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    addWeighted  as opencv_addWeighted,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from numpy import (
    ndarray     as numpy_ndarray,
    frombuffer  as numpy_frombuffer,
    concatenate as numpy_concatenate, 
    transpose   as numpy_transpose,
    full        as numpy_full, 
    zeros       as numpy_zeros, 
    expand_dims as numpy_expand_dims,
    squeeze     as numpy_squeeze,
    clip        as numpy_clip,
    mean        as numpy_mean,
    repeat      as numpy_repeat,
    max         as numpy_max, 
    float32,
    uint8
)

# GUI imports
from tkinter import StringVar
from tkinter import DISABLED
from customtkinter import (
    CTk,
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
version    = "3.9"
githubme   = "https://github.com/Djdefrag/QualityScaler"
telegramme = "https://linktr.ee/j3ngystudio"

app_name_color = "#DA70D6"
dark_color     = "#080808"

very_low_VRAM  = 4
low_VRAM       = 3
medium_VRAM    = 2.2
very_high_VRAM = 0.6

AI_LIST_SEPARATOR           = [ "----" ]
IRCNN_models_list           = [ "IRCNN_Mx1", "IRCNN_Lx1" ]
SRVGGNetCompact_models_list = [ "RealESR_Gx4", "RealSRx4_Anime" ]
RRDB_models_list            = [ "BSRGANx4", "BSRGANx2", "RealESRGANx4" ]


AI_models_list         = ( SRVGGNetCompact_models_list + AI_LIST_SEPARATOR + RRDB_models_list + AI_LIST_SEPARATOR + IRCNN_models_list )
gpus_list              = [ "GPU 1", "GPU 2", "GPU 3", "GPU 4" ]
image_extension_list   = [ ".png", ".jpg", ".bmp", ".tiff" ]
video_extension_list   = [ ".mp4 (x264)", ".mp4 (x265)", ".avi" ]
interpolation_list     = [ "Low", "Medium", "High", "Disabled" ]
AI_multithreading_list = [ "1 threads", "2 threads", "3 threads", "4 threads", "5 threads", "6 threads"]

OUTPUT_PATH_CODED    = "Same path as input files"
DOCUMENT_PATH        = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_UserPreference.json")
FFMPEG_EXE_PATH      = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH    = find_by_relative_path(f"Assets{os_separator}exiftool.exe")
FRAMES_FOR_CPU       = 30


if os_path_exists(FFMPEG_EXE_PATH): 
    print(f"[{app_name}] External ffmpeg.exe file found")
    os_environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

if os_path_exists(USER_PREFERENCE_PATH):
    print(f"[{app_name}] Preference file exist")
    with open(USER_PREFERENCE_PATH, "r") as json_file:
        json_data = json_load(json_file)
        default_AI_model          = json_data["default_AI_model"]
        default_AI_multithreading = json_data["default_AI_multithreading"]
        default_gpu               = json_data["default_gpu"]
        default_image_extension   = json_data["default_image_extension"]
        default_video_extension   = json_data["default_video_extension"]
        default_interpolation     = json_data["default_interpolation"]
        default_output_path       = json_data["default_output_path"]
        default_resize_factor     = json_data["default_resize_factor"]
        default_VRAM_limiter      = json_data["default_VRAM_limiter"]
        default_cpu_number        = json_data["default_cpu_number"]
else:
    print(f"[{app_name}] Preference file does not exist, using default coded value")
    default_AI_model          = AI_models_list[0]
    default_AI_multithreading = AI_multithreading_list[0]
    default_gpu               = gpus_list[0]
    default_image_extension   = image_extension_list[0]
    default_video_extension   = video_extension_list[0]
    default_interpolation     = interpolation_list[0]
    default_output_path       = OUTPUT_PATH_CODED
    default_resize_factor     = str(50)
    default_VRAM_limiter      = str(4)
    default_cpu_number        = str(int(os_cpu_count()/2))

COMPLETED_STATUS = "Completed"
ERROR_STATUS     = "Error"
STOP_STATUS      = "Stop"

offset_y_options = 0.105
row0_y = 0.52
row1_y = row0_y + offset_y_options
row2_y = row1_y + offset_y_options
row3_y = row2_y + offset_y_options
row4_y = row3_y + offset_y_options

offset_x_options = 0.28
column1_x = 0.5
column0_x = column1_x - offset_x_options
column2_x = column1_x + offset_x_options
column1_5_x = column1_x + offset_x_options/2

supported_file_extensions = [
    '.heic', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png',
    '.PNG', '.webp', '.WEBP', '.bmp', '.BMP', '.tif',
    '.tiff', '.TIF', '.TIFF', '.mp4', '.MP4', '.webm',
    '.WEBM', '.mkv', '.MKV', '.flv', '.FLV', '.gif',
    '.GIF', '.m4v', ',M4V', '.avi', '.AVI', '.mov',
    '.MOV', '.qt', '.3gp', '.mpg', '.mpeg'
]

supported_video_extensions = [
    '.mp4', '.MP4', '.webm', '.WEBM', '.mkv', '.MKV',
    '.flv', '.FLV', '.gif', '.GIF', '.m4v', ',M4V',
    '.avi', '.AVI', '.mov', '.MOV', '.qt', '.3gp',
    '.mpg', '.mpeg'
]



# AI -------------------

class AI:

    # CLASS INIT FUNCTIONS

    def __init__(
            self, 
            AI_model_name: str, 
            directml_gpu: str, 
            resize_factor: int,
            max_resolution: int
            ):
        
        # Passed variables
        self.AI_model_name  = AI_model_name
        self.directml_gpu   = directml_gpu
        self.resize_factor  = resize_factor
        self.max_resolution = max_resolution

        # Calculated variables
        self.AI_model_path    = find_by_relative_path(f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        self.inferenceSession = self._load_inferenceSession()
        self.upscale_factor   = self._get_upscale_factor()

    def _get_upscale_factor(self) -> int:
        if   "x1" in self.AI_model_name: return 1
        elif "x2" in self.AI_model_name: return 2
        elif "x4" in self.AI_model_name: return 4

    def _load_inferenceSession(self) -> onnxruntime_inferenceSession:        
        match self.directml_gpu:
            case 'GPU 1': directml_backend = [('DmlExecutionProvider', {"device_id": "0"})]
            case 'GPU 2': directml_backend = [('DmlExecutionProvider', {"device_id": "1"})]
            case 'GPU 3': directml_backend = [('DmlExecutionProvider', {"device_id": "2"})]
            case 'GPU 4': directml_backend = [('DmlExecutionProvider', {"device_id": "3"})]

        inference_session = onnxruntime_inferenceSession(path_or_bytes = self.AI_model_path, providers = directml_backend)

        return inference_session



    # INTERNAL CLASS FUNCTIONS

    def get_image_mode(self, image: numpy_ndarray) -> str:
        match image.shape:
            case (rows, cols):
                return "Grayscale"
            case (rows, cols, channels) if channels == 3:
                return "RGB"
            case (rows, cols, channels) if channels == 4:
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

    def resize_image_with_resize_factor(
            self,
            image: numpy_ndarray, 
            ) -> numpy_ndarray:
        
        old_height, old_width = self.get_image_resolution(image)

        new_width  = int(old_width * self.resize_factor)
        new_height = int(old_height * self.resize_factor)

        match self.resize_factor:
            case factor if factor > 1:
                return opencv_resize(image, (new_width, new_height), interpolation = INTER_LINEAR)
            case factor if factor < 1:
                return opencv_resize(image, (new_width, new_height), interpolation = INTER_AREA)
            case _:
                return image

    def resize_image_with_target_resolution(
            self,
            image: numpy_ndarray, 
            t_height: int,
            t_width: int
            ) -> numpy_ndarray:
        
        old_height, old_width = self.get_image_resolution(image)
        old_resolution = old_height + old_width
        new_resolution = t_height + t_width

        if new_resolution > old_resolution:
            return opencv_resize(image, (t_width, t_height), interpolation = INTER_LINEAR)
        else:
            return opencv_resize(image, (t_width, t_height), interpolation = INTER_AREA) 



    # VIDEO CLASS FUNCTIONS

    def calculate_multiframes_supported_by_gpu(self, video_frame_path: str) -> int:
        resized_video_frame  = self.resize_image_with_resize_factor(image_read(video_frame_path))
        height, width        = self.get_image_resolution(resized_video_frame)
        image_pixels         = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        frames_simultaneously = max_supported_pixels // image_pixels 

        print(f" Frames supported simultaneously by GPU: {frames_simultaneously}")

        return frames_simultaneously

    # TILLING FUNCTIONS

    def video_need_tilling(self, video_frame_path: str) -> bool:       
        resized_video_frame  = self.resize_image_with_resize_factor(image_read(video_frame_path))
        height, width        = self.get_image_resolution(resized_video_frame)
        image_pixels         = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        if image_pixels > max_supported_pixels:
            return True
        else:
            return False

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

    def calculate_tiles_number(
            self, 
            image: numpy_ndarray, 
            ) -> tuple:
        
        height, width = self.get_image_resolution(image)

        tiles_x = (width  + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution

        return tiles_x, tiles_y
    
    def split_image_into_tiles(
            self,
            image: numpy_ndarray, 
            tiles_x: int, 
            tiles_y: int
            ) -> list[numpy_ndarray]:

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

    def combine_tiles_into_image(
            self,
            image: numpy_ndarray,
            tiles: list[numpy_ndarray], 
            t_height: int, 
            t_width: int,
            num_tiles_x: int, 
            ) -> numpy_ndarray:

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
        # same performance (?)
        
        # io_binding = self.inferenceSession.io_binding()
        # io_binding.bind_cpu_input(self.inferenceSession.get_inputs()[0].name, image)
        # io_binding.bind_output(self.inferenceSession.get_outputs()[0].name, element_type = float32)
        # self.inferenceSession.run_with_iobinding(io_binding)
        # onnx_output = io_binding.copy_outputs_to_cpu()[0]

        onnx_input  = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]

        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))

        return onnx_output.astype(float32)

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:    
        match max_range:
            case 255:   return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)



    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        image_mode   = self.get_image_mode(image)
        image, range = self.normalize_image(image)
        image        = image.astype(float32)

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


    # EXTERNAL FUNCTION

    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:

        resized_image = self.resize_image_with_resize_factor(image)
        
        if self.image_need_tilling(resized_image):
            return self.AI_upscale_with_tilling(resized_image)
        else:
            return self.AI_upscale(resized_image)



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
        self._title = title        
        self._subtitle = subtitle
        self._default_value = default_value
        self._option_list = option_list
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

    def createEmptyLabel(
            self
            ) -> CTkLabel:
        
        return CTkLabel(master = self, 
                        fg_color = "transparent",
                        width    = 500,
                        height   = 17,
                        text     = '')

    def placeInfoMessageTitleSubtitle(
            self,
            ) -> None:

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

    def placeInfoMessageOptionsText(
            self,
            ) -> None:
        
        for option_text in self._option_list:
            optionLabel = CTkLabel(master = self,
                                    width  = 600,
                                    height = 45,
                                    corner_radius = 6,
                                    anchor     = 'w',
                                    justify    = "left",
                                    text_color = "#C0C0C0",
                                    fg_color   = "#282828",
                                    bg_color   = "transparent",
                                    font       = bold12,
                                    text       = option_text)
            
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
            resize_factor = 0,
            upscale_factor = 1,
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list      = selected_file_list
        self.resize_factor  = resize_factor
        self.upscale_factor = upscale_factor

        self.label_list = []
        self._create_widgets()

    def _destroy_(self) -> None:
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def _create_widgets(self) -> None:
        self.add_clean_button()
        index_row = 1
        for file_path in self.file_list:
            label = self.add_file_information(file_path, index_row)
            self.label_list.append(label)
            index_row +=1

    def add_file_information(self, file_path, index_row) -> CTkLabel:
        infos, icon = self.extract_file_info(file_path)
        label = CTkLabel(
            self, 
            text       = infos,
            image      = icon, 
            font       = bold12,
            text_color = "#C0C0C0",
            compound   = "left", 
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )      
        label.grid(
            row    = index_row, 
            column = 0,
            pady   = (3, 3), 
            padx   = (3, 3),
            sticky = "w")
        
        return label

    def add_clean_button(self) -> None:
        
        button = CTkButton(
            self, 
            image        = clear_icon,
            font         = bold11,
            text         = "CLEAN", 
            compound     = "left",
            width        = 100, 
            height       = 28,
            border_width = 1,
            fg_color     = "#282828",
            text_color   = "#E0E0E0",
            border_color = "#0096FF"
            )

        button.configure(command=lambda: self._destroy_())
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        
    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 50

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

            video_name = str(file_path.split("/")[-1])
            file_icon  = self.extract_file_icon(file_path)

            file_infos = (f"{video_name}\n"
                          f"Resolution {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames\n")
            
            if self.resize_factor != 0 and self.upscale_factor != 0:
                resized_height  = int(height * (self.resize_factor/100))
                resized_width   = int(width * (self.resize_factor/100))

                upscaled_height = int(resized_height * self.upscale_factor)
                upscaled_width  = int(resized_width * self.upscale_factor)

                file_infos += (f"AI input {self.resize_factor}% ➜ {resized_width}x{resized_height} \n"
                               f"AI output x{self.upscale_factor} ➜ {upscaled_width}x{upscaled_height}")

        else:
            image_name    = str(file_path.split("/")[-1])
            height, width = get_image_resolution(image_read(file_path))
            file_icon     = self.extract_file_icon(file_path)

            file_infos = (f"{image_name}\n"
                          f"Resolution {width}x{height}\n")
            
            if self.resize_factor != 0 and self.upscale_factor != 0:
                resized_height = int(height * (self.resize_factor/100))
                resized_width  = int(width * (self.resize_factor/100))

                upscaled_height = int(resized_height * self.upscale_factor)
                upscaled_width  = int(resized_width * self.upscale_factor)

                file_infos += (f"AI input {self.resize_factor}% ➜ {resized_width}x{resized_height} \n"
                               f"AI output x{self.upscale_factor} ➜ {upscaled_width}x{upscaled_height}")

        return file_infos, file_icon


    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        for label in self.label_list:
            label.grid_forget()
    
    def get_selected_file_list(self) -> list: 
        return self.file_list 

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_resize_factor(self, resize_factor) -> None:
        self.resize_factor = resize_factor
 

def update_file_widget(a, b, c) -> None:
    try:
        global scrollable_frame_file_list
        scrollable_frame_file_list
    except:
        return
    
    upscale_factor = get_upscale_factor()

    try:
        resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        resize_factor = 0
    
    scrollable_frame_file_list.clean_file_list()
    scrollable_frame_file_list.set_resize_factor(resize_factor)
    scrollable_frame_file_list.set_upscale_factor(upscale_factor)
    scrollable_frame_file_list._create_widgets()

def create_info_button(
        command: Callable, 
        text: str,
        width: int = 150
        ) -> CTkButton:
    
    return CTkButton(
        master  = window, 
        command = command,
        text          = text,
        fg_color      = "transparent",
        hover_color   = "#181818",
        text_color    = "#C0C0C0",
        anchor        = "w",
        height        = 22,
        width         = width,
        corner_radius = 10,
        font          = bold12,
        image         = info_icon
    )

def create_option_menu(
        command: Callable, 
        values: list,
        default_value: str) -> CTkOptionMenu:
    
    option_menu = CTkOptionMenu(
        master  = window, 
        command = command,
        values  = values,
        width              = 150,
        height             = 30,
        corner_radius      = 6,
        dropdown_font      = bold11,
        font               = bold11,
        anchor             = "center",
        text_color         = "#C0C0C0",
        fg_color           = "#000000",
        button_color       = "#000000",
        button_hover_color = "#000000",
        dropdown_fg_color  = "#000000"
    )
    option_menu.set(default_value)
    return option_menu

def create_text_box(
        textvariable: StringVar
        ) -> CTkEntry:
    
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        corner_radius = 6,
        width         = 150,
        height        = 30,
        font          = bold11,
        justify       = "center",
        text_color    = "#C0C0C0",
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
    )

def create_text_box_output_path(
        textvariable: StringVar
        ) -> CTkEntry:

    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        border_width  = 1,
        corner_radius = 6,
        width         = 300,
        height        = 30,
        font          = bold11,
        justify       = "center",
        text_color    = "#C0C0C0",
        fg_color      = "#000000",
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
        master     = window, 
        command    = command,
        text       = text,
        image      = icon,
        width      = width,
        height     = height,
        font         = bold11,
        border_width = 1,
        fg_color     = "#282828",
        text_color   = "#E0E0E0",
        border_color = border_color
    )



# File Utils functions ------------------------

def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir):
        remove_directory(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def stop_thread() -> None: 
    stop = 1 + "x"

def image_read(
        file_path: str, 
        flags: int = IMREAD_UNCHANGED
        ) -> numpy_ndarray: 
    
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_frombuffer(file.read(), uint8), flags)

def image_write(
        file_path: str, 
        file_data: numpy_ndarray
        ) -> None: 
    
    _, file_extension = os_path_splitext(file_path)
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)



# Image/video Utils functions ------------------------

def get_image_resolution(image: numpy_ndarray) -> tuple:
    height = image.shape[0]
    width  = image.shape[1]

    return height, width 

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate
   
def extract_video_frames_and_audio(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        target_directory: str,
        video_path: str, 
        cpu_number: int
    ) -> list[str]:

    create_dir(target_directory)

    # Audio extraction
    with VideoFileClip(video_path) as video_file_clip:
        try: 
            write_process_status(processing_queue, f"{file_number}. Extracting video audio")
            audio_path = f"{target_directory}{os_separator}audio.wav"
            video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
        except:
            pass

    # Video frame extraction
    frames_number_to_save = cpu_number * FRAMES_FOR_CPU

    video_capture = opencv_VideoCapture(video_path)
    frame_count   = int(video_capture.get(CAP_PROP_FRAME_COUNT))

    frames_to_save      = []
    frames_path_to_save = []
    video_frames_list   = []

    for frame_number in range(frame_count):
        success, frame = video_capture.read()
        if success:
            frames_to_save.append(frame)
            
            frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}.jpg"
            frames_path_to_save.append(frame_path)
            video_frames_list.append(frame_path)

            if len(frames_to_save) == frames_number_to_save:
                percentage_extraction = (frame_number / frame_count) * 100
                write_process_status(processing_queue, f"{file_number}. Extracting video frames ({round(percentage_extraction, 2)}%)")

                pool = ThreadPool(cpu_number)
                pool.starmap(image_write, zip(frames_path_to_save, frames_to_save))
                pool.close()
                pool.join()
                frames_to_save      = []
                frames_path_to_save = []

    video_capture.release()

    if len(frames_to_save) > 0:
        pool = ThreadPool(cpu_number)
        pool.starmap(image_write, zip(frames_path_to_save, frames_to_save))
        pool.close()
        pool.join()
    
    return video_frames_list

def video_reconstruction_by_frames(
        video_path: str,
        audio_path: str,
        video_output_path: str,
        upscaled_frame_list_paths: list[str], 
        cpu_number: int,
        selected_video_extension: str, 
        ) -> None:
        
    match selected_video_extension:
        case '.mp4 (x264)':
            selected_video_extension = '.mp4'
            codec = 'libx264'
        case '.mp4 (x265)':
            selected_video_extension = '.mp4'
            codec = 'libx265'
        case '.avi':
            selected_video_extension = '.avi'
            codec = 'png'

    frame_rate = get_video_fps(video_path)

    clip = ImageSequenceClip.ImageSequenceClip(
        sequence = upscaled_frame_list_paths, 
        fps = frame_rate,
    )
    clip.write_videofile(
        video_output_path,
        fps     = frame_rate,
        audio   = audio_path if os_path_exists(audio_path) else None,
        codec   = codec,
        bitrate = '12M',
        verbose = False,
        logger  = None,
        threads = cpu_number,
        preset  = "ultrafast"
    )

def check_video_upscaling_resume(
        target_directory: str,
        selected_AI_model: str,
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

def interpolate_images_and_save(
        target_path: str,
        starting_image: numpy_ndarray,
        upscaled_image: numpy_ndarray,
        starting_image_importance: float,
        ) -> None:
    
    def add_alpha_channel(image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full((image.shape[0], image.shape[1], 1), 255, dtype = uint8)
            image = numpy_concatenate((image, alpha), axis = 2)
        return image
    
    def get_image_mode(image: numpy_ndarray) -> str:
        match image.shape:
            case (rows, cols):
                return "Grayscale"
            case (rows, cols, channels) if channels == 3:
                return "RGB"
            case (rows, cols, channels) if channels == 4:
                return "RGBA"


    ZERO = 0
    upscaled_image_importance       = 1 - starting_image_importance
    starting_height, starting_width = get_image_resolution(starting_image)
    target_height, target_width     = get_image_resolution(upscaled_image)

    starting_resolution = starting_height + starting_width
    target_resolution   = target_height + target_width

    if starting_resolution > target_resolution:
        starting_image = opencv_resize(starting_image,(target_width, target_height), INTER_AREA)
    else:
        starting_image = opencv_resize(starting_image,(target_width, target_height), INTER_LINEAR)

    try: 
        if get_image_mode(starting_image) == "RGBA":
            starting_image = add_alpha_channel(starting_image)
            upscaled_image = add_alpha_channel(upscaled_image)

        interpolated_image = opencv_addWeighted(starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, ZERO)
        image_write(
            file_path = target_path, 
            file_data = interpolated_image
        )
    except:
        image_write(
            file_path = target_path, 
            file_data = upscaled_image
        )

def manage_upscaled_video_frame_save_async(
        upscaled_frame: numpy_ndarray,
        starting_frame: numpy_ndarray,
        upscaled_frame_path: str,
        selected_interpolation_factor: float
    ) -> None:

    if selected_interpolation_factor > 0:
        thread = Thread(
            target = interpolate_images_and_save,
            args = (
                upscaled_frame_path, 
                starting_frame,
                upscaled_frame,
                selected_interpolation_factor
            )
        )
    else:
        thread = Thread(
            target = image_write,
            args = (
                upscaled_frame_path, 
                upscaled_frame
            )
        )

    thread.start()

def update_process_status_videos(
        processing_queue: multiprocessing_Queue, 
        file_number: int, 
        frame_index: int, 
        how_many_frames: int,
        average_processing_time: float,
        ) -> None:

    if frame_index != 0 and (frame_index + 1) % 8 == 0:  
        remaining_frames = how_many_frames - frame_index
        remaining_time   = calculate_time_to_complete_video(average_processing_time, remaining_frames)
        if remaining_time != "":
            percent_complete = (frame_index + 1) / how_many_frames * 100 
            write_process_status(processing_queue, f"{file_number}. Upscaling video {percent_complete:.2f}% ({remaining_time})")

def copy_file_metadata(
        original_file_path: str, 
        upscaled_file_path: str
        ) -> None:
    
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
        subprocess_run(exiftool_cmd, check = True, shell = 'False')
    except:
        pass



# Core functions ------------------------

def check_upscale_steps() -> None:
    sleep(1)

    try:
        while True:
            actual_step = read_process_status()

            if actual_step == COMPLETED_STATUS:
                info_message.set(f"All files completed! :)")
                stop_upscale_process()
                stop_thread()

            elif actual_step == STOP_STATUS:
                info_message.set(f"Upscaling stopped")
                stop_upscale_process()
                stop_thread()

            elif ERROR_STATUS in actual_step:
                info_message.set(f"Error while upscaling :(")
                show_error_message(actual_step.replace(ERROR_STATUS, ""))
                stop_thread()

            else:
                info_message.set(actual_step)

            sleep(1)
    except:
        place_upscale_button()
        
def read_process_status() -> str:
    return processing_queue.get()

def write_process_status(
        processing_queue: multiprocessing_Queue,
        step: str
        ) -> None:
    
    print(f"{step}")
    while not processing_queue.empty(): processing_queue.get()
    processing_queue.put(f"{step}")

def stop_upscale_process() -> None:
    global process_upscale_orchestrator
    try:
        process_upscale_orchestrator
    except:
        pass
    else:
        process_upscale_orchestrator.kill()

def stop_button_command() -> None:
    stop_upscale_process()
    write_process_status(processing_queue, f"{STOP_STATUS}") 

def upscale_button_command() -> None: 
    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_AI_multithreading
    
    global selected_interpolation_factor
    global selected_image_extension
    global selected_video_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

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
        print(f"  Interpolation factor: {selected_interpolation_factor}")
        print(f"  Selected image output extension: {selected_image_extension}")
        print(f"  Selected video output extension: {selected_video_extension}")
        print(f"  Tiles resolution for selected GPU VRAM: {tiles_resolution}x{tiles_resolution}px")
        print(f"  Resize factor: {int(resize_factor * 100)}%")
        print(f"  Cpu number: {cpu_number}")
        print("=" * 50)

        place_stop_button()

        process_upscale_orchestrator = Process(
            target = upscale_orchestrator,
            args = (
                processing_queue, 
                selected_file_list, 
                selected_output_path.get(),
                selected_AI_model, 
                selected_gpu,
                selected_image_extension,
                tiles_resolution, 
                resize_factor, 
                cpu_number, 
                selected_video_extension,
                selected_interpolation_factor,
                selected_AI_multithreading
            )
        )
        process_upscale_orchestrator.start()

        thread_wait = Thread(target = check_upscale_steps)
        thread_wait.start()

def prepare_output_image_filename(
        image_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_image_extension: str,
        selected_interpolation_factor: float
        ) -> str:
        
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(image_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(image_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected image extension
    to_append += f"{selected_image_extension}"
        
    output_path += to_append

    return output_path

def prepare_output_video_frame_filename(
        frame_path: str, 
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
            
    file_path_no_extension, _ = os_path_splitext(frame_path)
    output_path = file_path_no_extension

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected image extension
    to_append += f".jpg"
        
    output_path += to_append

    return output_path

def prepare_output_video_filename(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_video_extension: str,
        selected_interpolation_factor: float
        ) -> str:
    
    match selected_video_extension:
        case '.mp4 (x264)':
            selected_video_extension = '.mp4'
        case '.mp4 (x265)':
            selected_video_extension = '.mp4'
        case '.avi':
            selected_video_extension = '.avi'

    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"
    
    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    # Selected video extension
    to_append += f"{selected_video_extension}"
        
    output_path += to_append

    return output_path

def prepare_output_video_frames_directory_name(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
    
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    match selected_interpolation_factor:
        case 0.3:
            to_append += "_Interpolation-Low"
        case 0.5:
            to_append += "_Interpolation-Medium"
        case 0.7:
            to_append += "_Interpolation-High"

    output_path += to_append

    return output_path

# ORCHESTRATOR

def upscale_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_output_path: str,
        selected_AI_model: str,
        selected_gpu: str,
        selected_image_extension: str,
        tiles_resolution: int,
        resize_factor: int,
        cpu_number: int,
        selected_video_extension: str,
        selected_interpolation_factor: float,
        selected_AI_multithreading: int
        ) -> None:

    write_process_status(processing_queue, f"Loading AI model")
    AI_instance = AI(selected_AI_model, selected_gpu, resize_factor, tiles_resolution)
    AI_instance_list = []
    AI_instance_list.append(AI_instance)

    if selected_AI_multithreading > 1:
        for _ in range(selected_AI_multithreading - 1):
            AI_instance_list.append(AI(selected_AI_model, selected_gpu, resize_factor, tiles_resolution))

    try:
        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(
                    processing_queue,
                    file_path, 
                    file_number,
                    selected_output_path, 
                    AI_instance,
                    AI_instance_list,
                    selected_AI_model,
                    resize_factor, 
                    cpu_number, 
                    selected_video_extension, 
                    selected_interpolation_factor,
                    selected_AI_multithreading
                )
            else:
                upscale_image(
                    processing_queue,
                    file_path, 
                    file_number,
                    selected_output_path,
                    AI_instance,
                    selected_AI_model,
                    selected_image_extension, 
                    resize_factor, 
                    selected_interpolation_factor
                )

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS} {str(exception)}")

# IMAGES

def upscale_image(
        processing_queue: multiprocessing_Queue,
        image_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_instance: AI,
        selected_AI_model: str,
        selected_image_extension: str,
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> None:
    
    starting_image = image_read(image_path)
    upscaled_image_path = prepare_output_image_filename(image_path, selected_output_path, selected_AI_model, resize_factor, selected_image_extension, selected_interpolation_factor)

    write_process_status(processing_queue, f"{file_number}. Upscaling image")
    upscaled_image = AI_instance.AI_orchestration(starting_image)

    if selected_interpolation_factor > 0:
        interpolate_images_and_save(
            upscaled_image_path,
            starting_image,
            upscaled_image,
            selected_interpolation_factor
        )

    else:
        image_write(
            file_path = upscaled_image_path,
            file_data = upscaled_image
        )

    copy_file_metadata(image_path, upscaled_image_path)

# VIDEOS

def upscale_video(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_instance: AI,
        AI_instance_list: list[AI],
        selected_AI_model: str,
        resize_factor: int, 
        cpu_number: int, 
        selected_video_extension: str,
        selected_interpolation_factor: float,
        selected_AI_multithreading: int
        ) -> None:

    global processed_frames_async
    global processing_times_async
    processed_frames_async = 0
    processing_times_async = []

    target_directory       = prepare_output_video_frames_directory_name(video_path, selected_output_path, selected_AI_model, resize_factor, selected_interpolation_factor)
    video_output_path      = prepare_output_video_filename(video_path, selected_output_path, selected_AI_model, resize_factor, selected_video_extension, selected_interpolation_factor)
    video_upscale_continue = check_video_upscaling_resume(target_directory, selected_AI_model)
    video_audio_path       = f"{target_directory}{os_separator}audio.wav"

    if video_upscale_continue:
        print(f"Resume upscaling")
        frame_list_paths = get_video_frames_for_upscaling_resume(target_directory, selected_AI_model)
    else:
        print(f"Upscaling from scrach")
        frame_list_paths = extract_video_frames_and_audio(processing_queue, file_number, target_directory, video_path, cpu_number)
    
    upscaled_frame_list_paths    = [prepare_output_video_frame_filename(frame_path, selected_AI_model, resize_factor, selected_interpolation_factor) for frame_path in frame_list_paths]
    video_need_tiles             = AI_instance.video_need_tilling(frame_list_paths[0])
    multiframes_supported_by_gpu = AI_instance.calculate_multiframes_supported_by_gpu(frame_list_paths[0])
    multiframes_number           = min(multiframes_supported_by_gpu, selected_AI_multithreading)

    write_process_status(processing_queue, f"{file_number}. Upscaling video") 
    if video_need_tiles or multiframes_number <= 1:
        upscale_video_frames(
            processing_queue,
            file_number,
            AI_instance,
            frame_list_paths,
            upscaled_frame_list_paths,
            selected_interpolation_factor
        )
    else:
        upscale_video_frames_multithreading(
            processing_queue,
            file_number,
            AI_instance_list,
            frame_list_paths,
            upscaled_frame_list_paths,
            multiframes_number,
            selected_interpolation_factor
        )

    check_forgotten_video_frames(processing_queue, file_number, AI_instance, frame_list_paths, upscaled_frame_list_paths, selected_interpolation_factor)

    write_process_status(processing_queue, f"{file_number}. Processing upscaled video")
    video_reconstruction_by_frames(video_path, video_audio_path, video_output_path, upscaled_frame_list_paths, cpu_number, selected_video_extension)
    copy_file_metadata(video_path, video_output_path)

def upscale_video_frames(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        AI_instance: AI,
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        selected_interpolation_factor: float
        ) -> None:
    
    frame_processing_times = []

    for frame_index, frame_path in enumerate(frame_list_paths):
        start_timer = timer()
        upscaled_frame_path = upscaled_frame_list_paths[frame_index]

        if not os_path_exists(upscaled_frame_path):
            starting_frame = image_read(frame_path)
            upscaled_frame = AI_instance.AI_orchestration(starting_frame)
            manage_upscaled_video_frame_save_async(upscaled_frame, starting_frame, upscaled_frame_path, selected_interpolation_factor)

        frame_processing_times.append(timer() - start_timer)
        if (frame_index + 1) % 8 == 0:
            average_processing_time = numpy_mean(frame_processing_times)
            update_process_status_videos(processing_queue, file_number, frame_index, len(frame_list_paths), average_processing_time)

        if (frame_index + 1) % 100 == 0: frame_processing_times = []

def upscale_video_frames_multithreading(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        AI_instance_list: list[AI],
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        multiframes_number: int,
        selected_interpolation_factor: float,
        ) -> None:
    
    def upscale_single_video_frame_async(
            processing_queue: multiprocessing_Queue,
            file_number: int,
            multiframes_number: int,
            total_video_frames: int,
            AI_instance: AI,
            frame_list_paths: list[str],
            upscaled_frame_list_paths: list[str],
            selected_interpolation_factor: float,
            ) -> None:

        global processed_frames_async
        global processing_times_async

        for frame_index in range(len(frame_list_paths)):
            start_timer = timer()
            upscaled_frame_path = upscaled_frame_list_paths[frame_index]

            if not os_path_exists(upscaled_frame_path):
                starting_frame = image_read(frame_list_paths[frame_index])
                upscaled_frame = AI_instance.AI_orchestration(starting_frame)

                manage_upscaled_video_frame_save_async(
                    upscaled_frame,
                    starting_frame,
                    upscaled_frame_path,
                    selected_interpolation_factor
                )

            processed_frames_async +=1
            processing_times_async.append(timer() - start_timer)

            if (processed_frames_async + 1) % 8 == 0:
                average_processing_time = float(numpy_mean(processing_times_async)/multiframes_number)
                update_process_status_videos(
                    processing_queue = processing_queue, 
                    file_number      = file_number,  
                    frame_index      = processed_frames_async, 
                    how_many_frames  = total_video_frames,
                    average_processing_time = average_processing_time
                )

            if (processed_frames_async + 1) % 100 == 0: processing_times_async = []
        
    
    total_video_frames         = len(frame_list_paths)
    chunk_size                 = total_video_frames // multiframes_number
    frame_list_chunks          = [frame_list_paths[i:i + chunk_size] for i in range(0, len(frame_list_paths), chunk_size)]
    upscaled_frame_list_chunks = [upscaled_frame_list_paths[i:i + chunk_size] for i in range(0, len(upscaled_frame_list_paths), chunk_size)]

    write_process_status(processing_queue, f"{file_number}. Upscaling video ({multiframes_number} threads)")

    pool = ThreadPool(multiframes_number)
    pool.starmap(
        upscale_single_video_frame_async,
        zip(
            repeat(processing_queue),
            repeat(file_number),
            repeat(multiframes_number),
            repeat(total_video_frames),
            AI_instance_list,
            frame_list_chunks,
            upscaled_frame_list_chunks,
            repeat(selected_interpolation_factor)
        )
    )
    pool.close()
    pool.join()

def check_forgotten_video_frames(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        AI_instance: AI,
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        selected_interpolation_factor: float,
        ):
    
    # Check if all the upscaled frames exist
    frame_path_todo_list          = []
    upscaled_frame_path_todo_list = []

    for frame_index in range(len(upscaled_frame_list_paths)):
        
        if not os_path_exists(upscaled_frame_list_paths[frame_index]):
            frame_path_todo_list.append(frame_list_paths[frame_index])
            upscaled_frame_path_todo_list.append(upscaled_frame_list_paths[frame_index]) 

    if len(upscaled_frame_path_todo_list) > 0:
        upscale_video_frames(
            processing_queue,
            file_number,
            AI_instance,
            frame_path_todo_list,
            upscaled_frame_path_todo_list,
            selected_interpolation_factor
        )



# GUI utils function ---------------------------

def opengithub() -> None:   
    open_browser(githubme, new=1)

def opentelegram() -> None: 
    open_browser(telegramme, new=1)

def check_if_file_is_video(
        file: str
        ) -> bool:
    
    return any(video_extension in file for video_extension in supported_video_extensions)

def check_supported_selected_files(
        uploaded_file_list: list
        ) -> list:
    
    return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

def user_input_checks() -> bool:
    global selected_file_list
    global selected_AI_model
    global selected_image_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    # Selected files 
    try: selected_file_list = scrollable_frame_file_list.get_selected_file_list()
    except:
        info_message.set("Please select a file")
        return False

    if len(selected_file_list) <= 0:
        info_message.set("Please select a file")
        return False


    # AI model
    if selected_AI_model == AI_LIST_SEPARATOR[0]:
        info_message.set("Please select the AI model")
        return False


    # File resize factor 
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_message.set("Resize % must be a numeric value")
        return False

    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_message.set("Resize % must be a value > 0")
        return False

    
    # Tiles resolution 
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_message.set("VRAM/RAM value must be a numeric value")
        return False

    if tiles_resolution > 0: 
        if selected_AI_model in RRDB_models_list:          
            vram_multiplier = very_high_VRAM
        elif selected_AI_model in SRVGGNetCompact_models_list: 
            vram_multiplier = medium_VRAM
        elif selected_AI_model in IRCNN_models_list:
            vram_multiplier = very_low_VRAM

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))
        tiles_resolution = int(selected_vram * 100)

        
    else:
        info_message.set("VRAM/RAM value must be > 0")
        return False


    # Cpu number 
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_message.set("Cpu number must be a numeric value")
        return False

    if cpu_number <= 0:         
        info_message.set("Cpu number value must be > 0")
        return False
    else: 
        cpu_number = int(cpu_number)

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
    if AI_LIST_SEPARATOR[0] in selected_AI_model: upscale_factor = 0
    elif 'x1' in selected_AI_model: upscale_factor = 1
    elif 'x2' in selected_AI_model: upscale_factor = 2
    elif 'x4' in selected_AI_model: upscale_factor = 4

    return upscale_factor

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        global scrollable_frame_file_list

        upscale_factor = get_upscale_factor()

        try:
            resize_factor = int(float(str(selected_resize_factor.get())))
        except:
            resize_factor = 0

        scrollable_frame_file_list = FileWidget(
            master = window, 
            selected_file_list = supported_files_list,
            resize_factor  = resize_factor,
            upscale_factor = upscale_factor,
            fg_color = dark_color, 
            bg_color = dark_color
        )
        
        scrollable_frame_file_list.place(
            relx = 0.0, 
            rely = 0.0, 
            relwidth  = 1.0, 
            relheight = 0.42
        )
        
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

def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu    
    selected_gpu = selected_option

def select_AI_multithreading_from_menu(selected_option: str) -> None:
    global selected_AI_multithreading
    selected_AI_multithreading = int(selected_option.split()[0])

def select_image_extension_from_menu(selected_option: str) -> None:
    global selected_image_extension   
    selected_image_extension = selected_option

def select_video_extension_from_menu(selected_option: str) -> None:
    global selected_video_extension   
    selected_video_extension = selected_option

def select_interpolation_from_menu(selected_option: str) -> None:
    global selected_interpolation_factor

    match selected_option:
        case "Disabled":
            selected_interpolation_factor = 0
        case "Low":
            selected_interpolation_factor = 0.3
        case "Medium":
            selected_interpolation_factor = 0.5
        case "High":
            selected_interpolation_factor = 0.7



# GUI info functions ---------------------------

def open_info_output_path():
    option_list = [
        "\n The default path is defined by the input files."
        + "\n For example uploading a file from the Download folder,"
        + "\n the app will save the generated files in the Download folder \n",

        " Otherwise it is possible to select the desired path using the SELECT button",
    ]

    MessageBox(
        messageType   = "info",
        title         = "Output path",
        subtitle      = "This widget allows to choose upscaled files path",
        default_value = default_output_path,
        option_list   = option_list
    )

def open_info_AI_model():
    option_list = [
        "\n IRCNN (2017) - Very simple and lightweight AI architecture\n" + 
        " Only denoising (no upscaling)\n" + 
        " Recommended for both image/video denoising\n" + 
        "  • IRCNN_Mx1 - (medium denoise)\n" +
        "  • IRCNN_Lx1 - (high denoise)\n",

        "\n SRVGGNetCompact (2022) - Fast and lightweight AI architecture\n" + 
        " Good-quality upscale\n" + 
        " Recommended for video upscaling\n" + 
        "  • RealESR_Gx4\n" + 
        "  • RealSRx4_Anime\n",

        "\n RRDB (2020) - Complex and heavy AI architecture\n" + 
        " High-quality upscale\n" + 
        " Recommended for image upscaling\n" +
        "  • BSRGANx2\n" + 
        "  • BSRGANx4\n" +
        "  • RealESRGANx4\n",

    ]

    MessageBox(
        messageType = "info",
        title       = "AI model",
        subtitle    = "This widget allows to choose between different AI models for upscaling",
        default_value = default_AI_model,
        option_list   = option_list
    )

def open_info_gpu():
    option_list = [
        "\n It is possible to select up to 4 GPUs, via the index (also visible in the Task Manager):\n" +
        "  • GPU 1 (GPU 0 in Task manager)\n" + 
        "  • GPU 2 (GPU 1 in Task manager)\n" + 
        "  • GPU 3 (GPU 2 in Task manager)\n" + 
        "  • GPU 4 (GPU 3 in Task manager)\n",

        "\n NOTES\n" +
        "  • Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be\n" +
        "  • For optimal performance, it is essential to regularly update your GPUs drivers\n" +
        "  • Selecting the index of a GPU not present in the PC will cause the app to use the CPU for AI operations\n"+
        "  • In the case of a single GPU, select 'GPU 1'\n"
    ]

    MessageBox(
        messageType = "info",
        title       = "GPU",
        subtitle    = "This widget allows to select the GPU for AI upscale",
        default_value = default_gpu,
        option_list   = option_list
    )

def open_info_AI_interpolation():
    option_list = [
        " Interpolation is the fusion of the upscaled image produced by AI and the original image",

        " \n INTERPOLATION OPTIONS\n" +
        "  • Disabled - 100% upscaled\n" + 
        "  • Low - 30% original / 70% upscaled\n" +
        "  • Medium - 50% original / 50% upscaled\n" +
        "  • High - 70% original / 30% upscaled\n",

        " \n NOTES\n" +
        "  • Can increase the quality of the final result\n" + 
        "  • Especially when using the tilling/merging function (with low VRAM)\n" +
        "  • Especially at low Input resolution % values (<50%) \n",

    ]

    MessageBox(
        messageType = "info",
        title       = "AI Interpolation", 
        subtitle    = "This widget allows to choose interpolation between upscaled and original image/frame",
        default_value = default_interpolation,
        option_list   = option_list
    )

def open_info_AI_multithreading():
    option_list = [
        " This option can improve video upscaling performance, especially with powerful GPUs",

        " \n AI MULTITHREADING OPTIONS\n"
        + "  • 1 threads - upscaling 1 frame\n" 
        + "  • 2 threads - upscaling 2 frame simultaneously\n" 
        + "  • 3 threads - upscaling 3 frame simultaneously\n" 
        + "  • 4 threads - upscaling 4 frame simultaneously\n" ,

        " \n NOTES \n"
        + "  • As the number of threads increases, the use of CPU, GPU and RAM memory also increases\n" 
        + "  • In particular, the GPU is put under a lot of stress, and may reach high temperatures\n" 
        + "  • Keep an eye on the temperature of your PC so that it doesn't overheat \n" 
        + "  • The app selects the most appropriate number of threads if the chosen number exceeds GPU capacity\n" ,

    ]

    MessageBox(
        messageType = "info",
        title       = "AI multithreading", 
        subtitle    = "This widget allows to choose how many video frames are upscaled simultaneously",
        default_value = default_AI_multithreading,
        option_list   = option_list
    )

def open_info_image_output():
    option_list = [
        " \n PNG\n  • very good quality\n  • slow and heavy file\n  • supports transparent images\n",
        " \n JPG\n  • good quality\n  • fast and lightweight file\n",
        " \n BMP\n  • highest quality\n  • slow and heavy file\n",
        " \n TIFF\n  • highest quality\n  • very slow and heavy file\n",
    ]

    MessageBox(
        messageType = "info",
        title       = "Image output",
        subtitle    = "This widget allows to choose the extension of upscaled images",
        default_value = default_image_extension,
        option_list   = option_list
    )

def open_info_video_extension():
    option_list = [
        "\n MP4 (x264)\n" + 
        "   • produces well compressed video using x264 codec\n",

        "\n MP4 (x265)\n" + 
        "   • produces well compressed video using x265 codec\n",

        "\n AVI\n" + 
        "   • produces the highest quality video\n" +
        "   • the video produced can also be of large size\n"
    ]

    MessageBox(
        messageType = "info",
        title = "Video output",
        subtitle = "This widget allows to choose the extension of the upscaled video",
        default_value = default_video_extension,
        option_list = option_list
    )

def open_info_vram_limiter():
    option_list = [
        " It is important to enter the correct value according to the VRAM of selected GPU ",
        " Selecting a value greater than the actual amount of GPU VRAM may result in upscale failure",
        " For integrated GPUs (Intel-HD series • Vega 3,5,7) - select 2 GB",
    ]

    MessageBox(
        messageType = "info",
        title       = "GPU Vram (GB)",
        subtitle    = "This widget allows to set a limit on the GPU VRAM memory usage",
        default_value = default_VRAM_limiter,
        option_list   = option_list
    )

def open_info_input_resolution():
    option_list = [
        " A high value (>70%) will create high quality photos/videos but will be slower",
        " While a low value (<40%) will create good quality photos/videos but will much faster",

        " \n For example, for a 1080p (1920x1080) image/video\n" + 
        " • Input resolution 25% => input to AI 270p (480x270)\n" +
        " • Input resolution 50% => input to AI 540p (960x540)\n" + 
        " • Input resolution 75% => input to AI 810p (1440x810)\n" + 
        " • Input resolution 100% => input to AI 1080p (1920x1080) \n",
    ]

    MessageBox(
        messageType = "info",
        title       = "Input resolution %",
        subtitle    = "This widget allows to choose the resolution input to the AI",
        default_value = default_resize_factor,
        option_list   = option_list
    )

def open_info_cpu():
    option_list = [
        " When possible the app will use the number of cpus selected",

        "\n Currently this value is used for: \n" +
        "  • video frames extraction \n" +
        "  • video encoding \n",
    ]

    MessageBox(
        messageType = "info",
        title       = "Cpu number",
        subtitle    = "This widget allows to choose how many cpus to devote to the app",
        default_value = default_cpu_number,
        option_list   = option_list
    )



# GUI place functions ---------------------------

def place_github_button():
    git_button = CTkButton(master  = window, 
                        command    = opengithub,
                        image      = logo_git,
                        width         = 30,
                        height        = 30,
                        border_width  = 1,
                        fg_color      = "transparent",
                        text_color    = "#C0C0C0",
                        border_color  = "#404040",
                        anchor        = "center",                           
                        text          = "", 
                        font          = bold11)
    
    git_button.place(relx = 0.045, rely = 0.87, anchor = "center")

def place_telegram_button():
    telegram_button = CTkButton(master = window, 
                            image      = logo_telegram,
                            command    = opentelegram,
                            width         = 30,
                            height        = 30,
                            border_width  = 1,
                            fg_color      = "transparent",
                            text_color    = "#C0C0C0",
                            border_color  = "#404040",
                            anchor        = "center",                           
                            text          = "", 
                            font          = bold11)
    telegram_button.place(relx = 0.045, rely = 0.93, anchor = "center")
 
def place_loadFile_section():
    background = CTkLabel(
        master   = window,
        text     = "",
        fg_color = dark_color
    )

    text_drop = """ SUPPORTED FILES \n\n IMAGES • jpg png tif bmp webp heic \n VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp """

    input_file_text = CTkLabel(
        master = window, 
        text       = text_drop,
        fg_color   = dark_color,
        bg_color   = dark_color,
        text_color = "#C0C0C0",
        width      = 300,
        height     = 150,
        font       = bold12,
        anchor     = "center"
    )
    
    input_file_button = CTkButton(
        master = window,
        command  = open_files_action, 
        text     = "SELECT FILES",
        width    = 140,
        height   = 30,
        font     = bold11,
        border_width = 1,
        fg_color     = "#282828",
        text_color   = "#E0E0E0",
        border_color = "#0096FF"
        )
    
    background.place(relx = 0.0, rely = 0.0, relwidth = 1.0, relheight = 0.42)
    input_file_text.place(relx = 0.5, rely = 0.20,  anchor = "center")
    input_file_button.place(relx = 0.5, rely = 0.35, anchor = "center")

def place_app_name():
    app_name_label = CTkLabel(
        master     = window, 
        text       = app_name + " " + version,
        text_color = app_name_color,
        font       = bold20,
        anchor     = "w"
    )
    
    app_name_label.place(relx = column0_x, rely = row0_y - 0.0175, anchor = "center")

def place_output_path_textbox():
    output_path_button  = create_info_button(open_info_output_path, "Output path", width = 300)
    output_path_textbox = create_text_box_output_path(selected_output_path) 
    select_output_path_button = create_active_button(
        command = open_output_path_action,
        text    = "SELECT",
        width   = 85,
        height  = 25
    )
  
    output_path_button.place(relx = column1_5_x, rely = row0_y - 0.05, anchor = "center")
    output_path_textbox.place(relx = column1_5_x, rely  = row0_y, anchor = "center")
    select_output_path_button.place(relx = column2_x, rely  = row0_y - 0.05, anchor = "center")

def place_AI_menu():
    AI_menu_button = create_info_button(open_info_AI_model, "AI model")
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list, default_AI_model)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.05, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_AI_interpolation_menu():
    interpolation_button = create_info_button(open_info_AI_interpolation, "AI Interpolation")
    interpolation_menu   = create_option_menu(select_interpolation_from_menu, interpolation_list, default_interpolation)
    
    interpolation_button.place(relx = column0_x, rely = row3_y - 0.05, anchor = "center")
    interpolation_menu.place(relx = column0_x, rely  = row3_y, anchor = "center")

def place_AI_multithreading_menu():
    AI_multithreading_button = create_info_button(open_info_AI_multithreading, "AI multithreading")
    AI_multithreading_menu   = create_option_menu(select_AI_multithreading_from_menu, AI_multithreading_list, default_AI_multithreading)
    
    AI_multithreading_button.place(relx = column0_x, rely = row2_y - 0.05, anchor = "center")
    AI_multithreading_menu.place(relx = column0_x, rely  = row2_y, anchor = "center")

def place_image_output_menu():
    file_extension_button = create_info_button(open_info_image_output, "Image output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list, default_image_extension)
    
    file_extension_button.place(relx = column2_x, rely = row1_y - 0.05, anchor = "center")
    file_extension_menu.place(relx = column2_x, rely = row1_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list, default_video_extension)
    
    video_extension_button.place(relx = column2_x, rely = row2_y - 0.05, anchor = "center")
    video_extension_menu.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_gpu_menu():
    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_gpu_from_menu, gpus_list, default_gpu)
    
    gpu_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    gpu_menu.place(relx = column1_x, rely  = row1_y, anchor = "center")

def place_vram_textbox():
    vram_button  = create_info_button(open_info_vram_limiter, "GPU Vram (GB)")
    vram_textbox = create_text_box(selected_VRAM_limiter) 
  
    vram_button.place(relx = column1_x, rely = row2_y - 0.05, anchor = "center")
    vram_textbox.place(relx = column1_x, rely  = row2_y, anchor = "center")

def place_input_resolution_textbox():
    resize_factor_button  = create_info_button(open_info_input_resolution, "Input resolution %")
    resize_factor_textbox = create_text_box(selected_resize_factor) 

    resize_factor_button.place(relx = column1_x, rely = row4_y - 0.05, anchor = "center")
    resize_factor_textbox.place(relx = column1_x, rely = row4_y, anchor = "center")

def place_cpu_textbox():
    cpu_button  = create_info_button(open_info_cpu, "CPU number")
    cpu_textbox = create_text_box(selected_cpu_number)

    cpu_button.place(relx = column1_x, rely = row3_y - 0.05, anchor = "center")
    cpu_textbox.place(relx = column1_x, rely  = row3_y, anchor = "center")

def place_message_label():
    message_label = CTkLabel(
        master  = window, 
        textvariable = info_message,
        height       = 25,
        font         = bold11,
        fg_color     = "#ffbf00",
        text_color   = "#000000",
        anchor       = "center",
        corner_radius = 12
    )
    message_label.place(relx = column2_x, rely = row4_y - 0.075, anchor = "center")

def place_stop_button(): 
    stop_button = create_active_button(
        command = stop_button_command,
        text    = "STOP",
        icon    = stop_icon,
        width   = 140,
        height  = 30,
        border_color = "#EC1D1D"
    )
    stop_button.place(relx = column2_x, rely = row4_y, anchor = "center")

def place_upscale_button(): 
    upscale_button = create_active_button(
        command = upscale_button_command,
        text    = "UPSCALE",
        icon    = upscale_icon,
        width   = 140,
        height  = 30
    )
    upscale_button.place(relx = column2_x, rely = row4_y, anchor = "center")
   


# Main functions ---------------------------

def on_app_close() -> None:
    window.grab_release()
    window.destroy()

    global selected_AI_model
    global selected_AI_multithreading
    global selected_gpu
    
    global selected_interpolation_factor
    global selected_image_extension
    global selected_video_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    AI_model_to_save          = f"{selected_AI_model}"
    AI_multithreading_to_save = f"{selected_AI_multithreading} threads"
    gpu_to_save               = selected_gpu
    image_extension_to_save   = selected_image_extension
    video_extension_to_save   = selected_video_extension
    interpolation_to_save= {
        0: "Disabled",
        0.3: "Low",
        0.5: "Medium",
        0.7: "High",
    }.get(selected_interpolation_factor)

    user_preference = {
        "default_AI_model":          AI_model_to_save,
        "default_AI_multithreading": AI_multithreading_to_save,
        "default_gpu":               gpu_to_save,
        "default_image_extension":   image_extension_to_save,
        "default_video_extension":   video_extension_to_save,
        "default_interpolation":     interpolation_to_save,
        "default_output_path":       selected_output_path.get(),
        "default_resize_factor":     str(selected_resize_factor.get()),
        "default_VRAM_limiter":      str(selected_VRAM_limiter.get()),
        "default_cpu_number":        str(selected_cpu_number.get()),
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
        window.geometry("675x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_app_name()
        place_output_path_textbox()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_AI_multithreading_menu()
        place_AI_interpolation_menu()

        place_gpu_menu()
        place_vram_textbox()
        place_cpu_textbox()
        place_input_resolution_textbox()

        place_image_output_menu()
        place_video_extension_menu()
        place_message_label()
        place_upscale_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing_freeze_support()
    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")
    
    processing_queue = multiprocessing_Queue(maxsize=1)

    window = CTk() 

    info_message            = StringVar()
    selected_output_path    = StringVar()
    selected_resize_factor  = StringVar()
    selected_VRAM_limiter   = StringVar()
    selected_cpu_number     = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_AI_multithreading
    global selected_image_extension
    global selected_video_extension
    global selected_interpolation_factor
    global tiles_resolution
    global resize_factor
    global cpu_number

    selected_file_list = []

    selected_AI_model          = default_AI_model
    selected_gpu               = default_gpu
    selected_image_extension   = default_image_extension
    selected_video_extension   = default_video_extension
    selected_AI_multithreading = int(default_AI_multithreading.split()[0])

    selected_interpolation_factor = {
        "Disabled": 0,
        "Low": 0.3,
        "Medium": 0.5,
        "High": 0.7,
    }.get(default_interpolation)

    selected_resize_factor.set(default_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_cpu_number.set(default_cpu_number)
    selected_output_path.set(default_output_path)

    info_message.set("Hi :)")
    selected_resize_factor.trace_add('write', update_file_widget)

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
    logo_git       = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(15, 15))
    logo_telegram  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(15, 15))
    stop_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon   = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(17, 17))

    app = App(window)
    window.update()
    window.mainloop()