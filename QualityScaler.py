
# Standard library imports
import sys
from shutil     import rmtree as remove_directory
from timeit     import default_timer as timer
from webbrowser import open as open_browser
from subprocess import run  as subprocess_run
from time       import sleep
from typing     import Callable

from threading import Thread
from multiprocessing import ( 
    Process, 
    Queue          as multiprocessing_Queue,
    freeze_support as multiprocessing_freeze_support
)
from multiprocessing.pool import ThreadPool

from os import (
    sep         as os_separator,
    devnull     as os_devnull,
    cpu_count   as os_cpu_count,
    makedirs    as os_makedirs,
    remove      as os_remove,
)

from os.path import (
    dirname  as os_path_dirname,
    abspath  as os_path_abspath,
    join     as os_path_join,
    exists   as os_path_exists,
    splitext as os_path_splitext
)


# Third-party library imports
from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from moviepy.editor import VideoFileClip 
from moviepy.video.io import ImageSequenceClip 

from onnx import load as onnx_load 
from onnxconverter_common import float16 as onnx_converter_float16 
from onnxruntime import InferenceSession as onnxruntime_inferenceSession 

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_GRAY2RGB,
    COLOR_RGB2BGRA,
    COLOR_RGB2GRAY,
    IMREAD_UNCHANGED,
    INTER_LINEAR,
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
    ndarray           as numpy_ndarray,
    ascontiguousarray as numpy_ascontiguousarray,
    frombuffer        as numpy_frombuffer,
    concatenate       as numpy_concatenate, 
    transpose         as numpy_transpose,
    full              as numpy_full, 
    zeros             as numpy_zeros, 
    expand_dims       as numpy_expand_dims,
    squeeze           as numpy_squeeze,
    clip              as numpy_clip,
    mean              as numpy_mean,
    max               as numpy_max, 
    float32,
    uint8
)

# GUI imports
from tkinter import StringVar
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
    set_default_color_theme,
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")


app_name = "QualityScaler"
version = "3.0"

app_name_color = "#DA70D6"
dark_color = "#080808"

githubme   = "https://github.com/Djdefrag/QualityScaler"
telegramme = "https://linktr.ee/j3ngystudio"

SRVGGNetCompact_vram_multiplier = 2.0
RRDB_vram_multiplier = 0.7
full_precision_vram_multiplier = 0.85

SRVGGNetCompact_models_list = [ 'RealESR_Gx4', 'RealSRx4_Anime' ]
RRDB_models_list = [ 'BSRGANx4', 'BSRGANx2', 'RealESRGANx4' ]

AI_models_list = (
                SRVGGNetCompact_models_list
                + RRDB_models_list 
                )

image_extension_list = [ '.png', '.jpg', '.bmp', '.tiff' ]
video_extension_list = [ '.mp4 (x264)', '.mp4 (x265)', '.avi' ]
interpolation_list   = [ 'Low', 'Medium', 'High', 'Disabled' ]
AI_precision_list    = [ 'Half precision', 'Full precision' ]

default_AI_model        = AI_models_list[0]
default_image_extension = image_extension_list[0]
default_video_extension = video_extension_list[0]
default_interpolation   = interpolation_list[0]
default_AI_precision    = AI_precision_list[0]
default_resize_factor   = str(50)
default_VRAM_limiter    = str(8)
default_cpu_number      = str(int(os_cpu_count()/2))

COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"

offset_y_options = 0.125
row0_y = 0.56
row1_y = row0_y + offset_y_options
row2_y = row1_y + offset_y_options
row3_y = row2_y + offset_y_options

offset_x_options = 0.28
column1_x = 0.5
column0_x = column1_x - offset_x_options
column2_x = column1_x + offset_x_options

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

def load_AI_model(
        selected_AI_model: str, 
        selected_half_precision: bool
        ) -> onnxruntime_inferenceSession:
        
    AI_model_path   = find_by_relative_path(f"AI-onnx{os_separator}{selected_AI_model}.onnx")
    AI_model_loaded = onnx_load(AI_model_path)

    if selected_half_precision:
        AI_model_loaded = onnx_converter_float16.convert_float_to_float16(
            model = AI_model_loaded,
            keep_io_types = True
        )

    AI_model = onnxruntime_inferenceSession(
        path_or_bytes = AI_model_loaded.SerializeToString(), 
        providers = [('DmlExecutionProvider', {'performance_preference': 'high_performance'})]
    )    

    return AI_model

def AI_upscale(
        AI_model: onnxruntime_inferenceSession, 
        half_precision: bool,
        image: numpy_ndarray, 
        ) -> numpy_ndarray:

    image = numpy_ascontiguousarray(image)
    image_mode = get_image_mode(image)
    image, range = normalize_image(image)
    image = image.astype(float32)

    match image_mode:
        case "RGB":
            image = preprocess_image(image)
            output_image = process_image_with_AI_model(AI_model, image)
            return postprocess_output(output_image, range)
        
        case "RGBA":
            alpha = image[:, :, 3]
            image = image[:, :, :3]
            image = opencv_cvtColor(image, COLOR_BGR2RGB)
            alpha = opencv_cvtColor(alpha, COLOR_GRAY2RGB)

            image = image.astype(float32)
            alpha = alpha.astype(float32)

            # Image
            image = preprocess_image(image)
            output_image = process_image_with_AI_model(AI_model, image)
            output_image = opencv_cvtColor(output_image, COLOR_RGB2BGRA)

            # Alpha
            alpha = preprocess_image(alpha)
            output_alpha = process_image_with_AI_model(AI_model, alpha)
            output_alpha = opencv_cvtColor(output_alpha, COLOR_RGB2GRAY)

            # Fusion Image + Alpha
            output_image[:, :, 3] = output_alpha
            return postprocess_output(output_image, range)
        
        case "Grayscale":
            image = opencv_cvtColor(image, COLOR_GRAY2RGB)
            image = preprocess_image(image)
            output_image = process_image_with_AI_model(AI_model, image)
            output_image = opencv_cvtColor(output_image, COLOR_RGB2GRAY)
            return postprocess_output(output_image, range)

def normalize_image(
        image: numpy_ndarray
        ) -> tuple:

    range = 65535 if numpy_max(image) > 256 else 255
    normalized_image = image / range

    return normalized_image, range

def preprocess_image(
        image: numpy_ndarray, 
        ) -> numpy_ndarray:
        
    image_transposed = numpy_transpose(image, (2, 0, 1))
    image_transposed_expanded = numpy_expand_dims(image_transposed, axis=0)

    return image_transposed_expanded
 
def process_image_with_AI_model(
        AI_model: onnxruntime_inferenceSession, 
        image: numpy_ndarray
        ) -> numpy_ndarray:
    
    onnx_input  = {AI_model.get_inputs()[0].name: image}
    onnx_output = AI_model.run(None, onnx_input)[0] 
    output_squeezed = numpy_squeeze(onnx_output, axis=0)
    output_squeezed_clamped = numpy_clip(output_squeezed, 0, 1)
    output_squeezed_clamped_transposed = numpy_transpose(output_squeezed_clamped, (1, 2, 0))

    return output_squeezed_clamped_transposed.astype(float32)

def postprocess_output(
        output_image: numpy_ndarray, 
        max_range: int
        ) -> numpy_ndarray:
    
    match max_range:
        case 255:
            postprocessed_image = (output_image * max_range).astype(uint8)
        case 65535:
            postprocessed_image = (output_image * max_range).round().astype(float32)

    return postprocessed_image
 


# GUI utils ---------------------------

class ScrollableImagesTextFrame(CTkScrollableFrame):

    def __init__(
            self, 
            master,
            selected_file_list, 
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list = selected_file_list
        self._create_widgets()

    def _create_widgets(
            self,
            ) -> None:
        
        self.add_clean_button()

        index_row = 1

        for file_path in self.file_list:

            if check_if_file_is_video(file_path):
                infos, icon = self.extract_video_info(file_path)
            else:
                infos, icon = self.extract_image_info(file_path)
        
            label = CTkLabel(
                self, 
                text       = infos,
                image      = icon, 
                font       = bold11,
                text_color = "#E0E0E0",
                compound   = "left", 
                padx       = 10,
                pady       = 5,
                anchor     = "center"
                )
                            
            label.grid(
                row = index_row, 
                column = 0, 
                pady = (3, 3), 
                padx = (3, 3), 
                sticky = "w"
                )
            
            index_row +=1

    def add_clean_button(
            self
            ) -> None:
        
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

        button.configure(command=lambda: self.clean_all_items())
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        
    def get_selected_file_list(
        self
        ) -> list: 
    
        return self.file_list  

    def clean_all_items(
            self
            ) -> None:
        
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def extract_image_info(
        self,
        image_file: str
        ) -> tuple:
    
        image_name = str(image_file.split("/")[-1])

        image = image_read(image_file)
        height, width = get_image_resolution(image)

        image_info = f"{image_name} • {width}x{height}"
        image_icon = CTkImage(pillow_image_open(image_file), size = (25, 25))

        return image_info, image_icon

    def extract_video_info(
        self,
        video_file: str
        ) -> tuple:
        
        cap          = opencv_VideoCapture(video_file)
        width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
        height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
        num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
        frame_rate   = cap.get(CAP_PROP_FPS)
        duration     = num_frames/frame_rate
        minutes      = int(duration/60)
        seconds      = duration % 60
        video_name   = str(video_file.split("/")[-1])
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False: break
            frame = opencv_cvtColor(frame, COLOR_BGR2RGB)
            video_icon = CTkImage(
                pillow_image_fromarray(frame, mode="RGB"), 
                size = (25, 25)
                )
            break
        cap.release()

        video_infos = f"{video_name} • {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames • {round(frame_rate, 2)}fps"
        
        return video_infos, video_icon
    
class CTkMessageBox(CTkToplevel):

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

def create_info_button(
        command: Callable, 
        text: str
        ) -> CTkButton:
    
    return CTkButton(
        master  = window, 
        command = command,
        text          = text,
        fg_color      = "transparent",
        hover_color   = "#181818",
        text_color    = "#C0C0C0",
        anchor        = "w",
        height        = 23,
        width         = 150,
        corner_radius = 12,
        font          = bold12,
        image         = info_icon
    )

def create_option_menu(
        command: Callable, 
        values: list
    ) -> CTkOptionMenu:
    
    return CTkOptionMenu(
        master  = window, 
        command = command,
        values  = values,
        width              = 150,
        height             = 31,
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

def create_text_box(
        textvariable: StringVar,
        ) -> CTkEntry:
    
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        border_width  = 1,
        corner_radius = 6,
        width         = 150,
        height        = 30,
        font          = bold11,
        justify       = "center",
        fg_color      = "#000000",
        border_color  = "#404040",
        )

def test_callback(a, b, c):
    print("Pippo")



#  Slice functions -------------------
    
def AI_upscale_with_tilling(
        AI_model: onnxruntime_inferenceSession, 
        image: numpy_ndarray,
        half_precision: bool,
        num_tiles_x: int, 
        num_tiles_y: int,
        target_height: int, 
        target_width: int
        ) -> numpy_ndarray:
    
    image_mode = get_image_mode(image)
    tiles_list = split_image_into_tiles(image, num_tiles_x, num_tiles_y)

    for tile_index in range(len(tiles_list)):
        tiles_list[tile_index] = AI_upscale(
            AI_model, 
            half_precision, 
            tiles_list[tile_index]
        )

    upscaled_image = combine_tiles_into_image(
        tiles_list, 
        image_mode, 
        target_height, 
        target_width, 
        num_tiles_x, 
        num_tiles_y
    )

    return upscaled_image

def split_image_into_tiles(
        image: numpy_ndarray, 
        num_tiles_x: int, 
        num_tiles_y: int
        ) -> list[numpy_ndarray]:

    img_height, img_width = get_image_resolution(image)

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

def combine_tiles_into_image(
        tiles: list[numpy_ndarray], 
        image_mode: str,
        image_target_height: int, 
        image_target_width: int,
        num_tiles_x: int, 
        num_tiles_y: int
        ) -> numpy_ndarray:
    
    if image_mode == 'Grayscale':
        tiled_image = numpy_zeros((image_target_height, image_target_width), dtype = uint8)
    else:
        tiled_image = numpy_zeros((image_target_height, image_target_width, 4), dtype = uint8)
    
    for tile_index in range(len(tiles)):
        actual_tile = tiles[tile_index]

        tile_height, tile_width = get_image_resolution(actual_tile)

        row     = tile_index // num_tiles_x
        col     = tile_index % num_tiles_x
        y_start = row * tile_height
        y_end   = y_start + tile_height
        x_start = col * tile_width
        x_end   = x_start + tile_width

        if image_mode == 'Grayscale':
            tiled_image[y_start:y_end, x_start:x_end] = actual_tile
        else:
            tiled_image[y_start:y_end, x_start:x_end] = add_alpha_channel(actual_tile)

    return tiled_image

def add_alpha_channel(
        tile: numpy_ndarray
        ) -> numpy_ndarray:
    
    if tile.shape[2] == 3:
        alpha_channel = numpy_full((tile.shape[0], tile.shape[1], 1), 255, dtype = uint8)
        tile          = numpy_concatenate((tile, alpha_channel), axis = 2)
    return tile

def file_need_tilling(
        file: numpy_ndarray, 
        max_tiles_resolution: int
        ) -> tuple:
    
    height, width = get_image_resolution(file)
    image_pixels = height * width
    max_supported_pixels = max_tiles_resolution * max_tiles_resolution

    if image_pixels > max_supported_pixels:
        num_tiles_horizontal = (width + max_tiles_resolution - 1) // max_tiles_resolution
        num_tiles_vertical   = (height + max_tiles_resolution - 1) // max_tiles_resolution
        return True, num_tiles_horizontal, num_tiles_vertical
    else:
        return False, None, None



# File Utils functions ------------------------

def find_by_relative_path(
        relative_path: str
        ) -> str:
    
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)

def remove_file(
        file_name: str
        ) -> None: 

    if os_path_exists(file_name): 
        os_remove(file_name)

def remove_dir(
        name_dir: str
        ) -> None:
    
    if os_path_exists(name_dir): 
        remove_directory(name_dir)

def create_dir(
        name_dir: str
        ) -> None:
    
    if os_path_exists(name_dir): remove_directory(name_dir)
    if not os_path_exists(name_dir): os_makedirs(name_dir, mode=0o777)

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

def image_write_async(
        file_path: str, 
        file_data: numpy_ndarray
    ) -> None: 

    image_write(
        file_path = file_path, 
        file_data = file_data
    )

def write_multiple_images_async(
        image_list: list[numpy_ndarray],
        image_paths: list[str],
        cpu_number: int
    ) -> None:

    pool = ThreadPool(cpu_number)
    pool.starmap(image_write_async, zip(image_paths, image_list))
    pool.close()
    pool.join()



# Image/video Utils functions ------------------------

def get_image_resolution(
        image: numpy_ndarray
        ) -> tuple:
    
    height = image.shape[0]
    width  = image.shape[1]

    return height, width 

def resize_image(
        file: numpy_ndarray, 
        resize_factor: int
        ) -> numpy_ndarray:
    
    old_height, old_width = get_image_resolution(file)

    new_width  = int(old_width * resize_factor)
    new_height = int(old_height * resize_factor)

    match resize_factor:
        case factor if factor > 1:
            return opencv_resize(file, (new_width, new_height), interpolation=INTER_CUBIC)
        case factor if factor < 1:
            return opencv_resize(file, (new_width, new_height), interpolation=INTER_LINEAR)
        case _:
            return file

def get_image_mode(
        image: numpy_ndarray
        ) -> str:
    
    match image.shape:
        case (rows, cols):
            return 'Grayscale'
        case (rows, cols, 3):
            return 'RGB'
        case (rows, cols, 4):
            return 'RGBA'

def extract_video_fps(
        video_path: str
        ) -> float:
    
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate
   
def extract_video_frames_and_audio(
        target_directory: str,
        video_path: str, 
        cpu_number: int
    ) -> tuple[list[str], str]:

    create_dir(target_directory)

    # Audio extraction
    with VideoFileClip(video_path) as video_file_clip:
        try: 
            audio_path = f"{target_directory}{os_separator}audio.mp3"
            video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
        except:
            pass

    # Video frame extraction
    frames_for_cpu = 25
    frames_number_to_save = cpu_number * frames_for_cpu

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
                write_multiple_images_async(
                    image_list = frames_to_save,
                    image_paths = frames_path_to_save,
                    cpu_number = cpu_number
                )
                frames_to_save      = []
                frames_path_to_save = []

    video_capture.release()

    if len(frames_to_save) > 0:
        write_multiple_images_async(
            image_list = frames_to_save,
            image_paths = frames_path_to_save,
            cpu_number = cpu_number
        )
    
    return video_frames_list, audio_path

def video_reconstruction_by_frames(
        video_path: str,
        audio_path: str,
        frames_upscaled_list: list[str], 
        selected_AI_model: str, 
        resize_factor: int, 
        cpu_number: int,
        selected_video_extension: str, 
        selected_interpolation_factor: float
        ) -> str:
        
    frame_rate = extract_video_fps(video_path)

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

    upscaled_video_path = prepare_output_video_filename(video_path, selected_AI_model, resize_factor, selected_video_extension, selected_interpolation_factor)

    clip = ImageSequenceClip.ImageSequenceClip(
        sequence = frames_upscaled_list, 
        fps = frame_rate
    )
    if os_path_exists(audio_path):
        clip.write_videofile(
            upscaled_video_path,
            fps     = frame_rate,
            audio   = audio_path,
            codec   = codec,
            bitrate = '16M',
            verbose = False,
            logger  = None,
            #ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
            threads = cpu_number
        )
    else:
        clip.write_videofile(
            upscaled_video_path,
            fps     = frame_rate,
            codec   = codec,
            bitrate = '16M',
            verbose = False,
            logger  = None,
            #ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
            threads = cpu_number
        ) 
        
    return upscaled_video_path
        
def interpolate_images_and_save(
        target_path: str,
        image1: numpy_ndarray,
        image2: numpy_ndarray,
        image1_importance: float,
        ) -> None:
    
    # image1 = original image
    # image2 = image produced by AI

    image2_importance = 1 - image1_importance
    target_height, target_width = get_image_resolution(image = image2)

    try: 
        image1 = opencv_resize(
            image1, 
            (target_width, target_height), 
            interpolation = INTER_CUBIC
        )
        
        image1 = add_alpha_channel(image1)
        image2 = add_alpha_channel(image2)
        interpolated_image = opencv_addWeighted(image1, image1_importance, image2, image2_importance, 0)
        image_write(
            file_path = target_path, 
            file_data = interpolated_image
        )
    except:
        image_write(
            file_path = target_path, 
            file_data = image2
        )

def interpolate_images_and_save_async(
        target_path: str,
        image1: numpy_ndarray,
        image2: numpy_ndarray,
        image1_importance: float,
        ) -> None:
    
    # image1 = original image
    # image2 = image produced by AI

    interpolate_images_and_save(
        target_path = target_path,
        image1 = image1,
        image2 = image2,
        image1_importance = image1_importance
    )

def get_upscaled_image_shape(
        image: numpy_ndarray, 
        upscaling_factor: int
        ) -> tuple[int, int]:
    
    image_to_upscale_height, image_to_upscale_width = get_image_resolution(image)

    target_height = image_to_upscale_height * upscaling_factor
    target_width  = image_to_upscale_width  * upscaling_factor
    
    return target_height, target_width

def get_video_info_for_upscaling(
        first_frame: str, 
        resize_factor: int, 
        upscaling_factor: int, 
        tiles_resolution: int
        ) -> tuple:
    
    first_frame = image_read(file_path = first_frame)
    first_frame_resized = resize_image(first_frame, resize_factor)

    # Tilling?
    need_tiles, num_tiles_x, num_tiles_y = file_need_tilling(first_frame_resized, tiles_resolution)
 
    # Resized resolution
    resized_height, resized_width = get_image_resolution(first_frame_resized)

    # Upscaling resolution
    frame_target_width  = resized_width * upscaling_factor
    frame_target_height = resized_height * upscaling_factor
        
    return frame_target_height, frame_target_width, need_tiles, num_tiles_x, num_tiles_y

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

def update_process_status_videos(
        processing_queue: multiprocessing_Queue, 
        file_number: int, 
        frame_index: int, 
        how_many_frames: int,
        average_processing_time: float,
        ) -> None:
    
    if frame_index != 0 and (frame_index + 1) % 4 == 0:    
        percent_complete = (frame_index + 1) / how_many_frames * 100 

        time_left = calculate_time_to_complete_video(
            time_for_frame   = average_processing_time,
            remaining_frames = how_many_frames - frame_index
        )
    
        write_process_status(processing_queue, f"{file_number}. Upscaling video {percent_complete:.2f}% ({time_left})")

def copy_file_metadata(
        original_file_path: str, 
        upscaled_file_path: str
        ) -> None:
    
    exiftool = "exiftool_12.70.exe"
    exiftool_path  = find_by_relative_path(f"Assets{os_separator}{exiftool}")

    exiftool_cmd = [
                    exiftool_path, 
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
                info_message.set(f"Error during upscale process :(")
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
    global selected_interpolation
    global selected_interpolation_factor
    global selected_half_precision
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
        print(f"  Selected AI model: {selected_AI_model}")
        print(f"  AI half precision: {selected_half_precision}")
        print(f"  Interpolation: {selected_interpolation}")
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
                selected_AI_model, 
                selected_image_extension,
                tiles_resolution, 
                resize_factor, 
                cpu_number, 
                selected_half_precision, 
                selected_video_extension,
                selected_interpolation, 
                selected_interpolation_factor
            )
        )
        process_upscale_orchestrator.start()

        thread_wait = Thread(
            target = check_upscale_steps
        )
        thread_wait.start()

def prepare_output_image_filename(
        image_path: str, 
        selected_AI_model: str, 
        resize_factor: int, 
        selected_image_extension: str,
        selected_interpolation_factor: float
        ) -> str:
    
    result_path, _ = os_path_splitext(image_path)

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
        
    result_path += to_append

    return result_path

def prepare_output_video_filename(
        video_path: str, 
        selected_AI_model: str, 
        resize_factor: int, 
        selected_video_extension: str,
        selected_interpolation_factor: float
        ) -> str:
    
    result_path, _ = os_path_splitext(video_path)

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
        
    result_path += to_append

    return result_path

def prepare_output_video_frames_directory_name(
        video_path: str, 
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
    
    result_path, _ = os_path_splitext(video_path)

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

    result_path += to_append

    return result_path

def upscale_image(
        processing_queue: multiprocessing_Queue,
        image_path: str, 
        file_number: int,
        AI_model: onnxruntime_inferenceSession, 
        selected_AI_model: str, 
        upscaling_factor: int,
        selected_image_extension: str,
        tiles_resolution: int, 
        resize_factor: int, 
        selected_half_precision: bool,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    target_image_path = prepare_output_image_filename(
        image_path        = image_path, 
        selected_AI_model = selected_AI_model, 
        resize_factor     = resize_factor, 
        selected_image_extension      = selected_image_extension, 
        selected_interpolation_factor = selected_interpolation_factor
    )
        
    starting_image   = image_read(file_path = image_path)
    image_to_upscale = resize_image(starting_image, resize_factor)

    target_height, target_width = get_upscaled_image_shape(
        image = image_to_upscale, 
        upscaling_factor = upscaling_factor
    )
    need_tiles, num_tiles_x, num_tiles_y = file_need_tilling(
        file = image_to_upscale, 
        max_tiles_resolution = tiles_resolution
    )

    write_process_status(processing_queue, f"{file_number}. Upscaling image")

    # Upscale image w/out tilling
    if need_tiles == False:
        upscaled_image = AI_upscale(
            AI_model       = AI_model, 
            half_precision = selected_half_precision, 
            image          = image_to_upscale
        )

    # Upscale image with tilling
    else:
        upscaled_image = AI_upscale_with_tilling(
            AI_model = AI_model, 
            image    = image_to_upscale,
            half_precision = selected_half_precision,  
            num_tiles_x    = num_tiles_x, 
            num_tiles_y    = num_tiles_y, 
            target_height  = target_height, 
            target_width   = target_width
        )

    # Interpolation
    if selected_interpolation:
        interpolate_images_and_save(
            target_path = target_image_path,
            image1 = starting_image,
            image2 = upscaled_image,
            image1_importance = selected_interpolation_factor
        )

    else:
        image_write(
            file_path = target_image_path,
            file_data = upscaled_image
        )

    # Metadata copy from original image
    copy_file_metadata(image_path, target_image_path)

def upscale_video(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        AI_model: onnxruntime_inferenceSession, 
        selected_AI_model: str, 
        upscaling_factor: int,
        tiles_resolution: int,
        resize_factor: int, 
        cpu_number: int, 
        selected_half_precision: bool, 
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    # Directory for video frames and audio
    target_directory = prepare_output_video_frames_directory_name(video_path, selected_AI_model, resize_factor, selected_interpolation_factor)

    # Extract video frames and audio
    write_process_status(processing_queue, f"{file_number}. Extracting video frames ({cpu_number} cpus)")
    frame_list_paths, audio_path = extract_video_frames_and_audio(
        target_directory = target_directory, 
        video_path = video_path,
        cpu_number = cpu_number
    )

    target_height, target_width, need_tiles, num_tiles_x, num_tiles_y = get_video_info_for_upscaling(
        first_frame   = frame_list_paths[0], 
        resize_factor = resize_factor, 
        upscaling_factor = upscaling_factor, 
        tiles_resolution = tiles_resolution
    ) 

    write_process_status(processing_queue, f"{file_number}. Upscaling video")  
    how_many_frames = len(frame_list_paths)
    frame_processing_times = []
    
    for frame_index in range(how_many_frames):
        start_timer = timer()

        frame_path       = frame_list_paths[frame_index]
        starting_frame   = image_read(file_path = frame_path)
        frame_to_upscale = resize_image(starting_frame, resize_factor)

        # Upscale frame w/out tilling
        if need_tiles == False:
            upscaled_frame = AI_upscale(
                AI_model       = AI_model, 
                half_precision = selected_half_precision, 
                image          = frame_to_upscale
        )

        # Upscale frame with tilling
        else:
            upscaled_frame = AI_upscale_with_tilling(
                AI_model = AI_model, 
                image    = frame_to_upscale,
                half_precision = selected_half_precision,  
                num_tiles_x    = num_tiles_x, 
                num_tiles_y    = num_tiles_y, 
                target_height  = target_height, 
                target_width   = target_width
            )

        # Interpolation
        if selected_interpolation:
            interpolate_thread = Thread(
                target = interpolate_images_and_save_async,
                args = (
                    frame_path, 
                    frame_to_upscale,
                    upscaled_frame,
                    selected_interpolation_factor
                    )
            )
            interpolate_thread.start()

        else:
            # Save frame overwriting existing frame
            write_thread = Thread(
                target = image_write_async, 
                args = (
                    frame_path, 
                    upscaled_frame
                )
            )
            write_thread.start()

        # Calculate processing time for each frame
        end_timer = timer()
        frame_processing_times.append(end_timer - start_timer)
        average_processing_time = float(numpy_mean(frame_processing_times))
    
        # Update process status every 4 frames
        update_process_status_videos(
            processing_queue = processing_queue, 
            file_number = file_number,  
            frame_index = frame_index, 
            how_many_frames = how_many_frames,
            average_processing_time = average_processing_time
        )

    # Upscaled video reconstuction
    write_process_status(processing_queue, f"{file_number}. Processing upscaled video")
    upscaled_video_path = video_reconstruction_by_frames(video_path, audio_path, frame_list_paths, selected_AI_model, resize_factor, cpu_number, selected_video_extension, selected_interpolation_factor)
    
    # Metadata copy from original video to upscaled video
    copy_file_metadata(video_path, upscaled_video_path)

    # Remove upscaled frames directory after video reconstruction
    remove_dir(target_directory)

def upscale_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_AI_model: str,
        selected_image_extension: str,
        tiles_resolution: int,
        resize_factor: int,
        cpu_number: int,
        selected_half_precision: bool,
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
        
    if   'x2' in selected_AI_model: upscaling_factor = 2
    elif 'x4' in selected_AI_model: upscaling_factor = 4

    try:
        write_process_status(processing_queue, f"Loading AI model")
        AI_model = load_AI_model(selected_AI_model, selected_half_precision)

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(
                    processing_queue,
                    file_path, 
                    file_number, 
                    AI_model, 
                    selected_AI_model,
                    upscaling_factor,
                    tiles_resolution,
                    resize_factor, 
                    cpu_number, 
                    selected_half_precision,
                    selected_video_extension, 
                    selected_interpolation, 
                    selected_interpolation_factor
                    )
            else:
                upscale_image(
                    processing_queue,
                    file_path, 
                    file_number,
                    AI_model, 
                    selected_AI_model, 
                    upscaling_factor, 
                    selected_image_extension, 
                    tiles_resolution, 
                    resize_factor, 
                    selected_half_precision, 
                    selected_interpolation,
                    selected_interpolation_factor
                    )

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS}{str(exception)}")



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
    global selected_half_precision
    global selected_image_extension
    global tiles_resolution
    global resize_factor
    global cpu_number

    is_ready = True

    # Selected files 
    try: selected_file_list = scrollable_frame_file_list.get_selected_file_list()
    except:
        info_message.set("Please select a file")
        is_ready = False

    if len(selected_file_list) <= 0:
        info_message.set("Please select a file")
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
        if selected_AI_model in RRDB_models_list:          
            vram_multiplier = RRDB_vram_multiplier
        elif selected_AI_model in SRVGGNetCompact_models_list: 
            vram_multiplier = SRVGGNetCompact_vram_multiplier

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))

        if selected_half_precision == True: 
            tiles_resolution = int(selected_vram * 100)

        elif selected_half_precision == False: 
            tiles_resolution = int(selected_vram * 100 * full_precision_vram_multiplier)
        
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

def show_error_message(
        exception: str
        ) -> None:
    
    messageBox_title = "Upscale error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text  = f"\n {str(exception)} \n"

    CTkMessageBox(messageType = "error", 
                title = messageBox_title, 
                subtitle = messageBox_subtitle,
                default_value = None,
                option_list = [messageBox_text])



# GUI select from menus functions ---------------------------

def select_AI_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_AI_model    
    selected_AI_model = selected_option

def select_AI_mode_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_half_precision
    match selected_option:
        case "Full precision":
            selected_half_precision = False
        case "Half precision":
            selected_half_precision = True

def select_image_extension_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_image_extension   
    selected_image_extension = selected_option

def select_video_extension_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_video_extension   
    selected_video_extension = selected_option

def select_interpolation_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_interpolation
    global selected_interpolation_factor

    if selected_option == "Disabled": 
        selected_interpolation = False
        selected_interpolation_factor = None
    else:
        selected_interpolation = True
        match selected_option:
            case "Low":
                selected_interpolation_factor = 0.3
            case "Medium":
                selected_interpolation_factor = 0.5
            case "High":
                selected_interpolation_factor = 0.7



# GUI info functions ---------------------------

def open_info_AI_model():
    option_list = [
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

        #"\n SAFM (2023) - Slow but lightweight AI architecture\n" + 
        #" High-quality upscale\n" +
        #" Recommended for image upscaling\n" + 
        #" Does not support Half-precision\n" +
        #"  • SAFMNLx4\n" + 
        #"  • SAFMNLx4_Real\n",
    ]

    CTkMessageBox(messageType = "info",
                    title = "AI model", 
                    subtitle = "This widget allows to choose between different AI models for upscaling",
                    default_value = "RealESR_Gx4",
                    option_list = option_list)

def open_info_AI_precision():
    option_list = [
        " \n HALF PRECISION\n" +
        "  • old GPUs are not compatible with this mode\n",

        " \n FULL PRECISION\n" +
        "  • compatible with all GPUs\n",

        " \n NOTES\n" +
        "  • both modes can offer a boost to performance\n" + 
        "  • this depends on the characteristics of the PC (GPU architecture, CPU power and others)\n" +
        "  • and on the characteristics of the AI-model selected\n" +
        "  • I recommend trying both modes and finding the fastest one for your PC \n"

    ]

    CTkMessageBox(messageType = "info",
                    title = "AI precision", 
                    subtitle = "This widget allows to choose the AI upscaling precision",
                    default_value = "Half precision",
                    option_list = option_list)

def open_info_interpolation():
    option_list = [
        " Interpolation is the fusion of the upscaled image produced by AI and the original image",
        " \n Increase the quality of the final result\n  • especially when using the tilling/merging function (with low VRAM) \n  • especially at low Input resolution % values (<50%)\n",
        " \n Levels of interpolation\n  • Disabled - 100% upscaled\n  • Low - 30% original / 70% upscaled\n  • Medium - 50% original / 50% upscaled\n  • High - 70% original / 30% upscaled\n"
    ]

    CTkMessageBox(messageType = "info",
                    title = "Interpolation", 
                    subtitle = "This widget allows to choose interpolation between upscaled and original image/frame",
                    default_value = "Low",
                    option_list = option_list)

def open_info_AI_output():
    option_list = [
        " \n PNG\n  • very good quality\n  • slow and heavy file\n  • supports transparent images\n",
        " \n JPG\n  • good quality\n  • fast and lightweight file\n",
        " \n BMP\n  • highest quality\n  • slow and heavy file\n",
        " \n TIFF\n  • highest quality\n  • very slow and heavy file\n",
    ]

    CTkMessageBox(messageType = "info",
                    title = "Image output", 
                    subtitle = "This widget allows to choose the extension of upscaled images",
                    default_value = ".png",
                    option_list = option_list)

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

    CTkMessageBox(messageType = "info",
                title = "Video output", 
                subtitle = "This widget allows to choose the extension of the upscaled video",
                default_value = ".mp4 (x264)",
                option_list = option_list)

def open_info_gpu():
    option_list = [
        "Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be",
        "For optimal results, it is essential to regularly update your GPU drivers"
    ]

    CTkMessageBox(messageType = "info",
                    title = "GPU", 
                    subtitle = "This widget allows to select the GPU for AI processing",
                    default_value = None,
                    option_list = option_list)

def open_info_vram_limiter():
    option_list = [
        " It is important to enter the correct value according to the VRAM memory of the chosen GPU",
        " Selecting a value greater than the actual amount of GPU VRAM may result in upscale failure",
        " For integrated GPUs (Intel-HD series • Vega 3,5,7) - select 2 GB",
    ]

    CTkMessageBox(messageType = "info",
                    title = "GPU Vram (GB)",
                    subtitle = "This widget allows to set a limit on the GPU VRAM memory usage",
                    default_value = "8 GB",
                    option_list = option_list)

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

    CTkMessageBox(messageType = "info",
                    title = "Input resolution %", 
                    subtitle = "This widget allows to choose the resolution input to the AI",
                    default_value = "60%",
                    option_list = option_list)

def open_info_cpu():
    option_list = [
        " When possible the app will use the number of cpus selected",
        " Currently this value is only used for the video encoding phase",
    ]

    default_cpus = str(int(os_cpu_count()/2))

    CTkMessageBox(messageType = "info",
                    title = "Cpu number",
                    subtitle = "This widget allows to choose how many cpus to devote to the app",
                    default_value = default_cpus,
                    option_list = option_list)



# GUI place functions ---------------------------

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        global scrollable_frame_file_list
        scrollable_frame_file_list = ScrollableImagesTextFrame(
            master = window, 
            selected_file_list = supported_files_list,
            fg_color = dark_color, 
            bg_color = dark_color
        )
        
        scrollable_frame_file_list.place(
            relx = 0.0, 
            rely = 0.0, 
            relwidth  = 1.0, 
            relheight = 0.45
        )
        
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")

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
    up_background = CTkLabel(master  = window, 
                        text     = "",
                        fg_color = dark_color,
                        font     = bold12,
                        anchor   = "w")
    
    up_background.place(relx = 0.0, rely = 0.0, relwidth  = 1.0, relheight = 0.45)

    text_drop = """ SUPPORTED FILES

IMAGES • jpg png tif bmp webp heic
VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp """

    input_file_text = CTkLabel(master = window, 
                        text       = text_drop,
                        fg_color   = dark_color,
                        bg_color   = dark_color,
                        text_color = "#C0C0C0",
                        width      = 300,
                        height     = 150,
                        font       = bold12,
                        anchor     = "center")
    
    input_file_button = CTkButton(master = window,
                            command  = open_files_action, 
                            text     = "SELECT FILES",
                            width    = 140,
                            height   = 30,
                            font     = bold11,
                            border_width = 1,
                            fg_color     = "#282828",
                            text_color   = "#E0E0E0",
                            border_color = "#0096FF")

    input_file_text.place(relx = 0.5, rely = 0.20,  anchor = "center")
    input_file_button.place(relx = 0.5, rely = 0.35, anchor = "center")

def place_app_name():
    app_name_label = CTkLabel(master     = window, 
                              text       = app_name + " " + version,
                              text_color = app_name_color,
                              font       = bold20,
                              anchor     = "w")
    
    app_name_label.place(relx = column0_x, rely = row0_y - 0.025, anchor = "center")

def place_AI_menu():
    AI_menu_button = create_info_button(open_info_AI_model, "AI model")
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.053, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_AI_mode_menu():
    AI_mode_button = create_info_button(open_info_AI_precision, "AI precision")
    AI_mode_menu   = create_option_menu(select_AI_mode_from_menu, AI_precision_list)
    
    AI_mode_button.place(relx = column0_x, rely = row2_y - 0.053, anchor = "center")
    AI_mode_menu.place(relx = column0_x, rely = row2_y, anchor = "center")

def place_interpolation_menu():
    interpolation_button = create_info_button(open_info_interpolation, "Interpolation")
    interpolation_menu   = create_option_menu(select_interpolation_from_menu, interpolation_list)
    
    interpolation_button.place(relx = column0_x, rely = row3_y - 0.053, anchor = "center")
    interpolation_menu.place(relx = column0_x, rely  = row3_y, anchor = "center")

def place_AI_output_menu():
    file_extension_button = create_info_button(open_info_AI_output, "Image output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column1_x, rely = row0_y - 0.053, anchor = "center")
    file_extension_menu.place(relx = column1_x, rely = row0_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    video_extension_menu.place(relx = column1_x, rely = row1_y, anchor = "center")

def place_vram_textbox():
    vram_button  = create_info_button(open_info_vram_limiter, "GPU Vram (GB)")
    vram_textbox = create_text_box(selected_VRAM_limiter) 
  
    vram_button.place(relx = column1_x, rely = row2_y - 0.053, anchor = "center")
    vram_textbox.place(relx = column1_x, rely  = row2_y, anchor = "center")

def place_input_resolution_textbox():
    resize_factor_button  = create_info_button(open_info_input_resolution, "Input resolution %")
    resize_factor_textbox = create_text_box(selected_resize_factor) 

    resize_factor_button.place(relx = column2_x, rely = row0_y - 0.053, anchor = "center")
    resize_factor_textbox.place(relx = column2_x, rely = row0_y, anchor = "center")

def place_cpu_textbox():
    cpu_button  = create_info_button(open_info_cpu, "CPU number")
    cpu_textbox = create_text_box(selected_cpu_number)

    cpu_button.place(relx = column2_x, rely = row1_y - 0.053, anchor = "center")
    cpu_textbox.place(relx = column2_x, rely  = row1_y, anchor = "center")

def place_message_label():
    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold11,
                            fg_color     = "#ffbf00",
                            text_color   = "#000000",
                            anchor       = "center",
                            corner_radius = 12)
    message_label.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_stop_button(): 
    stop_button = CTkButton(master     = window,
                            command    = stop_button_command, 
                            image      = stop_icon,
                            text       = "STOP",
                            width      = 140,
                            height     = 30,
                            font       = bold11,
                            border_width = 1,
                            fg_color     = "#282828",
                            text_color   = "#E0E0E0",
                            border_color = "#EC1D1D")
    stop_button.place(relx = column2_x, rely = row3_y, anchor = "center")

def place_upscale_button(): 
    upscale_button = CTkButton(master      = window, 
                                command    = upscale_button_command,
                                text       = "UPSCALE",
                                image      = upscale_icon,
                                width      = 140,
                                height     = 30,
                                font       = bold11,
                                border_width = 1,
                                fg_color     = "#282828",
                                text_color   = "#E0E0E0",
                                border_color = "#0096FF")
    upscale_button.place(relx = column2_x, rely = row3_y, anchor = "center")
   


# Main functions ---------------------------

def on_app_close() -> None:
    window.grab_release()
    window.destroy()
    stop_upscale_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        width        = 675
        height       = 675
        window.geometry("675x675")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_app_name()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_AI_mode_menu()
        place_interpolation_menu()

        place_AI_output_menu()
        place_video_extension_menu()
        place_vram_textbox()
        
        place_input_resolution_textbox()
        place_cpu_textbox()
        place_message_label()
        place_upscale_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing_freeze_support()

    processing_queue: multiprocessing_Queue = multiprocessing_Queue(maxsize=1)

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    info_message            = StringVar()
    selected_resize_factor  = StringVar()
    selected_VRAM_limiter   = StringVar()
    selected_cpu_number     = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_half_precision
    global selected_image_extension
    global selected_video_extension
    global selected_interpolation
    global selected_interpolation_factor
    global tiles_resolution
    global resize_factor
    global cpu_number

    selected_file_list: list[str] = []

    selected_AI_model        = default_AI_model
    selected_image_extension = default_image_extension
    selected_video_extension = default_video_extension

    selected_half_precision = True if default_AI_precision == "Half precision" else False

    if default_interpolation == "Disabled": 
        selected_interpolation = False
        selected_interpolation_factor = None
    else:
        selected_interpolation_factor_map = {
            "Low": 0.3,
            "Medium": 0.5,
            "High": 0.7
        }
        selected_interpolation = True
        selected_interpolation_factor = selected_interpolation_factor_map.get(default_interpolation)

    selected_resize_factor.set(default_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_cpu_number.set(default_cpu_number)

    info_message.set("Hi :)")
    selected_resize_factor.trace_add('write', test_callback)

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
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(16, 16))

    app = App(window)
    window.update()
    window.mainloop()