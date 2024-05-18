
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

from os import (
    sep       as os_separator,
    devnull   as os_devnull,
    environ   as os_environ,
    cpu_count as os_cpu_count,
    makedirs  as os_makedirs,
)

from os.path import (
    basename as os_path_basename,
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

from moviepy.editor   import VideoFileClip 
from moviepy.video.io import ImageSequenceClip 

from onnx                 import load as onnx_load 
from onnxruntime          import InferenceSession as onnxruntime_inferenceSession
from onnxconverter_common import convert_float_to_float16

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
    set_default_color_theme,
)



if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)



githubme   = "https://github.com/Djdefrag/QualityScaler"
telegramme = "https://linktr.ee/j3ngystudio"

app_name = "QualityScaler"
version  = "3.5"

app_name_color = "#DA70D6"
dark_color     = "#080808"

very_low_VRAM = 4
low_VRAM      = 3
medium_VRAM   = 2.2
high_VRAM     = 0.7
full_precision_vram_multiplier = 0.7

IRCNN_models_list           = [ 'IRCNNx1' ]
SRVGGNetCompact_models_list = [ 'RealESR_Gx4', 'RealSRx4_Anime' ]
RRDB_models_list            = [ 'BSRGANx4', 'BSRGANx2', 'RealESRGANx4' ]

AI_models_list         = ( IRCNN_models_list + SRVGGNetCompact_models_list + RRDB_models_list )
gpus_list              = [ 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4' ]
image_extension_list   = [ '.png', '.jpg', '.bmp', '.tiff' ]
video_extension_list   = [ '.mp4 (x264)', '.mp4 (x265)', '.avi' ]
interpolation_list     = [ 'Low', 'Medium', 'High', 'Disabled' ]
AI_precision_list      = [ 'Half precision', 'Full precision' ]
AI_multithreading_list = [ 'Disabled', '2 threads', '3 threads', '4 threads']

default_AI_model          = AI_models_list[0]
default_gpu               = gpus_list[0]
default_image_extension   = image_extension_list[0]
default_video_extension   = video_extension_list[0]
default_interpolation     = interpolation_list[0]
default_AI_precision      = AI_precision_list[0]
default_AI_multithreading = AI_multithreading_list[0]
default_output_path       = "Same path as input files"
default_resize_factor     = str(50)
default_VRAM_limiter      = str(8)
default_cpu_number        = str(int(os_cpu_count()/2))

FFMPEG_EXE_PATH   = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH = find_by_relative_path(f"Assets{os_separator}exiftool.exe")
FRAMES_FOR_CPU    = 30

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

def load_AI_model(
        selected_AI_model: str, 
        selected_gpu: str,
        selected_half_precision: bool
    ) -> onnxruntime_inferenceSession:

    AI_model_path   = find_by_relative_path(f"AI-onnx{os_separator}{selected_AI_model}.onnx")
    AI_model_loaded = onnx_load(AI_model_path)

    match selected_gpu:
        case 'GPU 1':
            backend = [('DmlExecutionProvider', {"device_id": "0"})]
        case 'GPU 2':
            backend = [('DmlExecutionProvider', {"device_id": "1"})]
        case 'GPU 3':
            backend = [('DmlExecutionProvider', {"device_id": "2"})]
        case 'GPU 4':
            backend = [('DmlExecutionProvider', {"device_id": "3"})]
    
    if selected_half_precision:
        AI_model_loaded = convert_float_to_float16(
            model = AI_model_loaded,
            keep_io_types = True
        )
    
    AI_model = onnxruntime_inferenceSession(
        path_or_bytes = AI_model_loaded.SerializeToString(), 
        providers     = backend
    )    

    return AI_model

def AI_upscale(
        AI_model: onnxruntime_inferenceSession, 
        image: numpy_ndarray
        ) -> numpy_ndarray:
    
    image        = numpy_ascontiguousarray(image)
    image_mode   = get_image_mode(image)
    image, range = normalize_image(image)
    image        = image.astype(float32)

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

def get_image_mode(image: numpy_ndarray) -> str:
    match image.shape:
        case (rows, cols):
            return "Grayscale"
        case (rows, cols, channels) if channels == 3:
            return "RGB"
        case (rows, cols, channels) if channels == 4:
            return "RGBA"

def normalize_image(image: numpy_ndarray) -> tuple:
    range            = 65535 if numpy_max(image) > 256 else 255
    normalized_image = image / range

    return normalized_image, range

def preprocess_image(image: numpy_ndarray) -> numpy_ndarray:
    image_transposed          = numpy_transpose(image, (2, 0, 1))
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
            return (output_image * max_range).astype(uint8)
        case 65535:
            return (output_image * max_range).round().astype(float32)
 


# GUI utils ---------------------------

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

class ScrollableImagesTextFrame_upscaler(CTkScrollableFrame):

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

    def clean_file_list(self) -> None:
        for label in self.label_list:
            label.grid_forget()

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
        label.grid(row = index_row, column = 0, 
                   pady = (3, 3), padx = (3, 3), 
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
        
    def get_selected_file_list(self) -> list: 
        return self.file_list  

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_resize_factor(self, resize_factor) -> None:
        self.resize_factor = resize_factor

    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 50

        if check_if_file_is_video(file_path):
            cap = opencv_VideoCapture(file_path)
            _, frame = cap.read()
            frame = opencv_cvtColor(frame, COLOR_BGR2RGB)
            ratio = min(max_size / frame.shape[0], max_size / frame.shape[1])
            icon = CTkImage(
                pillow_image_fromarray(frame, mode="RGB"), 
                size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
            )
            cap.release()
            return icon
        else:
            image = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)
            ratio = min(max_size / image.shape[0], max_size / image.shape[1])
            icon = CTkImage(
                pillow_image_fromarray(image, mode="RGB"), 
                size = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))
            )
            return icon

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
            
            file_icon = self.extract_file_icon(file_path)

            if self.resize_factor == 0:
                file_infos = (f"{video_name}\n"
                              f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height}\n")
            else:
                resized_height = int(height * (self.resize_factor/100))
                resized_width  = int(width * (self.resize_factor/100))

                upscaled_height = int(resized_height * self.upscale_factor)
                upscaled_width  = int(resized_width * self.upscale_factor)

                file_infos = (f"{video_name}\n"
                              f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height}\n"
                              f"AI input ({self.resize_factor}%) {resized_width}x{resized_height} ➜ {upscaled_width}x{upscaled_height}")

        else:
            image_name    = str(file_path.split("/")[-1])
            height, width = get_image_resolution(image_read(file_path))

            file_icon = self.extract_file_icon(file_path)

            if self.resize_factor == 0:
                file_infos = (f"{image_name}\n"
                              f"Resolution {width}x{height}\n")
            else:
                resized_height = int(height * (self.resize_factor/100))
                resized_width  = int(width * (self.resize_factor/100))

                upscaled_height = int(resized_height * self.upscale_factor)
                upscaled_width  = int(resized_width * self.upscale_factor)

                file_infos = (f"{image_name}\n"
                              f"Resolution {width}x{height}\n"
                              f"AI input ({self.resize_factor}%) {resized_width}x{resized_height} ➜ {upscaled_width}x{upscaled_height}")
        
        return file_infos, file_icon

def update_file_widget(a, b, c) -> None:
    try:
        global scrollable_frame_file_list
        scrollable_frame_file_list
    except:
        return
    
    global selected_AI_model
    if   'x1' in selected_AI_model: upscale_factor = 1
    elif 'x2' in selected_AI_model: upscale_factor = 2
    elif 'x4' in selected_AI_model: upscale_factor = 4

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
        values: list) -> CTkOptionMenu:
    
    return CTkOptionMenu(
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

def create_text_box(
        textvariable: StringVar
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
        text_color    = "#C0C0C0",
        fg_color      = "#000000",
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



#  Slice functions -------------------
    
def AI_upscale_with_tilling(
        AI_model: onnxruntime_inferenceSession, 
        image: numpy_ndarray,
        num_tiles_x: int, 
        num_tiles_y: int,
        target_height: int, 
        target_width: int
        ) -> numpy_ndarray:
    
    image_mode = get_image_mode(image)
    tiles_list = split_image_into_tiles(image, num_tiles_x, num_tiles_y)

    upscaled_tiles_list = []
    for tile in tiles_list:
        upscaled_tile = AI_upscale(AI_model, tile)
        upscaled_tiles_list.append(upscaled_tile)

    upscaled_image = combine_tiles_into_image(
        upscaled_tiles_list, 
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

def add_alpha_channel(tile: numpy_ndarray) -> numpy_ndarray:
    if tile.shape[2] == 3:
        alpha_channel = numpy_full((tile.shape[0], tile.shape[1], 1), 255, dtype = uint8)
        tile          = numpy_concatenate((tile, alpha_channel), axis = 2)
    return tile

def file_need_tilling(file: numpy_ndarray, max_tiles_resolution: int) -> bool:
    height, width = get_image_resolution(file)
    image_pixels  = height * width
    max_supported_pixels = max_tiles_resolution * max_tiles_resolution

    if image_pixels > max_supported_pixels:
        return True
    else:
        return False

def calculate_num_tiles(file: numpy_ndarray, max_tiles_resolution: int) -> tuple:
    height, width = get_image_resolution(file)

    num_tiles_horizontal = (width + max_tiles_resolution - 1) // max_tiles_resolution
    num_tiles_vertical   = (height + max_tiles_resolution - 1) // max_tiles_resolution

    return num_tiles_horizontal, num_tiles_vertical



# File Utils functions ------------------------

def remove_dir(name_dir: str) -> None:
    if os_path_exists(name_dir): 
        remove_directory(name_dir)

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
        file_data      = file.read()
        numpy_buffer   = numpy_frombuffer(file_data, uint8)
        opencv_decoded = opencv_imdecode(numpy_buffer, flags)
        return opencv_decoded

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

def resize_image(
        file: numpy_ndarray, 
        resize_factor: int
        ) -> numpy_ndarray:
    
    old_height, old_width = get_image_resolution(file)

    new_width  = int(old_width * resize_factor)
    new_height = int(old_height * resize_factor)

    match resize_factor:
        case factor if factor > 1:
            return opencv_resize(file, (new_width, new_height), interpolation = INTER_LINEAR)
        case factor if factor < 1:
            return opencv_resize(file, (new_width, new_height), interpolation = INTER_LINEAR)
        case _:
            return file

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
    ) -> tuple[list[str], str]:

    create_dir(target_directory)

    # Audio extraction
    with VideoFileClip(video_path) as video_file_clip:
        try: 
            write_process_status(processing_queue, f"{file_number}. Extracting video audio")
            audio_path = f"{target_directory}{os_separator}audio.mp3"
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
    
    return video_frames_list, audio_path

def video_reconstruction_by_frames(
        video_path: str,
        audio_path: str,
        selected_output_path: str,
        frames_upscaled_list: list[str], 
        selected_AI_model: str, 
        resize_factor: int, 
        cpu_number: int,
        selected_video_extension: str, 
        selected_interpolation_factor: float
        ) -> str:
        
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

    output_path = prepare_output_video_filename(video_path, selected_output_path, selected_AI_model, resize_factor, selected_video_extension, selected_interpolation_factor)
    frame_rate  = get_video_fps(video_path)

    clip = ImageSequenceClip.ImageSequenceClip(
        sequence = frames_upscaled_list, 
        fps = frame_rate,
    )
    clip.write_videofile(
        output_path,
        fps     = frame_rate,
        audio   = audio_path if os_path_exists(audio_path) else None,
        codec   = codec,
        bitrate = '16M',
        verbose = False,
        logger  = None,
        threads = cpu_number,
        preset  = "ultrafast"
    )

    return output_path

def interpolate_images_and_save(
        target_path: str,
        image1: numpy_ndarray,
        image2: numpy_ndarray,
        image1_importance: float,
        ) -> None:
    
    # image1 = original image
    # image2 = AI upscaled image

    image2_importance = 1 - image1_importance
    target_height, target_width = get_image_resolution(image = image2)

    try: 
        image1 = opencv_resize(image1,(target_width, target_height))
        
        image1 = add_alpha_channel(image1)
        image2 = add_alpha_channel(image2)
        interpolated_image = opencv_addWeighted(
            image1,
            image1_importance, 
            image2, 
            image2_importance, 
            0
        )
        image_write(
            file_path = target_path, 
            file_data = interpolated_image
        )
    except:
        image_write(
            file_path = target_path, 
            file_data = image2
        )

def get_upscaled_image_shape(
        image: numpy_ndarray, 
        upscale_factor: int
        ) -> tuple[int, int]:
    
    image_to_upscale_height, image_to_upscale_width = get_image_resolution(image)

    target_height = image_to_upscale_height * upscale_factor
    target_width  = image_to_upscale_width  * upscale_factor
    
    return target_height, target_width

def calculate_number_of_frames_supported_by_gpu(
        file: numpy_ndarray, 
        max_tiles_resolution: int
        ) -> int:
    
    height, width = get_image_resolution(file)
    image_pixels = height * width
    max_supported_pixels = max_tiles_resolution * max_tiles_resolution

    frames_simultaneously = max_supported_pixels // image_pixels 
    print(f" Frames supported simultaneously by GPU: {frames_simultaneously}")
    return frames_simultaneously

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

def manage_upscaled_video_frame_save_async(
        upscaled_frame: numpy_ndarray,
        starting_frame: numpy_ndarray,
        upscaled_frame_path: str,
        selected_interpolation: bool,
        selected_interpolation_factor: int
    ) -> None:

    if selected_interpolation:
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
    global selected_gpu
    global selected_AI_multithreading
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
        print(f"  Output path: {(selected_output_path.get())}")
        print(f"  Selected AI model: {selected_AI_model}")
        print(f"  Selected GPU: {selected_gpu}")
        print(f"  AI half precision: {selected_half_precision}")
        print(f"  AI multithreading: {selected_AI_multithreading}")
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
                selected_output_path.get(),
                selected_AI_model, 
                selected_gpu,
                selected_image_extension,
                tiles_resolution, 
                resize_factor, 
                cpu_number, 
                selected_half_precision, 
                selected_video_extension,
                selected_interpolation, 
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
        
    if selected_output_path == default_output_path:
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
    
    if selected_output_path == default_output_path:
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
    
    if selected_output_path == default_output_path:
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
        selected_half_precision: bool,
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float,
        selected_AI_multithreading: int
        ) -> None:
        
    if   'x1' in selected_AI_model: upscale_factor = 1
    elif 'x2' in selected_AI_model: upscale_factor = 2
    elif 'x4' in selected_AI_model: upscale_factor = 4

    write_process_status(processing_queue, f"Loading AI model")
    AI_model = load_AI_model(selected_AI_model, selected_gpu, selected_half_precision)

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
                    AI_model,
                    selected_AI_model,
                    selected_gpu, 
                    selected_half_precision,
                    upscale_factor,
                    tiles_resolution,
                    resize_factor, 
                    cpu_number, 
                    selected_video_extension, 
                    selected_interpolation, 
                    selected_interpolation_factor,
                    selected_AI_multithreading
                )
            else:
                upscale_image(
                    processing_queue,
                    file_path, 
                    file_number,
                    selected_output_path,
                    AI_model,
                    selected_AI_model,
                    upscale_factor, 
                    selected_image_extension, 
                    tiles_resolution, 
                    resize_factor, 
                    selected_interpolation,
                    selected_interpolation_factor
                )

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS}{str(exception)}")

# IMAGES

def upscale_image(
        processing_queue: multiprocessing_Queue,
        image_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_model: onnxruntime_inferenceSession,
        selected_AI_model: str,
        upscale_factor: int,
        selected_image_extension: str,
        tiles_resolution: int, 
        resize_factor: int, 
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    upscaled_image_path = prepare_output_image_filename(image_path, selected_output_path, selected_AI_model, resize_factor, selected_image_extension, selected_interpolation_factor)
    starting_image      = image_read(image_path)
    image_to_upscale    = resize_image(starting_image, resize_factor)
    need_tiles          = file_need_tilling(image_to_upscale, tiles_resolution)
    target_height, target_width = get_upscaled_image_shape(image_to_upscale, upscale_factor)
    if need_tiles:
        num_tiles_x, num_tiles_y = calculate_num_tiles(image_to_upscale, tiles_resolution)

    write_process_status(processing_queue, f"{file_number}. Upscaling image")

    if need_tiles:
        upscaled_image = AI_upscale_with_tilling(
            AI_model, 
            image_to_upscale,
            num_tiles_x, 
            num_tiles_y, 
            target_height, 
            target_width
        )
    else:
        upscaled_image = AI_upscale(AI_model, image_to_upscale)

    if selected_interpolation:
        interpolate_images_and_save(
            target_path = upscaled_image_path,
            image1 = starting_image,
            image2 = upscaled_image,
            image1_importance = selected_interpolation_factor
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
        AI_model: onnxruntime_inferenceSession,
        selected_AI_model: str,
        selected_gpu: str, 
        selected_half_precision: bool,
        upscale_factor: int,
        tiles_resolution: int,
        resize_factor: int, 
        cpu_number: int, 
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float,
        selected_AI_multithreading: int
        ) -> None:

    global processed_frames_async
    global frame_processing_times_async
    processed_frames_async = 0
    frame_processing_times_async = []

    need_tiles  = False
    num_tiles_x = None
    num_tiles_y = None
 
    target_directory = prepare_output_video_frames_directory_name(video_path, selected_output_path, selected_AI_model, resize_factor, selected_interpolation_factor)
    frame_list_paths, audio_path = extract_video_frames_and_audio(processing_queue, file_number, target_directory, video_path, cpu_number)
    first_frame      = resize_image(image_read(frame_list_paths[0]), resize_factor)
    need_tiles       = file_need_tilling(first_frame, tiles_resolution)
    target_height, target_width = get_upscaled_image_shape(first_frame, upscale_factor)

    if need_tiles:
        num_tiles_x, num_tiles_y = calculate_num_tiles(first_frame, tiles_resolution)

    multiframes_number = min(calculate_number_of_frames_supported_by_gpu(first_frame, tiles_resolution), selected_AI_multithreading)

    upscaled_frame_list_paths = [
        prepare_output_video_frame_filename(frame_path, selected_AI_model, resize_factor, selected_interpolation_factor)
        for frame_path in frame_list_paths
    ]

    write_process_status(processing_queue, f"{file_number}. Upscaling video") 
    if need_tiles or multiframes_number == 1:
        upscale_video_frames(
            processing_queue,
            file_number,
            AI_model,
            frame_list_paths,
            upscaled_frame_list_paths,
            need_tiles,
            num_tiles_x,
            num_tiles_y,
            target_height,
            target_width,
            resize_factor,
            selected_interpolation,
            selected_interpolation_factor
        )
        
    else:
        upscale_video_frames_multithreading(
            processing_queue,
            file_number,
            selected_AI_model,
            selected_gpu,
            selected_half_precision,
            frame_list_paths,
            upscaled_frame_list_paths,
            resize_factor,
            multiframes_number,
            selected_interpolation,
            selected_interpolation_factor
        )

    write_process_status(processing_queue, f"{file_number}. Processing upscaled video")
    upscaled_video_path = video_reconstruction_by_frames(
        video_path, 
        audio_path, 
        selected_output_path,
        upscaled_frame_list_paths, 
        selected_AI_model, 
        resize_factor, 
        cpu_number, 
        selected_video_extension, 
        selected_interpolation_factor
    )
    
    copy_file_metadata(video_path, upscaled_video_path)

    remove_dir(target_directory)
    
def upscale_video_frames(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        AI_model: onnxruntime_inferenceSession,
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        need_tiles: bool,
        num_tiles_x: int,
        num_tiles_y: int,
        target_height: int,
        target_width: int,
        resize_factor: int,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    frame_processing_times = []
    how_many_frames = len(frame_list_paths)

    for frame_index, frame_path in enumerate(frame_list_paths):
        start_timer    = timer()
        starting_frame = image_read(frame_path)
        resized_frame  = resize_image(starting_frame, resize_factor)
        if need_tiles:
            upscaled_frame = AI_upscale_with_tilling(AI_model, resized_frame, num_tiles_x, num_tiles_y, target_height, target_width)
        else:
            upscaled_frame = AI_upscale(AI_model, resized_frame)
            
        manage_upscaled_video_frame_save_async(
            upscaled_frame,
            starting_frame,
            upscaled_frame_list_paths[frame_index],
            selected_interpolation,
            selected_interpolation_factor
        )

        frame_processing_times.append(timer() - start_timer)
        if (frame_index + 1) % 8 == 0:
            average_processing_time = numpy_mean(frame_processing_times)
            update_process_status_videos(
                processing_queue,
                file_number,
                frame_index,
                how_many_frames,
                average_processing_time
            )

        if (frame_index + 1) % 100 == 0: frame_processing_times = []

def upscale_video_frames_multithreading(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        selected_AI_model: str, 
        selected_gpu: str, 
        selected_half_precision: bool,
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        resize_factor: int,
        multiframes_number: int,
        selected_interpolation: bool,
        selected_interpolation_factor: float,
        ) -> None:


    # INTERNAL FUNCTION
    
    def upscale_single_video_frame_async(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        multiframes_number: int,
        selected_AI_model: str, 
        selected_gpu: str, 
        selected_half_precision: bool,
        how_many_frames: int,
        frame_list_paths: list[str],
        upscaled_frame_list_paths: list[str],
        resize_factor: int,
        selected_interpolation: bool,
        selected_interpolation_factor: float,
        ) -> None:

        global processed_frames_async
        global frame_processing_times_async

        AI_model = load_AI_model(selected_AI_model, selected_gpu, selected_half_precision)
        
        for frame_index in range(len(frame_list_paths)):
            start_timer = timer()

            starting_frame = image_read(frame_list_paths[frame_index])
            resized_frame  = resize_image(starting_frame, resize_factor)
            upscaled_frame = AI_upscale(AI_model, resized_frame)
            manage_upscaled_video_frame_save_async(
                upscaled_frame,
                starting_frame,
                upscaled_frame_list_paths[frame_index],
                selected_interpolation,
                selected_interpolation_factor
            )

            processed_frames_async +=1
            frame_processing_times_async.append(timer() - start_timer)

            if (processed_frames_async + 1) % 8 == 0:
                average_processing_time = float(numpy_mean(frame_processing_times_async)/multiframes_number)
                update_process_status_videos(
                    processing_queue = processing_queue, 
                    file_number      = file_number,  
                    frame_index      = processed_frames_async, 
                    how_many_frames  = how_many_frames,
                    average_processing_time = average_processing_time
                )

            if (processed_frames_async + 1) % 100 == 0: frame_processing_times_async = []
            
    # -----------------
    
    threads    = multiframes_number
    chunk_size = len(frame_list_paths) // threads

    write_process_status(processing_queue, f"{file_number}. Upscaling video ({threads} threads)")

    frame_list_chunks          = [frame_list_paths[i:i + chunk_size] for i in range(0, len(frame_list_paths), chunk_size)]
    upscaled_frame_list_chunks = [upscaled_frame_list_paths[i:i + chunk_size] for i in range(0, len(upscaled_frame_list_paths), chunk_size)]

    pool = ThreadPool(threads)
    pool.starmap(
        upscale_single_video_frame_async,
        zip(
            repeat(processing_queue),
            repeat(file_number),
            repeat(threads),
            repeat(selected_AI_model),
            repeat(selected_gpu),
            repeat(selected_half_precision),
            repeat(len(frame_list_paths)),
            frame_list_chunks,
            upscaled_frame_list_chunks,
            repeat(resize_factor),
            repeat(selected_interpolation),
            repeat(selected_interpolation_factor)
        )
    )
    pool.close()
    pool.join()



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
            vram_multiplier = high_VRAM
        elif selected_AI_model in SRVGGNetCompact_models_list: 
            vram_multiplier = medium_VRAM
        elif selected_AI_model in IRCNN_models_list:
            vram_multiplier = very_low_VRAM

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

def show_error_message(exception: str) -> None:
    messageBox_title    = "Upscale error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text     = f"\n {str(exception)} \n"

    CTkMessageBox(
        messageType = "error",
        title = messageBox_title,
        subtitle = messageBox_subtitle,
        default_value = None,
        option_list = [messageBox_text]
    )

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        global scrollable_frame_file_list

        global selected_AI_model
        if   'x1' in selected_AI_model: upscale_factor = 1
        elif 'x2' in selected_AI_model: upscale_factor = 2
        elif 'x4' in selected_AI_model: upscale_factor = 4

        try:
            resize_factor = int(float(str(selected_resize_factor.get())))
        except:
            resize_factor = 0

        scrollable_frame_file_list = ScrollableImagesTextFrame_upscaler(
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
        selected_output_path.set(default_output_path)
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

def select_AI_precision_from_menu(selected_option: str) -> None:
    global selected_half_precision
    match selected_option:
        case "Full precision":
            selected_half_precision = False
        case "Half precision":
            selected_half_precision = True

def select_AI_multithreading_from_menu(selected_option: str) -> None:
    global selected_AI_multithreading
    if selected_option == "Disabled":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(selected_option.split()[0])

def select_image_extension_from_menu(selected_option: str) -> None:
    global selected_image_extension   
    selected_image_extension = selected_option

def select_video_extension_from_menu(selected_option: str) -> None:
    global selected_video_extension   
    selected_video_extension = selected_option

def select_interpolation_from_menu(selected_option: str) -> None:
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

def open_info_output_path():
    option_list = [
        "\n The default path is defined by the input files."
        + "\n For example uploading a file from the Download folder,"
        + "\n the app will save the upscaled files in the Download folder \n",

        " Otherwise it is possible to select the desired path using the SELECT button",
    ]

    CTkMessageBox(
        messageType = "info",
        title       = "Output path",
        subtitle    = "This widget allows to choose upscaled files path",
        default_value = default_output_path,
        option_list   = option_list
    )

def open_info_AI_model():
    option_list = [
        "\n IRCNN (2017) - Very simple and lightweight AI architecture\n" + 
        " Only denoising (no upscaling)\n" + 
        " Recommended for both image/video denoising\n" + 
        "  • IRCNNx1\n",

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

    CTkMessageBox(
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

    CTkMessageBox(
        messageType = "info",
        title       = "GPU",
        subtitle    = "This widget allows to select the GPU for AI upscale",
        default_value = default_gpu,
        option_list   = option_list
    )

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

    CTkMessageBox(
        messageType = "info",
        title       = "AI precision",
        subtitle    = "This widget allows to choose the AI upscaling precision",
        default_value = default_AI_precision,
        option_list   = option_list
    )

def open_info_interpolation():
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

    CTkMessageBox(
        messageType = "info",
        title       = "Interpolation", 
        subtitle    = "This widget allows to choose interpolation between upscaled and original image/frame",
        default_value = default_interpolation,
        option_list   = option_list
    )

def open_info_AI_multithreading():
    option_list = [
        " This option can improve video upscaling performance, especially with powerful GPUs",

        " \n AI MULTITHREADING OPTIONS\n"
        + "  • Disabed - upscaling 1 frame\n" 
        + "  • 2 threads - upscaling 2 frame simultaneously\n" 
        + "  • 3 threads - upscaling 3 frame simultaneously\n" 
        + "  • 4 threads - upscaling 4 frame simultaneously\n" ,

        " \n NOTES \n"
        + "  • As the number of threads increases, the use of CPU, GPU and RAM memory also increases\n" 
        + "  • In particular, the GPU is put under a lot of stress, and may reach high temperatures\n" 
        + "  • Keep an eye on the temperature of your PC so that it doesn't overheat \n" 
        + "  • The app selects the most appropriate number of threads if the chosen number exceeds GPU capacity\n" ,

    ]

    CTkMessageBox(
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

    CTkMessageBox(
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

    CTkMessageBox(
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

    CTkMessageBox(
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

    CTkMessageBox(
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

    CTkMessageBox(
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
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.05, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_AI_precision_menu():
    AI_mode_button = create_info_button(open_info_AI_precision, "AI precision")
    AI_mode_menu   = create_option_menu(select_AI_precision_from_menu, AI_precision_list)
    
    AI_mode_button.place(relx = column0_x, rely = row2_y - 0.05, anchor = "center")
    AI_mode_menu.place(relx = column0_x, rely = row2_y, anchor = "center")

def place_interpolation_menu():
    interpolation_button = create_info_button(open_info_interpolation, "Interpolation")
    interpolation_menu   = create_option_menu(select_interpolation_from_menu, interpolation_list)
    
    interpolation_button.place(relx = column0_x, rely = row4_y - 0.05, anchor = "center")
    interpolation_menu.place(relx = column0_x, rely  = row4_y, anchor = "center")

def place_AI_multithreading_menu():
    AI_multithreading_button = create_info_button(open_info_AI_multithreading, "AI multithreading")
    AI_multithreading_menu   = create_option_menu(select_AI_multithreading_from_menu, AI_multithreading_list)
    
    AI_multithreading_button.place(relx = column0_x, rely = row3_y - 0.05, anchor = "center")
    AI_multithreading_menu.place(relx = column0_x, rely  = row3_y, anchor = "center")

def place_image_output_menu():
    file_extension_button = create_info_button(open_info_image_output, "Image output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column2_x, rely = row1_y - 0.05, anchor = "center")
    file_extension_menu.place(relx = column2_x, rely = row1_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column2_x, rely = row2_y - 0.05, anchor = "center")
    video_extension_menu.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_gpu_menu():
    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_gpu_from_menu, gpus_list)
    
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
        place_AI_precision_menu()
        place_interpolation_menu()
        place_AI_multithreading_menu()

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

    processing_queue = multiprocessing_Queue(maxsize=1)

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    if os_path_exists(FFMPEG_EXE_PATH): 
        os_environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

    window = CTk() 

    info_message            = StringVar()
    selected_output_path    = StringVar()
    selected_resize_factor  = StringVar()
    selected_VRAM_limiter   = StringVar()
    selected_cpu_number     = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_half_precision
    global selected_AI_multithreading
    global selected_image_extension
    global selected_video_extension
    global selected_interpolation
    global selected_interpolation_factor
    global tiles_resolution
    global resize_factor
    global cpu_number

    selected_file_list = []

    selected_AI_model        = default_AI_model
    selected_gpu             = default_gpu
    selected_image_extension = default_image_extension
    selected_video_extension = default_video_extension

    selected_half_precision = True if default_AI_precision == "Half precision" else False

    if default_AI_multithreading == "Disabled":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(default_AI_multithreading.split()[0])

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
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(16, 16))

    app = App(window)
    window.update()
    window.mainloop()