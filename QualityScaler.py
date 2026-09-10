
# Standard library imports
import sys
from dataclasses import dataclass, field
from functools  import cache
from time       import sleep
from webbrowser import open as open_browser

from shutil     import rmtree as remove_directory, disk_usage as shutil_disk_usage
from timeit     import default_timer as timer

from typing    import Any, Callable, ClassVar, Optional
from threading import Thread
from queue     import Empty, Full
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import ( 
    Process        as multiprocessing_Process,
    Queue          as multiprocessing_Queue,
    Event          as multiprocessing_Event,
    Pool           as multiprocessing_Pool,
    Manager        as multiprocessing_Manager,
    freeze_support as multiprocessing_freeze_support
)

from json import (
    load  as json_load, 
    dumps as json_dumps
)

from os import (
    sep      as os_separator,
    devnull  as os_devnull,
    getpid   as os_getpid,
    makedirs as os_makedirs,
    listdir  as os_listdir,
    remove   as os_remove,
    fdopen   as os_fdopen,
    open     as os_open,
    O_WRONLY as os_O_WRONLY,
    O_CREAT  as os_O_CREAT
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

from subprocess import (
    Popen                as subprocess_Popen,
    PIPE                 as subprocess_PIPE,
    DEVNULL              as subprocess_DEVNULL,
    STARTUPINFO          as subprocess_STARTUPINFO,
    STARTF_USESHOWWINDOW as subprocess_STARTF_USESHOWWINDOW,
    TimeoutExpired       as subprocess_TimeoutExpired
)

if sys.platform == "win32":
    from winotify import Notification as winotify_Notification
    from win32api import SetFileAttributes as win32_SetFileAttributes
    from win32con import (
        FILE_ATTRIBUTE_NOT_CONTENT_INDEXED as win32_FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
        FILE_ATTRIBUTE_SYSTEM              as win32_FILE_ATTRIBUTE_SYSTEM,
        FILE_ATTRIBUTE_HIDDEN              as win32_FILE_ATTRIBUTE_HIDDEN,
    )

# Third-party library imports
from natsort import natsorted
from psutil import (
    Process             as psutil_Process,
    IDLE_PRIORITY_CLASS as psutil_IDLE_PRIORITY_CLASS,
    virtual_memory      as psutil_virtual_memory,
)
from onnxruntime import (
    InferenceSession          as onnxruntime_InferenceSession,
    SessionOptions            as onnxruntime_SessionOptions,
    GraphOptimizationLevel    as onnxruntime_GraphOptimizationLevel,
    get_available_providers   as onnxruntime_get_available_providers,
    get_version_string        as onnxruntime_get_version_string
)

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
    IMWRITE_JPEG_QUALITY,
    IMWRITE_PNG_COMPRESSION,
    INTER_LINEAR,
    INTER_AREA,
    CV_8U,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    resize       as opencv_resize,
    GaussianBlur as opencv_GaussianBlur,
    addWeighted  as opencv_addWeighted,
    setNumThreads as opencv_setNumThreads,
)

from numpy import (
    frombuffer        as numpy_frombuffer,
    concatenate       as numpy_concatenate,
    transpose         as numpy_transpose,
    ascontiguousarray as numpy_ascontiguousarray,
    expand_dims       as numpy_expand_dims,
    squeeze     as numpy_squeeze,
    clip        as numpy_clip,
    mean        as numpy_mean,
    repeat      as numpy_repeat,
    array_split as numpy_array_split,
    zeros       as numpy_zeros, 
    ones        as numpy_ones,
    arange      as numpy_arange,
    maximum     as numpy_maximum,
    max         as numpy_max, 
    ndarray     as numpy_ndarray,
    pad         as numpy_pad,
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
    CTkProgressBar,
    CTkScrollableFrame,
    CTkToplevel,
    CTkCanvas,
    filedialog,
    set_appearance_mode,
    set_default_color_theme,
    set_widget_scaling,
    set_window_scaling
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w", encoding="utf-8", errors="replace")
else:                  sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if sys.stderr is None: sys.stderr = open(os_devnull, "w", encoding="utf-8", errors="replace")
else:                  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)



app_name   = "QualityScaler"
version    = "2026.4"
githubme   = "https://github.com/Djdefrag/QualityScaler/releases"
telegramme = "https://linktr.ee/j3ngystudio"

app_name_color          = "#F274EE"
background_color        = "#000000"
widget_background_color = "#181818"
text_color              = "#C8C8C8"   # card value tone (was #B8B8B8)

# FileWidget card palette (loaded-files list)
CARD_BACKGROUND_COLOR   = "#1A1A1A"
CARD_BORDER_COLOR       = "#2B2B2B"
CARD_TITLE_COLOR        = "#ECECEC"
CARD_VALUE_COLOR        = "#C8C8C8"
CARD_MUTED_COLOR        = "#7E7E7E"
CARD_FAINT_COLOR        = "#6A6A6A"
CARD_ACCENT_COLOR       = "#5AA9FF"

# Resume badge (partially-processed videos)
RESUME_BADGE_COLOR      = "#16261C"
RESUME_ACCENT_COLOR     = "#4ADE80"
RESUME_BAR_DIM_COLOR    = "#1E4A33"   # dim green the bar fades in from while filling

# Status badge tint by processing state
MESSAGE_SUCCESS_COLOR   = "#4ADE80"   # completed
MESSAGE_WARNING_COLOR   = "#F0B85A"   # stopped
MESSAGE_ERROR_COLOR     = "#FF5C5C"   # error

# Shared UI style (option widgets), aligned with the file cards
UI_ACCENT_COLOR  = CARD_ACCENT_COLOR     # soft blue instead of the harsh #0096FF
UI_BORDER_COLOR  = CARD_BORDER_COLOR     # subtle border instead of #404040
UI_CORNER_RADIUS = 6                     # rounded corners like the cards


VRAM_model_usage = {
    'LVAx2':           2,
    'RealESR_Gx4':     2.5,
    'RealESR_Ax4':     2.5,
    'BSRGANx2':        0.8,
    'BSRGANx4':        0.75,
    'RealESRGANx4':    0.75,
    'MSharpx4':        1.5,
    'IRCNN_Mx1':       4,
    'IRCNN_Lx1':       4,
    'NomosWebPhoto_RealPLKSRx4': 1.5,  # Added PLKSR
    'SwinIRLargex4':              0.75, # Added SwinIR (smaller tiles due to high VRAM overhead)
}

MENU_LIST_SEPARATOR = [ "----" ]
LVA_models        = [ "LVAx2"                      ] 
RealESR_models    = [ "RealESR_Gx4", "RealESR_Ax4" ]
BSRGAN_models     = [ "BSRGANx2",    "BSRGANx4"    ]
RealESRGAN_models = [ "RealESRGANx4"               ]
MSharp_models     = [ "MSharpx4"                   ]
IRCNN_models      = [ "IRCNN_Mx1",   "IRCNN_Lx1"   ]
PLKSR_models  = [ "NomosWebPhoto_RealPLKSRx4" ]
SwinIR_models = [ "SwinIRLargex4" ]

AI_models_list = ( 
    LVA_models          + MENU_LIST_SEPARATOR +
    RealESR_models      + MENU_LIST_SEPARATOR + 
    BSRGAN_models       + MENU_LIST_SEPARATOR +
    RealESRGAN_models   + MENU_LIST_SEPARATOR +
    MSharp_models       + MENU_LIST_SEPARATOR +
    IRCNN_models        + MENU_LIST_SEPARATOR +
    PLKSR_models        + MENU_LIST_SEPARATOR +
    SwinIR_models
)

zoom_option_list       = [ "50%", "75%", "100%", "125%", "150%", "175%" ]
AI_multithreading_list = [ "OFF", "2 threads", "4 threads", "6 threads", "8 threads"]
sharpening_list        = [ "OFF", "Low", "High" ]
gpus_list              = [ "No GPU found" ]  # placeholder default, replaced at runtime by the detected GPUs
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
USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_{version}_userpreference.json")
FFMPEG_EXE_PATH      = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
LOGO_PNG_PATH        = find_by_relative_path(f"Assets{os_separator}logo.png")

COMPLETED_STATUS = "Completed"
ERROR_STATUS     = "Error"
STOP_STATUS      = "Stop"
CLOSE_APP_STATUS = "CloseApp"

MIN_FREE_DISK_SPACE_GB = 2


@dataclass
class UserPreferences:
    app_zoom:             str = "100%"
    ai_model:             str = AI_models_list[0]
    ai_multithreading:    str = AI_multithreading_list[0]
    gpu:                  str = gpus_list[0]
    keep_frames:          bool = True
    image_extension:      str = image_extension_list[0]
    video_extension:      str = video_extension_list[0]
    video_codec:          str = video_codec_list[0]
    sharpening:           str = sharpening_list[1]
    output_path:          str = OUTPUT_PATH_CODED
    input_resize_factor:  str = "50"
    output_resize_factor: str = "100"
    vram_limiter:         str = "4"


@dataclass
class ProcessingConfig:
    selected_file_list:         list[str]
    selected_output_path:       str
    selected_AI_model:          str
    selected_AI_multithreading: int
    input_resize_factor:        float
    output_resize_factor:       float
    selected_gpu:               str
    tiles_resolution:           int
    selected_sharpening_amount: float
    selected_keep_frames:       bool
    selected_image_extension:   str
    selected_video_extension:   str
    selected_video_codec:       str


@dataclass
class AppState:
    preferences:                   UserPreferences
    window:                        Optional[CTk] = None
    info_message:                  Optional[StringVar] = None
    selected_output_path:          Optional[StringVar] = None
    selected_input_resize_factor:  Optional[StringVar] = None
    selected_output_resize_factor: Optional[StringVar] = None
    selected_VRAM_limiter:         Optional[StringVar] = None
    selected_video_codec:          Optional[StringVar] = None
    file_widget:                   Optional["FileWidget"] = None
    process_upscale_orchestrator:  Optional[Any] = None
    process_status_q:              Optional[multiprocessing_Queue] = None
    video_frames_and_info_q:       Optional[multiprocessing_Queue] = None
    event_stop_upscale_process:    Optional[any] = None
    selected_file_list:            list[str] = field(default_factory=list)
    completed_video_files:         set = field(default_factory=set)


app_state: Optional[AppState] = None


# Vertical position (rely) of each row in the right-hand options panel
ROW_STEP   = 0.0825                  # gap between consecutive option rows
ROW_HEADER = 0.05                    # app title / zoom / links
_ROW_FIRST = 0.125                   # first option row (AI model)

def _row(index: int) -> float: return _ROW_FIRST + index * ROW_STEP

ROW_AI_MODEL          = _row(0)
ROW_AI_SHARPENING     = _row(1)
ROW_AI_MULTITHREADING = _row(2)
ROW_RESOLUTION        = _row(3)
ROW_GPU               = _row(4)
ROW_OUTPUT_FORMAT     = _row(5)
ROW_CODEC             = _row(6)
ROW_OUTPUT_PATH       = _row(9)
ROW_ACTIONS           = _row(10)

# Horizontal position (relx) of widgets, by column slot
COL_INFO_L = 0.625                   # left  info button
COL_INFO_R = 0.858                   # right info button
COL_TEXT_L = COL_INFO_L + 0.08       # left  text box    (input scale)
COL_TEXT_R = COL_INFO_R + 0.08       # right text box    (output scale, VRAM)
COL_MENU_L = COL_TEXT_L - 0.0127     # left  option menu (GPU, image, codec)
COL_MENU_R = COL_TEXT_R - 0.0127     # right option menu (video output, keep frames)

# Header / zoom / links area
COL_TITLE  = 0.66                    # app name
COL_ZOOM   = COL_TITLE + 0.2         # zoom menu, links, output path box
COL_MENU_C = COL_ZOOM + 0.0355       # centered single menu (AI model / sharpening / threads)

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

_supported_video_extensions_set = {ext.lower() for ext in supported_video_extensions}



# GPU -------------------

@dataclass
class GPU:
    # Domain model + registry for the system GPUs and their dedicated VRAM.
    name:      str    # short, readable display name, e.g. "RTX 5060 Ti"
    vram_gb:   float  # dedicated VRAM in GB
    device_id: int    # DirectML device_id (position among hardware adapters)
    vendor_id: int = 0  # PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel); 0 when unknown

    # Internal value used to let DirectML pick the best adapter when no concrete
    # GPU is available (empty registry / unknown selection). Never shown in the menu.
    AUTO:   ClassVar[str] = "Auto"
    # Placeholder shown in the dropdown when no GPU could be detected.
    NO_GPU: ClassVar[str] = "No GPU found"

    # PCI vendor IDs, used to auto-pick the matching hardware video encoder.
    VENDOR_NVIDIA: ClassVar[int] = 0x10DE
    VENDOR_AMD:    ClassVar[int] = 0x1002
    VENDOR_INTEL:  ClassVar[int] = 0x8086

    # Filled once at startup via GPU.detect(), ordered by device_id.
    detected: ClassVar[list["GPU"]] = []

    @property
    def vram_limit(self) -> int:
        # Dedicated VRAM floored to whole GB, never below 1.
        return max(1, int(self.vram_gb))

    @property
    def hardware_codec(self) -> Optional[str]:
        # Preferred H.264 hardware encoder for this GPU's vendor, or None if unknown.
        return {
            self.VENDOR_NVIDIA: "h264_nvenc",
            self.VENDOR_AMD:    "h264_amf",
            self.VENDOR_INTEL:  "h264_qsv",
        }.get(self.vendor_id)


    # PUBLIC REGISTRY API

    @classmethod
    def detect(cls) -> None:
        # Populate the registry with the GPUs found on this system.
        cls.detected = cls._enumerate()

    @classmethod
    def find(cls, name: str) -> Optional["GPU"]:
        return next((gpu for gpu in cls.detected if gpu.name == name), None)

    @classmethod
    def menu_list(cls) -> list[str]:
        # Dropdown options: every detected GPU (by name), or a single
        # "No GPU found" placeholder when none are available.
        return [gpu.name for gpu in cls.detected] if cls.detected else [cls.NO_GPU]

    @classmethod
    def default(cls) -> str:
        # First detected GPU name, or the "No GPU found" placeholder.
        return cls.detected[0].name if cls.detected else cls.NO_GPU

    @classmethod
    def device_id_for(cls, selected: str) -> str:
        # Map a GPU display name to its DirectML device_id ("0", "1", ...).
        # No GPUs / unknown selection fall back to "Auto" (DirectML picks the best adapter).
        if not cls.detected:
            return cls.AUTO
        gpu = cls.find(selected)
        return str(gpu.device_id) if gpu is not None else str(cls.detected[0].device_id)

    @classmethod
    def vram_for(cls, selected: str) -> Optional[int]:
        # Dedicated VRAM (GB, floored) for the selected GPU, or None if unknown.
        if not cls.detected:
            return None
        gpu = cls.find(selected)
        return gpu.vram_limit if gpu is not None else None

    @classmethod
    def codec_for(cls, selected: str) -> Optional[str]:
        # Hardware H.264 encoder matching the selected GPU's vendor, or None when
        # there are no GPUs / the selection or vendor is unknown.
        if not cls.detected:
            return None
        gpu = cls.find(selected)
        return gpu.hardware_codec if gpu is not None else None


    # NAME HELPER

    @staticmethod
    def _shorten_name(raw_name: str) -> str:
        # Produce a short, readable GPU label by dropping vendor/brand filler and trademark marks.
        # Examples:
        #   "NVIDIA GeForce RTX 5060 Ti"    -> "RTX 5060 Ti"
        #   "AMD Radeon RX 7900 XTX"        -> "RX 7900 XTX"
        #   "Intel(R) Iris(R) Xe Graphics"  -> "Iris Xe Graphics"
        if not raw_name:
            return raw_name

        cleaned = raw_name
        for marker in ("(R)", "(TM)", "(C)", "\u00ae", "\u2122", "\u00a9"):
            cleaned = cleaned.replace(marker, " ")

        GENERIC_WORDS = {"GRAPHICS", "SERIES", "FAMILY"}

        def _strip_words(text: str, words: set) -> str:
            return " ".join(t for t in text.split() if t.upper() not in words).strip()

        # First pass: drop vendor names and sub-brands (including "Radeon")
        primary = _strip_words(cleaned, {"NVIDIA", "GEFORCE", "AMD", "INTEL", "CORPORATION", "ADVANCED", "MICRO", "DEVICES", "RADEON"})

        # If only generic words survived (e.g. integrated "Radeon Graphics"), keep the sub-brand
        if not primary or all(token.upper() in GENERIC_WORDS for token in primary.split()):
            primary = _strip_words(cleaned, {"NVIDIA", "GEFORCE", "AMD", "INTEL", "CORPORATION", "ADVANCED", "MICRO", "DEVICES"})

        return primary if primary else raw_name.strip()


    # LOW-LEVEL DETECTION

    @staticmethod
    def _enumerate() -> list["GPU"]:
        # Enumerate DirectX adapters via DXGI and read their dedicated VRAM.
        # Works for every GPU vendor (Intel / AMD / NVIDIA) without extra dependencies.
        # device_id is the position among hardware adapters (== DirectML device_id).

        if sys.platform != "win32":
            return []

        import ctypes

        class _LUID(ctypes.Structure):
            _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class _DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description",           ctypes.c_wchar * 128),
                ("VendorId",              ctypes.c_uint),
                ("DeviceId",              ctypes.c_uint),
                ("SubSysId",              ctypes.c_uint),
                ("Revision",              ctypes.c_uint),
                ("DedicatedVideoMemory",  ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory",    ctypes.c_size_t),
                ("AdapterLuid",           _LUID),
                ("Flags",                 ctypes.c_uint),
            ]

        DXGI_ADAPTER_FLAG_SOFTWARE = 2
        IID_IDXGIFactory1 = _GUID(
            0x770aae78, 0xf26f, 0x4dba,
            (ctypes.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87)
        )

        def _com_method(obj_ptr, index, restype, argtypes):
            vtable    = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
            func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
            return ctypes.WINFUNCTYPE(restype, *argtypes)(func_addr)

        gpus:       list["GPU"]    = []
        seen_names: dict[str, int] = {}

        try:
            factory = ctypes.c_void_p()
            hr = ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
            if hr != 0 or not factory.value:
                return []

            # IDXGIFactory1::EnumAdapters1 -> vtable index 12 ; IUnknown::Release -> index 2
            EnumAdapters1  = _com_method(factory, 12, ctypes.c_int32, [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
            FactoryRelease = _com_method(factory, 2,  ctypes.c_ulong, [ctypes.c_void_p])

            try:
                adapter_index  = 0
                hardware_index = 0
                while True:
                    adapter = ctypes.c_void_p()
                    if EnumAdapters1(factory, adapter_index, ctypes.byref(adapter)) != 0:
                        break

                    # IDXGIAdapter1::GetDesc1 -> vtable index 10 ; IUnknown::Release -> index 2
                    desc           = _DXGI_ADAPTER_DESC1()
                    GetDesc1       = _com_method(adapter, 10, ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1)])
                    AdapterRelease = _com_method(adapter, 2,  ctypes.c_ulong, [ctypes.c_void_p])
                    GetDesc1(adapter, ctypes.byref(desc))

                    # Skip software adapters (WARP / Microsoft Basic Render Driver)
                    if not (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE):
                        vram_gb      = round(desc.DedicatedVideoMemory / (1024 ** 3), 1)
                        display_name = GPU._shorten_name(desc.Description)

                        # Disambiguate identical GPUs (e.g. two "RTX 5060 Ti" -> add " (2)")
                        if display_name in seen_names:
                            seen_names[display_name] += 1
                            display_name = f"{display_name} ({seen_names[display_name]})"
                        else:
                            seen_names[display_name] = 1

                        gpus.append(GPU(name=display_name, vram_gb=vram_gb, device_id=hardware_index, vendor_id=desc.VendorId))
                        hardware_index += 1

                    AdapterRelease(adapter)
                    adapter_index += 1
            finally:
                FactoryRelease(factory)

        except Exception as exception:
            print(f"[{app_name}] GPU detection failed: {exception}")
            return []

        return gpus



# AI engine -------------------
# Shared ONNX Runtime helpers, used both for the window title and to load the AI model.

def get_available_AI_providers() -> list[str]:
    # Execution providers exposed by the installed ONNX Runtime build.
    try:
        return list(onnxruntime_get_available_providers())
    except Exception:
        return []

def is_directml_available() -> bool:
    return any("Dml" in provider or "DirectML" in provider for provider in get_available_AI_providers())

def get_AI_providers() -> list[str]:
    # Prefer DirectML (GPU); fall back to CPU when DirectML is not available.
    return ["DmlExecutionProvider"] if is_directml_available() else ["CPUExecutionProvider"]

def get_AI_engine_info() -> str:
    # Human-readable engine label, e.g. "AI engine 1.x.y + DirectML".
    try:
        AI_engine_version = onnxruntime_get_version_string()
        AI_provider_name  = "DirectML" if is_directml_available() else "CPU"
        return f"AI engine {AI_engine_version} + {AI_provider_name}"
    except Exception:
        return ""



# AI -------------------

def get_model_upscale_factor(selected_AI_model: str) -> int:
    # Upscale factor encoded in the model name (e.g. "BSRGANx4" -> 4).
    if   "x1" in selected_AI_model: return 1
    elif "x2" in selected_AI_model: return 2
    elif "x3" in selected_AI_model: return 3
    elif "x4" in selected_AI_model: return 4
    else:                           return 1

class AI_upscale:

    # CLASS INIT FUNCTIONS

    def __init__(
            self, 
            selected_AI_model:   str, 
            selected_gpu:        str, 
            input_resize_factor: float,
            tiles_resolution:    int,
        ):
        
        # Passed variables
        self.selected_AI_model   = selected_AI_model
        self.selected_gpu        = selected_gpu
        self.input_resize_factor = input_resize_factor
        self.tiles_resolution    = tiles_resolution
        
        # ONNX file path resolver
        def resolve_model_path(model_name: str) -> str:
            base_folder = f"AI-onnx{os_separator}"
            candidates = [
                f"{base_folder}{model_name}_fp16.onnx",
                f"{base_folder}{model_name}_fp32.onnx",
                f"{base_folder}{model_name}.onnx"
            ]
            for path in candidates:
                full_path = find_by_relative_path(path)
                if os_path_exists(full_path):
                    return full_path
            # Fallback to default naming convention if not found yet
            return find_by_relative_path(f"{base_folder}{model_name}_fp16.onnx")

        self.selected_AI_model_path = resolve_model_path(self.selected_AI_model)
        self.upscale_factor         = get_model_upscale_factor(self.selected_AI_model)

        # Variable assigned later 
        self.inferenceSession = None
        self.input_name       = None
        self.onnx_input       = None

    def _load_inferenceSession(self) -> onnxruntime_InferenceSession:

        providers = get_AI_providers()

        # DirectML accepts a target device_id / performance hint; other providers take no options.
        # provider_options must align 1:1 with providers, so build it per-provider.
        provider_options = []
        for provider in providers:
            if provider == "DmlExecutionProvider":
                # selected_gpu is already resolved to a DirectML device_id ("0", "1", ...) or "Auto"
                if self.selected_gpu == "Auto":
                    provider_options.append({"performance_preference": "high_performance"})
                else:
                    provider_options.append({"device_id": str(self.selected_gpu)})
            else:
                provider_options.append({})

        sess_options                          = onnxruntime_SessionOptions()
        sess_options.enable_profiling         = False
        sess_options.intra_op_num_threads     = 1
        sess_options.graph_optimization_level = onnxruntime_GraphOptimizationLevel.ORT_ENABLE_ALL

        inference_session = onnxruntime_InferenceSession(
            path_or_bytes    = self.selected_AI_model_path, 
            sess_options     = sess_options,
            providers        = providers,
            provider_options = provider_options,
        )

        return inference_session



    # INTERNAL CLASS FUNCTIONS

    def calculate_target_resolution(self, image: numpy_ndarray) -> tuple:
        height, width = get_image_resolution(image)
        target_height = height * self.upscale_factor
        target_width  = width  * self.upscale_factor

        return target_height, target_width

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        return resize_with_factor(image, self.input_resize_factor, INTER_AREA)



    # TILLING FUNCTIONS

    def image_need_tilling(self, image: numpy_ndarray) -> bool:
        height, width = get_image_resolution(image)
        return (height * width) > (self.tiles_resolution * self.tiles_resolution)

    def calculate_tiles_number(self, image: numpy_ndarray) -> tuple:
        
        height, width = get_image_resolution(image)

        tiles_x = (width  + self.tiles_resolution - 1) // self.tiles_resolution
        tiles_y = (height + self.tiles_resolution - 1) // self.tiles_resolution

        return tiles_x, tiles_y
    

    # AI CLASS FUNCTIONS

    def normalize_image(self, image: numpy_ndarray, original_dtype) -> tuple:
        # uint8 sources always have max_range 255 - skip the full-array max() scan (repeated per tile when tiling).
        if original_dtype == uint8:
            max_range = 255.0
        else:
            max_val   = numpy_max(image)
            max_range = 65535.0 if max_val > 256 else 255.0
        image /= max_range

        return image, max_range
    
    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_ascontiguousarray(numpy_transpose(image, (2, 0, 1)))
        image = numpy_expand_dims(image, axis=0)

        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:
        self.onnx_input[self.input_name] = image
        return self.inferenceSession.run(None, self.onnx_input)[0]

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))

        return onnx_output

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:    
        onnx_output *= max_range
        return onnx_output.astype(uint8) if max_range == 255 else onnx_output.astype(float32)



    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        original_dtype = image.dtype
        image          = image.astype(float32, copy=False)
        image_mode     = get_image_mode(image)
        image, norm_range = self.normalize_image(image, original_dtype)

        match image_mode:
            case "RGB":
                image = self.preprocess_image(image)
                onnx_output  = self.onnxruntime_inference(image)
                onnx_output  = self.postprocess_output(onnx_output)
                output_image = self.de_normalize_image(onnx_output, norm_range)

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
                output_image = self.de_normalize_image(onnx_output_image, norm_range)

                return output_image
            
            case "Grayscale":
                image = opencv_cvtColor(image, COLOR_GRAY2RGB)
                
                image = self.preprocess_image(image)
                onnx_output  = self.onnxruntime_inference(image)
                onnx_output  = self.postprocess_output(onnx_output)
                output_image = opencv_cvtColor(onnx_output, COLOR_RGB2GRAY)
                output_image = self.de_normalize_image(output_image, norm_range)

                return output_image

            case _:
                return image

    def _tile_feather_weights(self, length: int, ramp: int) -> numpy_ndarray:
        # Linear ramp 0->1 over the first/last "ramp" pixels, flat 1 in the middle,
        # so overlapping tiles fade into each other instead of leaving a visible seam.
        weights = numpy_ones(length, dtype = float32)
        r       = min(ramp, length // 2)  # r <= length//2 keeps the two ramps from overlapping
        if r > 0:
            edge               = numpy_arange(1, r + 1, dtype = float32) / (r + 1)
            weights[:r]        = edge
            weights[length-r:] = edge[::-1]
        return weights

    def AI_upscale_with_tilling(self, image: numpy_ndarray) -> numpy_ndarray:
        OVERLAP = 16

        tiles_x, tiles_y      = self.calculate_tiles_number(image)
        img_height, img_width = get_image_resolution(image)
        t_height, t_width     = self.calculate_target_resolution(image)
        uf                    = self.upscale_factor
        ramp                  = OVERLAP * uf

        tile_w = img_width  // tiles_x
        tile_h = img_height // tiles_y

        # Pad with reflect so every tile has OVERLAP pixels of context on each side
        pad_cfg = ((OVERLAP, OVERLAP), (OVERLAP, OVERLAP), (0, 0)) if len(image.shape) == 3 else ((OVERLAP, OVERLAP), (OVERLAP, OVERLAP))
        padded  = numpy_pad(image, pad_cfg, mode = 'reflect')

        image_mode = get_image_mode(image)
        channels   = 1 if image_mode == "Grayscale" else (4 if image_mode == "RGBA" else 3)

        # Accumulators padded by "ramp" on each side: every tile (core + overlap) then writes
        # fully in-bounds at the simple positive offset (cy0*uf, cx0*uf) -> no clipping needed.
        accumulator = numpy_zeros((t_height + 2*ramp, t_width + 2*ramp, channels), dtype = float32)
        weights     = numpy_zeros((t_height + 2*ramp, t_width + 2*ramp, 1),        dtype = float32)

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                cx0 = tx * tile_w
                cy0 = ty * tile_h
                cx1 = img_width  if tx == tiles_x - 1 else (tx + 1) * tile_w
                cy1 = img_height if ty == tiles_y - 1 else (ty + 1) * tile_h

                # Tile = core + OVERLAP on every side, taken from the reflect-padded image
                tile          = padded[cy0 : cy1 + 2*OVERLAP, cx0 : cx1 + 2*OVERLAP]
                upscaled_tile = self.AI_upscale(tile)
                if upscaled_tile.ndim == 2:
                    upscaled_tile = upscaled_tile[:, :, None]

                th, tw = upscaled_tile.shape[0], upscaled_tile.shape[1]

                # Feather mask: fades to 0 at the tile edges so neighbours blend in the overlap
                wy   = self._tile_feather_weights(th, ramp)
                wx   = self._tile_feather_weights(tw, ramp)
                mask = (wy[:, None] * wx[None, :])[:, :, None]

                # Top-left position inside the padded accumulator (always >= 0)
                y0 = cy0 * uf
                x0 = cx0 * uf

                accumulator[y0 : y0 + th, x0 : x0 + tw] += upscaled_tile.astype(float32) * mask
                weights[y0 : y0 + th, x0 : x0 + tw]     += mask

        # Remove the padding, normalize the blend, round, and cast back to uint8
        accumulator = accumulator[ramp : ramp + t_height, ramp : ramp + t_width]
        weights     = weights[ramp : ramp + t_height, ramp : ramp + t_width]
        numpy_maximum(weights, 1e-6, out = weights)
        blended = accumulator / weights + 0.5

        if image_mode == "Grayscale":
            return blended[:, :, 0].astype(uint8)
        return blended.astype(uint8)


    # PUBLIC FUNCTION

    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:
        if self.inferenceSession is None:
            self.inferenceSession = self._load_inferenceSession()
            self.input_name       = self.inferenceSession.get_inputs()[0].name
            self.onnx_input       = { self.input_name: None }
            
        resized_image = self.resize_with_input_factor(image)

        if self.image_need_tilling(resized_image):
            upscaled_image = self.AI_upscale_with_tilling(resized_image)
        else:
            upscaled_image = self.AI_upscale(resized_image)
                 
        return upscaled_image
    


# Upscale task -------------------

def _build_name_suffix(
        selected_AI_model:          str,
        input_resize_factor:        float,
        output_resize_factor:       float,
        selected_sharpening_amount: float
        ) -> str:

    suffix  = f"_{selected_AI_model}"
    suffix += f"_InputR-{str(int(input_resize_factor * 100))}"
    suffix += f"_OutputR-{str(int(output_resize_factor * 100))}"

    match selected_sharpening_amount:
        case 0.3: suffix += "_Sharpening-Low"
        case 0.5: suffix += "_Sharpening-High"

    return suffix

def _build_output_path_base(source_path: str, selected_output_path: str) -> str:
    # Output path without suffix/extension: next to the source file, or inside the chosen folder.
    if selected_output_path == OUTPUT_PATH_CODED:
        base, _ = os_path_splitext(source_path)
        return base
    base, _ = os_path_splitext(os_path_basename(source_path))
    return f"{selected_output_path}{os_separator}{base}"

class VideoUpscaleTask:

    def __init__(
            self, 
            video_path:                 str,
            selected_output_path:       str,
            selected_AI_model:          str,
            selected_AI_multithreading: int, 
            selected_gpu:               str,
            tiles_resolution:           int,
            input_resize_factor:        float,
            output_resize_factor:       float,
            selected_sharpening_amount: float,
            selected_video_extension:   str,
            selected_video_codec:       str,
            ) -> None:
        
        # Passed variables
        self.video_path                 = video_path
        self.selected_output_path       = selected_output_path
        self.selected_AI_model          = selected_AI_model
        self.selected_AI_multithreading = selected_AI_multithreading
        self.selected_gpu               = selected_gpu
        self.tiles_resolution           = tiles_resolution
        self.input_resize_factor        = input_resize_factor
        self.output_resize_factor       = output_resize_factor
        self.selected_sharpening_amount = selected_sharpening_amount
        self.selected_video_codec       = selected_video_codec
        self.selected_video_extension   = selected_video_extension

        # Calculated variables

        # Upscale factor
        self.upscale_factor = get_model_upscale_factor(selected_AI_model)
        
        # 1. Target directory
        self.target_directory = self._prepare_output_video_directory_name(
            video_path                 = self.video_path,
            selected_output_path       = self.selected_output_path, 
            selected_AI_model          = self.selected_AI_model,
            input_resize_factor        = self.input_resize_factor, 
            output_resize_factor       = self.output_resize_factor, 
            selected_sharpening_amount = self.selected_sharpening_amount
        )

        # 2. Video output path
        self.video_output_path = self._prepare_output_video_filename(
            video_path                 = self.video_path, 
            selected_output_path       = self.selected_output_path, 
            selected_AI_model          = self.selected_AI_model, 
            input_resize_factor        = self.input_resize_factor, 
            output_resize_factor       = self.output_resize_factor, 
            selected_video_extension   = self.selected_video_extension, 
            selected_sharpening_amount = self.selected_sharpening_amount
        )

        # 3. FFMPEG encoding infos
        self.video_fps            = get_video_fps(self.video_path)
        self.effective_codec      = {"x264": "libx264", "x265": "libx265"}.get(self.selected_video_codec, self.selected_video_codec)
        self.ffmpeg_txt_file_path = f"{os_path_splitext(self.video_output_path)[0]}.txt"

        # Variable calculated later
        self.extracted_frames_paths  = None
        self.extracted_frames_number = None
        self.upscaled_frame_paths    = None
        self.original_width          = None
        self.original_height         = None
        self.AI_input_height         = None
        self.AI_input_width          = None
        self.target_height           = None
        self.target_width            = None
        self.optimal_threads_number  = None
        self.frames_paths_to_upscale = None
        self.frames_chunks_list      = None
        self.upscaled_frames_counter = None

    def _complete_init(self, extracted_frames_paths: list[str]):
        
        # Passed variables
        self.extracted_frames_paths = extracted_frames_paths

        # Calculated variables
        
        # 1. Number of extracted frames
        self.extracted_frames_number = len(self.extracted_frames_paths)

        # 2. Original video resolution / AI input resolution / video output resolution
        self.original_height, self.original_width = self._get_video_resolution(image_read(self.extracted_frames_paths[0]))

        self.AI_input_height, self.AI_input_width = self._calculate_input_resolution(
            original_height     = self.original_height,
            original_width      = self.original_width,
            input_resize_factor = self.input_resize_factor
        )
        self.target_height, self.target_width = self._calculate_output_resolution(
            AI_input_height      = self.AI_input_height,
            AI_input_width       = self.AI_input_width,
            upscale_factor       = self.upscale_factor,
            output_resize_factor = self.output_resize_factor
        )

        # 3. Upscaled frames paths
        self.upscaled_frame_paths = self._prepare_upscaled_frame_path_list(
            extracted_frames_paths     = self.extracted_frames_paths,
            selected_AI_model          = self.selected_AI_model,
            input_resize_factor        = self.input_resize_factor,
            output_resize_factor       = self.output_resize_factor,
            selected_sharpening_amount = self.selected_sharpening_amount
        )

        # 4. Optimal threads number
        self.optimal_threads_number = self._calculate_optimal_threads_number(
            tiles_resolution           = self.tiles_resolution,
            selected_AI_multithreading = self.selected_AI_multithreading
        )

        # 5. Already upscaled filter
        self.frames_paths_to_upscale = [
            (i, o) for i, o in zip(self.extracted_frames_paths, self.upscaled_frame_paths) 
            if not os_path_exists(o)
        ]
        
        # 6. Frames paths chuncks
        self.frames_chunks_list = [
            list(c) for c in numpy_array_split(self.frames_paths_to_upscale, self.optimal_threads_number)
        ]

        # 7. Already upscaled counter
        self.upscaled_frames_counter = self.extracted_frames_number - len(self.frames_paths_to_upscale)
        
        self._log_task_infos()



    # Class debug logs

    def _log_task_infos(self) -> None:        
        info_message = (
            f"[VideoUpscaleTask Created]\n"
            f"  > Input:  {self.video_path}\n"
            f"  > Output: {self.video_output_path}\n"
            f"  AI INFO:\n"
            f"      - AI Model:     {self.selected_AI_model}\n"
            f"      - Sharpening:   {self.selected_sharpening_amount}\n"
            f"      - GPU:          {self.selected_gpu}\n"
            f"      - Threads:      x{self.optimal_threads_number}\n"
            f"  RESOLUTIONS INFO:\n"
            f"      - Video Input:  {self.original_width}x{self.original_height}\n"
            f"      - AI Input:     {self.AI_input_width}x{self.AI_input_height}\n"
            f"      - AI Scale:     x{self.upscale_factor}\n"
            f"      - Out Factor:   x{self.output_resize_factor}\n"
            f"      - Final Output: {self.target_width}x{self.target_height}\n"
            f"  FRAMES INFO:\n"
            f"      - Total frames: {self.extracted_frames_number}\n"
            f"      - To upscale:   {len(self.frames_paths_to_upscale)}\n"
            f"      - Already done: {self.upscaled_frames_counter}"
        )
        
        print(info_message)



    # Functions to calculate video file names

    def _prepare_output_video_filename(
            self,
            video_path:                 str, 
            selected_output_path:       str,
            selected_AI_model:          str, 
            input_resize_factor:        float, 
            output_resize_factor:       float,
            selected_video_extension:   str,
            selected_sharpening_amount: float
            ) -> str:

        # The output filename is the output directory name plus the chosen video extension.
        output_path  = self._prepare_output_video_directory_name(
            video_path                 = video_path,
            selected_output_path       = selected_output_path,
            selected_AI_model          = selected_AI_model,
            input_resize_factor        = input_resize_factor,
            output_resize_factor       = output_resize_factor,
            selected_sharpening_amount = selected_sharpening_amount,
        )
        output_path += selected_video_extension

        return output_path

    def _prepare_output_video_directory_name(
            self,
            video_path:                 str, 
            selected_output_path:       str,
            selected_AI_model:          str, 
            input_resize_factor:        float, 
            output_resize_factor:       float,
            selected_sharpening_amount: float
            ) -> str:
        
        output_path  = _build_output_path_base(video_path, selected_output_path)
        output_path += _build_name_suffix(selected_AI_model, input_resize_factor, output_resize_factor, selected_sharpening_amount)

        return output_path

    def _prepare_output_video_frame_filename(
            self,
            frame_path:                 str, 
            selected_AI_model:          str, 
            input_resize_factor:        float, 
            output_resize_factor:       float,
            selected_sharpening_amount: float
            ) -> str:
                
        file_path_no_extension, _ = os_path_splitext(frame_path)
        output_path  = file_path_no_extension
        output_path += _build_name_suffix(selected_AI_model, input_resize_factor, output_resize_factor, selected_sharpening_amount)
        output_path += ".jpg"

        return output_path

    def _prepare_upscaled_frame_path_list(
            self,
            extracted_frames_paths:     list[str],
            selected_AI_model:          str,
            input_resize_factor:        int,
            output_resize_factor:       str,
            selected_sharpening_amount: float
            ) -> list[str]:

        return [
            self._prepare_output_video_frame_filename(
                frame_path,
                selected_AI_model,
                input_resize_factor,
                output_resize_factor,
                selected_sharpening_amount
            )
            for frame_path in extracted_frames_paths
        ]



    # Functions to calculate resolutions

    def _get_video_resolution(self, frame: numpy_ndarray) -> tuple[int, int]:
        return frame.shape[0], frame.shape[1] # Ritorna (Altezza, Larghezza)

    def _calculate_input_resolution(self, original_height: int, original_width: int, input_resize_factor: float) -> tuple[int, int]:
        
        aspect_ratio    = original_width / original_height
        AI_input_width  = round((original_width * input_resize_factor) / 2) * 2
        AI_input_height = round((AI_input_width / aspect_ratio) / 2) * 2

        return AI_input_height, AI_input_width
    
    def _calculate_output_resolution(
            self, 
            AI_input_height:      int,
            AI_input_width:       int,
            upscale_factor:       int,
            output_resize_factor: float
        ) -> tuple[int, int]:

        aspect_ratio  = AI_input_width / AI_input_height
        target_width  = round((AI_input_width * upscale_factor * output_resize_factor) / 2) * 2
        target_height = round((target_width / aspect_ratio) / 2) * 2

        return target_height, target_width


    # Functions to calculate optimal threads number

    def _calculate_optimal_threads_number(self, tiles_resolution: int, selected_AI_multithreading: int) -> int:
        
        # 1 Calculate resized frame pixels

        AI_input_pixels_number = self.AI_input_height * self.AI_input_width

        # 2. Calculate max supported pixels
        max_supported_pixels                = tiles_resolution * tiles_resolution
        max_supported_frames_simultaneously = int(max_supported_pixels // AI_input_pixels_number) 

        # 3. Calculate the suitable number of threads to use
        optimal_threads_number = min(max_supported_frames_simultaneously, selected_AI_multithreading)
        if optimal_threads_number <= 0: optimal_threads_number = 1

        return optimal_threads_number



# File / Media Utils -------------------

def image_read(file_path: str) -> numpy_ndarray: 
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_frombuffer(file.read(), uint8), IMREAD_UNCHANGED)
    
def image_write(
        file_path: str, 
        file_data: numpy_ndarray, 
        file_extension: str = ".jpg",
        jpeg_quality: int = 95,
        png_compression: int = 1,
        ) -> None: 
    
    encode_params = []
    ext_lower = file_extension.lower()
    if ext_lower in (".jpg", ".jpeg"):
        encode_params = [IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif ext_lower == ".png":
        encode_params = [IMWRITE_PNG_COMPRESSION, png_compression]

    opencv_imencode(file_extension, file_data, encode_params)[1].tofile(file_path)

def delete_file(file_path: str) -> None:
    if os_path_exists(file_path): os_remove(file_path)

def check_if_file_is_video(file: str) -> bool:
    return os_path_splitext(file)[1].lower() in _supported_video_extensions_set

def prepare_output_image_filename(
        image_path:                 str, 
        selected_output_path:       str,
        selected_AI_model:          str, 
        input_resize_factor:        float, 
        output_resize_factor:       float,
        selected_image_extension:   str,
        selected_sharpening_amount: float
        ) -> str:
        
    output_path  = _build_output_path_base(image_path, selected_output_path)
    output_path += _build_name_suffix(selected_AI_model, input_resize_factor, output_resize_factor, selected_sharpening_amount)
    output_path += selected_image_extension

    return output_path

def get_subprocess_startupinfo() -> Optional[subprocess_STARTUPINFO]:
    # Hide the child process (FFMPEG) console window on Windows; None elsewhere.
    if sys.platform != "win32":
        return None
    startupinfo = subprocess_STARTUPINFO()
    startupinfo.dwFlags |= subprocess_STARTF_USESHOWWINDOW
    return startupinfo

def count_ffmpeg_frames(stdout, counter: list) -> None:
    # FFMPEG -progress writes "frame=N" lines; keep the latest count in counter[0].
    for raw_line in stdout:
        line = raw_line.decode("utf-8", errors = "replace").strip()
        if line.startswith("frame="):
            try: counter[0] = int(line.split("=", 1)[1])
            except ValueError: pass

def run_ffmpeg_with_progress(
        command:          list[str],
        frame_counter:    list[int],
        total_frames:     int,
        status_prefix:    str,
        process_status_q: multiprocessing_Queue,
        event_stop:       multiprocessing_Event, # type: ignore
        startupinfo:      Optional[subprocess_STARTUPINFO],
        idle_priority:    bool = False,
        ) -> Optional[subprocess_Popen]:
    # Run an FFMPEG command, reporting "<status_prefix> N%" progress until it finishes.
    # Returns the finished process, or None if it was stopped early via event_stop.
    ffmpeg_process = subprocess_Popen(
        command,
        stdin       = subprocess_DEVNULL,
        stdout      = subprocess_PIPE,
        stderr      = subprocess_DEVNULL,
        startupinfo = startupinfo
    )

    if idle_priority:
        try: psutil_Process(ffmpeg_process.pid).nice(psutil_IDLE_PRIORITY_CLASS)
        except Exception: pass

    progress_thread = Thread(target = count_ffmpeg_frames, args = (ffmpeg_process.stdout, frame_counter), daemon = True)
    progress_thread.start()

    while ffmpeg_process.poll() is None:
        if event_stop.is_set():
            print("[FFMPEG] Terminating early due to stop event")
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=10)
            except subprocess_TimeoutExpired:
                print("[FFMPEG] Did not terminate in time, killing it")
                ffmpeg_process.kill()
                ffmpeg_process.wait(timeout=5)
            progress_thread.join(timeout=5)
            return None
        percent = int((frame_counter[0] / total_frames) * 100) if total_frames > 0 else 0
        write_process_status(process_status_q, f"{status_prefix} {percent}%")
        sleep(1)

    progress_thread.join(timeout=5)
    return ffmpeg_process

def sanitize_fps(frame_rate: float) -> float:
    # Fall back to 30 fps when the source reports an invalid/zero frame rate.
    return frame_rate if frame_rate and frame_rate > 0 else 30.0

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = sanitize_fps(video_capture.get(CAP_PROP_FPS))
    video_capture.release()
    return frame_rate

def get_image_resolution(image: numpy_ndarray) -> tuple:
    # Return height x width
    return image.shape[0], image.shape[1] 

def get_image_mode(image: numpy_ndarray) -> str:
    shape = image.shape
    if len(shape) == 2:                      return "Grayscale"
    elif len(shape) == 3 and shape[2] == 3:  return "RGB"
    elif len(shape) == 3 and shape[2] == 4:  return "RGBA"

def resize_with_factor(image: numpy_ndarray, factor: float, interpolation: int) -> numpy_ndarray:
    # Resize by a scale factor, forcing even width/height (required by video encoders).
    height, width = get_image_resolution(image)
    new_width  = int(width  * factor)
    new_height = int(height * factor)
    if new_width  % 2 != 0: new_width  += 1
    if new_height % 2 != 0: new_height += 1
    return opencv_resize(image, (new_width, new_height), interpolation = interpolation)

def resize_with_output_factor(image: numpy_ndarray, output_resize_factor: float) -> numpy_ndarray:
    return resize_with_factor(image, output_resize_factor, INTER_LINEAR)

def sharpen_and_save(
        target_path:    str,
        upscaled_image: numpy_ndarray,
        amount:         float,
        file_extension: str = ".jpg"
        ) -> None:
    # Unsharp mask on the AI output only: no original-image pixels are reinjected, so no
    # source noise/compression artifacts can leak back into the result.
    try:
        blurred = opencv_GaussianBlur(upscaled_image, (0, 0), sigmaX = 1.2)
        result  = opencv_addWeighted(upscaled_image, 1 + amount, blurred, -amount, 0, dtype = CV_8U)
        image_write(target_path, result, file_extension)

    except Exception as exception:
        print(f"[{app_name}] AI sharpening failed, saving unsharpened upscaled image: {exception}")
        image_write(target_path, upscaled_image, file_extension)



# Upscale pipeline --------------------

def prevent_sleep() -> None:
    # Keep Windows from sleeping while a batch is running (doesn't force the display to stay on).
    if sys.platform != "win32": return
    import ctypes
    try:
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as e:
        print(f"[{app_name}] Warning: could not prevent system sleep: {e}")

def allow_sleep() -> None:
    if sys.platform != "win32": return
    import ctypes
    try:
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception as e:
        print(f"[{app_name}] Warning: could not restore system sleep state: {e}")

def show_completion_notification() -> None:
    # Windows toast so the user gets pinged even if the app isn't in focus.
    if sys.platform != "win32": return
    try:
        winotify_Notification(
            app_id = app_name,
            title  = app_name,
            msg    = "All files completed! :)",
            icon   = LOGO_PNG_PATH,
        ).show()
    except Exception as e:
        print(f"[{app_name}] Warning: could not show system notification: {e}")

def check_upscale_steps() -> None:
    sleep(1)

    while True:
        actual_step = app_state.process_status_q.get()
        print(f"[{app_name}] check_upscale_steps - {actual_step}")

        if actual_step == CLOSE_APP_STATUS: break

        elif actual_step == STOP_STATUS:
            allow_sleep()
            app_state.info_message.set("Upscaling stopped")
            App.place_upscale_button()
            app_state.window.after(0, lambda: update_file_widget(1, 2, 3))
            if app_state.file_widget is not None: app_state.window.after(0, app_state.file_widget.clear_active_highlight)
            break

        elif actual_step == COMPLETED_STATUS:
            allow_sleep()
            app_state.info_message.set("All files completed! :)")
            for file_path in app_state.selected_file_list:
                if check_if_file_is_video(file_path): app_state.completed_video_files.add(_completed_video_key(file_path))
            stop_upscale_process()
            App.place_upscale_button()
            app_state.window.after(0, lambda: update_file_widget(1, 2, 3))
            if app_state.file_widget is not None: app_state.window.after(0, app_state.file_widget.clear_active_highlight)
            show_completion_notification()
            break

        elif ERROR_STATUS in actual_step:
            allow_sleep()
            app_state.info_message.set("Error while upscaling :(")
            error_to_show = actual_step.replace(ERROR_STATUS, "")
            show_error_message(error_to_show.strip())
            stop_upscale_process()
            App.place_upscale_button()
            app_state.window.after(0, lambda: update_file_widget(1, 2, 3))
            if app_state.file_widget is not None: app_state.window.after(0, app_state.file_widget.clear_active_highlight)
            break

        else:
            app_state.info_message.set(actual_step)
            try:
                file_number = int(actual_step.split('.')[0])
                if app_state.file_widget is not None:
                    app_state.window.after(0, lambda fn = file_number: app_state.file_widget.highlight_active_file(fn))
            except (ValueError, IndexError):
                pass

        sleep(0.25)
        
def write_process_status(process_status_q: multiprocessing_Queue, step: str) -> None:
    
    while not process_status_q.empty(): process_status_q.get()
    process_status_q.put(f"{step}")

def stop_upscale_process() -> None:
    print(f"[{app_name}] stop_upscale_process - setting upscale process stop event")
    app_state.event_stop_upscale_process.set()

    sleep(1)

    if app_state.process_upscale_orchestrator is not None:
        print(f"[{app_name}] stop_upscale_process - waiting for upscale orchestrator to terminate")
        app_state.process_upscale_orchestrator.kill()
        app_state.process_upscale_orchestrator = None
        print(f"[{app_name}] stop_upscale_process - upscale orchestrator terminated")

    try:
        while not app_state.video_frames_and_info_q.empty(): app_state.video_frames_and_info_q.get_nowait()
        print(f"[{app_name}] stop_upscale_process - video_frames_and_info_q cleared")
    except Exception as e:
        print(f"[{app_name}] Warning clearing video_frames_and_info_q: {e}")

    write_process_status(app_state.process_status_q, STOP_STATUS)
    app_state.event_stop_upscale_process.clear()

def stop_button_command() -> None:
    stop_upscale_process()

# ORCHESTRATOR

def upscale_button_command() -> None: 
    processing_config = build_processing_config()

    if processing_config is not None:
        app_state.info_message.set("Loading")

        print("=" * 50)
        print("> Starting upscale:")
        print(f"    Files to upscale: {len(processing_config.selected_file_list)}")
        print(f"    Output path: {processing_config.selected_output_path}")
        print(f"    Selected AI model: {processing_config.selected_AI_model}")
        print(f"    Sharpening amount: {processing_config.selected_sharpening_amount}")
        print(f"    AI multithreading: {processing_config.selected_AI_multithreading}")
        print(f"    Selected GPU: {processing_config.selected_gpu}")
        print(f"    Tiles resolution for selected GPU VRAM: {processing_config.tiles_resolution}x{processing_config.tiles_resolution}px")
        print(f"    Selected image output extension: {processing_config.selected_image_extension}")
        print(f"    Selected video output extension: {processing_config.selected_video_extension}")
        print(f"    Selected video output codec: {processing_config.selected_video_codec}")
        print(f"    Input resize factor: {int(processing_config.input_resize_factor * 100)}%")
        print(f"    Output resize factor: {int(processing_config.output_resize_factor * 100)}%")
        print(f"    Save frames: {processing_config.selected_keep_frames}")
        print("=" * 50)

        App.place_stop_button()

        app_state.completed_video_files.clear()
        app_state.event_stop_upscale_process.clear()
        while not app_state.process_status_q.empty():        app_state.process_status_q.get_nowait()
        while not app_state.video_frames_and_info_q.empty(): app_state.video_frames_and_info_q.get_nowait()

        app_state.process_upscale_orchestrator = multiprocessing_Process(
            target = upscale_orchestrator,
            args = (
                app_state.process_status_q,
                app_state.video_frames_and_info_q,
                app_state.event_stop_upscale_process,
                processing_config.selected_file_list,
                processing_config.selected_output_path,
                processing_config.selected_AI_model,
                processing_config.selected_AI_multithreading,
                processing_config.input_resize_factor,
                processing_config.output_resize_factor,
                processing_config.selected_gpu,
                processing_config.tiles_resolution,
                processing_config.selected_sharpening_amount,
                processing_config.selected_keep_frames,
                processing_config.selected_image_extension,
                processing_config.selected_video_extension,
                processing_config.selected_video_codec,
            )
        )
        prevent_sleep()
        app_state.process_upscale_orchestrator.start()

        Thread(target = check_upscale_steps).start()

def upscale_orchestrator(
        process_status_q:           multiprocessing_Queue,
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event, # type: ignore

        selected_file_list:         list[str],
        selected_output_path:       str,
        selected_AI_model:          str,
        selected_AI_multithreading: int,
        input_resize_factor:        float,
        output_resize_factor:       float,
        selected_gpu:               str,
        tiles_resolution:           int,
        selected_sharpening_amount: float,
        selected_keep_frames:       bool,
        selected_image_extension:   str,
        selected_video_extension:   str,
        selected_video_codec:       str,
        ) -> None:

    try:
        AI_instance_for_images = None

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            if event_stop_upscale_process.is_set(): return
            
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if not os_path_exists(file_path):
                raise FileNotFoundError(f"File not found (moved, renamed or deleted?): {file_path}")

            if check_if_file_is_video(file_path):
                upscale_video(
                    process_status_q            = process_status_q,
                    video_frames_and_info_q     = video_frames_and_info_q,
                    event_stop_upscale_process  = event_stop_upscale_process,
                    video_path                  = file_path, 
                    file_number                 = file_number,
                    selected_output_path        = selected_output_path, 
                    selected_AI_model           = selected_AI_model, 
                    selected_sharpening_amount  = selected_sharpening_amount,
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
                if AI_instance_for_images is None:
                    AI_instance_for_images = AI_upscale(selected_AI_model, selected_gpu, input_resize_factor, tiles_resolution)
                
                upscale_image(
                    process_status_q         = process_status_q,
                    image_path               = file_path, 
                    file_number              = file_number,
                    selected_output_path     = selected_output_path,
                    AI_instance              = AI_instance_for_images,
                    selected_AI_model        = selected_AI_model,
                    selected_image_extension = selected_image_extension, 
                    input_resize_factor      = input_resize_factor, 
                    output_resize_factor     = output_resize_factor,
                    selected_sharpening_amount = selected_sharpening_amount
                )

        if not event_stop_upscale_process.is_set(): write_process_status(process_status_q, f"{COMPLETED_STATUS}")

    except Exception as exception:
        error_message = str(exception)
        write_process_status(process_status_q, f"{ERROR_STATUS} {error_message}")
 
# IMAGES

def upscale_image(
        process_status_q:           multiprocessing_Queue,
        image_path:                 str, 
        file_number:                int,
        selected_output_path:       str,
        AI_instance:                AI_upscale,
        selected_AI_model:          str,
        selected_image_extension:   str,
        input_resize_factor:        float, 
        output_resize_factor:       float,
        selected_sharpening_amount: float
        ) -> None:
    
    write_process_status(process_status_q, f"{file_number}. Upscaling image")

    # 1. Read the image file
    starting_image = image_read(image_path)

    # 2. Prepare upscaled image path
    upscaled_image_path = prepare_output_image_filename(image_path, selected_output_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_image_extension, selected_sharpening_amount)
    
    # 3. Upscale the image
    upscaled_image = AI_instance.AI_orchestration(starting_image)
    
    # 4. Resize the image with output scale
    upscaled_image = resize_with_output_factor(upscaled_image, output_resize_factor) 

    # 5. Sharpen the upscaled image OR save it as-is
    if selected_sharpening_amount > 0:
        sharpen_and_save(
            target_path    = upscaled_image_path, 
            upscaled_image = upscaled_image, 
            amount         = selected_sharpening_amount, 
            file_extension = selected_image_extension
        )
    else:
        image_write(
            file_path      = upscaled_image_path, 
            file_data      = upscaled_image, 
            file_extension = selected_image_extension
        )

# VIDEOS

# Function executed as process

def upscale_video_frames_async(
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event, # type: ignore
        video_upscale_task:         VideoUpscaleTask,
        frame_chunk:                list[tuple[str, str]]
        ) -> None:
    
    process_pid = os_getpid()
    psutil_Process(process_pid).nice(psutil_IDLE_PRIORITY_CLASS)

    opencv_setNumThreads(1)

    AI_instance = AI_upscale(
        selected_AI_model   = video_upscale_task.selected_AI_model, 
        selected_gpu        = video_upscale_task.selected_gpu, 
        input_resize_factor = video_upscale_task.input_resize_factor, 
        tiles_resolution    = video_upscale_task.tiles_resolution
    )

    for input_path, output_path in frame_chunk:
        
        if event_stop_upscale_process.is_set():
            print(f"[Upscale process {process_pid}] Terminating early due to stop event")
            break

        start_timer = timer()
                
        # Upscale frame
        starting_frame = image_read(input_path)
        upscaled_frame = AI_instance.AI_orchestration(starting_frame)

        # Calculate processing time
        end_timer       = timer()
        processing_time = round((end_timer - start_timer), 3)

        # Add things in queue
        success = False
        while not success:
            try:
                video_frames_and_info_q.put_nowait(
                    {
                        "upscaled_frame":      upscaled_frame,
                        "upscaled_frame_path": output_path,
                        "processing_time":     processing_time
                    }
                )
                success = True
                break
            except Full:
                sleep(0.1)

    if event_stop_upscale_process.is_set():
        print(f"[Upscale process {process_pid}] Terminated")
    else:
        print(f"[Upscale process {process_pid}] finished the job")

# -------------------------------------

def upscale_video(
        process_status_q:           multiprocessing_Queue,
        video_frames_and_info_q:    multiprocessing_Queue,
        event_stop_upscale_process: multiprocessing_Event, # type: ignore
        video_path:                 str, 
        file_number:                int,
        selected_output_path:       str,
        selected_AI_model:          str,
        selected_sharpening_amount: float,
        selected_AI_multithreading: int,
        selected_gpu:               str,
        input_resize_factor:        float,
        output_resize_factor:       float,
        tiles_resolution:           int, 
        selected_video_extension:   str,
        selected_video_codec:       str,
        selected_keep_frames:       bool,
        ) -> None:
    
    # Internal functions

    def create_dir(name_dir: str) -> None:
        
        if os_path_exists(name_dir):     remove_directory(name_dir)
        if not os_path_exists(name_dir): os_makedirs(name_dir, mode=0o777)

        if sys.platform == "win32":
            try:
                win32_SetFileAttributes(name_dir, win32_FILE_ATTRIBUTE_NOT_CONTENT_INDEXED)
            except Exception as e:
                print(f"[create_dir] Error setting NOT_CONTENT_INDEXED attribute: {e}")

            desktop_ini = os_path_join(name_dir, "desktop.ini")
            try:
                with open(desktop_ini, "w", encoding="utf-8") as f: f.write("[.ShellClassInfo]\nNoIndexing=1\n")
            except Exception as e:
                print(f"[create_dir] Error creating desktop.ini: {e}")

            try:
                win32_SetFileAttributes(name_dir,    win32_FILE_ATTRIBUTE_SYSTEM)
                win32_SetFileAttributes(desktop_ini, win32_FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                print(f"[create_dir] Error setting folder/ini attributes: {e}")

    def get_frames_for_resume(target_directory: str, selected_AI_model: str) -> list[str]:

        if not os_path_exists(target_directory): return []
        directory_files      = os_listdir(target_directory)
        upscaled_frames_path = [f for f in directory_files if selected_AI_model in f]
        if len(upscaled_frames_path) <= 1: return []
        original_frames_path = [f for f in directory_files if f.endswith('.jpg') and selected_AI_model not in f]

        return natsorted([os_path_join(target_directory, f) for f in original_frames_path])

    def extract_video_frames(
            process_status_q:           multiprocessing_Queue,
            event_stop_upscale_process: multiprocessing_Event, # type: ignore
            file_number:                int,
            target_directory:           str,
            video_path:                 str,
            ) -> list[str]:

        extracted_frame_count = [0]

        # 1. Get total number of frames and fps
        video_capture       = opencv_VideoCapture(video_path)
        video_frames_number = int(video_capture.get(CAP_PROP_FRAME_COUNT))
        video_fps           = sanitize_fps(video_capture.get(CAP_PROP_FPS))
        video_capture.release()

        # 2. Create directory to extract frames
        create_dir(target_directory)

        # 3. Create FFMPEG command to extract video frames
        # -progress pipe:1 writes structured progress ("frame=N" lines) to stdout
        # -nostats suppresses the default stderr stats overlay
        output_pattern = os_path_join(target_directory, "frame_%03d.jpg")
        extraction_command = [
            FFMPEG_EXE_PATH,
            "-y",
            "-loglevel",   "error",
            "-progress",   "pipe:1",
            "-nostats",
            "-threads",    "0",
            "-err_detect", "ignore_err",
            "-hwaccel",    "auto",
            "-i",          video_path,
            "-vf",         f"fps={video_fps}",
            "-an",
            "-qscale:v",   "3",
            output_pattern
        ]

        # 4. Execute FFMPEG command
        startupinfo = get_subprocess_startupinfo()

        ffmpeg_process = None
        try:
            ffmpeg_process = run_ffmpeg_with_progress(
                command          = extraction_command,
                frame_counter    = extracted_frame_count,
                total_frames     = video_frames_number,
                status_prefix    = f"{file_number}. Extracting video frames",
                process_status_q = process_status_q,
                event_stop       = event_stop_upscale_process,
                startupinfo      = startupinfo,
                idle_priority    = True,
            )
            if ffmpeg_process is None:
                return []

        except Exception as e:
            write_process_status(process_status_q, f"{ERROR_STATUS} Frame extraction failed: {e}")
            if ffmpeg_process: ffmpeg_process.kill()
            return []

        # 5. Get extracted frames paths and return
        extracted_files = [
            os_path_join(target_directory, f)
            for f in natsorted(os_listdir(target_directory))
            if f.endswith(".jpg") and f.startswith("frame_")
        ]

        return extracted_files

    def calculate_time_to_complete_video(time_for_frame: float, remaining_frames: int) -> str:
        
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

    def update_video_process_status(
            process_status_q:        multiprocessing_Queue, 
            file_number:             int,
            upscaled_count:          int,
            extracted_frames_number: int,
            average_processing_time: float
            ) -> None:

        frames_left = extracted_frames_number - upscaled_count
        time_left   = calculate_time_to_complete_video(average_processing_time, frames_left)

        if time_left != "":
            percent = int((upscaled_count / extracted_frames_number) * 100)
            write_process_status(process_status_q, f"{file_number}. Upscaling video {percent}% ({time_left})")

    def manage_upscaled_frames_save_on_disk(
            process_status_q:                multiprocessing_Queue,
            video_frames_and_info_q:         multiprocessing_Queue,
            event_stop_upscale_process:      multiprocessing_Event, # type: ignore
            event_stop_upscaled_save_thread: multiprocessing_Event, # type: ignore
            file_number:                     int,
            video_upscale_task:              VideoUpscaleTask,
            ) -> None:

        opencv_setNumThreads(1)

        def _internal_save_frame(
                upscaled_frame:             numpy_ndarray, 
                upscaled_frame_path:        str, 
                selected_sharpening_amount: float
                ) -> None:

            if selected_sharpening_amount > 0:
                sharpen_and_save(upscaled_frame_path, upscaled_frame, selected_sharpening_amount)
            else:
                image_write(upscaled_frame_path, upscaled_frame, jpeg_quality=90)


        # Main
        current_upscaled_count = video_upscale_task.upscaled_frames_counter
        UPDATE_STATUS_TIMER    = 3.0
        processing_times_list  = []
        last_update_time       = timer()

        with ThreadPoolExecutor(max_workers=4) as executor:
            threads_set = set()

            while True:
                if event_stop_upscale_process.is_set():
                    print("[Frames saving thread] terminating by upscale stop event")
                    break

                if event_stop_upscaled_save_thread.is_set() and video_frames_and_info_q.empty():
                    print("[Frames saving thread] terminating correctly")
                    break

                try:
                    item = video_frames_and_info_q.get_nowait()
                    current_upscaled_count += 1
                except Empty:
                    sleep(0.1)
                    continue

                upscaled_frame      = item["upscaled_frame"]
                upscaled_frame_path = item["upscaled_frame_path"]
                processing_time     = item["processing_time"]

                processing_times_list.append(processing_time/video_upscale_task.optimal_threads_number)

                threads_set.add(
                    executor.submit(
                        _internal_save_frame,
                        upscaled_frame,
                        upscaled_frame_path,
                        video_upscale_task.selected_sharpening_amount
                    )
                )

                now = timer()
                if now - last_update_time >= UPDATE_STATUS_TIMER:
                    last_update_time = now

                    done_threads = {t for t in threads_set if t.done()}
                    threads_set -= done_threads

                    if processing_times_list:
                        update_video_process_status(
                            process_status_q        = process_status_q,
                            file_number             = file_number,
                            upscaled_count          = current_upscaled_count,
                            extracted_frames_number = video_upscale_task.extracted_frames_number,
                            average_processing_time = numpy_mean(processing_times_list)
                        )
                        processing_times_list = []

            for t in threads_set: t.result()

    def upscale_video_frames(
            process_status_q:           multiprocessing_Queue,
            video_frames_and_info_q:    multiprocessing_Queue,
            event_stop_upscale_process: multiprocessing_Event, # type: ignore
            file_number:                int,
            video_upscale_task:         VideoUpscaleTask
            ) -> None:

        event_stop_upscaled_save_thread = multiprocessing_Event()
        
        if not video_upscale_task.frames_paths_to_upscale: # 2. NO frames to upscale - set event save thread to close and exit
            write_process_status(process_status_q, f"{file_number}. All frames already upscaled!")
            event_stop_upscaled_save_thread.set()
            return
        else: # 3. If there are frames to upscale - start thread to save upscaled frames on disk
            save_thread = Thread(
                target = manage_upscaled_frames_save_on_disk,
                args = (
                    process_status_q, 
                    video_frames_and_info_q, 
                    event_stop_upscale_process,
                    event_stop_upscaled_save_thread,
                    file_number, 
                    video_upscale_task
                )
            )
            save_thread.start()
            
            # 4. Start upscale process
            write_process_status(process_status_q, f"{file_number}. Upscaling video ({video_upscale_task.optimal_threads_number} threads)")
            with multiprocessing_Pool(video_upscale_task.optimal_threads_number) as pool:
                pool.starmap(
                    upscale_video_frames_async,
                    zip(
                        repeat(video_frames_and_info_q),
                        repeat(event_stop_upscale_process),
                        repeat(video_upscale_task),
                        video_upscale_task.frames_chunks_list,
                    )
                )

            # 5. Signal save thread and wait for it to finish
            write_process_status(process_status_q, f"{file_number}. Finalizing upscaling")
            event_stop_upscaled_save_thread.set()
            save_thread.join(timeout=30)
            if save_thread.is_alive():
                print(f"[{file_number}] Warning: save thread did not finish within timeout")

    def encode_upscaled_video(process_status_q: multiprocessing_Queue, video_upscale_task: VideoUpscaleTask) -> None:

        encoded_frame_count = [0]

        # Cleaning files from previous encoding
        delete_file(video_upscale_task.ffmpeg_txt_file_path)

        # Create a file .txt with all upscaled video frames paths || this file is essential
        with os_fdopen(os_open(video_upscale_task.ffmpeg_txt_file_path, os_O_WRONLY | os_O_CREAT, 0o777), 'w', encoding="utf-8") as txt:
            for frame_path in video_upscale_task.upscaled_frame_paths:
                if os_path_exists(frame_path):
                    safe_path = os_path_abspath(frame_path).replace("\\", "/").replace("'", "'\\''")
                    txt.write(f"file '{safe_path}' \n")

        # Create the upscaled video trying with selected codec OR x264 codec fallback
        codecs_to_try = [video_upscale_task.effective_codec, "libx264"]

        total_frames = video_upscale_task.extracted_frames_number

        startupinfo = get_subprocess_startupinfo()

        for current_codec in codecs_to_try:
            print(f"[FFMPEG] upscaled video encoding with ({current_codec})")
            encoded_frame_count[0] = 0
            ffmpeg_process = None

            try:
                encoding_command = [
                    str(FFMPEG_EXE_PATH),
                    "-y",
                    "-loglevel",   "error",
                    "-progress",   "pipe:1",
                    "-nostats",
                    "-f",          "concat",
                    "-safe",       "0",
                    "-r",          str(video_upscale_task.video_fps),
                    "-i",          str(video_upscale_task.ffmpeg_txt_file_path),
                    "-i",          str(video_upscale_task.video_path),
                    "-map",        "0:v:0",
                    "-map",        "1:a?",
                    "-c:v",        str(current_codec),
                    "-c:a",        "copy",
                    "-g",          str(video_upscale_task.video_fps),
                    "-vf",         f"scale={video_upscale_task.target_width}:{video_upscale_task.target_height},format=yuv420p",
                    "-color_range","tv",
                    "-movflags",   "+faststart",
                    "-b:v",        "50000k",
                    str(video_upscale_task.video_output_path)
                ]

                ffmpeg_process = run_ffmpeg_with_progress(
                    command          = encoding_command,
                    frame_counter    = encoded_frame_count,
                    total_frames     = total_frames,
                    status_prefix    = f"{file_number}. Encoding upscaled video",
                    process_status_q = process_status_q,
                    event_stop       = event_stop_upscale_process,
                    startupinfo      = startupinfo,
                    idle_priority    = True,
                )
                if ffmpeg_process is None:
                    return

                if ffmpeg_process.returncode != 0:
                    raise RuntimeError(f"ffmpeg exited with code {ffmpeg_process.returncode}")

                delete_file(video_upscale_task.ffmpeg_txt_file_path)
                print(f"[FFMPEG] encoding completed with ({current_codec})")
                break

            except Exception as e:
                if ffmpeg_process:
                    ffmpeg_process.kill()
                    try:
                        ffmpeg_process.wait(timeout=10)
                    except subprocess_TimeoutExpired:
                        print("[FFMPEG] Process did not exit after kill()")
                if current_codec != "libx264":
                    write_process_status(process_status_q, f"{file_number}. Encoding upscaled video (x264 fallback)")
                    delete_file(video_upscale_task.video_output_path)
                    continue
                else:
                    raise RuntimeError(f"Video encoding failed with all codecs: {e}")


    
    # Main function

    # 1. Preparation
    video_upscale_task = VideoUpscaleTask(
        video_path                  = video_path,
        selected_output_path        = selected_output_path,
        selected_AI_model           = selected_AI_model,
        selected_AI_multithreading  = selected_AI_multithreading,
        selected_gpu                = selected_gpu,
        tiles_resolution            = tiles_resolution,
        input_resize_factor         = input_resize_factor,
        output_resize_factor        = output_resize_factor,
        selected_sharpening_amount  = selected_sharpening_amount,
        selected_video_extension    = selected_video_extension,
        selected_video_codec        = selected_video_codec
    )
        
    extracted_frames_paths = get_frames_for_resume(video_upscale_task.target_directory, video_upscale_task.selected_AI_model)

    if extracted_frames_paths:
        write_process_status(process_status_q, f"{file_number}. Resume video upscaling")
    else:
        write_process_status(process_status_q, f"{file_number}. Extracting video frames")
        extracted_frames_paths = extract_video_frames(
            process_status_q           = process_status_q,
            event_stop_upscale_process = event_stop_upscale_process,
            file_number                = file_number,
            target_directory           = video_upscale_task.target_directory,
            video_path                 = video_path
        )
    
    if not extracted_frames_paths: return

    video_upscale_task._complete_init(extracted_frames_paths)


    # 2. Upscaling video frames
    write_process_status(process_status_q, f"{file_number}. Upscaling video") 
    upscale_video_frames(
        process_status_q           = process_status_q, 
        video_frames_and_info_q    = video_frames_and_info_q,
        event_stop_upscale_process = event_stop_upscale_process,
        file_number                = file_number,
        video_upscale_task         = video_upscale_task
    )


    # 3. Video encoding
    if event_stop_upscale_process.is_set(): return
    write_process_status(process_status_q, f"{file_number}. Encoding upscaled video")
    encode_upscaled_video(process_status_q, video_upscale_task)


    # 4. Delete frames folder
    if selected_keep_frames == False:
        write_process_status(process_status_q, f"{file_number}. Removing video frames")
        if os_path_exists(video_upscale_task.target_directory): remove_directory(video_upscale_task.target_directory)



# GUI widgets -----------------------

def lerp_hex(color_a: str, color_b: str, t: float) -> str:
    # Linear interpolation between two #RRGGBB colours; t in [0, 1].
    a = (int(color_a[1:3], 16), int(color_a[3:5], 16), int(color_a[5:7], 16))
    b = (int(color_b[1:3], 16), int(color_b[3:5], 16), int(color_b[5:7], 16))
    r, g, blue = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{r:02X}{g:02X}{blue:02X}"


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

        self.configure(fg_color = background_color)

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

    def _ok_event(self, event = None) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(self) -> None:
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
            title_subtitle_text_color = UI_ACCENT_COLOR
        elif self._messageType == "error":
            title_subtitle_text_color = "#FF5C5C"

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
                text_color = CARD_MUTED_COLOR,
                font       = bold17,
                text       = f"Default: {self._default_value}"
                )
        
        subtitleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = CARD_VALUE_COLOR,
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
                text_color    = CARD_VALUE_COLOR,
                fg_color      = CARD_BACKGROUND_COLOR,
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

    def _copy_log_to_clipboard(self, copy_button: CTkButton) -> None:
        self.clipboard_clear()
        self.clipboard_append("\n".join(self._option_list))
        self.update()
        copy_button.configure(text = "Copied!")

    def placeInfoMessageOkButton(self) -> None:
        
        self._ctkwidgets_index += 1

        button_row = CTkFrame(self, fg_color = "transparent")
        button_row.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = (10, 20), pady = (10, 20), sticky = "e")

        if self._messageType == "error":
            copy_button = CTkButton(
                master  = button_row,
                text    = 'Copy log',
                width   = 125,
                font          = bold11,
                border_width  = 2,
                corner_radius = UI_CORNER_RADIUS,
                fg_color      = CARD_BACKGROUND_COLOR,
                hover_color   = widget_background_color,
                text_color    = "#E0E0E0",
                border_color  = UI_ACCENT_COLOR
            )
            copy_button.configure(command = lambda: self._copy_log_to_clipboard(copy_button))
            copy_button.pack(side = "left", padx = (0, 10))

        ok_button = CTkButton(
            master  = button_row,
            command = self._ok_event,
            text    = 'OK',
            width   = 125,
            font          = bold11,
            border_width  = 2,
            corner_radius = UI_CORNER_RADIUS,
            fg_color      = CARD_BACKGROUND_COLOR,
            hover_color   = widget_background_color,
            text_color    = "#E0E0E0",
            border_color  = UI_ACCENT_COLOR
        )
        ok_button.pack(side = "left")

    def _create_widgets(self) -> None:

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
        if app_state is not None:
            app_state.file_widget = None
            app_state.selected_file_list = []
        self.destroy()
        App.place_loadFile_section()

    def _create_widgets(self) -> None:
        self.add_clean_button()
        self._render_cards()

    def _render_cards(self) -> None:
        for file_path in self.file_list:
            item = self._create_file_card(file_path)
            if item is not None:
                self.ui_components.append(item)

    def _remove_file(self, file_path: str) -> None:
        if app_state is not None and app_state.process_upscale_orchestrator is not None: return # ignore while an upscale is running
        if file_path not in self.file_list: return

        self.file_list.remove(file_path)
        if not self.file_list:
            self._destroy_()
            return

        self.clean_file_list()
        self._render_cards()

    def _create_file_card(self, file_path) -> Optional[dict]:
        is_video, width, height, num_frames, frame_rate = self._read_media_properties(file_path)
        file_icon = self.extract_file_icon(file_path)

        # Card container
        card = CTkFrame(self, fg_color = CARD_BACKGROUND_COLOR, corner_radius = 12, border_width = 2, border_color = CARD_BORDER_COLOR)
        card.grid(row = self.index_row, column = 0, columnspan = 3, padx = 6, pady = (3, 9), sticky = "ew")
        card.grid_columnconfigure(1, weight = 1)

        # Thumbnail
        thumbnail = CTkLabel(card, text = "", image = file_icon)
        thumbnail.grid(row = 0, column = 0, padx = (12, 14), pady = 12, sticky = "n")

        # Remove-file button
        remove_button = CTkButton(card, command = lambda: self._remove_file(file_path), text = "X", width = 24, height = 24, font = bold11, fg_color = "transparent", hover_color = "#8C3B3B", border_width = 2, border_color = MESSAGE_ERROR_COLOR, text_color = MESSAGE_ERROR_COLOR, corner_radius = UI_CORNER_RADIUS)
        remove_button.grid(row = 0, column = 2, padx = (0, 8), pady = 12, sticky = "n")
        for widget in (remove_button, remove_button._text_label):
            widget.bind("<Enter>", lambda e: remove_button._text_label.configure(fg = "#FFFFFF"), add = "+")
            widget.bind("<Leave>", lambda e: remove_button._text_label.configure(fg = MESSAGE_ERROR_COLOR), add = "+")

        # Text content column
        content = CTkFrame(card, fg_color = "transparent")
        content.grid(row = 0, column = 1, sticky = "ew", padx = (0, 14), pady = 11)
        content.grid_columnconfigure(0, weight = 1)

        content_row = 0

        # File name
        name_label = CTkLabel(content, text = os_path_basename(file_path), font = bold14, text_color = CARD_TITLE_COLOR, anchor = "w", justify = "left")
        name_label.grid(row = content_row, column = 0, sticky = "ew")
        content_row += 1

        # Source meta line (duration / resolution / fps for videos, resolution for images)
        meta_label = CTkLabel(content, text = self._format_source_meta(is_video, width, height, num_frames, frame_rate), font = bold12, text_color = CARD_MUTED_COLOR, anchor = "w")
        meta_label.grid(row = content_row, column = 0, sticky = "ew", pady = (2, 0))
        content_row += 1

        dynamic_section = self._create_dynamic_section(content, content_row, is_video, file_path, width, height)

        self.index_row += 1
        return {"card": card, "content": content, "dynamic_row": content_row, "is_video": is_video, "file_path": file_path, "width": width, "height": height, "dynamic_section": dynamic_section}

    def _create_dynamic_section(
            self,
            content,
            start_row,
            is_video,
            file_path,
            width,
            height
            ) -> Optional[CTkFrame]:

        resume_percent  = get_video_resume_progress(file_path) if is_video else None
        has_pipeline    = self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0
        display_percent = resume_percent if resume_percent is not None else 0

        if not is_video and not has_pipeline:
            return None

        section = CTkFrame(content, fg_color = "transparent")
        section.grid_columnconfigure(0, weight = 1)
        section.grid(row = start_row, column = 0, sticky = "ew")

        inner_row = 0

        if is_video:
            badge = self._create_resume_badge(section, display_percent)
            badge.grid(row = inner_row, column = 0, sticky = "ew", pady = (8, 0))
            inner_row += 1

        if has_pipeline:
            separator = CTkFrame(section, fg_color = CARD_BORDER_COLOR, height = 1)
            separator.grid(row = inner_row, column = 0, sticky = "ew", pady = (7, 6))
            inner_row += 1

            pipeline_rows  = self._compute_pipeline_rows(width, height)
            pipeline_table = self._create_pipeline_table(section, pipeline_rows)
            pipeline_table.grid(row = inner_row, column = 0, sticky = "ew")

        return section

    def add_clean_button(self) -> None:

        button = CTkButton(
            master        = self, 
            command       = self._destroy_,
            text          = "CLEAN",
            image         = clear_icon,
            width         = 90, 
            height        = 28,
            font          = bold11,
            border_width  = 2,
            corner_radius = UI_CORNER_RADIUS,
            fg_color      = "#282828",
            text_color    = "#E0E0E0",
            border_color  = UI_ACCENT_COLOR
        )
        
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        



    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 60

        if check_if_file_is_video(file_path):
            video_cap    = opencv_VideoCapture(file_path)
            ret, frame   = video_cap.read()
            video_cap.release()
            if not ret or frame is None:
                return CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")), size=(60, 60))
            source_icon = opencv_cvtColor(frame, COLOR_BGR2RGB)
        else:
            source_icon = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)

        ratio       = min(max_size / source_icon.shape[0], max_size / source_icon.shape[1])
        new_width   = int(source_icon.shape[1] * ratio)
        new_height  = int(source_icon.shape[0] * ratio)
        source_icon = opencv_resize(source_icon,(new_width, new_height))
        ctk_icon    = CTkImage(pillow_image_fromarray(source_icon, mode="RGB"), size = (new_width, new_height))

        return ctk_icon
        
    def _read_media_properties(self, file_path) -> tuple[bool, int, int, int, float]:
        if check_if_file_is_video(file_path):
            cap        = opencv_VideoCapture(file_path)
            width      = round(cap.get(CAP_PROP_FRAME_WIDTH))
            height     = round(cap.get(CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
            frame_rate = sanitize_fps(cap.get(CAP_PROP_FPS))
            cap.release()
            return True, width, height, num_frames, frame_rate

        height, width = get_image_resolution(image_read(file_path))
        return False, width, height, 0, 0.0

    def _format_source_meta(self, is_video, width, height, num_frames, frame_rate) -> str:
        if not is_video:
            return f"{width}×{height}"
        duration = num_frames / frame_rate if frame_rate else 0
        minutes  = int(duration / 60)
        seconds  = round(duration % 60)
        return f"{minutes}m {seconds}s  |  {width}×{height}  |  {round(frame_rate, 1)} fps"

    def _compute_pipeline_rows(self, width, height) -> list[tuple]:
        input_width   = int(width  * (self.input_resize_factor  / 100))
        input_height  = int(height * (self.input_resize_factor  / 100))
        ai_width      = int(input_width  * self.upscale_factor)
        ai_height     = int(input_height * self.upscale_factor)
        output_width  = int(ai_width  * (self.output_resize_factor / 100))
        output_height = int(ai_height * (self.output_resize_factor / 100))

        # (label, detail, resolution, is_ai)
        return [
            ("Input",  f"{self.input_resize_factor}%",  f"{input_width}×{input_height}",   False),
            ("AI",     f"x{self.upscale_factor}",       f"{ai_width}×{ai_height}",         True),
            ("Output", f"{self.output_resize_factor}%", f"{output_width}×{output_height}", False),
        ]

    def _create_resume_badge(self, parent, percent) -> CTkFrame:
        badge = CTkFrame(parent, fg_color = RESUME_BADGE_COLOR, corner_radius = 8)
        badge.grid_columnconfigure(1, weight = 1)

        CTkLabel(badge, text = "Completed", font = bold12, text_color = RESUME_ACCENT_COLOR, anchor = "w").grid(row = 0, column = 0, padx = (10, 10), pady = 7, sticky = "w")

        bar = CTkProgressBar(badge, height = 8, corner_radius = 4, fg_color = CARD_BORDER_COLOR, progress_color = RESUME_ACCENT_COLOR)
        bar.set(0)
        bar.grid(row = 0, column = 1, pady = 7, sticky = "ew")

        percent_label = CTkLabel(badge, text = "0%", font = bold12, text_color = RESUME_ACCENT_COLOR, width = 42, anchor = "e")
        percent_label.grid(row = 0, column = 2, padx = (10, 10), pady = 7, sticky = "e")

        self._animate_progress_bar(bar, percent / 100, percent_label, percent)

        return badge

    def _animate_progress_bar(self, bar, target, percent_label = None, percent = 0, current = 0.0) -> None:
        # Ease-out fill from 0 to the target value, with the colour fading in from a dim
        # tone to the full accent and the percentage counting up; stops if destroyed.
        if not bar.winfo_exists(): return
        current += (target - current) * 0.08
        ratio = current / target if target > 0 else 1.0
        bar.configure(progress_color = lerp_hex(RESUME_BAR_DIM_COLOR, RESUME_ACCENT_COLOR, min(1.0, ratio)))
        if percent_label is not None and percent_label.winfo_exists():
            percent_label.configure(text = f"{round(current * 100)}%")
        if target - current < 0.005:
            bar.set(target)
            if percent_label is not None and percent_label.winfo_exists():
                percent_label.configure(text = f"{percent}%")
            if percent >= 100:
                self._glow_bar(bar)
            else:
                bar.configure(progress_color = RESUME_ACCENT_COLOR)
            return
        bar.set(current)
        bar.after(20, lambda: self._animate_progress_bar(bar, target, percent_label, percent, current))

    def _glow_bar(self, bar, step = 0, sequence = None) -> None:
        # Slow double pulse when the bar reaches 100%, settling back on the accent.
        if not bar.winfo_exists(): return
        if sequence is None:
            peak = "#C4FFDC"
            up   = [lerp_hex(RESUME_ACCENT_COLOR, peak, i / 6) for i in range(7)]
            down = [lerp_hex(peak, RESUME_ACCENT_COLOR, i / 6) for i in range(1, 7)]
            sequence = up + down + up + down   # two slow pulses
        if step >= len(sequence):
            bar.configure(progress_color = RESUME_ACCENT_COLOR)
            return
        bar.configure(progress_color = sequence[step])
        bar.after(45, lambda: self._glow_bar(bar, step + 1, sequence))

    def _create_pipeline_table(self, parent, pipeline_rows) -> CTkFrame:
        table = CTkFrame(parent, fg_color = "transparent")
        table.grid_columnconfigure(0, minsize = 96)   # label
        table.grid_columnconfigure(1, minsize = 96)   # detail (same width as label -> equal spacing)
        table.grid_columnconfigure(2, weight = 1)     # resolution (fills remaining space)

        for row_index, (label, detail, resolution, is_ai) in enumerate(pipeline_rows):
            label_color  = CARD_ACCENT_COLOR if is_ai else CARD_MUTED_COLOR
            detail_color = CARD_ACCENT_COLOR if is_ai else CARD_FAINT_COLOR
            value_color  = CARD_ACCENT_COLOR if is_ai else CARD_VALUE_COLOR
            CTkLabel(table, text = label,      font = bold12, text_color = label_color,  anchor = "w", height = 20).grid(row = row_index, column = 0, sticky = "w", pady = 0)
            CTkLabel(table, text = detail,     font = bold12, text_color = detail_color, anchor = "w", height = 20).grid(row = row_index, column = 1, sticky = "w", pady = 0)
            CTkLabel(table, text = resolution, font = bold12, text_color = value_color,  anchor = "w", height = 20).grid(row = row_index, column = 2, sticky = "w", padx = (0, 14), pady = 0)

        return table


    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        self.index_row = 1
        for item in self.ui_components: item["card"].destroy()
        self.ui_components = []

    def refresh_pipeline(self) -> None:
        for item in self.ui_components:
            if item["dynamic_section"] is not None:
                item["dynamic_section"].destroy()
            item["dynamic_section"] = self._create_dynamic_section(item["content"], item["dynamic_row"], item["is_video"], item["file_path"], item["width"], item["height"])
    
    def get_selected_file_list(self) -> list: 
        return self.file_list 

    def highlight_active_file(self, file_number: int) -> None:
        # Accent border on the card currently being processed (1-based, matches processing order).
        for index, item in enumerate(self.ui_components):
            if not item["card"].winfo_exists(): continue
            item["card"].configure(border_color = CARD_ACCENT_COLOR if index == file_number - 1 else CARD_BORDER_COLOR)

    def clear_active_highlight(self) -> None:
        for item in self.ui_components:
            if item["card"].winfo_exists(): item["card"].configure(border_color = CARD_BORDER_COLOR)

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_input_resize_factor(self, input_resize_factor) -> None:
        self.input_resize_factor = input_resize_factor

    def set_output_resize_factor(self, output_resize_factor) -> None:
        self.output_resize_factor = output_resize_factor
 
# GUI state & validation -----------------

def get_values_for_file_widget() -> tuple:
    # Upscale factor
    upscale_factor = get_upscale_factor()

    # Input resolution %
    try:
        input_resize_factor = int(float(str(app_state.selected_input_resize_factor.get())))
    except Exception:
        input_resize_factor = 0

    # Output resolution %
    try:
        output_resize_factor = int(float(str(app_state.selected_output_resize_factor.get())))
    except Exception:
        output_resize_factor = 0

    return upscale_factor, input_resize_factor, output_resize_factor

def update_file_widget(a, b, c) -> None:
    file_widget = None if app_state is None else app_state.file_widget
    if file_widget is None:
        return
        
    upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_resize_factor)
    file_widget.set_output_resize_factor(output_resize_factor)
    file_widget.refresh_pipeline()

def get_current_ai_model() -> str:
    return app_state.preferences.ai_model

def get_current_ai_multithreading() -> int:
    if app_state.preferences.ai_multithreading == "OFF":
        return 1
    return int(app_state.preferences.ai_multithreading.split()[0])

def get_current_sharpening_amount() -> float:
    return {
        "OFF": 0,
        "Low": 0.3,
        "High": 0.5,
    }.get(app_state.preferences.sharpening, 0)

def _completed_video_key(video_path: str) -> tuple:
    # Ties the "already completed" flag to the settings used when it was completed, so
    # changing AI model/sharpening/resize factors doesn't keep showing a stale 100% badge.
    return (
        video_path,
        app_state.preferences.ai_model,
        app_state.preferences.sharpening,
        str(app_state.selected_input_resize_factor.get()),
        str(app_state.selected_output_resize_factor.get()),
    )

def get_video_resume_progress(video_path: str) -> Optional[int]:
    # Extension is ignored on purpose: extracted frames are always ".jpg" and upscaled
    # frames always carry the AI model name, whatever output extension is selected.
    if _completed_video_key(video_path) in app_state.completed_video_files: return 100

    selected_AI_model          = get_current_ai_model()
    selected_output_path       = app_state.selected_output_path.get()
    selected_sharpening_amount = get_current_sharpening_amount()

    try:
        input_resize_factor  = int(float(str(app_state.selected_input_resize_factor.get()))) / 100
        output_resize_factor = int(float(str(app_state.selected_output_resize_factor.get()))) / 100
    except Exception:
        return None

    target_directory  = _build_output_path_base(video_path, selected_output_path)
    target_directory += _build_name_suffix(selected_AI_model, input_resize_factor, output_resize_factor, selected_sharpening_amount)

    if not os_path_exists(target_directory): return None

    directory_files  = os_listdir(target_directory)
    upscaled_frames  = [f for f in directory_files if selected_AI_model in f]
    if len(upscaled_frames) <= 1: return None

    extracted_frames = [f for f in directory_files if f.endswith(".jpg") and selected_AI_model not in f]
    if not extracted_frames: return None

    return min(100, int(len(upscaled_frames) / len(extracted_frames) * 100))

def build_processing_config() -> Optional[ProcessingConfig]:
    # Selected files 
    try:
        selected_file_list = app_state.file_widget.get_selected_file_list()
    except Exception:
        app_state.info_message.set("Please select a file")
        return None

    if len(selected_file_list) <= 0:
        app_state.info_message.set("Please select a file")
        return None

    app_state.selected_file_list = selected_file_list


    # Output disk space
    disk_space_warning = check_disk_space(app_state.selected_output_path.get(), selected_file_list)
    if disk_space_warning is not None:
        app_state.info_message.set("Not enough disk space")
        show_disk_space_error_message(disk_space_warning)
        return None


    # AI model
    selected_AI_model = get_current_ai_model()
    if selected_AI_model == MENU_LIST_SEPARATOR[0]:
        app_state.info_message.set("Please select the AI model")
        return None


    # Input resize factor 
    try:
        input_resize_factor = int(float(str(app_state.selected_input_resize_factor.get())))
    except Exception:
        app_state.info_message.set("Input resolution % must be a number")
        return None

    if input_resize_factor > 0: input_resize_factor = input_resize_factor/100
    else:
        app_state.info_message.set("Input resolution % must be a value > 0")
        return None


    # Output resize factor 
    try:
        output_resize_factor = int(float(str(app_state.selected_output_resize_factor.get())))
    except Exception:
        app_state.info_message.set("Output resolution % must be a number")
        return None

    if output_resize_factor > 0: output_resize_factor = output_resize_factor/100
    else:
        app_state.info_message.set("Output resolution % must be a value > 0")
        return None

    
    # VRAM limiter -> tiles resolution
    try:
        vram_limiter_gb = int(float(str(app_state.selected_VRAM_limiter.get())))
    except Exception:
        app_state.info_message.set("GPU VRAM value must be a number")
        return None

    if vram_limiter_gb <= 0:
        app_state.info_message.set("GPU VRAM value must be a value > 0")
        return None

    vram_multiplier  = VRAM_model_usage.get(selected_AI_model, 1)
    tiles_resolution = int(vram_multiplier * vram_limiter_gb * 100)

    return ProcessingConfig(
        selected_file_list         = selected_file_list,
        selected_output_path       = app_state.selected_output_path.get(),
        selected_AI_model          = selected_AI_model,
        selected_AI_multithreading = get_current_ai_multithreading(),
        input_resize_factor        = input_resize_factor,
        output_resize_factor       = output_resize_factor,
        selected_gpu               = GPU.device_id_for(app_state.preferences.gpu),
        tiles_resolution           = tiles_resolution,
        selected_sharpening_amount = get_current_sharpening_amount(),
        selected_keep_frames       = app_state.preferences.keep_frames,
        selected_image_extension   = app_state.preferences.image_extension,
        selected_video_extension   = app_state.preferences.video_extension,
        selected_video_codec       = app_state.preferences.video_codec,
    )

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

def show_disk_space_error_message(warning: str) -> None:
    messageBox_title    = "Not enough disk space"
    messageBox_subtitle = "Free up some space on the output drive and try again"
    messageBox_text     = f"\n {warning} \n"

    MessageBox(
        messageType   = "error",
        title         = messageBox_title,
        subtitle      = messageBox_subtitle,
        default_value = None,
        option_list   = [messageBox_text]
    )

def check_disk_space(output_path: str, selected_file_list: list[str]) -> Optional[str]:
    # Images are tiny; only videos (extracted frames + re-encode) can meaningfully fill a disk.
    video_file_list = [f for f in selected_file_list if check_if_file_is_video(f)]
    if not video_file_list: return None

    # No output folder selected -> output is saved next to each source file.
    if output_path == OUTPUT_PATH_CODED:
        check_path = os_path_dirname(video_file_list[0])
    else:
        check_path = output_path if os_path_exists(output_path) else os_path_dirname(output_path)
    if not check_path or not os_path_exists(check_path): return None

    try:
        free_gb = shutil_disk_usage(check_path).free / (1024 ** 3)
    except Exception:
        return None

    if free_gb < MIN_FREE_DISK_SPACE_GB:
        return f"Not enough free disk space on the output drive ({free_gb:.1f} GB free, at least {MIN_FREE_DISK_SPACE_GB} GB recommended)"

    return None

def get_upscale_factor() -> int:
    selected_AI_model = get_current_ai_model()
    if MENU_LIST_SEPARATOR[0] in selected_AI_model: return 0
    return get_model_upscale_factor(selected_AI_model)

# GUI widget factories ------------------

def place_option_background(row: float) -> None:
    background = App.create_option_background()
    background.place(relx = 0.75, rely = row, relwidth = 0.48, anchor = "center")

# GUI callbacks ----------------------

def apply_app_zoom(zoom: float) -> None:
    set_window_scaling(zoom)
    set_widget_scaling(zoom)

def open_files_action():

    def check_supported_selected_files(uploaded_file_list: list) -> list:
        return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

    app_state.info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:

        upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

        app_state.selected_file_list = supported_files_list
        app_state.file_widget = FileWidget(
            master               = App.get_app_window(), 
            selected_file_list   = supported_files_list,
            upscale_factor       = upscale_factor,
            input_resize_factor  = input_resize_factor,
            output_resize_factor = output_resize_factor,
            fg_color             = background_color, 
            bg_color             = background_color
        )
        app_state.file_widget.place(relx = 0.0, rely = 0.0, relwidth = 0.5, relheight = 1.0)
        app_state.info_message.set("Ready")
    else: 
        app_state.info_message.set("Not supported files :(")

def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        app_state.selected_output_path.set(OUTPUT_PATH_CODED)
        app_state.preferences.output_path = OUTPUT_PATH_CODED
    else:
        app_state.selected_output_path.set(asked_selected_output_path)
        app_state.preferences.output_path = asked_selected_output_path

    update_file_widget(1, 2, 3)

def apply_auto_vram_for_gpu(selected_gpu: str, fallback: Optional[str] = None) -> None:
    detected_vram = GPU.vram_for(selected_gpu)
    if   detected_vram is not None: value = str(detected_vram)
    elif fallback      is not None: value = fallback
    else:                           return

    app_state.selected_VRAM_limiter.set(value)
    app_state.preferences.vram_limiter = value

def apply_auto_codec_for_gpu(selected_gpu: str) -> None:
    codec = GPU.codec_for(selected_gpu)
    if codec is None or codec not in video_codec_list:
        return

    app_state.preferences.video_codec = codec
    if app_state.selected_video_codec is not None:
        app_state.selected_video_codec.set(codec)

def open_info_messagebox(title: str, subtitle: str, option_list: list) -> None:
    MessageBox(
        messageType   = "info",
        title         = title,
        subtitle      = subtitle,
        default_value = None,
        option_list   = option_list
    )



# App lifecycle ----------------------

def save_user_choices_in_json() -> None:
    app_state.preferences.output_path = app_state.selected_output_path.get()
    app_state.preferences.input_resize_factor = str(app_state.selected_input_resize_factor.get())
    app_state.preferences.output_resize_factor = str(app_state.selected_output_resize_factor.get())
    app_state.preferences.vram_limiter = str(app_state.selected_VRAM_limiter.get())

    user_preference = {
        "default_app_zoom":             app_state.preferences.app_zoom,
        "default_AI_model":             app_state.preferences.ai_model,
        "default_AI_multithreading":    app_state.preferences.ai_multithreading,
        "default_gpu":                  app_state.preferences.gpu,
        "default_keep_frames":          "ON" if app_state.preferences.keep_frames else "OFF",
        "default_image_extension":      app_state.preferences.image_extension,
        "default_video_extension":      app_state.preferences.video_extension,
        "default_video_codec":          app_state.preferences.video_codec,
        "default_sharpening":           app_state.preferences.sharpening,
        "default_output_path":          app_state.preferences.output_path,
        "default_input_resize_factor":  app_state.preferences.input_resize_factor,
        "default_output_resize_factor": app_state.preferences.output_resize_factor,
        "default_VRAM_limiter":         app_state.preferences.vram_limiter,
    }
    user_preference_json = json_dumps(user_preference)
    with open(USER_PREFERENCE_PATH, "w", encoding="utf-8") as preference_file:
        preference_file.write(user_preference_json)

def load_user_preferences() -> UserPreferences:
    if os_path_exists(USER_PREFERENCE_PATH):
        try:
            with open(USER_PREFERENCE_PATH, "r", encoding="utf-8") as json_file:
                json_data = json_load(json_file)
            print(f"[{app_name}] Preference file exist")
        except Exception as e:
            print(f"[{app_name}] Preference file is corrupted ({e}), using default coded value")
            return UserPreferences()
        return UserPreferences(
            app_zoom             = json_data.get("default_app_zoom",             "100%"),
            ai_model             = json_data.get("default_AI_model",             AI_models_list[0]),
            ai_multithreading    = json_data.get("default_AI_multithreading",    AI_multithreading_list[0]),
            gpu                  = json_data.get("default_gpu",                  gpus_list[0]),
            keep_frames          = json_data.get("default_keep_frames",          keep_frames_list[1]) == "ON",
            image_extension      = json_data.get("default_image_extension",      image_extension_list[0]),
            video_extension      = json_data.get("default_video_extension",      video_extension_list[0]),
            video_codec          = json_data.get("default_video_codec",          video_codec_list[0]),
            sharpening           = json_data.get("default_sharpening",           sharpening_list[1]),
            output_path          = json_data.get("default_output_path",          OUTPUT_PATH_CODED),
            input_resize_factor  = json_data.get("default_input_resize_factor",  str(50)),
            output_resize_factor = json_data.get("default_output_resize_factor", str(100)),
            vram_limiter         = json_data.get("default_VRAM_limiter",         str(4)),
        )

    print(f"[{app_name}] Preference file does not exist, using default coded value")
    return UserPreferences()

def on_app_close() -> None:
    # 1. Save user choices in file
    save_user_choices_in_json()

    # 2. Destroy app window
    app_state.window.grab_release()
    app_state.window.destroy()

    # 3. Stop upscale process and thread check_upscale_step
    write_process_status(app_state.process_status_q, f"{CLOSE_APP_STATUS}")
    stop_upscale_process()

class App():

    # Init -------------------------------------------------

    def __init__(self, window) -> None:
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title(get_AI_engine_info())
        window.geometry("1000x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        self.place_loadFile_section()

        self.place_app_name()
        self.place_app_zoom_and_links()
        self.place_AI_menu()
        self.place_AI_sharpening_menu()
        self.place_AI_multithreading_menu()
        self.place_input_output_resolution_textboxs()
        self.place_gpu_gpuVRAM_menus()
        self.place_image_video_output_menus()
        self.place_video_codec_keep_frames_menus()
        self.place_output_path_textbox()

        self.place_message_label()
        self.place_upscale_button()

    # Widget placement (place_*) ---------------------------

    @staticmethod
    def place_at(widget, relx: float, rely: float) -> None:
        widget.place(relx = relx, rely = rely, anchor = "center")

    @staticmethod
    def place_loadFile_section() -> None:
        background = App.create_panel_background()

        text_drop = (" SUPPORTED FILES \n\n "
                   + "IMAGES - jpg png tif bmp webp heic \n " 
                   + "VIDEOS - mp4 webm mkv flv gif avi mov mpg qt 3gp ")

        input_file_text = CTkLabel(
            master     = App.get_app_window(), 
            text       = text_drop,
            fg_color   = background_color,
            bg_color   = background_color,
            text_color = CARD_MUTED_COLOR,
            width      = 300,
            height     = 150,
            font       = bold13,
            anchor     = "center"
        )
    
        input_file_button = CTkButton(
            master       = App.get_app_window(),
            command      = open_files_action, 
            text         = "SELECT FILES",
            width        = 140,
            height       = 30,
            font         = bold12,
            border_width  = 2,
            corner_radius = UI_CORNER_RADIUS,
            fg_color      = "#282828",
            text_color    = "#E0E0E0",
            border_color  = UI_ACCENT_COLOR
        )
    
        background.place(relx = 0.0, rely = 0.0, relwidth = 0.5, relheight = 1.0)
        App.place_at(input_file_text, 0.25, 0.4)
        App.place_at(input_file_button, 0.25, 0.5)

    @staticmethod
    def place_app_name() -> None:
        background = App.create_panel_background()
        app_name_label = CTkLabel(
            master     = App.get_app_window(), 
            text       = app_name + " " + version,
            fg_color   = background_color,
            bg_color   = background_color,
            text_color = app_name_color,
            font       = bold18,
            anchor     = "w"
        )
        background.place(relx = 0.5, rely = 0.0, relwidth = 0.5, relheight = 1.0)
        App.place_at(app_name_label, COL_TITLE - 0.055, ROW_HEADER)

    @staticmethod
    def place_app_zoom_and_links() -> None:

        # App zoom menu
        label_app_zoom = CTkLabel(
            master     = App.get_app_window(),
            text       = "App zoom",
            width      = 50,
            height     = 22,
            fg_color   = "transparent",
            bg_color   = background_color,
            text_color = CARD_TITLE_COLOR,
            font       = bold13,
            anchor     = "w"
        )
        zoom_option_menu = App.create_option_menu(
            command       = App.select_app_zoom, 
            values        = zoom_option_list, 
            default_value = app_state.preferences.app_zoom, 
            width         = 71
        )
        App.place_at(label_app_zoom, COL_ZOOM-0.06, ROW_HEADER)
        App.place_at(zoom_option_menu, COL_ZOOM+0.0155, ROW_HEADER)

        def opentelegram() -> None: open_browser(telegramme, new=1)
        def opengithub()   -> None: open_browser(githubme, new=1)

        # Telegram button
        telegram_button = App.create_link_button(command = opentelegram, icon = logo_telegram)
        App.place_at(telegram_button, COL_ZOOM+0.075, ROW_HEADER)

        # Github button
        git_button = App.create_link_button(command = opengithub, icon = logo_git)
        App.place_at(git_button, COL_ZOOM+0.11, ROW_HEADER)

    @staticmethod
    def place_AI_menu() -> None:

        def open_info_AI_model():
            option_list = [
                "\n"
                "LVAx2"
                "\n"
                " - Target: Live-action video upscaling"
                "\n"
                " - Tips: AI sharpening - OFF/Low"
                "\n",

                "\n"
                "RealESR_Gx4 - RealESR_Ax4"
                "\n"
                " - Target: Animated/degraded live-action video upscaling"
                "\n"
                " - Tips: AI sharpening - Low for animation, High for live-action videos"
                "\n",

                "\n"
                "BSRGANx2 - BSRGANx4"
                "\n"
                " - Target: High-quality image upscaling"
                "\n"
                " - Tips: can be used to upscale videos but will be slow"
                "\n",

                "\n"
                "RealESRGANx4"
                "\n"
                " - Target: High-quality image upscaling"
                "\n"
                " - Tips: can be used to upscale videos but will be slow"
                "\n",

                "\n"
                "MSharpx4"
                "\n"
                " - Target: Image upscaling and sharpening"
                "\n"
                " - Tips: to use on good quality photos (not too much noise)"
                "\n",

                "\n"
                "IRCNN_Mx1 - IRCNN_Lx1"
                "\n"
                " - Target: Video/image denoising"
                "\n"
                " - Tips: AI sharpening - OFF"
                "\n",

                "\n"
                "NomosWebPhoto_RealPLKSRx4"
                "\n"
                " - Target: Fast, natural photorealistic image upscaling"
                "\n"
                " - Tips: Great for web photos; avoids harsh edge artifacts and halos"
                "\n",

                "\n"
                "SwinIRLargex4"
                "\n"
                " - Target: Ultra high-fidelity image & artwork upscaling"
                "\n"
                " - Tips: Preserves fine texture and line art without artifacts; slower processing time"
                "\n",
            ]

            open_info_messagebox("AI model", "Select the AI model based on your content type", option_list)

        row = ROW_AI_MODEL
        place_option_background(row)

        info_button = App.create_info_button(open_info_AI_model, "AI model")
        option_menu = App.create_option_menu(App.select_AI_from_menu, AI_models_list, app_state.preferences.ai_model)

        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_C, row)

    @staticmethod
    def place_AI_sharpening_menu() -> None:

        def open_info_AI_sharpening():
            option_list = [
                " Sharpening applies an unsharp mask to the AI result to bring back perceived detail",

                " \n SHARPENING OPTIONS\n" +
                "  - [OFF] No sharpening is applied\n" +
                "  - [Low] Light sharpening\n" +
                "  - [High] Stronger sharpening\n",

                " \n NOTES\n" +
                "  - Useful when the AI output looks too soft or smooth\n" +
                "  - Works only on the AI output, so it can't reintroduce source noise/compression artifacts\n",
            ]

            open_info_messagebox("AI sharpening", "Sharpen the AI output to fine-tune detail", option_list)

        row = ROW_AI_SHARPENING

        place_option_background(row)

        info_button = App.create_info_button(open_info_AI_sharpening, "AI sharpening")
        option_menu = App.create_option_menu(App.select_sharpening_from_menu, sharpening_list, app_state.preferences.sharpening)

        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_C, row)

    @staticmethod
    def place_AI_multithreading_menu() -> None:

        def open_info_AI_multithreading():
            option_list = [
                " This option can enhance video upscaling performance, especially on powerful GPUs.",

                " \n AI MULTITHREADING OPTIONS\n"
                + "  - OFF - Processes one frame at a time.\n"
                + "  - 2 threads - Processes two frames simultaneously.\n"
                + "  - 4 threads - Processes four frames simultaneously.\n"
                + "  - 6 threads - Processes six frames simultaneously.\n"
                + "  - 8 threads - Processes eight frames simultaneously.\n",

                " \n NOTES\n"
                + "  - Higher thread counts increase CPU, GPU, and RAM usage.\n"
                + "  - The GPU may be heavily stressed, potentially reaching high temperatures.\n"
                + "  - Monitor your system's temperature to prevent overheating.\n"
                + "  - If the chosen thread count exceeds GPU capacity, the app automatically selects an optimal value.\n",
            ]

            open_info_messagebox("AI multithreading", "Process multiple video frames in parallel to speed up upscaling", option_list)

        row = ROW_AI_MULTITHREADING
        place_option_background(row)

        info_button = App.create_info_button(open_info_AI_multithreading, "AI multithreading")
        option_menu = App.create_option_menu(App.select_AI_multithreading_from_menu, AI_multithreading_list, app_state.preferences.ai_multithreading)

        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_C, row)

    @staticmethod
    def place_input_output_resolution_textboxs() -> None:

        def open_info_input_resolution():
            option_list = [
                " A high value (>50%) will create high quality photos/videos but will be slower",
                " While a low value (<50%) will create good quality photos/videos but will much faster",

                " \n For example, for a 1080p (1920x1080) image/video\n" + 
                " - Input scale 25% => input to AI 270p (480x270)\n" +
                " - Input scale 50% => input to AI 540p (960x540)\n" + 
                " - Input scale 75% => input to AI 810p (1440x810)\n" + 
                " - Input scale 100% => input to AI 1080p (1920x1080) \n",
            ]

            open_info_messagebox("Input resolution %", "Controls the resolution fed to the AI — lower is faster, higher is sharper", option_list)

        def open_info_output_resolution():
            option_list = [
                " 100% keeps the exact resolution produced by the AI upscaling",
                " A lower value (<100%) will downscale the AI result to a smaller resolution, saving space and processing time",
                " A higher value (>100%) will further upscale the AI output, increasing size but not adding real details",

                "\n For example, if the AI generates a 4K (3840x2160) image/video\n" +
                " - Output scale 50%  => final output 1920x1080 (downscaled)\n" +
                " - Output scale 100% => final output 3840x2160 (AI native)\n" +
                " - Output scale 200% => final output 7680x4320 (8K, interpolated)\n",
            ]

            open_info_messagebox("Output resolution %", "Controls the final output resolution after AI upscaling", option_list)


        row = ROW_RESOLUTION

        place_option_background(row)

        # Input scale %
        info_button = App.create_info_button(open_info_input_resolution, "Input scale %")
        option_menu = App.create_text_box(App.get_input_resize_factor_var(), width = little_textbox_width) 

        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_TEXT_L, row)

        # Output scale %
        info_button = App.create_info_button(open_info_output_resolution, "Output scale %")
        option_menu = App.create_text_box(App.get_output_resize_factor_var(), width = little_textbox_width)  

        App.place_at(info_button, COL_INFO_R, row)
        App.place_at(option_menu, COL_TEXT_R, row)

    @staticmethod
    def place_gpu_gpuVRAM_menus() -> None:

        def open_info_gpu():
            option_list = [
                "\n The app automatically detects the GPUs installed on your system\n" +
                "  - Each entry is a GPU detected on your PC, listed by name\n" +
                "  - If no GPU is detected the menu shows \"No GPU found\"\n",

                "\n NOTES\n" +
                "  - Keep in mind that the more powerful the chosen GPU is, the faster the upscaling will be\n" +
                "  - For optimal performance, it is essential to regularly update your GPU drivers\n" +
                "  - If no GPU is detected the app may fall back to the CPU\n"
            ]

            open_info_messagebox("GPU", "Select which GPU to use for AI inference", option_list)

        def open_info_vram_limiter():
            option_list = [
                " This value is auto-detected from the selected GPU's dedicated VRAM at startup",
                " You can still override it manually if the detected value is not correct",
                " Make sure to enter the correct value based on the selected GPU's VRAM",
                " Setting a value higher than the available VRAM may cause upscale failure",
                " For Intel integrated GPUs (UHD / Iris Xe) or AMD integrated GPUs (Vega 3/5/7), select 2 GB to avoid issues",
            ]

            open_info_messagebox("GPU VRAM (GB)", "Match this value to your GPU's available VRAM to avoid out-of-memory errors", option_list)

        row = ROW_GPU

        place_option_background(row)

        # GPU
        gpu_menu_list = GPU.menu_list()
        if app_state.preferences.gpu not in gpu_menu_list:
            app_state.preferences.gpu = GPU.default()

        info_button = App.create_info_button(open_info_gpu, "GPU")
        option_menu = App.create_option_menu(App.select_gpu_from_menu, gpu_menu_list, app_state.preferences.gpu, width = little_menu_width) 

        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_L, row)

        # GPU VRAM
        info_button = App.create_info_button(open_info_vram_limiter, "GPU VRAM (GB)")
        option_menu = App.create_text_box(App.get_vram_limiter_var(), width = little_textbox_width)  

        App.place_at(info_button, COL_INFO_R, row)
        App.place_at(option_menu, COL_TEXT_R, row)

    @staticmethod
    def place_image_video_output_menus() -> None:

        def open_info_image_output():
            option_list = [
                " \n PNG\n"
                " - Very good quality\n"
                " - Slow and heavy file\n"
                " - Supports transparent images\n"
                " - Lossless compression (no quality loss)\n"
                " - Ideal for graphics, web images, and screenshots\n",

                " \n JPG\n"
                " - Good quality\n"
                " - Fast and lightweight file\n"
                " - Lossy compression (some quality loss)\n"
                " - Ideal for photos and web images\n"
                " - Does not support transparency\n",

                " \n BMP\n"
                " - Highest quality\n"
                " - Slow and heavy file\n"
                " - Uncompressed format (large file size)\n"
                " - Ideal for raw images and high-detail graphics\n"
                " - Does not support transparency\n",

                " \n TIFF\n"
                " - Highest quality\n"
                " - Very slow and heavy file\n"
                " - Supports both lossless and lossy compression\n"
                " - Often used in professional photography and printing\n"
                " - Supports multiple layers and transparency\n",
            ]


            open_info_messagebox("Image output", "Choose the output format for upscaled images", option_list)

        def open_info_video_extension():
            option_list = [
                " \n MP4\n"
                " - Most widely supported format\n"
                " - Good quality with efficient compression\n"
                " - Fast and lightweight file\n"
                " - Ideal for streaming and general use\n",

                " \n MKV\n"
                " - High-quality format with multiple audio and subtitle tracks support\n"
                " - Larger file size compared to MP4\n"
                " - Supports almost any codec\n"
                " - Ideal for high-quality videos and archiving\n",

                " \n AVI\n"
                " - Older format with high compatibility\n"
                " - Larger file size due to less efficient compression\n"
                " - Supports multiple codecs but lacks modern features\n"
                " - Ideal for older devices and raw video storage\n",

                " \n MOV\n"
                " - High-quality format developed by Apple\n"
                " - Large file size due to less compression\n"
                " - Best suited for editing and high-quality playback\n"
                " - Compatible mainly with macOS and iOS devices\n",
            ]

            open_info_messagebox("Video output", "Choose the container format for upscaled videos", option_list)

        row = ROW_OUTPUT_FORMAT

        place_option_background(row)

        # Image output
        info_button = App.create_info_button(open_info_image_output, "Image ext.")
        option_menu = App.create_option_menu(App.select_image_extension_from_menu, image_extension_list, app_state.preferences.image_extension, width = little_menu_width)
        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_L, row)

        # Video output
        info_button = App.create_info_button(open_info_video_extension, "Video ext.")
        option_menu = App.create_option_menu(App.select_video_extension_from_menu, video_extension_list, app_state.preferences.video_extension, width = little_menu_width)
        App.place_at(info_button, COL_INFO_R, row)
        App.place_at(option_menu, COL_MENU_R, row)

    @staticmethod
    def place_video_codec_keep_frames_menus() -> None:

        def open_info_video_codec():
            option_list = [
                " \n SOFTWARE ENCODING (CPU)\n"
                " - x264 | H.264 software encoding\n"
                " - x265 | HEVC (H.265) software encoding\n",

                " \n NVIDIA GPU ENCODING (NVENC - Optimized for NVIDIA GPU)\n"
                " - h264_nvenc | H.264 hardware encoding\n"
                " - hevc_nvenc | HEVC (H.265) hardware encoding\n",

                " \n AMD GPU ENCODING (AMF - Optimized for AMD GPU)\n"
                " - h264_amf | H.264 hardware encoding\n"
                " - hevc_amf | HEVC (H.265) hardware encoding\n",

                " \n INTEL GPU ENCODING (QSV - Optimized for Intel GPU)\n"
                " - h264_qsv | H.264 hardware encoding\n"
                " - hevc_qsv | HEVC (H.265) hardware encoding\n"
            ]


            open_info_messagebox("Video codec", "Choose the encoder used to compress the upscaled video", option_list)

        def open_info_keep_frames():
            option_list = [
                "\n ON \n" + 
                " The app does NOT delete the video frames after creating the upscaled video \n",

                "\n OFF \n" + 
                " The app deletes the video frames after creating the upscaled video \n"
            ]

            open_info_messagebox("Keep frames", "Choose whether to keep the extracted frames folder after encoding", option_list)


        row = ROW_CODEC

        place_option_background(row)

        # Video codec
        info_button = App.create_info_button(open_info_video_codec, "Video codec")
        option_menu = App.create_option_menu(App.select_video_codec_from_menu, video_codec_list, app_state.preferences.video_codec, width = little_menu_width, variable = App.get_video_codec_var())
        App.place_at(info_button, COL_INFO_L, row)
        App.place_at(option_menu, COL_MENU_L, row)

        # Keep frames
        info_button = App.create_info_button(open_info_keep_frames, "Keep frames")
        option_menu = App.create_option_menu(App.select_save_frame_from_menu, keep_frames_list, "ON" if app_state.preferences.keep_frames else "OFF", width = little_menu_width)
        App.place_at(info_button, COL_INFO_R, row)
        App.place_at(option_menu, COL_MENU_R, row)

    @staticmethod
    def place_output_path_textbox() -> None:

        def open_info_output_path():
            option_list = [
                  "\n The default path is defined by the input files."
                + "\n For example: selecting a file from the Download folder,"
                + "\n the app will save upscaled files in the Download folder \n",

                " Otherwise it is possible to select the desired path using the SELECT button",
            ]

            open_info_messagebox("Output path", "Choose where upscaled files are saved", option_list)

        background    = App.create_option_background()
        info_button   = App.create_info_button(open_info_output_path, "Output path")
        option_menu   = App.create_text_box(App.get_output_path_var(), width = 250, state = DISABLED) 
        active_button = App.create_active_button(
            command = open_output_path_action, 
            text    = "SELECT", 
            icon    = None, 
            width   = 60, 
            height  = 26
        )
  
        background.place(   relx = 0.75,                 rely = ROW_OUTPUT_PATH, relwidth = 0.48,  anchor = "center")
        App.place_at(info_button, COL_INFO_L, ROW_OUTPUT_PATH)
        App.place_at(active_button, COL_INFO_L + 0.052, ROW_OUTPUT_PATH)
        App.place_at(option_menu, COL_ZOOM - 0.008, ROW_OUTPUT_PATH)

    @staticmethod
    def place_message_label() -> None:
        message_label = CTkLabel(
            master        = App.get_app_window(), 
            textvariable  = App.get_info_message_var(),
            height        = 25,
            width         = 250,
            font          = bold11,
            bg_color      = background_color,
            fg_color      = UI_ACCENT_COLOR,
            text_color    = "#0A0A0A",
            anchor        = "center",
            corner_radius = UI_CORNER_RADIUS
        )

        triangle_dimension = 14
        zero = 0
        triangle_pointer = CTkCanvas(
            App.get_app_window(), 
            width   = triangle_dimension, 
            height  = triangle_dimension, 
            bg      = background_color, 
            highlightthickness = 0
        )
        triangle_item = triangle_pointer.create_polygon(
            triangle_dimension, zero,
            zero,               (triangle_dimension/2),
            triangle_dimension, triangle_dimension,
            fill = UI_ACCENT_COLOR
        )
        App.place_at(triangle_pointer, 0.716, ROW_ACTIONS)
        App.place_at(message_label, 0.85, ROW_ACTIONS)

        # Tint the status badge by state (working / completed / stopped / error)
        App._message_label         = message_label
        App._message_triangle      = triangle_pointer
        App._message_triangle_item = triangle_item
        App._message_color         = UI_ACCENT_COLOR
        App.get_info_message_var().trace_add("write", App._refresh_message_color)

    @staticmethod
    def _refresh_message_color(*_args) -> None:
        label = getattr(App, "_message_label", None)
        if label is None or not label.winfo_exists(): return

        message = App.get_info_message_var().get().lower()
        if   "error" in message:     target = MESSAGE_ERROR_COLOR
        elif "completed" in message: target = MESSAGE_SUCCESS_COLOR
        elif "stopped" in message:   target = MESSAGE_WARNING_COLOR
        else:                        target = UI_ACCENT_COLOR

        App._animate_message_color(target)

    @staticmethod
    def _set_message_color(color) -> None:
        label = getattr(App, "_message_label", None)
        if label is not None and label.winfo_exists():
            label.configure(fg_color = color)
        triangle = getattr(App, "_message_triangle", None)
        item     = getattr(App, "_message_triangle_item", None)
        if triangle is not None and triangle.winfo_exists() and item is not None:
            triangle.itemconfig(item, fill = color)
        App._message_color = color

    @staticmethod
    def _animate_message_color(target) -> None:
        start = getattr(App, "_message_color", UI_ACCENT_COLOR)
        if start == target:
            App._set_message_color(target)
            return

        App._message_color_token = getattr(App, "_message_color_token", 0) + 1
        token  = App._message_color_token
        window = App.get_app_window()
        steps  = 10

        def run(step) -> None:
            if token != getattr(App, "_message_color_token", 0): return
            label = getattr(App, "_message_label", None)
            if label is None or not label.winfo_exists(): return
            App._set_message_color(lerp_hex(start, target, min(1.0, step / steps)))
            if step >= steps:
                App._set_message_color(target)
                return
            window.after(24, lambda: run(step + 1))

        run(1)

    @staticmethod
    def place_stop_button() -> None: 
        stop_button = App.create_active_button(
            command      = stop_button_command,
            text         = "STOP",
            icon         = stop_icon,
            width        = 150,
            height       = 30,
            border_color = "#EC1D1D"
        )
        App.place_at(stop_button, 0.62, ROW_ACTIONS)

    @staticmethod
    def place_upscale_button() -> None: 
        upscale_button = App.create_active_button(
            command = upscale_button_command,
            text    = "UPSCALE",
            icon    = upscale_icon,
            width   = 150,
            height  = 30
        )
        App.place_at(upscale_button, 0.62, ROW_ACTIONS)


    # Menu callbacks (select_*) ----------------------------

    @staticmethod
    def select_app_zoom(selected_option: str) -> None:
        app_state.preferences.app_zoom = selected_option
        apply_app_zoom(float(selected_option.replace("%", "")) / 100)

    @staticmethod
    def select_AI_from_menu(selected_option: str) -> None:
        app_state.preferences.ai_model = selected_option
        update_file_widget(1, 2, 3)

    @staticmethod
    def select_AI_multithreading_from_menu(selected_option: str) -> None:
        app_state.preferences.ai_multithreading = selected_option

    @staticmethod
    def select_sharpening_from_menu(selected_option: str) -> None:
        app_state.preferences.sharpening = selected_option
        update_file_widget(1, 2, 3)

    @staticmethod
    def select_gpu_from_menu(selected_option: str) -> None:
        app_state.preferences.gpu = selected_option
        apply_auto_vram_for_gpu(selected_option)
        apply_auto_codec_for_gpu(selected_option)

    @staticmethod
    def select_save_frame_from_menu(selected_option: str):
        app_state.preferences.keep_frames = selected_option == "ON"

    @staticmethod
    def select_image_extension_from_menu(selected_option: str) -> None:
        app_state.preferences.image_extension = selected_option

    @staticmethod
    def select_video_extension_from_menu(selected_option: str) -> None:
        app_state.preferences.video_extension = selected_option

    @staticmethod
    def select_video_codec_from_menu(selected_option: str) -> None:
        app_state.preferences.video_codec = selected_option

    # Widget factories (create_*) --------------------------

    @staticmethod
    def create_option_background() -> CTkFrame:
        return CTkFrame(
            master   = App.get_app_window(),
            bg_color = background_color,
            fg_color = CARD_BACKGROUND_COLOR,
            height   = 46,
            corner_radius = 12,
            border_width  = 1,
            border_color  = CARD_BORDER_COLOR
        )

    @staticmethod
    def create_panel_background() -> CTkFrame:
        return CTkFrame(
            master        = App.get_app_window(),
            fg_color      = background_color,
            corner_radius = 0,
            border_width  = 0
        )

    @staticmethod
    def create_info_button(command: Callable, text: str, width: int = 200) -> CTkFrame:

        frame = CTkFrame(
            master   = App.get_app_window(), 
            fg_color = CARD_BACKGROUND_COLOR, 
            height   = 25
        )

        button = CTkButton(
            master        = frame,
            command       = command,
            font          = bold14,
            text          = "?",
            border_width  = 0,
            fg_color      = CARD_BACKGROUND_COLOR,
            hover_color   = CARD_BACKGROUND_COLOR,
            text_color    = CARD_MUTED_COLOR,
            width         = 20,
            height        = 20,
            corner_radius = UI_CORNER_RADIUS
        )
        button.bind("<Enter>", lambda e: button.configure(text_color = UI_ACCENT_COLOR))
        button.bind("<Leave>", lambda e: button.configure(text_color = CARD_MUTED_COLOR))
        button.grid(row=0, column=0, padx=(0, 7), pady=0, sticky="w")

        label = CTkLabel(
            master     = frame,
            text       = text,
            width      = width,
            height     = 22,
            fg_color   = "transparent",
            bg_color   = CARD_BACKGROUND_COLOR,
            text_color = CARD_VALUE_COLOR,
            font       = bold13,
            anchor     = "w"
        )
        label.grid(row=0, column=1, pady=0, sticky="w")

        frame.grid_propagate(False)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        return frame

    @staticmethod
    def create_option_menu(
            command:       Callable, 
            values:        list,
            default_value: str,
            border_color:  str = UI_BORDER_COLOR, 
            border_width:  int = 1,
            width:         int = 159,
            height:        int = 26,
            variable:      Optional[StringVar] = None
        ) -> CTkFrame:

        total_width  = (width + 2 * border_width)
        total_height = (height + 2 * border_width)

        frame = CTkFrame(
            master        = App.get_app_window(),
            fg_color      = background_color,
            width         = total_width,
            height        = total_height,
            border_width  = 0,
            corner_radius = UI_CORNER_RADIUS,
        )

        option_menu = CTkOptionMenu(
            master             = frame, 
            command            = command,
            values             = values,
            variable           = variable,
            width              = width,
            height             = height,
            corner_radius      = UI_CORNER_RADIUS,
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

    @staticmethod
    def create_text_box(
            textvariable: StringVar, 
            width:        int,
            height:       int = 26,
            state:        str = "normal"
        ) -> CTkEntry:

        return CTkEntry(
            master        = App.get_app_window(), 
            textvariable  = textvariable,
            corner_radius = UI_CORNER_RADIUS,
            width         = width,
            height        = height,
            font          = bold11,
            justify       = "center",
            text_color    = text_color,
            fg_color      = "#000000",
            border_width  = 2,
            border_color  = UI_BORDER_COLOR,
            state         = state,
        )

    @staticmethod
    def create_active_button(
            command:      Callable,
            text:         str,
            icon:         CTkImage,
            width:        int = 140,
            height:       int = 30,
            border_color: str = UI_ACCENT_COLOR
        ) -> CTkButton:

        button = CTkButton(
            master        = App.get_app_window(), 
            text          = text,
            image         = icon,
            width         = width,
            height        = height,
            font          = bold11,
            border_width  = 2,
            corner_radius = UI_CORNER_RADIUS,
            fg_color      = "#282828",
            text_color    = "#E0E0E0",
            border_color  = border_color
        )

        def _press_feedback() -> None:
            # Brief darken on click, then restore, before running the real command.
            if button.winfo_exists():
                button.configure(fg_color = "#1E1E1E")
                button.after(120, lambda: button.winfo_exists() and button.configure(fg_color = "#282828"))
            command()

        button.configure(command = _press_feedback)
        return button

    @staticmethod
    def create_link_button(command: Callable, icon: CTkImage) -> CTkButton:

        button = CTkButton(
            master        = App.get_app_window(),
            command       = command,
            image         = icon,
            width         = 30,
            height        = 30,
            border_width  = 2,
            corner_radius = UI_CORNER_RADIUS,
            bg_color      = background_color,
            fg_color      = CARD_BACKGROUND_COLOR,
            hover_color   = widget_background_color,
            text_color    = text_color,
            border_color  = CARD_BORDER_COLOR,
            anchor        = "center",
            text          = "", 
            font          = bold11
        )
        button.bind("<Enter>", lambda e: button.configure(border_color = UI_ACCENT_COLOR))
        button.bind("<Leave>", lambda e: button.configure(border_color = CARD_BORDER_COLOR))
        return button

    # State accessors (get_*) ------------------------------

    @staticmethod
    def get_app_window() -> CTk:
        return app_state.window

    @staticmethod
    def get_info_message_var() -> StringVar:
        return app_state.info_message

    @staticmethod
    def get_output_path_var() -> StringVar:
        return app_state.selected_output_path

    @staticmethod
    def get_input_resize_factor_var() -> StringVar:
        return app_state.selected_input_resize_factor

    @staticmethod
    def get_output_resize_factor_var() -> StringVar:
        return app_state.selected_output_resize_factor

    @staticmethod
    def get_vram_limiter_var() -> StringVar:
        return app_state.selected_VRAM_limiter

    @staticmethod
    def get_video_codec_var() -> StringVar:
        return app_state.selected_video_codec



# Main functions ---------------------------

if __name__ == "__main__":
    multiprocessing_freeze_support()

    preferences = load_user_preferences()
    app_state = AppState(preferences = preferences)

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")
    apply_app_zoom(float(preferences.app_zoom.replace("%", "")) / 100)

    free_ram_gb = psutil_virtual_memory().available / (1024**3)
    if   free_ram_gb < 8:  queue_maxsize = 30
    elif free_ram_gb < 16: queue_maxsize = 50
    elif free_ram_gb < 32: queue_maxsize = 100
    elif free_ram_gb < 64: queue_maxsize = 150
    else:                  queue_maxsize = 200
    print(f"[{app_name}] free RAM: {free_ram_gb:.2f} GB - queue_maxsize = {queue_maxsize}")
    
    multiprocessing_manager    = multiprocessing_Manager()
    process_status_q           = multiprocessing_manager.Queue(maxsize=1)
    video_frames_and_info_q    = multiprocessing_manager.Queue(maxsize=queue_maxsize)
    event_stop_upscale_process = multiprocessing_manager.Event()

    app_state.window                       = CTk()
    app_state.info_message                 = StringVar()
    app_state.selected_output_path         = StringVar()
    app_state.selected_input_resize_factor = StringVar()
    app_state.selected_output_resize_factor = StringVar()
    app_state.selected_VRAM_limiter        = StringVar()
    app_state.selected_video_codec         = StringVar()
    app_state.process_status_q             = process_status_q
    app_state.video_frames_and_info_q      = video_frames_and_info_q
    app_state.event_stop_upscale_process   = event_stop_upscale_process

    GPU.detect()
    if GPU.detected:
        print(f"[{app_name}] Detected GPUs:")
        for gpu in GPU.detected:
            print(f"    - GPU {gpu.device_id + 1}: {gpu.name} ({gpu.vram_gb} GB VRAM)")
    else:
        print(f"[{app_name}] GPU detection unavailable, using saved value")

    # Normalize a stale/unknown saved GPU (e.g. from another PC) to a safe default
    if preferences.gpu not in GPU.menu_list(): preferences.gpu = GPU.default()

    app_state.selected_input_resize_factor.set(preferences.input_resize_factor)
    app_state.selected_output_resize_factor.set(preferences.output_resize_factor)
    apply_auto_vram_for_gpu(preferences.gpu, fallback = preferences.vram_limiter)
    apply_auto_codec_for_gpu(preferences.gpu)
    app_state.selected_video_codec.set(preferences.video_codec)
    app_state.selected_output_path.set(preferences.output_path)
    app_state.selected_file_list = []

    app_state.info_message.set("Hi :)")
    app_state.selected_input_resize_factor.trace_add('write', update_file_widget)
    app_state.selected_output_resize_factor.trace_add('write', update_file_widget)

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
    logo_git      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(18, 18))
    logo_telegram = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(16, 16))
    stop_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon    = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(18, 18))

    app = App(app_state.window)
    app_state.window.update()
    app_state.window.mainloop()