
# Standard library imports
import sys
from shutil          import rmtree as remove_directory
from timeit          import default_timer as timer
from subprocess      import run  as subprocess_run
from webbrowser      import open as open_browser
from time            import sleep

from threading       import Thread
from multiprocessing import ( 
    Process, 
    Queue          as multiprocessing_Queue,
    freeze_support as multiprocessing_freeze_support
)

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
from PIL.Image        import open as pillow_image_open
from moviepy.editor   import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from download import download

from torch import (
    device          as torch_device,
    inference_mode  as torch_inference_mode,
    tensor          as torch_tensor,
    zeros           as torch_zeros,
    cat             as torch_cat,
    load            as torch_load,
    ones            as torch_ones,
    sqrt            as torch_sqrt,
    from_numpy      as torch_from_numpy,
    set_num_threads as torch_set_num_threads,
)

from torch.nn import (
    init as torch_nn_init,
    Sequential,
    Conv2d,
    Module,
    ModuleList,
    Parameter,
    PixelShuffle,
    GELU,
    PReLU,
    LeakyReLU
)

from torch.nn.functional import (
    interpolate as torch_nn_interpolate, 
    layer_norm  as torch_nn_layer_norm,
    adaptive_max_pool2d as torch_nn_adaptive_max_pool2d, 
)

from torch_directml import (
    device       as directml_device,
    device_count as directml_device_count,
    device_name  as directml_device_name,
    gpu_memory   as directml_gpu_memory,
    has_float64_support as directml_has_float64_support
)

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
    max         as numpy_max, 
    float32, 
    uint16,
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
version = "2.12"

app_name_color = "#DA70D6"
dark_color = "#080808"

githubme   = "https://github.com/Djdefrag/QualityScaler"
telegramme = "https://linktr.ee/j3ngystudio"

SRVGGNetCompact_vram_multiplier = 1.7
RRDB_vram_multiplier = 0.8
SAFMN_vram_multiplier = 1.8

SRVGGNetCompact_models_list = [ 'RealESR_Gx4', 'RealSRx4_Anime' ]
BSRGAN_models_list = [ 'BSRGANx4', 'BSRGANx2' ]
SAFMN_models_list  = [ 'SAFMNLx4', 'SAFMNLx4_Real']

AI_models_list = (
                SRVGGNetCompact_models_list
                + BSRGAN_models_list 
                + SAFMN_models_list 
                )


image_extension_list = [ '.png', '.jpg', '.bmp', '.tiff' ]
video_extension_list = [ '.mp4 (x264)', '.mp4 (x265)', '.avi' ]
interpolation_list   = [ 'Low', 'Medium', 'High', 'Disabled' ]
AI_modes_list        = [ 'Half precision', 'Full precision' ]

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



# AI models -------------------

def get_model_url(model_name: str) -> str:
    return f"https://zackees.github.io/ai-image-video-models/{model_name}"

def torch_load_model(model_path, map_location = None) -> Module:
    print(f"Loading model from {model_path}")
    model_name = os.path.basename(model_path)
    donwload_url = get_model_url(model_name)
    download(donwload_url, model_path, replace = False, timeout=60)
    return torch_load(model_path, map_location = map_location)

@torch_inference_mode(True)
def load_AI_model(
        selected_AI_model: str, 
        backend: directml_device, 
        half_precision: bool
        ) -> any:
    
    model_path = find_by_relative_path(f"AI{os_separator}{selected_AI_model}.pth")

    # BSRGAN
    if selected_AI_model == 'BSRGANx4':
        AI_model = BSRGANx4_Net()
        pretrained_model = torch_load(model_path)

    elif selected_AI_model == 'BSRGANx2':
        AI_model = BSRGANx2_Net()
        pretrained_model = torch_load(model_path)

    # SRVGGNetCompact
    elif selected_AI_model == 'RealESR_Gx4':
        AI_model = SRVGGNetCompact_Plus()
        pretrained_model = torch_load(model_path, map_location = torch_device('cpu'))['params']

    elif selected_AI_model == 'RealSRx4_Anime':
        AI_model = SRVGGNetCompact()
        pretrained_model = torch_load_model(model_path, map_location = torch_device('cpu'))['params']

    # SAFMNet
    elif selected_AI_model == 'SAFMNLx4_Real':
        AI_model = SAFM_Net()
        pretrained_model = torch_load(model_path)['params']
    
    elif selected_AI_model == 'SAFMNLx4':
        AI_model = SAFM_Net()
        pretrained_model = torch_load(model_path)['params']
        
    AI_model.load_state_dict(pretrained_model, strict = True)
    AI_model.eval()

    if half_precision: 
        AI_model = AI_model.half()

    AI_model = AI_model.to(backend)     

    return AI_model

@torch_inference_mode(True)
def AI_enhance(
        AI_model: any, 
        backend: directml_device,
        half_precision: bool,
        image: numpy_ndarray, 
        image_mode: str, 
        ) -> numpy_ndarray:
    
    image, max_range = normalize_image(image.astype(float32))

    if image_mode == "RGB":
        image        = preprocess_image(image, half_precision, backend)
        output_image = process_image_with_model(AI_model, image)
        output_image = postprocess_output(output_image, max_range)
        return output_image
    
    elif image_mode == 'RGBA':
        alpha = image[:, :, 3]
        image = image[:, :, :3]
        image = opencv_cvtColor(image, COLOR_BGR2RGB)
        alpha = opencv_cvtColor(alpha, COLOR_GRAY2RGB)

        # Image
        image        = preprocess_image(image, half_precision, backend)
        output_image = process_image_with_model(AI_model, image)
        output_image = opencv_cvtColor(output_image, COLOR_RGB2BGRA)

        # Alpha
        alpha        = preprocess_image(alpha, half_precision, backend)
        output_alpha = process_image_with_model(AI_model, alpha)
        output_alpha = opencv_cvtColor(output_alpha, COLOR_RGB2GRAY)

        # Fusion Image + Alpha
        output_image[:, :, 3] = output_alpha
        output_image = postprocess_output(output_image, max_range)
        return output_image

    elif image_mode == 'Grayscale':
        image        = opencv_cvtColor(image, COLOR_GRAY2RGB)
        image        = preprocess_image(image, half_precision, backend)
        output_image = process_image_with_model(AI_model, image)
        output_image = opencv_cvtColor(output_image, COLOR_RGB2GRAY)
        output_image = postprocess_output(output_image, max_range)
        return output_image

def AI_enhance_tiles(
        AI_model: any, 
        backend: directml_device, 
        half_precision: bool,
        image_to_upscale: numpy_ndarray, 
        image_mode: str, 
        num_tiles_x: int, 
        num_tiles_y: int, 
        target_height: int, 
        target_width: int
        ) -> numpy_ndarray:
    
    tiles_list = split_image_into_tiles(image_to_upscale, num_tiles_x, num_tiles_y)

    for tile_index in range(len(tiles_list)):
        tile = tiles_list[tile_index]
        tile_upscaled = AI_enhance(AI_model, backend, half_precision, tile, image_mode)
        tiles_list[tile_index] = tile_upscaled

    image_upscaled = combine_tiles_into_image(tiles_list, image_mode, target_height, target_width, num_tiles_x, num_tiles_y)

    return image_upscaled

def normalize_image(
        image: numpy_ndarray
        ) -> tuple:

    if numpy_max(image) > 256: 
        max_range = 65535
    else:
        max_range = 255

    normalized_image = image / max_range

    return normalized_image, max_range

def preprocess_image(
        image: numpy_ndarray, 
        half_precision: bool, 
        backend: directml_device
        ) -> torch_tensor:
    
    image = torch_from_numpy(numpy_transpose(image, (2, 0, 1))).float()
    if half_precision: 
        return image.unsqueeze(0).half().to(backend)
    else:              
        return image.unsqueeze(0).to(backend)

def process_image_with_model(
        AI_model: any, 
        image: torch_tensor
        ) -> numpy_ndarray:
    
    output_image = AI_model(image)
    output_image = output_image.squeeze().float().clamp(0, 1).cpu().numpy()
    output_image = numpy_transpose(output_image, (1, 2, 0))
    
    return output_image
    
def postprocess_output(
        output_image: numpy_ndarray, 
        max_range: int
        ) -> numpy_ndarray:
    
    if max_range == 255:
        return (output_image * max_range).round().astype(uint8) 
    elif max_range == 65535:
        return (output_image * max_range).round().astype(uint16)
 


# BSRGAN Architecture

class ResidualDenseBlock_5C(Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch_nn_init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch_cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch_cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch_cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch_cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB_BSRGAN(Module):

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

class BSRGANx4_Net(Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(BSRGANx4_Net, self).__init__()

        self.conv_first = Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = Sequential(*[RRDB_BSRGAN(nf, gc) for _ in range(nb)])
        self.trunk_conv = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1    = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2    = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv     = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last  = Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu      = LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch_nn_init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        fea   = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea   = fea + trunk

        fea = self.lrelu(self.upconv1(torch_nn_interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(torch_nn_interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class BSRGANx2_Net(Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(BSRGANx2_Net, self).__init__()

        self.conv_first = Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = Sequential(*[RRDB_BSRGAN(nf, gc) for _ in range(nb)])
        self.trunk_conv = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1    = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv     = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last  = Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu      = LeakyReLU(negative_slope=0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch_nn_init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        fea   = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea   = fea + trunk

        fea = self.lrelu(self.upconv1(torch_nn_interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

# SAFM Architecture

class LayerNorm_SAFM(Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = Parameter(torch_ones(normalized_shape))
        self.bias   = Parameter(torch_zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]: raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return torch_nn_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch_sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CCM(Module):

    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = Sequential(
            Conv2d(dim, hidden_dim, 3, 1, 1),
            GELU(), 
            Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class SAFM(Module):

    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = ModuleList([Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = torch_nn_adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = torch_nn_interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch_cat(out, dim=1))
        out = self.act(out) * x
        return out

class AttBlock(Module):

    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm_SAFM(dim) 
        self.norm2 = LayerNorm_SAFM(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x
        
class SAFM_Net(Module):

    def __init__(self, dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()

        self.to_feat = Conv2d(3, dim, 3, 1, 1)
        self.feats   = Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img  = Sequential(
            Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

# SRVGGNetCompact

class SRVGGNetCompact(Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat   = num_feat
        self.num_conv   = num_conv
        self.upscale    = upscale

        self.body = ModuleList()
        self.body.append(Conv2d(num_in_ch, num_feat, 3, 1, 1))
        activation = PReLU(num_parameters=num_feat)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(Conv2d(num_feat, num_feat, 3, 1, 1))
            activation = PReLU(num_parameters=num_feat)
            self.body.append(activation)

        self.body.append(Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = torch_nn_interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

class SRVGGNetCompact_Plus(Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super(SRVGGNetCompact_Plus, self).__init__()
        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat   = num_feat
        self.num_conv   = num_conv
        self.upscale    = upscale

        self.body = ModuleList()
        self.body.append(Conv2d(num_in_ch, num_feat, 3, 1, 1))
        activation = PReLU(num_parameters=num_feat)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(Conv2d(num_feat, num_feat, 3, 1, 1))
            activation = PReLU(num_parameters=num_feat)
            self.body.append(activation)

        self.body.append(Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = torch_nn_interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out



#  Slice functions -------------------

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

def file_need_tilling(
        image: numpy_ndarray, 
        tiles_resolution: int
        ) -> tuple:
    
    height, width = get_image_resolution(image)
    image_pixels = height * width
    max_supported_pixels = tiles_resolution * tiles_resolution

    if image_pixels > max_supported_pixels:
        num_tiles_horizontal = (width + tiles_resolution - 1) // tiles_resolution
        num_tiles_vertical   = (height + tiles_resolution - 1) // tiles_resolution
        return True, num_tiles_horizontal, num_tiles_vertical
    else:
        return False, None, None
    
def add_alpha_channel(
        tile: numpy_ndarray
        ) -> numpy_ndarray:
    
    if tile.shape[2] == 3:
        alpha_channel = numpy_full((tile.shape[0], tile.shape[1], 1), 255, dtype = uint8)
        tile          = numpy_concatenate((tile, alpha_channel), axis = 2)
    return tile



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
    
    if os_path_exists(name_dir): 
        remove_directory(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def stop_thread() -> None: 
    stop = 1 + "x"



# Image/video Utils functions ------------------------

def image_write(
        file_path: str, 
        file_data: numpy_ndarray
        ) -> None: 
    
    _, file_extension = os_path_splitext(file_path)
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def image_read(
        file_path: str, 
        flags: int = IMREAD_UNCHANGED
        ) -> numpy_ndarray: 
    
    with open(file_path, 'rb') as file:
        image_data    = file.read()
        image_buffer  = numpy_frombuffer(image_data, uint8)
        image_decoded = opencv_imdecode(image_buffer, flags)
        return image_decoded

def get_image_resolution(
        image: numpy_ndarray
        ) -> tuple:
    
    height = image.shape[0]
    width  = image.shape[1]

    return height, width 

def resize_file(
        file: numpy_ndarray, 
        resize_factor: int
        ) -> numpy_ndarray:
    
    old_height, old_width = get_image_resolution(file)

    new_width  = int(old_width * resize_factor)
    new_height = int(old_height * resize_factor)

    if resize_factor > 1: 
        return opencv_resize(file, (new_width, new_height), interpolation = INTER_CUBIC)
    elif resize_factor < 1: 
        return opencv_resize(file, (new_width, new_height), interpolation = INTER_LINEAR)
    else:                   
        return file

def get_image_mode(
        image: numpy_ndarray
        ) -> str:
    
    if len(image.shape) == 2: 
        return 'Grayscale'
    elif image.shape[2] == 3: 
        return 'RGB' 
    elif image.shape[2] == 4: 
        return 'RGBA'
    else:                     
        return 'Unknown'

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
    ) -> list[str]:

    create_dir(target_directory)

    with VideoFileClip(video_path) as video_file_clip:
        try: 
            audio_path = f"{target_directory}{os_separator}audio.mp3"
            video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
        except:
            pass
        
        video_frame_rate = extract_video_fps(video_path)
        frames_sequence_path = f"{target_directory}{os_separator}frame_%01d.jpg"
        video_frames_list = video_file_clip.write_images_sequence(
            nameformat = frames_sequence_path, 
            fps        = video_frame_rate,
            verbose    = False, 
            withmask   = True,
            logger     = None, 
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
        ) -> None:
        
    frame_rate = extract_video_fps(video_path)

    if selected_video_extension == '.mp4 (x264)':  
        selected_video_extension = '.mp4'
        codec = 'libx264'
    elif selected_video_extension == '.mp4 (x265)':  
        selected_video_extension = '.mp4'
        codec = 'libx265'
    elif selected_video_extension == '.avi': 
        selected_video_extension = '.avi' 
        codec = 'png'

    upscaled_video_path = prepare_output_video_filename(video_path, selected_AI_model, resize_factor, selected_video_extension, selected_interpolation_factor)

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os_path_exists(audio_path):
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            audio   = audio_path,
                            codec   = codec,
                            bitrate = '16M',
                            verbose = False,
                            logger  = None,
                            ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                             fps     = frame_rate,
                             codec   = codec,
                             bitrate = '16M',
                             verbose = False,
                             logger  = None,
                             ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
                             threads = cpu_number)  
        
    return upscaled_video_path
        
def interpolate_images(
        starting_image: numpy_ndarray,
        upscaled_image: numpy_ndarray,
        image_target_height: int, 
        image_target_width: int,
        selected_interpolation_factor: float,
        ) -> numpy_ndarray:
    
    starting_image_importance = selected_interpolation_factor
    upscaled_image_importance = 1 - starting_image_importance

    try: 
        starting_image = opencv_resize(starting_image, (image_target_width, image_target_height), interpolation = INTER_CUBIC)
        
        starting_image     = add_alpha_channel(starting_image)
        upscaled_image     = add_alpha_channel(upscaled_image)
        interpolated_image = opencv_addWeighted(upscaled_image, upscaled_image_importance, starting_image, starting_image_importance, 0)
        return interpolated_image
    except:
        return upscaled_image

def get_final_image_shape(
        image_to_upscale: numpy_ndarray, 
        upscaling_factor: int
        ) -> tuple:
    
    image_to_upscale_height, image_to_upscale_width = get_image_resolution(image_to_upscale)

    target_height = image_to_upscale_height * upscaling_factor
    target_width  = image_to_upscale_width  * upscaling_factor
    
    return target_height, target_width

def calculate_time_to_complete_video(
        start_timer: float, 
        end_timer: float, 
        how_many_frames: int, 
        index_frame: int
        ) -> str:
    
    seconds_for_frame = round(end_timer - start_timer, 2)
    frames_left       = how_many_frames - (index_frame + 1)
    seconds_left      = seconds_for_frame * frames_left

    hours_left   = seconds_left // 3600
    minutes_left = (seconds_left % 3600) // 60
    seconds_left = round((seconds_left % 3600) % 60)

    time_left = ""

    if int(hours_left) > 0: 
        time_left = f"{int(hours_left):02d}h"
    
    if int(minutes_left) > 0: 
        time_left = f"{time_left}{int(minutes_left):02d}m"

    if seconds_left > 0: 
        time_left = f"{time_left}{seconds_left:02d}s"

    return time_left        

def get_video_info_for_upscaling(
        frame_list_paths: list, 
        resize_factor: int, 
        upscaling_factor: int, 
        tiles_resolution: int
        ) -> tuple:
    
    first_frame = image_read(frame_list_paths[0])

    # Tilling?
    first_frame_resized = resize_file(first_frame, resize_factor)
    need_tiles, num_tiles_x, num_tiles_y = file_need_tilling(first_frame_resized, tiles_resolution)
 
    # Resizing shapes
    height, width = get_image_resolution(first_frame)
    frame_resized_width  = int(width * resize_factor)
    frame_resized_height = int(height * resize_factor)

    # Upscaling shapes
    frame_target_width  = frame_resized_width * upscaling_factor
    frame_target_height = frame_resized_height * upscaling_factor
        
    return frame_target_height, frame_target_width, need_tiles, num_tiles_x, num_tiles_y

def update_process_status_videos(
        processing_queue: multiprocessing_Queue, 
        file_number: int, 
        start_timer: float, 
        index_frame: int, 
        how_many_frames: int
        ) -> None:
    
    if index_frame != 0 and (index_frame + 1) % 4 == 0:    
        percent_complete = (index_frame + 1) / how_many_frames * 100 
        end_timer        = timer()
        time_left        = calculate_time_to_complete_video(start_timer, end_timer, how_many_frames, index_frame)
    
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
    except Exception as ex:
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
                error_message = f"Error during upscale process :("
                error = actual_step.replace(ERROR_STATUS, "")
                info_message.set(error_message)
                show_error_message(error)
                stop_thread()

            else:
                info_message.set(actual_step)

            sleep(1)
    except:
        place_upscale_button()
        
def read_process_status() -> None:
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
    global half_precision
    global selected_AI_device 
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
        print(f"  AI half precision: {half_precision}")
        print(f"  Interpolation: {selected_interpolation}")
        print(f"  Interpolation factor: {selected_interpolation_factor}")
        print(f"  Selected GPU: {directml_device_name(selected_AI_device)}")
        print(f"  Selected image output extension: {selected_image_extension}")
        print(f"  Selected video output extension: {selected_video_extension}")
        print(f"  Tiles resolution for selected GPU VRAM: {tiles_resolution}x{tiles_resolution}px")
        print(f"  Resize factor: {int(resize_factor * 100)}%")
        print(f"  Cpu number: {cpu_number}")
        print("=" * 50)

        backend = torch_device(directml_device(selected_AI_device))

        place_stop_button()

        process_upscale_orchestrator = Process(
                                        target = upscale_orchestrator,
                                        args   = (processing_queue, selected_file_list, selected_AI_model, backend, selected_image_extension, 
                                                  tiles_resolution, resize_factor, cpu_number, half_precision, selected_video_extension,
                                                  selected_interpolation, selected_interpolation_factor)
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
    if selected_interpolation_factor == 0.3:
        to_append += f"_Interpolatation-Low"

    elif selected_interpolation_factor == 0.5:
        to_append += f"_Interpolatation-Medium"

    elif selected_interpolation_factor == 0.7:
        to_append += f"_Interpolatation-High"

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
    if selected_interpolation_factor == 0.3:
        to_append += f"_Interpolatation-Low"

    elif selected_interpolation_factor == 0.5:
        to_append += f"_Interpolatation-Medium"

    elif selected_interpolation_factor == 0.7:
        to_append += f"_Interpolatation-High"

    # Selected video extension
    to_append += f"{selected_video_extension}"
        
    result_path += to_append

    return result_path

def prepare_output_video_frames_directory_filename(
        video_path: str, 
        selected_AI_model: str, 
        resize_factor: int, 
        selected_interpolation_factor: float
        ) -> str:
    
    result_path, _  = os_path_splitext(video_path)

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Selected intepolation
    if selected_interpolation_factor == 0.3:
        to_append += f"_Interpolatation-Low"

    elif selected_interpolation_factor == 0.5:
        to_append += f"_Interpolatation-Medium"

    elif selected_interpolation_factor == 0.7:
        to_append += f"_Interpolatation-High"

    result_path += to_append

    return result_path

def upscale_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_AI_model: str,
        backend: directml_device, 
        selected_image_extension: str,
        tiles_resolution: int,
        resize_factor: int,
        cpu_number: int,
        half_precision: bool,
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    write_process_status(processing_queue, f"Loading")
    
    if   'x2' in selected_AI_model: upscaling_factor = 2
    elif 'x4' in selected_AI_model: upscaling_factor = 4

    torch_set_num_threads(2)

    try:
        write_process_status(processing_queue, f"Loading AI model")
        AI_model = load_AI_model(selected_AI_model, backend, half_precision)

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(processing_queue, 
                                file_path, 
                                file_number, 
                                AI_model, 
                                selected_AI_model, 
                                upscaling_factor, 
                                backend, 
                                tiles_resolution, 
                                resize_factor, 
                                cpu_number, 
                                half_precision, 
                                selected_video_extension, 
                                selected_interpolation, 
                                selected_interpolation_factor)
            else:
                upscale_image(processing_queue, 
                                file_path, 
                                file_number, 
                                AI_model, 
                                selected_AI_model, 
                                upscaling_factor, 
                                backend, 
                                selected_image_extension, 
                                tiles_resolution, 
                                resize_factor, 
                                half_precision, 
                                selected_interpolation, 
                                selected_interpolation_factor)

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS}{str(exception)}")

def upscale_image(
        processing_queue: multiprocessing_Queue,
        image_path: str, 
        file_number: int,
        AI_model: any, 
        selected_AI_model: str, 
        upscaling_factor: int,
        backend: directml_device, 
        selected_image_extension: str,
        tiles_resolution: int, 
        resize_factor: int, 
        half_precision: bool,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    result_image_path = prepare_output_image_filename(image_path, selected_AI_model, resize_factor, selected_image_extension, selected_interpolation_factor)
        
    starting_image   = image_read(image_path)
    image_to_upscale = resize_file(starting_image, resize_factor)
    image_mode       = get_image_mode(starting_image)

    target_height, target_width = get_final_image_shape(image_to_upscale, upscaling_factor)
    need_tiles, num_tiles_x, num_tiles_y = file_need_tilling(image_to_upscale, tiles_resolution)

    write_process_status(processing_queue, f"{file_number}. Upscaling image")

    # Upscale image w/out tilling
    if need_tiles == False:
        image_upscaled = AI_enhance(AI_model, backend, half_precision, image_to_upscale, image_mode)

    # Upscale image with tilling
    else:
        image_upscaled = AI_enhance_tiles(AI_model, backend, half_precision, image_to_upscale, image_mode, num_tiles_x, num_tiles_y, target_height, target_width)

    # Interpolation
    if selected_interpolation:
        image_upscaled = interpolate_images(starting_image, image_upscaled, target_height, target_width, selected_interpolation_factor)
    
    # Save image
    image_write(result_image_path, image_upscaled)

    # Metadata copy from original image
    copy_file_metadata(image_path, result_image_path)

def upscale_video(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        AI_model: any, 
        selected_AI_model: str, 
        upscaling_factor: int,
        backend: directml_device, 
        tiles_resolution: int,
        resize_factor: int, 
        cpu_number: int, 
        half_precision: bool, 
        selected_video_extension: str,
        selected_interpolation: bool,
        selected_interpolation_factor: float
        ) -> None:
    
    # Directory for video frames and audio
    target_directory = prepare_output_video_frames_directory_filename(video_path, selected_AI_model, resize_factor, selected_interpolation_factor)

    # Extract video frames and audio
    write_process_status(processing_queue, f"{file_number}. Extracting video frames and audio")
    frame_list_paths, audio_path = extract_video_frames_and_audio(target_directory, video_path)

    target_height, target_width, need_tiles, num_tiles_x, num_tiles_y = get_video_info_for_upscaling(frame_list_paths, resize_factor, upscaling_factor, tiles_resolution)
    how_many_frames = len(frame_list_paths) 

    write_process_status(processing_queue, f"{file_number}. Upscaling video")  
    for index_frame in range(how_many_frames):
        start_timer = timer()

        starting_frame   = image_read(frame_list_paths[index_frame])
        frame_to_upscale = resize_file(starting_frame, resize_factor)
        image_mode       = get_image_mode(starting_frame)

        # Upscale frame w/out tilling
        if need_tiles == False:
            frame_upscaled = AI_enhance(AI_model, backend, half_precision, frame_to_upscale, image_mode)

        # Upscale frame with tilling
        else:
            frame_upscaled = AI_enhance_tiles(AI_model, backend, half_precision, frame_to_upscale, image_mode, num_tiles_x, num_tiles_y, target_height, target_width)

        # Interpolation
        if selected_interpolation:
            frame_upscaled = interpolate_images(starting_frame, frame_upscaled, target_height, target_width, selected_interpolation_factor)

        # Save frame overwriting existing frame
        image_write(frame_list_paths[index_frame], frame_upscaled)
    
        # Update process status every 4 frames
        update_process_status_videos(processing_queue, file_number, start_timer, index_frame, how_many_frames)

    # Upscaled video reconstuction
    write_process_status(processing_queue, f"{file_number}. Processing upscaled video")
    upscaled_video_path = video_reconstruction_by_frames(video_path, audio_path, frame_list_paths, selected_AI_model, resize_factor, cpu_number, selected_video_extension, selected_interpolation_factor)
    
    # Metadata copy from original video to upscaled video
    copy_file_metadata(video_path, upscaled_video_path)

    # Remove upscaled frames directory after video reconstruction
    remove_dir(target_directory)



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

def user_input_checks() -> None:
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
        if   selected_AI_model in BSRGAN_models_list:          
            vram_multiplier = RRDB_vram_multiplier
        elif selected_AI_model in SRVGGNetCompact_models_list: 
            vram_multiplier = SRVGGNetCompact_vram_multiplier
        elif selected_AI_model in SAFMN_models_list:           
            vram_multiplier = SAFMN_vram_multiplier

        selected_vram = (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))

        if half_precision == True: 
            tiles_resolution = int(selected_vram * 100)

            if selected_AI_model in SAFMN_models_list: 
                info_message.set("SAFM not compatible with Half precision")
                is_ready = False

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

def extract_image_info(
        image_file: str
        ) -> tuple:
    
    image_name = str(image_file.split("/")[-1])

    image  = image_read(image_file)
    width  = int(image.shape[1])
    height = int(image.shape[0])

    image_label = f"{image_name} • {width}x{height}"

    ctkimage = CTkImage(pillow_image_open(image_file), size = (25, 25))

    return image_label, ctkimage

def extract_video_info(
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
        image_write("temp.jpg", frame)
        break
    cap.release()

    video_label = f"{video_name} • {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames • {round(frame_rate, 2)}fps"

    ctkimage = CTkImage(pillow_image_open("temp.jpg"), size = (25, 25))
    
    return video_label, ctkimage

def show_error_message(
        exception: str
        ) -> None:
    
    messageBox_title = "Upscale error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text  = f" {str(exception)} "

    CTkInfoMessageBox(messageType = "error", 
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
    
    global half_precision
    if   selected_option == "Full precision": 
        half_precision = False
    elif selected_option == "Half precision": 
        half_precision = True

def select_gpu_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_AI_device    
    for device in gpu_list:
        if device.name == selected_option:
            selected_AI_device = device.index

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
        if selected_option == "Low": 
            selected_interpolation_factor = 0.3
        elif selected_option == "Medium": 
            selected_interpolation_factor = 0.5
        elif selected_option == "High": 
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
        "  • BSRGANx4\n",

        "\n SAFM (2023) - Slow but lightweight AI architecture\n" + 
        " High-quality upscale\n" +
        " Recommended for image upscaling\n" + 
        " Does not support Half-precision\n" +
        "  • SAFMNLx4\n" + 
        "  • SAFMNLx4_Real\n",
    ]

    CTkInfoMessageBox(messageType = "info",
                    title = "AI model", 
                    subtitle = "This widget allows to choose between different AI models for upscaling",
                    default_value = "RealESR_Gx4",
                    option_list = option_list)

def open_info_AI_precision():
    option_list = [
        " \n Half precision\n  • 50% less GPU memory\n  • 30-70% slower\n  • old GPUs are not compatible with this mode \n",
        " \n Full precision (>=8GB VRAM recommended)\n  • 50% more GPU memory\n  • 30-70% faster\n  • compatible with all GPUs \n",
    ]

    CTkInfoMessageBox(messageType = "info",
                    title = "AI precision", 
                    subtitle = "This widget allows to choose the AI upscaling mode",
                    default_value = "Half precision",
                    option_list = option_list)

def open_info_interpolation():
    option_list = [
        " Interpolation is the fusion of the upscaled image produced by AI and the original image",
        " \n Increase the quality of the final result\n  • especially when using the tilling/merging function (with low VRAM) \n  • especially at low Input resolution % values (<50%)\n",
        " \n Levels of interpolation\n  • Disabled - 100% upscaled\n  • Low - 30% original / 70% upscaled\n  • Medium - 50% original / 50% upscaled\n  • High - 70% original / 30% upscaled\n"
    ]

    CTkInfoMessageBox(messageType = "info",
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

    CTkInfoMessageBox(messageType = "info",
                    title = "Image output", 
                    subtitle = "This widget allows to choose the extension of upscaled images",
                    default_value = ".png",
                    option_list = option_list)

def open_info_video_extension():
    option_list = [
        " MP4 (x264) - produces well compressed video using x264 codec",
        " MP4 (x265) - produces well compressed video using x265 codec",
        " AVI - produces the highest quality video"
    ]

    CTkInfoMessageBox(messageType = "info",
                title = "Video output", 
                subtitle = "This widget allows to choose the extension of the upscaled video",
                default_value = ".mp4 (x264)",
                option_list = option_list)

def open_info_gpu():
    option_list = [
        "Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be",
        "For optimal results, it is essential to regularly update your GPU drivers"
    ]

    CTkInfoMessageBox(messageType = "info",
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

    CTkInfoMessageBox(messageType = "info",
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

    CTkInfoMessageBox(messageType = "info",
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

    CTkInfoMessageBox(messageType = "info",
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
        scrollable_frame_file_list = ScrollableImagesTextFrame(master = window, fg_color = dark_color, bg_color = dark_color)
        scrollable_frame_file_list.place(relx = 0.0, 
                                        rely = 0.0, 
                                        relwidth  = 1.0,  
                                        relheight = 0.45)
        
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

def place_github_button():
    git_button = CTkButton(master      = window, 
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
    telegram_button = CTkButton(master     = window, 
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

    input_file_text = CTkLabel(master      = window, 
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
    AI_mode_menu   = create_option_menu(select_AI_mode_from_menu, AI_modes_list)
    
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

def place_gpu_menu():

    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_gpu_from_menu, gpu_list_names)
    
    gpu_button.place(relx = 0.5, rely = row2_y - 0.053, anchor = "center")
    gpu_menu.place(relx = 0.5, rely  = row2_y, anchor = "center")

def place_vram_textbox():

    vram_button  = create_info_button(open_info_vram_limiter, "GPU Vram (GB)")
    vram_textbox = create_text_box(selected_VRAM_limiter) 
  
    vram_button.place(relx = column1_x, rely = row3_y - 0.053, anchor = "center")
    vram_textbox.place(relx = column1_x, rely  = row3_y, anchor = "center")

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
                            font         = bold10,
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
   


# GUI utils ---------------------------

class Gpu:
    def __init__(
            self, 
            index: int, 
            name: str, 
            memory: any, 
            float64: bool
            ) -> None:
        
        self.index: int    = index
        self.name: str     = name
        self.memory: any   = memory
        self.float64: bool = float64

class ScrollableImagesTextFrame(CTkScrollableFrame):

    def __init__(
            self, 
            master, 
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)
        self.file_list   = []
        self.label_list  = []
        self.button_list = []

    def get_selected_file_list(
            self: any
            ) -> list: 
        
        return self.file_list

    def add_clean_button(
            self: any
            ) -> None:
        
        label  = CTkLabel(self, text = "")
        button = CTkButton(self, 
                        image        = clear_icon,
                        font         = bold11,
                        text         = "CLEAN", 
                        compound     = "left",
                        width        = 100, 
                        height       = 28,
                        border_width = 1,
                        fg_color     = "#282828",
                        text_color   = "#E0E0E0",
                        border_color = "#0096FF")

        button.configure(command=lambda: self.clean_all_items())
        
        button.grid(row = len(self.button_list), column=1, pady=(7, 7), padx = (0, 7))
        self.label_list.append(label)
        self.button_list.append(button)

    def add_item(
            self: any, 
            text_to_show: str, 
            file_element: str, 
            image: CTkImage = None
            ) -> None:
        
        label = CTkLabel(self, 
                    text       = text_to_show,
                    font       = bold11,
                    image      = image, 
                    text_color = "#E0E0E0",
                    compound   = "left", 
                    padx       = 10,
                    pady       = 5,
                    anchor     = "center")
                        
        label.grid(row = len(self.label_list), column = 0, pady = (3, 3), padx = (3, 3), sticky = "w")
        self.label_list.append(label)
        self.file_list.append(file_element)    

    def clean_all_items(
            self: any
            ) -> None:
        
        self.label_list  = []
        self.button_list = []
        self.file_list   = []
        self.destroy()
        place_loadFile_section()

class CTkInfoMessageBox(CTkToplevel):

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
        command: any, 
        text: str
        ) -> CTkButton:
    
    return CTkButton(master  = window, 
                    command  = command,
                    text          = text,
                    fg_color      = "transparent",
                    text_color    = "#C0C0C0",
                    anchor        = "w",
                    height        = 23,
                    width         = 150,
                    corner_radius = 12,
                    font          = bold12,
                    image         = info_icon)

def create_option_menu(
        command: any, 
        values: list
        ) -> CTkOptionMenu:
    
    return CTkOptionMenu(master = window, 
                        command = command,
                        values  = values,
                        width              = 150,
                        height             = 31,
                        anchor             = "center",
                        dropdown_font      = bold10,
                        font               = bold11,
                        text_color         = "#C0C0C0",
                        fg_color           = "#000000",
                        button_color       = "#000000",
                        button_hover_color = "#000000",
                        dropdown_fg_color  = "#000000")

def create_text_box(
        textvariable: StringVar
        ) -> CTkEntry:
    
    return CTkEntry(master        = window, 
                    textvariable  = textvariable,
                    border_width  = 1,
                    width         = 150,
                    height        = 30,
                    font          = bold10,
                    justify       = "center",
                    fg_color      = "#000000",
                    border_color  = "#404040")



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
        place_gpu_menu()
        place_vram_textbox()
        
        place_input_resolution_textbox()
        place_cpu_textbox()
        place_message_label()
        place_upscale_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing_freeze_support()

    processing_queue = multiprocessing_Queue(maxsize=1)

    gpu_list_names = []
    gpu_list = []

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    how_many_gpus = directml_device_count()
    for index in range(how_many_gpus): 
        gpu_index   = index
        gpu_name    = directml_device_name(index)
        gpu_memory  = directml_gpu_memory(index)
        gpu_float64 = directml_has_float64_support(index)
        gpu = Gpu(gpu_index, gpu_name, gpu_memory, gpu_float64)

        gpu_list.append(gpu)
        gpu_list_names.append(gpu_name)

    window = CTk() 

    global selected_file_list
    global selected_AI_model
    global half_precision
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global selected_interpolation
    global selected_interpolation_factor
    global tiles_resolution
    global resize_factor
    global cpu_number

    selected_file_list = []

    if AI_modes_list[0] == "Half precision": 
        half_precision = True
    elif AI_modes_list[0] == "Full precision": 
        half_precision = False

    if interpolation_list[0] == "Disabled": 
        selected_interpolation = False
        selected_interpolation_factor = None
    else:
        selected_interpolation = True
        if interpolation_list[0] == "Low": 
            selected_interpolation_factor = 0.3
        elif interpolation_list[0] == "Medium":
            selected_interpolation_factor = 0.5
        elif interpolation_list[0] == "High": 
            selected_interpolation_factor = 0.7

    selected_AI_device       = 0
    selected_AI_model        = AI_models_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    info_message            = StringVar()
    selected_resize_factor  = StringVar()
    selected_VRAM_limiter   = StringVar()
    selected_cpu_number     = StringVar()

    info_message.set("Hi :)")
    selected_resize_factor.set("60")
    selected_VRAM_limiter.set("8")
    selected_cpu_number.set(str(int(os_cpu_count()/2)))

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
