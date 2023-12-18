
# Standard library imports

import os
from download import download
import sys
from shutil          import rmtree
from timeit          import default_timer as timer
from time            import sleep
from math            import gcd  as math_gcd
from subprocess      import run  as subprocess_run
from webbrowser      import open as open_browser

from threading       import Thread
from multiprocessing import ( 
    Process, 
    freeze_support 
)

from os import (
    sep         as os_separator,
    devnull     as os_devnull,
    chmod       as os_chmod,
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

from torch import (
    device          as torch_device,
    no_grad         as torch_no_grad,
    zeros           as torch_zeros,
    cat             as torch_cat,
    load            as torch_load,
    ones            as torch_ones,
    sqrt            as torch_sqrt,
    from_numpy      as torch_from_numpy,
    set_num_threads as torch_set_num_threads,
    chunk           as torch_chunk,
)

from torch.nn.functional import (
    adaptive_max_pool2d, 
    interpolate as torch_nn_interpolate, 
    layer_norm  as torch_nn_layer_norm,
    gelu        as torch_functional_gelu,
    normalize   as torch_functional_normalize,
    pad         as torch_functional_pad 
)

from torch.nn import (
    init as torch_nn_init,
    Tanh,
    Sequential,
    Conv2d,
    Linear,
    Module,
    ModuleList,
    Parameter,
    PixelShuffle,
    GELU,
    PReLU,
    ReLU,
    LeakyReLU
)

from torch_directml import (
     device       as directml_device,
     device_count as directml_device_count,
     device_name  as directml_device_name
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
    INTER_AREA,
    INTER_LINEAR,
    VideoCapture as opencv_VideoCapture,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    addWeighted  as opencv_addWeighted,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from numpy import (
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

# changelog 2.8
# added SAFMN AI architecture
# removed RealESRGAN
# updated exiftool version to version 12.68
# changed images read and write supporting file with special characters
# lot of bugfixes and improvements

app_name  = "QualityScaler"
version   = "2.8"
app_name_color = "#DA70D6"

githubme    = "https://github.com/Djdefrag/QualityScaler"
telegramme  = "https://linktr.ee/j3ngystudio"


# SRVGGNetCompact
SRVGGNetCompact_vram_multiplier = 1.8
SRVGGNetCompact_models_list     = [ 'RealESR_Gx4', 'RealSRx4_Anime' ]

# RRDB models
RRDB_vram_multiplier   = 0.9
BSRGAN_models_list     = [ 'BSRGANx4', 'BSRGANx2' ]

# SAFM models
SAFMN_vram_multiplier = 2
SAFMN_models_list     = [ 'SAFMNLx4', 'SAFMNLx4_Real']

# DITN models
DITN_vram_multiplier = 2
DITN_models_list     = [ 'DITN_x4' ]


AI_models_list = (
                SRVGGNetCompact_models_list
                + BSRGAN_models_list 
                + SAFMN_models_list 
                # + DITN_models_list
                )




image_extension_list  = [ '.jpg', '.png', '.bmp', '.tiff' ]
video_extension_list  = [ '.mp4', '.avi', '.webm' ]
interpolation_list    = [ 'Enabled', 'Disabled' ]
AI_modes_list         = [ "Half precision", "Full precision" ]

log_file_path  = f"{app_name}.log"
temp_dir       = f"{app_name}_temp"
audio_path     = f"{app_name}_temp{os_separator}audio.mp3"
frame_sequence = f"{app_name}_temp{os_separator}frame_%01d.jpg"

device_list_names  = []
device_list        = []

dark_color       = "#080808"
offset_y_options = 0.125
row0_y           = 0.56
row1_y           = row0_y + offset_y_options
row2_y           = row1_y + offset_y_options
row3_y           = row2_y + offset_y_options

offset_x_options = 0.28
column1_x        = 0.5
column0_x        = column1_x - offset_x_options
column2_x        = column1_x + offset_x_options

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

supported_file_extensions = [
                            '.heic', 
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



# AI models -------------------


def get_model_url(model_name: str) -> str:
    return f"https://zackees.github.io/ai-image-video-models/{model_name}"

def torch_load_model(model_path, map_location = None) -> Module:
    print(f"Loading model from {model_path}")
    model_name = os.path.basename(model_path)
    donwload_url = get_model_url(model_name)
    download(donwload_url, model_path, replace = False, timeout=60)
    return torch_load(model_path, map_location = map_location)

def load_AI_model(selected_AI_model, 
                  upscaling_factor, 
                  backend, 
                  half_precision):
    
    update_process_status(f"Loading AI model")
    model_path = find_by_relative_path(f"AI{os_separator}{selected_AI_model}.pth")
    
    with torch_no_grad():

        if selected_AI_model in BSRGAN_models_list:
            model = BSRGAN_Net(sf = upscaling_factor)
            pretrained_model = torch_load_model(model_path, map_location = torch_device('cpu'))

        elif selected_AI_model in SRVGGNetCompact_models_list:
            if 'RealESR_Gx4' in selected_AI_model:
                model = SRVGGNetCompact(num_conv = 32)
            else:
                model = SRVGGNetCompact(num_conv = 16)
            pretrained_model = torch_load_model(model_path, map_location = torch_device('cpu'))['params']

        elif selected_AI_model in SAFMN_models_list:
            model = SAFMN(dim = 128, n_blocks = 16)
            pretrained_model = torch_load_model(model_path, map_location = torch_device('cpu'))['params']
        
        elif selected_AI_model in DITN_models_list:
            model = DITN_Real()
            pretrained_model = torch_load_model(model_path)
            
        model.load_state_dict(pretrained_model, strict = True)
        model.eval()
        if half_precision: model = model.half()
        model = model.to(backend, non_blocking = True)

    return model



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

class BSRGAN_Net(Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(BSRGAN_Net, self).__init__()
        self.sf = sf

        self.conv_first = Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = Sequential(*[RRDB_BSRGAN(nf, gc) for _ in range(nb)])
        self.trunk_conv = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1    = Conv2d(nf, nf, 3, 1, 1, bias=True)

        if sf == 4: self.upconv2 = Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv    = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu     = LeakyReLU(negative_slope=0.2, inplace=True)

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
        if self.sf == 4: fea = self.lrelu(self.upconv2(torch_nn_interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

# SRVGGNetCompact

class SRVGGNetCompact(Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = ModuleList()
        # the first conv
        self.body.append(Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = torch_nn_interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
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
                s = adaptive_max_pool2d(xc[i], p_size)
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
        
class SAFMN(Module):

    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = Conv2d(3, dim, 3, 1, 1)

        self.feats = Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = Sequential(
            Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x



# DITN

class FeedForward(Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch_functional_gelu(x1) * x2
        x = self.project_out(x)
        return x

class ISA(Module):
    def __init__(self, dim, bias):
        super(ISA, self).__init__()
        self.temperature = Parameter(torch_ones(1, 1, 1))
        self.qkv = Linear(dim, dim*3)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.data.shape
        x = x.view(b, c, -1).transpose(-1, -2)
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = torch_functional_normalize(q, dim=-1)
        k = torch_functional_normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out

class SDA(Module):
    def __init__(self, n_feats, LayerNorm_type='WithBias'):
        super(SDA, self).__init__()
        i_feats = 2 * n_feats
        self.scale = Parameter(torch_zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.DConvs = Sequential(
            Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats),
            Conv2d(n_feats, n_feats, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats, dilation=3),
            Conv2d(n_feats, n_feats, 1, 1, 0))

        self.proj_first = Sequential(
            Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = Sequential(
            Conv2d(n_feats, n_feats, 1, 1, 0))
        self.dim = n_feats
    
    def forward(self, x):
        x = self.proj_first(x)
        a, x = torch_chunk(x, 2, dim=1)
        a = self.DConvs(a)
        x = self.proj_last(x * a) * self.scale

        return x

class ITL(Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super(ITL, self).__init__()
        self.attn = ISA(n_feats, bias)
        self.act  = Tanh()
        self.conv1 = Conv2d(n_feats, n_feats, 1)
        self.conv2 = Conv2d(n_feats, n_feats, 1)

        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x

class SAL(Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super(SAL, self).__init__()
        self.SDA = SDA(n_feats)
        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)
        self.act = Tanh()
        self.conv1 = Conv2d(n_feats, n_feats, 1)
        self.conv2 = Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        x = x + self.SDA(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x
    
class UpsampleOneStep(Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(PixelShuffle(scale))

        super(UpsampleOneStep, self).__init__(*m)

class UFONE(Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, ITL_blocks, SAL_blocks, patch_size):
        super(UFONE, self).__init__()
        ITL_body  = [ITL(dim, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(ITL_blocks)]
        self.ITLs = Sequential(*ITL_body)
        SAL_body  = [SAL(dim, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(SAL_blocks)]
        self.SALs = Sequential(*SAL_body)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.data.shape

        local_features = x.view(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        local_features = local_features.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)
        local_features = self.ITLs(local_features)
        local_features = local_features.view(B, H//self.patch_size, W//self.patch_size, C, self.patch_size, self.patch_size)
        global_features = local_features.permute(0, 3, 1, 4, 2, 5).contiguous().view(B,C,H,W)
        global_features = self.SALs(global_features)
        return global_features
    
class DITN_Real(Module):
    def __init__(self, 
        inp_channels = 3, 
        dim          = 60,
        ITL_blocks   = 4,
        SAL_blocks   = 4,
        UFONE_blocks = 1,
        ffn_expansion_factor = 2,
        bias           = False,
        LayerNorm_type = 'WithBias',
        patch_size = 8,
        upscale    = 4
    ):

        super(DITN_Real, self).__init__()
        self.sft = Conv2d(inp_channels, dim, 3, 1, 1)

        self.patch_size = patch_size

        ## UFONE Block1
        UFONE_body = [UFONE(dim, ffn_expansion_factor, bias, LayerNorm_type, ITL_blocks, SAL_blocks, patch_size) for _ in range(UFONE_blocks)]
        self.UFONE = Sequential(*UFONE_body)

        self.conv_after_body = Conv2d(dim, dim, 3, 1, 1)
        self.upsample        = UpsampleOneStep(upscale, dim, 3)
        self.dim         = dim
        self.patch_sizes = [patch_size, patch_size]
        self.scale       = upscale
        self.SAL_blocks  = SAL_blocks
        self.ITL_blocks  = ITL_blocks

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.patch_sizes[0]
        for i in range(1, len(self.patch_sizes)):
            wsize = wsize*self.patch_sizes[i] // math_gcd(wsize, self.patch_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = torch_functional_pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, input_image):
        _, _, old_h, old_w = input_image.shape
        input_image = self.check_image_size(input_image)
        sft = self.sft(input_image)

        local_features = self.UFONE(sft)

        local_features = self.conv_after_body(local_features)
        out_dec_level1 = self.upsample(local_features + sft)

        return out_dec_level1[:, :, 0:old_h * self.scale, 0:old_w * self.scale]



# AI inference -------------------

def get_image_mode(image):
    if len(image.shape) == 2: 
        return 'Grayscale'  # Immagine in scala di grigi
    elif image.shape[2] == 3: 
        return 'RGB'        # RGB
    elif image.shape[2] == 4: 
        return 'RGBA'       # RGBA
    else:                     
        return 'Unknown'
    
def preprocess_image(image, half_precision, backend):
    image = torch_from_numpy(numpy_transpose(image, (2, 0, 1))).float()
    if half_precision: image = image.unsqueeze(0).half().to(backend, non_blocking=True)
    else:              image = image.unsqueeze(0).to(backend, non_blocking=True)
    
    return image

def postprocess_output(output_image, max_range):
    output = (output_image * max_range).round().astype(uint16 if max_range == 65535 else uint8)
    return output

def process_image_with_model(AI_model, image):
    output_image = AI_model(image)
    output_image = output_image.squeeze().float().clamp(0, 1).cpu().numpy()
    output_image = numpy_transpose(output_image, (1, 2, 0))
    
    return output_image

def AI_enhance(AI_model, image, backend, half_precision):
    with torch_no_grad():
        image = image.astype(float32)
        max_range = 255
        if numpy_max(image) > 256: max_range = 65535
        
        image /= max_range
        img_mode = get_image_mode(image)

        if img_mode == "RGB":
            image        = preprocess_image(image, half_precision, backend)
            output_image = process_image_with_model(AI_model, image)
            output_image = postprocess_output(output_image, max_range)
            return output_image
        
        elif img_mode == 'RGBA':
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

        elif img_mode == 'Grayscale':
            image        = opencv_cvtColor(image, COLOR_GRAY2RGB)
            image        = preprocess_image(image, half_precision, backend)
            output_image = process_image_with_model(AI_model, image)
            output_image = opencv_cvtColor(output_image, COLOR_RGB2GRAY)
            output_image = postprocess_output(output_image, max_range)
            return output_image
        


# Classes and utils -------------------

class Gpu:
    def __init__(self, index, name):
        self.name   = name
        self.index  = index

for index in range(directml_device_count()): 
    gpu = Gpu(index = index, name = directml_device_name(index))
    device_list.append(gpu)
    device_list_names.append(gpu.name)

class ScrollableImagesTextFrame(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.label_list  = []
        self.button_list = []
        self.file_list   = []

    def get_selected_file_list(self): return self.file_list

    def add_clean_button(self):
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
        button.grid(row = len(self.button_list), column=1, pady=(0, 10), padx = 5)
        self.label_list.append(label)
        self.button_list.append(button)

    def add_item(self, text_to_show, file_element, image = None):
        label = CTkLabel(self, 
                        text          = text_to_show,
                        font          = bold11,
                        image         = image, 
                        text_color    = "#E0E0E0",
                        compound      = "left", 
                        padx          = 10,
                        pady          = 5,
                        anchor        = "center")
                        
        label.grid(row = len(self.label_list), column = 0, pady = (3, 3), padx = (3, 3), sticky = "w")
        self.label_list.append(label)
        self.file_list.append(file_element)    

    def clean_all_items(self):
        self.label_list  = []
        self.button_list = []
        self.file_list   = []
        self.destroy()
        place_loadFile_section()

class CTkMessageBox(CTkToplevel):
    def __init__(self,
                 title: str = "CTkDialog",
                 text: str = "CTkDialog",
                 type: str = "info"):

        super().__init__()

        self._running: bool = False
        self._title = title
        self._text = text
        self.type = type

        self.title('')
        self.lift()                          # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()                       # make other windows not clickable

    def _create_widgets(self):

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self._text = '\n' + self._text +'\n'

        if self.type == "info":
            color_for_messagebox_title = "#0096FF"
        elif self.type == "error":
            color_for_messagebox_title = "#ff1a1a"


        self._titleLabel = CTkLabel(master  = self,
                                    width      = 500,
                                    anchor     = 'w',
                                    justify    = "left",
                                    fg_color   = "transparent",
                                    text_color = color_for_messagebox_title,
                                    font       = bold24,
                                    text       = self._title)
        
        self._titleLabel.grid(row=0, column=0, columnspan=2, padx=30, pady=20, sticky="ew")

        self._label = CTkLabel(master = self,
                                width      = 550,
                                wraplength = 550,
                                corner_radius = 10,
                                anchor     = 'w',
                                justify    = "left",
                                text_color = "#C0C0C0",
                                bg_color   = "transparent",
                                fg_color   = "#303030",
                                font       = bold12,
                                text       = self._text)
        
        self._label.grid(row=1, column=0, columnspan=2, padx=30, pady=5, sticky="ew")

        self._ok_button = CTkButton(master  = self,
                                    command = self._ok_event,
                                    text    = 'OK',
                                    width   = 125,
                                    font         = bold11,
                                    border_width = 1,
                                    fg_color     = "#282828",
                                    text_color   = "#E0E0E0",
                                    border_color = "#0096FF")
        
        self._ok_button.grid(row=2, column=1, columnspan=1, padx=(10, 20), pady=(10, 20), sticky="e")

    def _ok_event(self, event = None):
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self.grab_release()
        self.destroy()

def create_info_button(command, text):
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

def create_option_menu(command, values):
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

def create_text_box(textvariable):
    return CTkEntry(master        = window, 
                    textvariable  = textvariable,
                    border_width  = 1,
                    width         = 150,
                    height        = 30,
                    font          = bold10,
                    justify       = "center",
                    fg_color      = "#000000",
                    border_color  = "#404040")



#  Slice functions -------------------

def split_image_into_tiles(image, 
                           num_tiles_x, 
                           num_tiles_y):

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
                             image_target_height, 
                             image_target_width,
                             num_tiles_x, 
                             num_tiles_y):

    tiled_image = numpy_zeros((image_target_height, image_target_width, 4), dtype = uint8)

    for tile_index in range(len(tiles)):
        actual_tile = tiles[tile_index]

        tile_height, tile_width, _ = actual_tile.shape
        row     = tile_index // num_tiles_x
        col     = tile_index % num_tiles_x
        y_start = row * tile_height
        y_end   = y_start + tile_height
        x_start = col * tile_width
        x_end   = x_start + tile_width

        tiled_image[y_start:y_end, x_start:x_end] = add_alpha_channel(actual_tile)

    return tiled_image

def image_need_tiles(image, tiles_resolution):
    height, width, _ = image.shape

    num_tiles_horizontal = (width + tiles_resolution - 1) // tiles_resolution
    num_tiles_vertical   = (height + tiles_resolution - 1) // tiles_resolution

    total_tiles = num_tiles_horizontal * num_tiles_vertical

    if total_tiles <= 1:
        return False, 1, 1
    else:
        return True, num_tiles_horizontal, num_tiles_vertical
    
def video_need_tiles(height, width, tiles_resolution):
    num_tiles_horizontal = (width + tiles_resolution - 1) // tiles_resolution
    num_tiles_vertical   = (height + tiles_resolution - 1) // tiles_resolution

    total_tiles = num_tiles_horizontal * num_tiles_vertical

    if total_tiles <= 1:
        return False, 1, 1
    else:
        return True, num_tiles_horizontal, num_tiles_vertical

def add_alpha_channel(tile):
    if tile.shape[2] == 3:  # Check if the tile does not have an alpha channel
        alpha_channel = numpy_full((tile.shape[0], tile.shape[1], 1), 255, dtype = uint8)
        tile          = numpy_concatenate((tile, alpha_channel), axis = 2)

    return tile

def fix_tile_shape(tile, tile_upscaled, upscaling_factor):
    tile_height, tile_width, _ = tile.shape
    target_tile_height = tile_height * upscaling_factor
    target_tile_width  = tile_width  * upscaling_factor

    tile_upscaled = opencv_resize(tile_upscaled, 
                                  (target_tile_width, target_tile_height),
                                  interpolation = INTER_LINEAR)

    return tile_upscaled



# File Utils functions ------------------------

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)

def remove_file(file_name): 
    if os_path_exists(file_name): 
        os_remove(file_name)

def remove_dir(name_dir):
    if os_path_exists(name_dir): 
        rmtree(name_dir)

def create_temp_dir(name_dir):
    if os_path_exists(name_dir): 
        rmtree(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def write_in_log_file(text_to_insert):
    with open(log_file_path,'w') as log_file: 
        os_chmod(log_file_path, 0o777)
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    with open(log_file_path,'r') as log_file: 
        os_chmod(log_file_path, 0o777)
        step = log_file.readline()
    log_file.close()
    return step

def remove_temp_files():
    remove_dir(temp_dir)
    remove_file(log_file_path)

def stop_thread(): 
    stop = 1 + "x"



# Image/video Utils functions ------------------------

def image_write(file_path, file_data): 
    _, file_extension = os_path_splitext(file_path)
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def image_read(file_path, flags = IMREAD_UNCHANGED): 
    with open(file_path, 'rb') as file:
        image_data = file.read()
        image = opencv_imdecode(numpy_frombuffer(image_data, uint8), flags)
        return image

def resize_image(image, resize_factor, resize_algorithm):
    old_height, old_width, _ = image.shape
    new_width  = int(old_width * resize_factor)
    new_height = int(old_height * resize_factor)

    resized_image = opencv_resize(image, (new_width, new_height), interpolation = resize_algorithm)
    return resized_image       

def check_if_file_need_resize(file, resize_factor):
    if resize_factor > 1:
        file_to_upscale = resize_image(file, resize_factor, INTER_LINEAR)
    elif resize_factor < 1:
        file_to_upscale = resize_image(file, resize_factor, INTER_AREA)
    else:
        file_to_upscale = file

    return file_to_upscale

def extract_video_frames_and_audio(video_path, file_number):
    video_frames_list = []
    cap          = opencv_VideoCapture(video_path)
    frame_rate   = cap.get(CAP_PROP_FPS)
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
    cap          = opencv_VideoCapture(input_video_path)
    frame_rate   = cap.get(CAP_PROP_FPS)
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
    if os_path_exists(audio_path) and selected_video_extension != '.webm':
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
        
    return upscaled_video_path

def interpolate_images(starting_image, 
                       upscaled_image, 
                       image_target_height, 
                       image_target_width):
    
    # Check the best rescale algorithm
    # INTER_AREA for downscale
    # INTER_LINEAR for upscale
    _, starting_image_width, _ = starting_image.shape
    if image_target_width > starting_image_width:
        resize_algorithm = INTER_LINEAR
    else: 
        resize_algorithm = INTER_AREA
    
    starting_image = opencv_resize(starting_image, (image_target_width, image_target_height), interpolation = resize_algorithm)
    
    starting_image     = add_alpha_channel(starting_image)
    upscaled_image     = add_alpha_channel(upscaled_image)
    interpolated_image = opencv_addWeighted(upscaled_image, 0.5, starting_image, 0.5, 0)

    return interpolated_image

def get_final_image_shape(image_to_upscale, upscaling_factor):
    image_to_upscale_height, image_to_upscale_width, _ = image_to_upscale.shape
    target_height = image_to_upscale_height * upscaling_factor
    target_width  = image_to_upscale_width  * upscaling_factor
    
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

    if int(hours_left)   > 0: time_left = f"{int(hours_left):02d}h"
    
    if int(minutes_left) > 0: time_left = f"{time_left}{int(minutes_left):02d}m"

    if seconds_left      > 0: time_left = f"{time_left}{seconds_left:02d}s"

    return time_left        

def copy_file_metadata(original_file_path, upscaled_file_path):
    
    exiftool_path  = find_by_relative_path(f"Assets{os_separator}exiftool_12.68.exe")

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

def check_upscale_steps():
    sleep(2)
    try:
        while True:
            actual_step = read_log_file()

            info_message.set(actual_step)

            if "All files completed! :)" in actual_step or "Upscaling stopped" in actual_step:
                stop_upscale_process()
                remove_temp_files()
                stop_thread()
            elif "Error during upscale process" in actual_step:
                info_message.set('Error during upscale process :(')
                show_error_message(actual_step.replace("Error during upscale process", ""))
                remove_temp_files()
                stop_thread()

            sleep(2)
    except:
        place_upscale_button()
        
def update_process_status(actual_process_phase):
    print(f"{actual_process_phase}")
    write_in_log_file(actual_process_phase) 

def stop_upscale_process():
    global process_upscale_orchestrator
    try:
        process_upscale_orchestrator
    except:
        pass
    else:
        process_upscale_orchestrator.terminate()
        process_upscale_orchestrator.join()

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

        thread_wait = Thread(target = check_upscale_steps, daemon = True)
        thread_wait.start()

def prepare_output_image_filename(image_path, 
                                  selected_AI_model, 
                                  resize_factor, 
                                  selected_image_extension, 
                                  selected_interpolation):
    
    result_path, _    = os_path_splitext(image_path)
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
    
    result_path, _    = os_path_splitext(video_path)
    resize_percentage = str(int(resize_factor * 100)) + "%"

    if selected_interpolation:
        to_append = f"_{selected_AI_model}_{resize_percentage}_interpolated{selected_video_extension}"
    else:
        to_append = f"_{selected_AI_model}_{resize_percentage}{selected_video_extension}"

    result_path += to_append

    return result_path

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
    
    torch_set_num_threads(2)

    upscaling_factor = 2 if 'x2' in selected_AI_model else 4

    try:
        AI_model = load_AI_model(selected_AI_model, upscaling_factor, backend, half_precision)

        how_many_files = len(selected_file_list)

        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(file_path, 
                            file_number,
                            AI_model, 
                            selected_AI_model, 
                            upscaling_factor,
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
                            upscaling_factor,
                            backend, 
                            selected_image_extension, 
                            tiles_resolution, 
                            resize_factor, 
                            half_precision, 
                            selected_interpolation)

        update_process_status(f"All files completed! :)")

    except Exception as exception:
        update_process_status(f"Error during upscale process {str(exception)}")

# Images

def upscale_image(image_path, 
                  file_number,
                  AI_model, 
                  selected_AI_model, 
                  upscaling_factor,
                  backend, 
                  selected_image_extension, 
                  tiles_resolution, 
                  resize_factor, 
                  half_precision,
                  selected_interpolation):
    
    result_image_path = prepare_output_image_filename(image_path, 
                                                      selected_AI_model, 
                                                      resize_factor, 
                                                      selected_image_extension,
                                                      selected_interpolation)
        
    starting_image   = image_read(image_path)
    image_to_upscale = check_if_file_need_resize(starting_image, resize_factor)

    image_target_height, image_target_width = get_final_image_shape(image_to_upscale, upscaling_factor)
    need_tiles, num_tiles_x, num_tiles_y    = image_need_tiles(image_to_upscale, tiles_resolution)

    update_process_status(f"{file_number}. Upscaling image")

    # Upscale image w/out tilling
    if need_tiles == False:
        image_upscaled = AI_enhance(AI_model, image_to_upscale, backend, half_precision)

    # Upscale image with tilling
    else:
        tiles_list = split_image_into_tiles(image_to_upscale, num_tiles_x, num_tiles_y)

        how_many_tiles = len(tiles_list)

        for tile_index in range(how_many_tiles):
            tile = tiles_list[tile_index]

            tile_upscaled = AI_enhance(AI_model, tile, backend, half_precision)
            tile_upscaled = fix_tile_shape(tile, tile_upscaled, upscaling_factor)
            tiles_list[tile_index] = tile_upscaled

        image_upscaled = combine_tiles_into_image(tiles_list, image_target_height, image_target_width, num_tiles_x, num_tiles_y)
    
    # Interpolation
    if selected_interpolation:
        image_upscaled = interpolate_images(starting_image, image_upscaled, image_target_height, image_target_width)
    
    # Save image
    image_write(result_image_path, image_upscaled)

    # Metadata copy from original image
    copy_file_metadata(image_path, result_image_path)

# Videos

def get_video_info_for_upscaling(frame_list_paths, 
                                 resize_factor, 
                                 upscaling_factor, 
                                 tiles_resolution):
    
    first_frame = image_read(frame_list_paths[0])

    height, width, _ = first_frame.shape
 
    # Resizing shapes
    frame_resized_width  = int(width * resize_factor)
    frame_resized_height = int(height * resize_factor)

    # Tilling?
    need_tiles, num_tiles_x, num_tiles_y = video_need_tiles(frame_resized_height, frame_resized_width, tiles_resolution)

    # Upscaling shapes
    frame_target_width  = frame_resized_width * upscaling_factor
    frame_target_height = frame_resized_height * upscaling_factor
        
    return frame_target_height, frame_target_width, need_tiles, num_tiles_x, num_tiles_y

def upscale_video(video_path, 
                  file_number,
                  AI_model, 
                  selected_AI_model, 
                  upscaling_factor,
                  backend, 
                  selected_image_extension, 
                  tiles_resolution,
                  resize_factor, 
                  cpu_number, 
                  half_precision, 
                  selected_video_extension,
                  selected_interpolation):
    
    create_temp_dir(temp_dir)

    frames_upscaled_paths_list = [] 
    frame_list_paths = extract_video_frames_and_audio(video_path, file_number)
    how_many_frames  = len(frame_list_paths) 

    update_process_status(f"{file_number}. Upscaling video")  
    frame_target_height, frame_target_width, need_tiles, num_tiles_x, num_tiles_y = get_video_info_for_upscaling(frame_list_paths, resize_factor, upscaling_factor, tiles_resolution)

    for index_frame in range(how_many_frames):

        start_timer = timer()

        result_frame_path = prepare_output_image_filename(frame_list_paths[index_frame], 
                                                         selected_AI_model, 
                                                         resize_factor, 
                                                         selected_image_extension, 
                                                         selected_interpolation)

        starting_frame   = image_read(frame_list_paths[index_frame])
        frame_to_upscale = check_if_file_need_resize(starting_frame, resize_factor)

        # Upscale frame w/out tilling
        if need_tiles == False:
            frame_upscaled = AI_enhance(AI_model, frame_to_upscale, backend, half_precision)

        # Upscale frame with tilling
        else:
            tiles_list = split_image_into_tiles(frame_to_upscale, num_tiles_x, num_tiles_y)
            
            how_many_tiles = len(tiles_list)

            for tile_index in range(how_many_tiles):
                tile = tiles_list[tile_index]
                tile_upscaled          = AI_enhance(AI_model, tile, backend, half_precision)
                tile_upscaled          = fix_tile_shape(tile, tile_upscaled, upscaling_factor)
                tiles_list[tile_index] = tile_upscaled

            frame_upscaled = combine_tiles_into_image(tiles_list, frame_target_height, frame_target_width, num_tiles_x, num_tiles_y)

        # Interpolation
        if selected_interpolation:
            frame_upscaled = interpolate_images(starting_frame, frame_upscaled, frame_target_height, frame_target_width)

        # Save frame
        image_write(result_frame_path, frame_upscaled)
    
        frames_upscaled_paths_list.append(result_frame_path)

        # Update process status every 6 frames
        if index_frame != 0 and (index_frame + 1) % 6 == 0: 
            end_timer        = timer()    
            percent_complete = (index_frame + 1) / how_many_frames * 100 
            time_left        = calculate_time_to_complete_video(start_timer, end_timer, how_many_frames, index_frame)
        
            update_process_status(f"{file_number}. Upscaling video {percent_complete:.2f}% ({time_left})")

    upscaled_video_path = video_reconstruction_by_frames(video_path, 
                                                        file_number,
                                                        frames_upscaled_paths_list, 
                                                        selected_AI_model, 
                                                        resize_factor, 
                                                        cpu_number, 
                                                        selected_video_extension,
                                                        selected_interpolation)
    
    copy_file_metadata(video_path, upscaled_video_path)



# GUI utils function ---------------------------

def opengithub():   
    open_browser.open(githubme, new=1)

def opentelegram(): 
    open_browser.open(telegramme, new=1)

def check_if_file_is_video(file):
    return any(video_extension in file for video_extension in supported_video_extensions)

def check_supported_selected_files(uploaded_file_list):
    return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

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
        elif selected_AI_model in SRVGGNetCompact_models_list: vram_multiplier = SRVGGNetCompact_vram_multiplier
        elif selected_AI_model in SAFMN_models_list:           vram_multiplier = SAFMN_vram_multiplier
        elif selected_AI_model in DITN_models_list:            vram_multiplier = DITN_vram_multiplier

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

def extract_image_info(image_file):
    image_name = str(image_file.split("/")[-1])

    image  = image_read(image_file, IMREAD_UNCHANGED)
    width  = int(image.shape[1])
    height = int(image.shape[0])

    image_label = f"IMAGE • {image_name} • {width}x{height}"

    ctkimage = CTkImage(pillow_image_open(image_file), size = (25, 25))

    return image_label, ctkimage

def extract_video_info(video_file):
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

    video_label = f"VIDEO • {video_name} • {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames • {round(frame_rate, 2)}fps"

    ctkimage = CTkImage(pillow_image_open("temp.jpg"), size = (25, 25))
    
    return video_label, ctkimage

def show_error_message(exception):
    messageBox_title = "Upscale error"

    messageBox_text  = str(exception) + "\n\n" + "Please report the error on Github/Telegram"

    CTkMessageBox(text = messageBox_text, title = messageBox_title, type = "error")



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
    if new_value == 'Enabled':
        selected_interpolation = True
    elif new_value == 'Disabled':
        selected_interpolation = False



# GUI info functions ---------------------------

def open_info_AI_model():
    messageBox_title = "AI model"
    messageBox_text = """This widget allows to choose between different AIs.

SRVGGNetCompact (2022)
    • Fast and lightweight AI • Good-quality upscale
    • Recommended for video upscaling
        • RealESR_Gx4 (default)
        • RealSRx4_Anime    

RRDB (2020)
    • Complex and heavy AI architecture • High-quality upscale
    • Recommended for image upscaling
        • BSRGANx4 
        • BSRGANx2

SAFMN (2023)
    • Slow but lightweight AI • Highest-quality upscale
    • Recommended for image upscaling
    • Does not support Half-precision
        • SAFMNLx4 
        • SAFMNLx4_Real


AI model tagged with _x4 will upscale with factor 4
AI model tagged with _x2 will upscale with factor 2""" 

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_device():
    messageBox_title = "GPU"

    messageBox_text = """This widget allows you to select the GPU for AI processing.

 • Keep in mind that the more powerful your GPU is, faster the upscaling will be
 • For optimal results, it's essential to regularly update your GPU drivers"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_AI_output():
    messageBox_title = "AI output"

    messageBox_text = """This widget allows to choose the extension of upscaled image/frame.

 • jpg • good quality • very fast
 • png • very good quality • slow • supports transparent images
 • bmp • highest quality • slow
 • tiff • highest quality • very slow"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_input_resolution():
    messageBox_title = "Input resolution %"

    messageBox_text = """This widget allows to choose the resolution input to the AI.

For example for a 100x100px image/video:
 • Input resolution 50% => input to AI 50x50px
 • Input resolution 100% => input to AI 100x100px
 • Input resolution 200% => input to AI 200x200px """

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_vram_limiter():
    messageBox_title = "GPU VRAM (GB)"

    messageBox_text = """This widget allows to set a limit on the gpu's VRAM memory usage.

 • For a gpu with 4 GB of Vram you must select 4
 • For a gpu with 6 GB of Vram you must select 6
 • For a gpu with 8 GB of Vram you must select 8
 • For integrated gpus (Intel-HD series • Vega 3,5,7) you must select 2

Selecting a value greater than the actual amount of gpu VRAM may result in upscale failure """

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_cpu():
    messageBox_title = "Cpu number"

    messageBox_text = """This widget allows you to choose how many cpus to devote to the app.
    
Where possible the app will use the number of cpus selected."""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_AI_precision():
    messageBox_title = "AI precision"

    messageBox_text = """This widget allows you to choose the AI upscaling mode.

 • Full precision (>=8GB Vram recommended)
    • compatible with all GPUs 
    • uses 50% more GPU memory
    • is 30-70% faster
  
 • Half precision
    • some old GPUs are not compatible with this mode
    • uses 50% less GPU memory
    • is 30-70% slower"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_video_extension():
    messageBox_title = "Video output"

    messageBox_text = """This widget allows you to choose the video output.

 • .mp4  • produces good quality and well compressed video
 • .avi  • produces the highest quality video
 • .webm • produces low quality but light video"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_interpolation():
    messageBox_title = "Interpolation"

    messageBox_text = """This widget allows you to choose interpolating the upscaled image/frame with the original image/frame.

 • Is the fusion of the original image with the image produced by the AI
 • Increase the quality of the final result
     • especially when using the tilling/merging function
     • especially at low "Input resolution %" values (<50%)"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)



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
    
    up_background.place(relx = 0.0, 
                        rely = 0.0, 
                        relwidth  = 1.0,  
                        relheight = 0.45)

    text_drop = """ •  SUPPORTED FILES  •

IMAGES • jpg png tif bmp webp heic
VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp"""

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
                                width      = 140,
                                height     = 30,
                                font       = bold11,
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

    file_extension_button = create_info_button(open_info_AI_output, "AI output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column1_x, rely = row0_y - 0.053, anchor = "center")
    file_extension_menu.place(relx = column1_x, rely = row0_y, anchor = "center")

def place_video_extension_menu():

    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    video_extension_menu.place(relx = column1_x, rely = row1_y, anchor = "center")

def place_gpu_menu():

    gpu_button = create_info_button(open_info_device, "GPU")
    gpu_menu   = create_option_menu(select_AI_device_from_menu, device_list_names)
    
    gpu_button.place(relx = 0.5, rely = row2_y - 0.053, anchor = "center")
    gpu_menu.place(relx = 0.5, rely  = row2_y, anchor = "center")

def place_vram_textbox():

    vram_button  = create_info_button(open_info_vram_limiter,  "GPU VRAM (GB)")
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
   


# Main functions ---------------------------

def on_app_close():
    window.grab_release()
    window.destroy()
    stop_upscale_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None

        window.title('')
        width        = 675
        height       = 675
        window.geometry("675x675")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        window.protocol("WM_DELETE_WINDOW", on_app_close)

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
    freeze_support()

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

    if interpolation_list[0]   == "Disabled": selected_interpolation = False
    elif interpolation_list[0] == "Enabled": selected_interpolation = True

    selected_AI_device       = 0
    selected_AI_model        = AI_models_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    info_message            = StringVar()
    selected_resize_factor  = StringVar()
    selected_VRAM_limiter   = StringVar()
    selected_cpu_number     = StringVar()

    info_message.set("Hi :)")
    selected_resize_factor.set("50")
    selected_VRAM_limiter.set("8")
    selected_cpu_number.set(str(int(os_cpu_count()/2)))

    font   = "Segoe UI"    
    bold8  = CTkFont(family = font, size = 8, weight = "bold")
    bold9  = CTkFont(family = font, size = 9, weight = "bold")
    bold10 = CTkFont(family = font, size = 10, weight = "bold")
    bold11 = CTkFont(family = font, size = 11, weight = "bold")
    bold12 = CTkFont(family = font, size = 12, weight = "bold")
    bold18 = CTkFont(family = font, size = 18, weight = "bold")
    bold19 = CTkFont(family = font, size = 19, weight = "bold")
    bold20 = CTkFont(family = font, size = 20, weight = "bold")
    bold21 = CTkFont(family = font, size = 21, weight = "bold")
    bold24 = CTkFont(family = font, size = 24, weight = "bold")

    # Images
    logo_git       = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(15, 15))
    logo_telegram  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(15, 15))
    stop_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon   = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(14, 14))

    app = App(window)
    window.update()
    window.mainloop()