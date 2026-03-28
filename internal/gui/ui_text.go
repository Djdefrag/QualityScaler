package gui

import "fmt"

type Language string

type TextKey string

type HelpTopic string

type HelpContent struct {
	Title    string
	Subtitle string
	Items    []string
}

const (
	LangZH Language = "zh-CN"
	LangEN Language = "en-US"
)

const (
	TextInputFiles         TextKey = "input_files"
	TextAddFiles           TextKey = "add_files"
	TextClearList          TextKey = "clear_list"
	TextInfo               TextKey = "info"
	TextSystemInfo         TextKey = "system_info"
	TextAppVersion         TextKey = "app_version"
	TextConfig             TextKey = "config"
	TextAIBackendSettings  TextKey = "ai_backend_settings"
	TextOutputSettings     TextKey = "output_settings"
	TextAIModel            TextKey = "ai_model"
	TextGPU                TextKey = "gpu"
	TextPerformanceMode    TextKey = "performance_mode"
	TextMultithreading     TextKey = "multithreading"
	TextVRAM               TextKey = "vram"
	TextBlending           TextKey = "blending"
	TextInputScale         TextKey = "input_scale"
	TextOutputScale        TextKey = "output_scale"
	TextResolutionPreview  TextKey = "resolution_preview"
	TextSourceResolution   TextKey = "source_resolution"
	TextInputResolution    TextKey = "input_resolution"
	TextAIModelScaleDisplay TextKey = "ai_model_scale_display"
	TextOutputResolution   TextKey = "output_resolution"
	TextUnknownResolution  TextKey = "unknown_resolution"
	TextImageFormat        TextKey = "image_format"
	TextVideoFormat        TextKey = "video_format"
	TextVideoCodec         TextKey = "video_codec"
	TextKeepFrames         TextKey = "keep_frames"
	TextOutputFolder       TextKey = "output_folder"
	TextSelect             TextKey = "select"
	TextLanguage           TextKey = "language"
	TextTheme              TextKey = "theme"
	TextUIFontSize         TextKey = "ui_font_size"
	TextAppearanceApplied  TextKey = "appearance_applied"
	TextStatusAndProgress  TextKey = "status_and_progress"
	TextStageETA           TextKey = "stage_eta"
	TextFileETA            TextKey = "file_eta"
	TextTotalETA           TextKey = "total_eta"
	TextReady              TextKey = "ready"
	TextStartUpscale       TextKey = "start_upscale"
	TextStop               TextKey = "stop"
	TextSelectFilesTitle   TextKey = "select_files_title"
	TextMediaFilesFilter   TextKey = "media_files_filter"
	TextSelectOutputFolder TextKey = "select_output_folder"
	TextLoadedFiles        TextKey = "loaded_files"
	TextFileListCleared    TextKey = "file_list_cleared"
	TextSelectOneFile      TextKey = "select_one_file"
	TextProcessing         TextKey = "processing"
	TextCalculating        TextKey = "calculating"
	TextStopped            TextKey = "stopped"
	TextErrorPrefix        TextKey = "error_prefix"
	TextTaskErrorTitle     TextKey = "task_error_title"
	TextCompleted          TextKey = "completed"
	TextInvalidInputScale  TextKey = "invalid_input_scale"
	TextInvalidOutputScale TextKey = "invalid_output_scale"
	TextInvalidVRAM        TextKey = "invalid_vram"
	TextLangSwitchHint     TextKey = "lang_switch_hint"
)

const (
	HelpAIModel     HelpTopic = "ai_model"
	HelpGPU         HelpTopic = "gpu"
	HelpThreading   HelpTopic = "threading"
	HelpVRAM        HelpTopic = "vram"
	HelpBlending    HelpTopic = "blending"
	HelpInputScale  HelpTopic = "input_scale"
	HelpOutputScale HelpTopic = "output_scale"
	HelpImageFormat HelpTopic = "image_format"
	HelpVideoFormat HelpTopic = "video_format"
	HelpVideoCodec  HelpTopic = "video_codec"
	HelpKeepFrames  HelpTopic = "keep_frames"
	HelpOutputPath  HelpTopic = "output_path"
)

var textMap = map[Language]map[TextKey]string{
	LangZH: {
		TextInputFiles:         "输入文件",
		TextAddFiles:           "添加文件",
		TextClearList:          "清空列表",
		TextInfo:               "信息",
		TextSystemInfo:         "系统信息",
		TextAppVersion:         "程序版本:",
		TextConfig:             "参数配置",
		TextAIBackendSettings:  "AI 与后端",
		TextOutputSettings:     "输出设置",
		TextAIModel:            "AI 模型:",
		TextGPU:                "GPU:",
		TextPerformanceMode:    "性能模式:",
		TextMultithreading:     "并行线程(每视频):",
		TextVRAM:               "显存(GB):",
		TextBlending:           "融合强度:",
		TextInputScale:         "输入缩放 %:",
		TextOutputScale:        "输出缩放 %:",
		TextResolutionPreview:  "分辨率预览",
		TextSourceResolution:   "源分辨率: %s",
		TextInputResolution:    "输入分辨率(应用输入缩放): %s",
		TextAIModelScaleDisplay:"模型倍率(%s): ×%d",
		TextOutputResolution:   "输出分辨率(模型+输出缩放): %s",
		TextUnknownResolution:  "未选择文件",
		TextImageFormat:        "图片格式:",
		TextVideoFormat:        "视频格式:",
		TextVideoCodec:         "视频编码:",
		TextKeepFrames:         "保留帧:",
		TextOutputFolder:       "输出目录:",
		TextSelect:             "选择...",
		TextLanguage:           "语言:",
		TextTheme:              "主题:",
		TextUIFontSize:         "字号:",
		TextAppearanceApplied:  "界面外观已应用。",
		TextStatusAndProgress:  "状态与进度",
		TextStageETA:           "当前状态剩余: %s",
		TextFileETA:            "本文件剩余: %s",
		TextTotalETA:           "总剩余: %s",
		TextReady:              "就绪",
		TextStartUpscale:       "开始超分",
		TextStop:               "停止",
		TextSelectFilesTitle:   "选择文件",
		TextMediaFilesFilter:   "媒体文件|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff;*.mp4;*.mkv;*.avi;*.mov;*.webm;*.flv;*.gif",
		TextSelectOutputFolder: "选择输出目录",
		TextLoadedFiles:        "已加载 %d 个文件",
		TextFileListCleared:    "文件列表已清空",
		TextSelectOneFile:      "请至少选择一个文件",
		TextProcessing:         "超分处理中... | AI后端: %s | 模式: %s",
		TextCalculating:        "计算中...",
		TextStopped:            "已停止",
		TextErrorPrefix:        "错误: %s",
		TextTaskErrorTitle:     "超分任务错误",
		TextCompleted:          "处理完成",
		TextInvalidInputScale:  "输入缩放 %% 必须大于 0",
		TextInvalidOutputScale: "输出缩放 %% 必须大于 0",
		TextInvalidVRAM:        "GPU 显存(GB)必须大于 0",
		TextLangSwitchHint:     "语言已切换；界面文案已更新。",
	},
	LangEN: {
		TextInputFiles:         "Input Files",
		TextAddFiles:           "Add Files",
		TextClearList:          "Clear List",
		TextInfo:               "Information",
		TextSystemInfo:         "System Info",
		TextAppVersion:         "App Version:",
		TextConfig:             "Configuration",
		TextAIBackendSettings:  "AI & Backend",
		TextOutputSettings:     "Output Settings",
		TextAIModel:            "AI Model:",
		TextGPU:                "GPU:",
		TextPerformanceMode:    "Performance Mode:",
		TextMultithreading:     "Multithreading (per video):",
		TextVRAM:               "VRAM (GB):",
		TextBlending:           "Blending:",
		TextInputScale:         "Input Scale %:",
		TextOutputScale:        "Output Scale %:",
		TextResolutionPreview:  "Resolution Preview",
		TextSourceResolution:   "Source Resolution: %s",
		TextInputResolution:    "Input Resolution (after input scale): %s",
		TextAIModelScaleDisplay:"Model Scale (%s): ×%d",
		TextOutputResolution:   "Output Resolution (model + output scale): %s",
		TextUnknownResolution:  "No file selected",
		TextImageFormat:        "Image Format:",
		TextVideoFormat:        "Video Format:",
		TextVideoCodec:         "Video Codec:",
		TextKeepFrames:         "Keep Frames:",
		TextOutputFolder:       "Output Folder:",
		TextSelect:             "Select...",
		TextLanguage:           "Language:",
		TextTheme:              "Theme:",
		TextUIFontSize:         "Font Size:",
		TextAppearanceApplied:  "Appearance settings applied.",
		TextStatusAndProgress:  "Status & Progress",
		TextStageETA:           "Current Stage ETA: %s",
		TextFileETA:            "Current File ETA: %s",
		TextTotalETA:           "Total ETA: %s",
		TextReady:              "Ready",
		TextStartUpscale:       "UPSCALE",
		TextStop:               "STOP",
		TextSelectFilesTitle:   "Select files",
		TextMediaFilesFilter:   "Media files|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff;*.mp4;*.mkv;*.avi;*.mov;*.webm;*.flv;*.gif",
		TextSelectOutputFolder: "Select output folder",
		TextLoadedFiles:        "Loaded %d file(s)",
		TextFileListCleared:    "File list cleared",
		TextSelectOneFile:      "Please select at least one file",
		TextProcessing:         "Upscaling... | AI backend: %s | Mode: %s",
		TextCalculating:        "Calculating...",
		TextStopped:            "Stopped",
		TextErrorPrefix:        "Error: %s",
		TextTaskErrorTitle:     "Upscale error",
		TextCompleted:          "Completed",
		TextInvalidInputScale:  "input scale %% must be > 0",
		TextInvalidOutputScale: "output scale %% must be > 0",
		TextInvalidVRAM:        "GPU VRAM (GB) must be > 0",
		TextLangSwitchHint:     "Language switched. UI text has been refreshed.",
	},
}

var languageMenu = map[Language][]string{
	LangZH: {"中文", "English"},
	LangEN: {"Chinese", "English"},
}

var themeMenu = map[Language][]string{
	LangZH: {"高对比蓝", "柔和灰"},
	LangEN: {"High Contrast Blue", "Soft Gray"},
}

var fontScaleMenu = map[Language][]string{
	LangZH: {"紧凑", "标准", "大号"},
	LangEN: {"Compact", "Standard", "Large"},
}

func normalizeLanguage(v string) Language {
	switch v {
	case string(LangEN):
		return LangEN
	default:
		return LangZH
	}
}

func languageByIndex(index int) Language {
	if index == 1 {
		return LangEN
	}
	return LangZH
}

func languageIndex(lang Language) int {
	if lang == LangEN {
		return 1
	}
	return 0
}

func languageOptions(base Language) []string {
	if options, ok := languageMenu[base]; ok {
		return options
	}
	return languageMenu[LangZH]
}

func themeOptions(base Language) []string {
	if options, ok := themeMenu[base]; ok {
		return options
	}
	return themeMenu[LangZH]
}

func fontScaleOptions(base Language) []string {
	if options, ok := fontScaleMenu[base]; ok {
		return options
	}
	return fontScaleMenu[LangZH]
}

func tr(lang Language, key TextKey, args ...interface{}) string {
	lang = normalizeLanguage(string(lang))
	if v, ok := textMap[lang][key]; ok {
		if len(args) > 0 {
			return fmt.Sprintf(v, args...)
		}
		return v
	}
	if v, ok := textMap[LangEN][key]; ok {
		if len(args) > 0 {
			return fmt.Sprintf(v, args...)
		}
		return v
	}
	return string(key)
}

func helpContent(lang Language, topic HelpTopic) HelpContent {
	lang = normalizeLanguage(string(lang))
	help := map[Language]map[HelpTopic]HelpContent{
		LangZH: {
			HelpAIModel: {
				Title:    "AI 模型",
				Subtitle: "选择不同模型会影响清晰度、速度与显存占用。",
				Items: []string{
					"RealESR 系列适合通用放大，动画可优先 RealESR_Animex4。",
					"BSRGAN 偏向降噪与细节增强。",
					"IRCNN 更适合去噪修复场景。",
				},
			},
			HelpGPU: {
				Title:    "GPU",
				Subtitle: "选择用于 AI 推理的显卡。",
				Items: []string{
					"Auto 会自动选择高性能 GPU。",
					"GPU 1 对应任务管理器中的 GPU 0。",
					"若选择不存在的 GPU，可能回退到 CPU。",
				},
			},
			HelpThreading: {
				Title:    "并行线程",
				Subtitle: "仅对视频逐帧处理生效。",
				Items: []string{
					"线程越多速度可能更快，但占用更多显存和内存。",
					"显存不足时建议降低线程数或切换 Balanced。",
				},
			},
			HelpVRAM: {
				Title:    "GPU 显存(GB)",
				Subtitle: "限制推理可使用的显存预算。",
				Items: []string{
					"建议设置为显卡实际可用显存。",
					"集成显卡建议从 2GB 开始尝试。",
				},
			},
			HelpBlending: {
				Title:    "融合强度",
				Subtitle: "将原图与超分结果按比例融合。",
				Items: []string{
					"OFF = 不融合，保留 AI 输出。",
					"Low/Medium/High 会逐步提升原图权重。",
				},
			},
			HelpInputScale: {
				Title:    "输入缩放 %",
				Subtitle: "先缩放输入再送入 AI。",
				Items: []string{
					"高数值质量更好，但速度更慢。",
					"低数值更快，适合预览或低性能设备。",
				},
			},
			HelpOutputScale: {
				Title:    "输出缩放 %",
				Subtitle: "对 AI 输出再次缩放。",
				Items: []string{
					"100% 保持 AI 原生结果。",
					"小于 100% 可减小体积与处理耗时。",
					"大于 100% 仅插值放大，不会增加真实细节。",
				},
			},
			HelpImageFormat: {
				Title:    "图片格式",
				Subtitle: "选择图片输出扩展名。",
				Items: []string{
					"PNG 质量高且无损，体积较大。",
					"JPG 体积小，存在有损压缩。",
					"BMP/TIFF 适合高质量归档。",
				},
			},
			HelpVideoFormat: {
				Title:    "视频格式",
				Subtitle: "选择视频封装格式。",
				Items: []string{
					"MP4 兼容性最好。",
					"MKV 适合高质量与多音轨。",
					"AVI/MOV 适合特定工具链。",
				},
			},
			HelpVideoCodec: {
				Title:    "视频编码",
				Subtitle: "选择输出视频编码器。",
				Items: []string{
					"x264/x265 为 CPU 软件编码。",
					"*_nvenc/*_amf/*_qsv 为硬件编码。",
					"硬件编码更快，但质量和兼容性因设备而异。",
				},
			},
			HelpKeepFrames: {
				Title:    "保留帧",
				Subtitle: "控制是否保留视频中间帧文件。",
				Items: []string{
					"ON：保留中间帧，便于排错或复用。",
					"OFF：任务完成后自动清理临时帧。",
				},
			},
			HelpOutputPath: {
				Title:    "输出目录",
				Subtitle: "选择处理结果的保存位置。",
				Items: []string{
					"默认与输入文件同目录。",
					"可点击“选择...”指定统一输出目录。",
				},
			},
		},
		LangEN: {
			HelpAIModel: {
				Title:    "AI Model",
				Subtitle: "Different models balance quality, speed, and VRAM usage.",
				Items: []string{
					"RealESR models are good for general upscaling; RealESR_Animex4 for anime.",
					"BSRGAN focuses on denoise and detail enhancement.",
					"IRCNN is suitable for restoration/denoise cases.",
				},
			},
			HelpGPU: {
				Title:    "GPU",
				Subtitle: "Select the GPU used for AI inference.",
				Items: []string{
					"Auto chooses a high-performance GPU.",
					"GPU 1 maps to GPU 0 in Task Manager.",
					"Selecting a non-existent GPU may fall back to CPU.",
				},
			},
			HelpThreading: {
				Title:    "Multithreading",
				Subtitle: "Applies to per-frame video processing.",
				Items: []string{
					"More threads can be faster but use more VRAM and RAM.",
					"Lower threads if you hit VRAM limits.",
				},
			},
			HelpVRAM: {
				Title:    "GPU VRAM (GB)",
				Subtitle: "Limit the inference VRAM budget.",
				Items: []string{
					"Use a value close to available VRAM.",
					"For integrated GPUs, start from 2 GB.",
				},
			},
			HelpBlending: {
				Title:    "Blending",
				Subtitle: "Blend original content with AI output.",
				Items: []string{
					"OFF keeps pure AI output.",
					"Low/Medium/High increase original-image weight.",
				},
			},
			HelpInputScale: {
				Title:    "Input Scale %",
				Subtitle: "Resize input before AI inference.",
				Items: []string{
					"Higher values improve quality but reduce speed.",
					"Lower values are faster and useful for preview.",
				},
			},
			HelpOutputScale: {
				Title:    "Output Scale %",
				Subtitle: "Resize the AI output again.",
				Items: []string{
					"100% keeps native AI output size.",
					"<100% reduces output size and processing costs.",
					">100% is interpolation only and adds no real details.",
				},
			},
			HelpImageFormat: {
				Title:    "Image Format",
				Subtitle: "Choose output image extension.",
				Items: []string{
					"PNG: high quality, lossless, larger files.",
					"JPG: smaller files, lossy compression.",
					"BMP/TIFF: useful for archival workflows.",
				},
			},
			HelpVideoFormat: {
				Title:    "Video Format",
				Subtitle: "Choose output container format.",
				Items: []string{
					"MP4 has the widest compatibility.",
					"MKV is good for quality and multiple tracks.",
					"AVI/MOV fit specific legacy workflows.",
				},
			},
			HelpVideoCodec: {
				Title:    "Video Codec",
				Subtitle: "Choose output encoder.",
				Items: []string{
					"x264/x265 are CPU software codecs.",
					"*_nvenc/*_amf/*_qsv are hardware codecs.",
					"Hardware encoding is faster; quality varies by device.",
				},
			},
			HelpKeepFrames: {
				Title:    "Keep Frames",
				Subtitle: "Keep or delete temporary extracted frames.",
				Items: []string{
					"ON keeps intermediate frames for debugging/reuse.",
					"OFF removes temporary frames after success.",
				},
			},
			HelpOutputPath: {
				Title:    "Output Folder",
				Subtitle: "Choose where processed files are saved.",
				Items: []string{
					"Default is the input file directory.",
					"Use Select... to pick a fixed output folder.",
				},
			},
		},
	}

	if byLang, ok := help[lang]; ok {
		if content, ok := byLang[topic]; ok {
			return content
		}
	}
	return HelpContent{Title: "Info", Subtitle: "No help available for this item."}
}
