package app

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
)

const (
	AppName        = "QualityScaler-Go"
	AppVersion     = "2026.1-mvp"
	OutputPathCode = "Same path as input files"
)

// BuildEdition represents the build edition of the application
type BuildEdition string

const (
	EditionUnknown  BuildEdition = ""
	EditionOnnxCPU  BuildEdition = "onnx-cpu"
	EditionOnnxCUDA BuildEdition = "onnx-cuda"
	EditionTensorRT BuildEdition = "tensorrt-gpu"
	EditionFull     BuildEdition = "full"
)

var (
	AIModels = []string{
		"LVAx2",
		"RealESR_Gx4",
		"RealESR_Animex4",
		"BSRGANx2",
		"BSRGANx4",
		"MSharpx4",
		"IRCNN_Mx1",
		"IRCNN_Lx1",
	}
	AIThreads        = []string{"OFF", "2 threads", "4 threads", "6 threads", "8 threads"}
	PerformanceModes = []string{"Balanced", "Extreme Performance", "Quality"}
	Blending         = []string{"OFF", "Low", "Medium", "High"}
	GPUs             = []string{"Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4"}
	KeepFrames       = []string{"OFF", "ON"}
	ImageExts        = []string{".jpg", ".png", ".bmp", ".tiff"}
	VideoExts        = []string{".mp4", ".mkv", ".avi", ".mov"}
	VideoCodecs      = []string{
		"x264", "x265", "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv",
	}
	SupportedExts = map[string]struct{}{
		".heic": {}, ".jpg": {}, ".jpeg": {}, ".png": {}, ".webp": {}, ".bmp": {}, ".tif": {}, ".tiff": {},
		".mp4": {}, ".webm": {}, ".mkv": {}, ".flv": {}, ".gif": {}, ".m4v": {}, ".avi": {}, ".mov": {},
		".qt": {}, ".3gp": {}, ".mpg": {}, ".mpeg": {}, ".vob": {},
	}
	SupportedVideoExts = map[string]struct{}{
		".mp4": {}, ".webm": {}, ".mkv": {}, ".flv": {}, ".gif": {}, ".m4v": {}, ".avi": {}, ".mov": {},
		".qt": {}, ".3gp": {}, ".mpg": {}, ".mpeg": {}, ".vob": {},
	}
)

type Preferences struct {
	UILanguage        string `json:"ui_language"`
	UITheme           string `json:"ui_theme"`
	UIFontScale       string `json:"ui_font_scale"`
	AIModel           string `json:"ai_model"`
	AIThreading       string `json:"ai_threading"`
	PerformanceMode   string `json:"performance_mode"`
	GPU               string `json:"gpu"`
	KeepFrames        string `json:"keep_frames"`
	ImageExtension    string `json:"image_extension"`
	VideoExtension    string `json:"video_extension"`
	VideoCodec        string `json:"video_codec"`
	Blending          string `json:"blending"`
	OutputPath        string `json:"output_path"`
	InputScalePercent string `json:"input_scale_percent"`
	OutScalePercent   string `json:"output_scale_percent"`
	VRAMGB            string `json:"vram_gb"`
}

// DetectBuildEdition detects the current build edition from executable name
func DetectBuildEdition() BuildEdition {
	exe, err := os.Executable()
	if err != nil {
		return EditionUnknown
	}
	exeName := strings.ToLower(filepath.Base(exe))

	// Check for edition markers in filename
	for _, edition := range []BuildEdition{EditionOnnxCPU, EditionOnnxCUDA, EditionTensorRT, EditionFull} {
		if strings.Contains(exeName, string(edition)) {
			return edition
		}
	}

	// Check for qualityscaler_tensorrt.dll presence (tensorrt-gpu/full editions)
	if _, err := os.Stat("qualityscaler_tensorrt.dll"); err == nil {
		// Has TensorRT, but also check for AI-onnx (full edition)
		if _, err := os.Stat("AI-onnx"); err == nil {
			return EditionFull
		}
		return EditionTensorRT
	}

	// Check for CUDA provider DLL (onnx-cuda/tensorrt-gpu/full editions)
	if _, err := os.Stat("onnxruntime_providers_cuda.dll"); err == nil {
		return EditionOnnxCUDA
	}

	// Default: onnx-cpu
	return EditionOnnxCPU
}

// DefaultPreferences returns default preferences based on detected build edition
func DefaultPreferences() Preferences {
	edition := DetectBuildEdition()

	switch edition {
	case EditionTensorRT, EditionFull:
		// TensorRT/GPU editions: optimize for performance
		return Preferences{
			UILanguage:        "zh-CN",
			UITheme:           "contrast",
			UIFontScale:       "standard",
			AIModel:           AIModels[0],         // LVAx2
			AIThreading:       AIThreads[0],        // OFF (default, user can enable)
			PerformanceMode:   PerformanceModes[1], // Extreme Performance
			GPU:               GPUs[0],
			KeepFrames:        KeepFrames[0],
			ImageExtension:    ImageExts[0],
			VideoExtension:    VideoExts[1],
			VideoCodec:        VideoCodecs[2], // h264_nvenc (GPU accelerated)
			Blending:          Blending[0],    // OFF (TensorRT doesn't need blending)
			OutputPath:        "",              // Empty by default to avoid path leakage
			InputScalePercent: "25",
			OutScalePercent:   "100",
			VRAMGB:            "12",
		}
	case EditionOnnxCUDA:
		// ONNX+CUDA edition: balanced performance
		return Preferences{
			UILanguage:        "zh-CN",
			UITheme:           "contrast",
			UIFontScale:       "standard",
			AIModel:           AIModels[0],         // LVAx2
			AIThreading:       AIThreads[0],        // OFF (default, user can enable)
			PerformanceMode:   PerformanceModes[0], // Balanced
			GPU:               GPUs[0],
			KeepFrames:        KeepFrames[0],
			ImageExtension:    ImageExts[0],
			VideoExtension:    VideoExts[1],
			VideoCodec:        VideoCodecs[0], // x264 (balanced)
			Blending:          Blending[3],    // High (ONNX may need blending)
			OutputPath:        "",              // Empty by default to avoid path leakage
			InputScalePercent: "25",
			OutScalePercent:   "100",
			VRAMGB:            "8",
		}
	default:
		// ONNX-CPU edition: conservative settings
		return Preferences{
			UILanguage:        "zh-CN",
			UITheme:           "contrast",
			UIFontScale:       "standard",
			AIModel:           AIModels[0],         // LVAx2
			AIThreading:       AIThreads[0],        // OFF (default, user can enable)
			PerformanceMode:   PerformanceModes[2], // Quality
			GPU:               GPUs[0],
			KeepFrames:        KeepFrames[0],
			ImageExtension:    ImageExts[0],
			VideoExtension:    VideoExts[1],
			VideoCodec:        VideoCodecs[0], // x264
			Blending:          Blending[3],    // High
			OutputPath:        "",              // Empty by default to avoid path leakage
			InputScalePercent: "25",
			OutScalePercent:   "100",
			VRAMGB:            "4",
		}
	}
}

func PreferenceFilePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if home == "" {
		return "", errors.New("home directory is empty")
	}
	return filepath.Join(home, "Documents", "QualityScaler_2026.1_userpreference_go.json"), nil
}

func LoadPreferences() Preferences {
	prefs := DefaultPreferences()
	path, err := PreferenceFilePath()
	if err != nil {
		return prefs
	}

	b, err := os.ReadFile(path)
	if err != nil {
		return prefs
	}
	_ = json.Unmarshal(b, &prefs)
	return prefs
}

func SavePreferences(prefs Preferences) error {
	path, err := PreferenceFilePath()
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(prefs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func FindByRelativePath(rel string) string {
	exe, err := os.Executable()
	if err == nil {
		candidate := filepath.Join(filepath.Dir(exe), rel)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	cwd, err := os.Getwd()
	if err == nil {
		candidate := filepath.Join(cwd, rel)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	return rel
}
