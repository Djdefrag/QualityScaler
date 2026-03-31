package gui

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"qualityscaler-go/internal/app"
	"qualityscaler-go/internal/core"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/lxn/walk"
	. "github.com/lxn/walk/declarative"
)

var ffmpegVideoResolutionRE = regexp.MustCompile(`(?m)Video:.*?(\d{2,5})x(\d{2,5})`)
var genericResolutionRE = regexp.MustCompile(`(?m)(\d{2,5})x(\d{2,5})`)

type AppWindow struct {
	mw                   *walk.MainWindow
	uiFontFamily         string
	uiLang               Language
	fileList             *walk.ListBox
	language             *walk.ComboBox
	theme                *walk.ComboBox
	fontSize             *walk.ComboBox
	statusLabel          *walk.Label
	progressBar          *walk.ProgressBar
	stageEtaLabel        *walk.Label
	fileEtaLabel         *walk.Label
	totalEtaLabel        *walk.Label
	inputFilesLabel      *walk.Label
	infoLabel            *walk.Label
	configLabel          *walk.Label
	languageLabel        *walk.Label
	themeLabel           *walk.Label
	fontSizeLabel        *walk.Label
	appVersionLabel      *walk.Label
	aiModelLabel         *walk.Label
	gpuLabel             *walk.Label
	perfModeLabel        *walk.Label
	aiThreadsLabel       *walk.Label
	vramLabel            *walk.Label
	blendingLabel        *walk.Label
	inputScaleLabel      *walk.Label
	outputScaleLabel     *walk.Label
	resPreviewTitleLabel *walk.Label
	sourceResLabel       *walk.Label
	inputResLabel        *walk.Label
	modelScaleLabel      *walk.Label
	finalResLabel        *walk.Label
	imageFormatLabel     *walk.Label
	videoFormatLabel     *walk.Label
	videoCodecLabel      *walk.Label
	keepFramesLabel      *walk.Label
	outputFolderLabel    *walk.Label
	systemInfoGroup      *walk.GroupBox
	aiBackendGroup       *walk.GroupBox
	outputSettingsGroup  *walk.GroupBox
	statusGroup          *walk.GroupBox
	addFilesBtn          *walk.PushButton
	clearFilesBtn        *walk.PushButton
	selectOutputBtn      *walk.PushButton
	outputPath           *walk.LineEdit
	inputScale           *walk.LineEdit
	outputScale          *walk.LineEdit
	vramGB               *walk.LineEdit
	aiModel              *walk.ComboBox
	aiThreads            *walk.ComboBox
	perfMode             *walk.ComboBox
	blending             *walk.ComboBox
	gpu                  *walk.ComboBox
	keepFrames           *walk.ComboBox
	imageExt             *walk.ComboBox
	videoExt             *walk.ComboBox
	videoCodec           *walk.ComboBox
	upscaleButton        *walk.PushButton
	stopButton           *walk.PushButton
	files                []string
	prefs                app.Preferences
	ffmpegPath           string
	isApplyingLang       bool
	lastProgress         int
	mu                   sync.Mutex
	cancel               context.CancelFunc
}

func Run() error {
	aiAvailable := core.InitializeAI()

	aw := &AppWindow{
		prefs:        app.LoadPreferences(),
		ffmpegPath:   app.FindByRelativePath(filepath.Join("Assets", "ffmpeg.exe")),
		uiFontFamily: preferredUIFont(),
	}
	aw.uiLang = normalizeLanguage(aw.prefs.UILanguage)
	if err := aw.build(); err != nil {
		return err
	}

	aw.applyPrefs()
	aw.applyLocalizedText()

	modelName := strings.TrimSpace(aw.prefs.AIModel)
	if modelName == "" {
		modelName = app.AIModels[0]
	}

	if aiAvailable {
		if err := core.WarmupAISession(modelName); err != nil {
			detail := core.AIStatusDetail()
			aw.setStatus("就绪 | AI后端: " + core.AIBackend() + " | " + detail)
			walk.MsgBox(
				aw.mw,
				"AI 运行诊断",
				"模型初始化失败: "+modelName+"\n\n后端: "+core.AIBackend()+"\n详情: "+detail+"\n错误: "+err.Error()+"\n\n程序可能回退到 CPU 模式。",
				walk.MsgBoxIconWarning,
			)
		} else {
			aw.setStatus("就绪 | AI后端: " + core.AIBackend())
			if core.AIBackend() != "CUDA (GPU 0)" {
				walk.MsgBox(
					aw.mw,
					"AI 运行诊断",
					"ONNX 已加载，但 CUDA 未启用。\n\n后端: "+core.AIBackend()+"\n详情: "+core.AIStatusDetail()+"\n\n请检查 ONNX GPU DLL 与 CUDA/cuDNN 依赖。",
					walk.MsgBoxIconWarning,
				)
			}
		}
	} else {
		detail := core.AIStatusDetail()
		missing := core.MissingRuntimeDLLs()
		if len(missing) > 0 {
			detail += "\nMissing: " + strings.Join(missing, ", ")
		}
		aw.setStatus("就绪 | AI后端: CPU 回退 | " + detail)
		walk.MsgBox(
			aw.mw,
			"AI 运行环境未就绪",
			"AI 无法使用 GPU，可能是运行时文件缺失或加载失败。\n\n"+detail+"\n\n请将 ONNX Runtime GPU DLL 放在程序根目录或 Assets。",
			walk.MsgBoxIconWarning,
		)
	}

	aw.stopButton.SetEnabled(false)
	aw.mw.Closing().Attach(func(canceled *bool, reason walk.CloseReason) {
		_ = app.SavePreferences(aw.capturePrefs())
		aw.stopProcessing()
	})
	aw.mw.Run()
	return nil
}

func (a *AppWindow) build() error {
	var icon interface{}
	if i, err := walk.NewIconFromFile(filepath.Join("Assets", "logo.ico")); err == nil {
		icon = i
	}

	// Helper to load image safely, ignore error (return nil/empty)
	imageFromFile := func(name string) interface{} {
		path := filepath.Join("Assets", name)
		if _, err := walk.NewImageFromFile(path); err == nil {
			return path
		}
		return nil
	}

	mainAccent := walk.RGB(32, 96, 255)
	mainBackground := walk.RGB(238, 242, 248)
	panelBackground := walk.RGB(255, 255, 255)
	textPrimary := walk.RGB(33, 37, 45)
	textSecondary := walk.RGB(92, 102, 120)

	labelWithHelp := func(lbl **walk.Label, key TextKey, topic HelpTopic) Composite {
		return Composite{
			Layout: HBox{Margins: Margins{}, Spacing: 4},
			Children: []Widget{
				Label{AssignTo: lbl, Text: tr(a.uiLang, key), Font: Font{PointSize: 10}, TextColor: textPrimary},
				PushButton{Text: "i", Font: Font{Bold: true, PointSize: 8}, MinSize: Size{Width: 22, Height: 22}, MaxSize: Size{Width: 22, Height: 22}, OnClicked: func() { a.showHelp(topic) }},
			},
		}
	}

	return MainWindow{
		AssignTo:   &a.mw,
		Title:      fmt.Sprintf("%s %s", app.AppName, app.AppVersion),
		Icon:       icon,
		MinSize:    Size{Width: 1180, Height: 800},
		Font:       Font{Family: a.uiFontFamily, PointSize: 9},
		Background: SolidColorBrush{Color: mainBackground},
		Layout:     HBox{Margins: Margins{Left: 20, Top: 20, Right: 20, Bottom: 20}, Spacing: 18},
		Children: []Widget{
			Composite{
				Layout:     VBox{Margins: Margins{Left: 10, Top: 10, Right: 10, Bottom: 10}, Spacing: 10},
				MaxSize:    Size{Width: 360, Height: 0},
				Background: SolidColorBrush{Color: panelBackground},
				Children: []Widget{
					Label{AssignTo: &a.inputFilesLabel, Text: tr(a.uiLang, TextInputFiles), Font: Font{Bold: true, PointSize: 10}, TextColor: mainAccent},
					ListBox{AssignTo: &a.fileList, MinSize: Size{Width: 320, Height: 220}, OnCurrentIndexChanged: a.updateResolutionPreview},
					Composite{
						Layout: HBox{Margins: Margins{}, Spacing: 8},
						Children: []Widget{
							PushButton{AssignTo: &a.addFilesBtn, Text: tr(a.uiLang, TextAddFiles), OnClicked: a.addFiles, MinSize: Size{Width: 120, Height: 32}},
							PushButton{AssignTo: &a.clearFilesBtn, Text: tr(a.uiLang, TextClearList), OnClicked: a.clearFiles, Image: imageFromFile("clear_icon.png"), MinSize: Size{Width: 120, Height: 32}},
						},
					},
					VSpacer{Size: 8},
					Label{AssignTo: &a.infoLabel, Text: tr(a.uiLang, TextInfo), Font: Font{Bold: true, PointSize: 12}, TextColor: mainAccent},
					GroupBox{
						AssignTo: &a.systemInfoGroup,
						Title:    tr(a.uiLang, TextSystemInfo),
						Layout:   Grid{Columns: 2, Spacing: 8},
						Children: []Widget{
							Label{AssignTo: &a.appVersionLabel, Text: tr(a.uiLang, TextAppVersion)},
							Label{Text: app.AppVersion},
						},
					},
				},
			},
			Composite{
				Layout:     VBox{Margins: Margins{Left: 14, Top: 14, Right: 14, Bottom: 14}, Spacing: 12},
				Background: SolidColorBrush{Color: panelBackground},
				Children: []Widget{
					Composite{
						Layout: HBox{Margins: Margins{}, Spacing: 10},
						Children: []Widget{
							Label{AssignTo: &a.configLabel, Text: tr(a.uiLang, TextConfig), Font: Font{Bold: true, PointSize: 12}, TextColor: mainAccent},
							HSpacer{},
							Label{AssignTo: &a.languageLabel, Text: tr(a.uiLang, TextLanguage), Font: Font{PointSize: 10}, TextColor: textPrimary},
							ComboBox{AssignTo: &a.language, Editable: true, Model: languageOptions(a.uiLang), MinSize: Size{Width: 100, Height: 0}, OnCurrentIndexChanged: a.onLanguageChanged},
							Label{AssignTo: &a.themeLabel, Text: tr(a.uiLang, TextTheme), Font: Font{PointSize: 10}, TextColor: textPrimary},
							ComboBox{AssignTo: &a.theme, Editable: true, Model: themeOptions(a.uiLang), MinSize: Size{Width: 120, Height: 0}, OnCurrentIndexChanged: a.onThemeChanged},
							Label{AssignTo: &a.fontSizeLabel, Text: tr(a.uiLang, TextUIFontSize), Font: Font{PointSize: 10}, TextColor: textPrimary},
							ComboBox{AssignTo: &a.fontSize, Editable: true, Model: fontScaleOptions(a.uiLang), MinSize: Size{Width: 100, Height: 0}, OnCurrentIndexChanged: a.onFontScaleChanged},
						},
					},
					GroupBox{
						AssignTo: &a.aiBackendGroup,
						Title:    tr(a.uiLang, TextAIBackendSettings),
						Font:     Font{Bold: true, PointSize: 10},
						Layout:   Grid{Columns: 4, Spacing: 10},
						Children: []Widget{
							labelWithHelp(&a.aiModelLabel, TextAIModel, HelpAIModel), ComboBox{AssignTo: &a.aiModel, Editable: true, Model: app.AIModels, MinSize: Size{Width: 170, Height: 0}, OnCurrentIndexChanged: a.updateResolutionPreview},
							labelWithHelp(&a.gpuLabel, TextGPU, HelpGPU), ComboBox{AssignTo: &a.gpu, Editable: true, Model: app.GPUs, MinSize: Size{Width: 170, Height: 0}},
							labelWithHelp(&a.perfModeLabel, TextPerformanceMode, HelpThreading), ComboBox{AssignTo: &a.perfMode, Editable: true, Model: app.PerformanceModes, MinSize: Size{Width: 170, Height: 0}},
							labelWithHelp(&a.aiThreadsLabel, TextMultithreading, HelpThreading), ComboBox{AssignTo: &a.aiThreads, Editable: true, Model: app.AIThreads, MinSize: Size{Width: 170, Height: 0}},
							labelWithHelp(&a.vramLabel, TextVRAM, HelpVRAM), LineEdit{AssignTo: &a.vramGB, MaxSize: Size{Width: 72, Height: 0}},
							labelWithHelp(&a.blendingLabel, TextBlending, HelpBlending), ComboBox{AssignTo: &a.blending, Editable: true, Model: app.Blending, MinSize: Size{Width: 170, Height: 0}},
							labelWithHelp(&a.inputScaleLabel, TextInputScale, HelpInputScale), LineEdit{AssignTo: &a.inputScale, MaxSize: Size{Width: 72, Height: 0}, OnTextChanged: a.updateResolutionPreview},
							labelWithHelp(&a.outputScaleLabel, TextOutputScale, HelpOutputScale), LineEdit{AssignTo: &a.outputScale, MaxSize: Size{Width: 72, Height: 0}, OnTextChanged: a.updateResolutionPreview},
						},
					},
					GroupBox{
						Title:  tr(a.uiLang, TextResolutionPreview),
						Layout: VBox{Margins: Margins{Left: 8, Top: 8, Right: 8, Bottom: 8}, Spacing: 6},
						Children: []Widget{
							Label{AssignTo: &a.resPreviewTitleLabel, Text: tr(a.uiLang, TextResolutionPreview), Font: Font{Bold: true, PointSize: 10}, TextColor: mainAccent},
							Label{AssignTo: &a.sourceResLabel, Text: tr(a.uiLang, TextSourceResolution, tr(a.uiLang, TextUnknownResolution)), Font: Font{PointSize: 10}, TextColor: textSecondary},
							Label{AssignTo: &a.inputResLabel, Text: tr(a.uiLang, TextInputResolution, tr(a.uiLang, TextUnknownResolution)), Font: Font{PointSize: 10}, TextColor: textSecondary},
							Label{AssignTo: &a.modelScaleLabel, Text: tr(a.uiLang, TextAIModelScaleDisplay, app.AIModels[0], core.ModelScale(app.AIModels[0])), Font: Font{PointSize: 10}, TextColor: textPrimary},
							Label{AssignTo: &a.finalResLabel, Text: tr(a.uiLang, TextOutputResolution, tr(a.uiLang, TextUnknownResolution)), Font: Font{Bold: true, PointSize: 11}, TextColor: mainAccent},
						},
					},
					GroupBox{
						AssignTo: &a.outputSettingsGroup,
						Title:    tr(a.uiLang, TextOutputSettings),
						Font:     Font{Bold: true, PointSize: 10},
						Layout:   Grid{Columns: 4, Spacing: 10},
						Children: []Widget{
							labelWithHelp(&a.imageFormatLabel, TextImageFormat, HelpImageFormat), ComboBox{AssignTo: &a.imageExt, Editable: true, Model: app.ImageExts},
							labelWithHelp(&a.videoFormatLabel, TextVideoFormat, HelpVideoFormat), ComboBox{AssignTo: &a.videoExt, Editable: true, Model: app.VideoExts},
							labelWithHelp(&a.videoCodecLabel, TextVideoCodec, HelpVideoCodec), ComboBox{AssignTo: &a.videoCodec, Editable: true, Model: app.VideoCodecs},
							labelWithHelp(&a.keepFramesLabel, TextKeepFrames, HelpKeepFrames), ComboBox{AssignTo: &a.keepFrames, Editable: true, Model: app.KeepFrames},
						},
					},
					Composite{
						Layout: HBox{Margins: Margins{}, Spacing: 8},
						Children: []Widget{
							Composite{Layout: HBox{Margins: Margins{}, Spacing: 4}, Children: []Widget{Label{AssignTo: &a.outputFolderLabel, Text: tr(a.uiLang, TextOutputFolder)}, PushButton{Text: "i", MinSize: Size{Width: 22, Height: 22}, MaxSize: Size{Width: 22, Height: 22}, OnClicked: func() { a.showHelp(HelpOutputPath) }}}},
							LineEdit{AssignTo: &a.outputPath, ReadOnly: true},
							PushButton{AssignTo: &a.selectOutputBtn, Text: tr(a.uiLang, TextSelect), OnClicked: a.selectOutputPath, MinSize: Size{Width: 90, Height: 32}},
						},
					},
					VSpacer{},
					GroupBox{
						AssignTo: &a.statusGroup,
						Title:    tr(a.uiLang, TextStatusAndProgress),
						Layout:   VBox{Margins: Margins{Left: 12, Top: 12, Right: 12, Bottom: 12}, Spacing: 8},
						Children: []Widget{
							Label{AssignTo: &a.stageEtaLabel, Text: tr(a.uiLang, TextStageETA, "--:--:--"), Font: Font{Family: a.uiFontFamily, PointSize: 10}, TextColor: textSecondary},
							Label{AssignTo: &a.fileEtaLabel, Text: tr(a.uiLang, TextFileETA, "--:--:--"), Font: Font{Family: a.uiFontFamily, PointSize: 10}, TextColor: textSecondary},
							Label{AssignTo: &a.totalEtaLabel, Text: tr(a.uiLang, TextTotalETA, "--:--:--"), Font: Font{Family: a.uiFontFamily, PointSize: 10}, TextColor: textSecondary},
							ProgressBar{AssignTo: &a.progressBar, MinSize: Size{Height: 20}},
							Label{AssignTo: &a.statusLabel, Text: tr(a.uiLang, TextReady), Font: Font{Family: a.uiFontFamily, PointSize: 10, Bold: true}, TextColor: walk.RGB(255, 255, 255), Background: SolidColorBrush{Color: walk.RGB(30, 88, 220)}, MinSize: Size{Height: 30}},
						},
					},
					Composite{
						Layout: HBox{Margins: Margins{Top: 12}, Spacing: 12},
						Children: []Widget{
							PushButton{AssignTo: &a.upscaleButton, Text: tr(a.uiLang, TextStartUpscale), Font: Font{Bold: true, PointSize: 10}, OnClicked: a.startProcessing, Image: imageFromFile("upscale_icon.png"), MinSize: Size{Width: 190, Height: 46}},
							PushButton{AssignTo: &a.stopButton, Text: tr(a.uiLang, TextStop), Font: Font{Bold: true, PointSize: 10}, OnClicked: a.stopProcessing, Image: imageFromFile("stop_icon.png"), MinSize: Size{Width: 190, Height: 46}},
						},
					},
				},
			},
		},
	}.Create()
}

func (a *AppWindow) addFiles() {
	dlg := new(walk.FileDialog)
	dlg.Title = tr(a.uiLang, TextSelectFilesTitle)
	dlg.Filter = tr(a.uiLang, TextMediaFilesFilter)
	ok, err := dlg.ShowOpenMultiple(a.mw)
	if err != nil || !ok {
		return
	}
	for _, path := range dlg.FilePaths {
		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := app.SupportedExts[ext]; ok {
			a.files = append(a.files, path)
		}
	}
	a.refreshFileList()
	a.updateResolutionPreview()
	a.setStatus(tr(a.uiLang, TextLoadedFiles, len(a.files)))
}

func (a *AppWindow) clearFiles() {
	a.files = nil
	a.refreshFileList()
	a.updateResolutionPreview()
	a.setStatus(tr(a.uiLang, TextFileListCleared))
}

func (a *AppWindow) refreshFileList() {
	if a.fileList == nil {
		return
	}
	_ = a.fileList.SetModel(a.files)
}

func (a *AppWindow) selectOutputPath() {
	dlg := new(walk.FileDialog)
	dlg.Title = tr(a.uiLang, TextSelectOutputFolder)
	ok, err := dlg.ShowBrowseFolder(a.mw)
	if err != nil || !ok {
		return
	}
	a.outputPath.SetText(dlg.FilePath)
}

func (a *AppWindow) applyPrefs() {
	a.uiLang = normalizeLanguage(a.prefs.UILanguage)
	if a.language != nil {
		a.language.SetModel(languageOptions(a.uiLang))
		a.language.SetCurrentIndex(languageIndex(a.uiLang))
	}
	if a.theme != nil {
		a.theme.SetModel(themeOptions(a.uiLang))
		a.theme.SetCurrentIndex(themeIndex(a.prefs.UITheme))
	}
	if a.fontSize != nil {
		a.fontSize.SetModel(fontScaleOptions(a.uiLang))
		a.fontSize.SetCurrentIndex(fontScaleIndex(a.prefs.UIFontScale))
	}
	a.aiModel.SetCurrentIndex(indexOf(app.AIModels, a.prefs.AIModel))
	a.aiThreads.SetCurrentIndex(indexOf(app.AIThreads, a.prefs.AIThreading))
	a.perfMode.SetCurrentIndex(indexOf(app.PerformanceModes, a.prefs.PerformanceMode))
	a.blending.SetCurrentIndex(indexOf(app.Blending, a.prefs.Blending))
	a.gpu.SetCurrentIndex(indexOf(app.GPUs, a.prefs.GPU))
	a.keepFrames.SetCurrentIndex(indexOf(app.KeepFrames, a.prefs.KeepFrames))
	a.imageExt.SetCurrentIndex(indexOf(app.ImageExts, a.prefs.ImageExtension))
	a.videoExt.SetCurrentIndex(indexOf(app.VideoExts, a.prefs.VideoExtension))
	a.videoCodec.SetCurrentIndex(indexOf(app.VideoCodecs, a.prefs.VideoCodec))
	a.inputScale.SetText(a.prefs.InputScalePercent)
	a.outputScale.SetText(a.prefs.OutScalePercent)
	a.vramGB.SetText(a.prefs.VRAMGB)
	if a.prefs.OutputPath == "" {
		a.outputPath.SetText(app.OutputPathCode)
	} else {
		a.outputPath.SetText(a.prefs.OutputPath)
	}
	a.applyAppearance()
}

func (a *AppWindow) onLanguageChanged() {
	if a.isApplyingLang {
		return
	}
	if a.language == nil {
		return
	}
	a.uiLang = languageByIndex(a.language.CurrentIndex())
	a.applyLocalizedText()
	a.setStatus(tr(a.uiLang, TextLangSwitchHint))
}

func (a *AppWindow) onThemeChanged() {
	if a.theme == nil {
		return
	}
	a.prefs.UITheme = themeFromIndex(a.theme.CurrentIndex())
	a.applyAppearance()
	a.setStatus(tr(a.uiLang, TextAppearanceApplied))
}

func (a *AppWindow) onFontScaleChanged() {
	if a.fontSize == nil {
		return
	}
	a.prefs.UIFontScale = fontScaleFromIndex(a.fontSize.CurrentIndex())
	a.applyAppearance()
	a.setStatus(tr(a.uiLang, TextAppearanceApplied))
}

func (a *AppWindow) applyLocalizedText() {
	a.isApplyingLang = true
	defer func() { a.isApplyingLang = false }()

	if a.language != nil {
		a.language.SetModel(languageOptions(a.uiLang))
		a.language.SetCurrentIndex(languageIndex(a.uiLang))
	}
	if a.inputFilesLabel != nil {
		a.inputFilesLabel.SetText(tr(a.uiLang, TextInputFiles))
	}
	if a.infoLabel != nil {
		a.infoLabel.SetText(tr(a.uiLang, TextInfo))
	}
	if a.configLabel != nil {
		a.configLabel.SetText(tr(a.uiLang, TextConfig))
	}
	if a.languageLabel != nil {
		a.languageLabel.SetText(tr(a.uiLang, TextLanguage))
	}
	if a.themeLabel != nil {
		a.themeLabel.SetText(tr(a.uiLang, TextTheme))
	}
	if a.fontSizeLabel != nil {
		a.fontSizeLabel.SetText(tr(a.uiLang, TextUIFontSize))
	}
	if a.theme != nil {
		a.theme.SetModel(themeOptions(a.uiLang))
		a.theme.SetCurrentIndex(themeIndex(a.prefs.UITheme))
	}
	if a.fontSize != nil {
		a.fontSize.SetModel(fontScaleOptions(a.uiLang))
		a.fontSize.SetCurrentIndex(fontScaleIndex(a.prefs.UIFontScale))
	}
	if a.systemInfoGroup != nil {
		a.systemInfoGroup.SetTitle(tr(a.uiLang, TextSystemInfo))
	}
	if a.appVersionLabel != nil {
		a.appVersionLabel.SetText(tr(a.uiLang, TextAppVersion))
	}
	if a.aiBackendGroup != nil {
		a.aiBackendGroup.SetTitle(tr(a.uiLang, TextAIBackendSettings))
	}
	if a.outputSettingsGroup != nil {
		a.outputSettingsGroup.SetTitle(tr(a.uiLang, TextOutputSettings))
	}
	if a.statusGroup != nil {
		a.statusGroup.SetTitle(tr(a.uiLang, TextStatusAndProgress))
	}
	if a.aiModelLabel != nil {
		a.aiModelLabel.SetText(tr(a.uiLang, TextAIModel))
	}
	if a.gpuLabel != nil {
		a.gpuLabel.SetText(tr(a.uiLang, TextGPU))
	}
	if a.perfModeLabel != nil {
		a.perfModeLabel.SetText(tr(a.uiLang, TextPerformanceMode))
	}
	if a.aiThreadsLabel != nil {
		a.aiThreadsLabel.SetText(tr(a.uiLang, TextMultithreading))
	}
	if a.vramLabel != nil {
		a.vramLabel.SetText(tr(a.uiLang, TextVRAM))
	}
	if a.blendingLabel != nil {
		a.blendingLabel.SetText(tr(a.uiLang, TextBlending))
	}
	if a.inputScaleLabel != nil {
		a.inputScaleLabel.SetText(tr(a.uiLang, TextInputScale))
	}
	if a.outputScaleLabel != nil {
		a.outputScaleLabel.SetText(tr(a.uiLang, TextOutputScale))
	}
	if a.imageFormatLabel != nil {
		a.imageFormatLabel.SetText(tr(a.uiLang, TextImageFormat))
	}
	if a.videoFormatLabel != nil {
		a.videoFormatLabel.SetText(tr(a.uiLang, TextVideoFormat))
	}
	if a.videoCodecLabel != nil {
		a.videoCodecLabel.SetText(tr(a.uiLang, TextVideoCodec))
	}
	if a.keepFramesLabel != nil {
		a.keepFramesLabel.SetText(tr(a.uiLang, TextKeepFrames))
	}
	if a.outputFolderLabel != nil {
		a.outputFolderLabel.SetText(tr(a.uiLang, TextOutputFolder))
	}
	if a.addFilesBtn != nil {
		a.addFilesBtn.SetText(tr(a.uiLang, TextAddFiles))
	}
	if a.clearFilesBtn != nil {
		a.clearFilesBtn.SetText(tr(a.uiLang, TextClearList))
	}
	if a.selectOutputBtn != nil {
		a.selectOutputBtn.SetText(tr(a.uiLang, TextSelect))
	}
	if a.upscaleButton != nil {
		a.upscaleButton.SetText(tr(a.uiLang, TextStartUpscale))
	}
	if a.stopButton != nil {
		a.stopButton.SetText(tr(a.uiLang, TextStop))
	}
	if a.resPreviewTitleLabel != nil {
		a.resPreviewTitleLabel.SetText(tr(a.uiLang, TextResolutionPreview))
	}
	a.updateResolutionPreview()
	a.setETAs("--:--:--", "--:--:--", "--:--:--")
}

func (a *AppWindow) updateResolutionPreview() {
	model := a.optionText(a.aiModel, app.AIModels, app.AIModels[0])
	modelScale := core.ModelScale(model)

	if a.modelScaleLabel != nil {
		a.modelScaleLabel.SetText(tr(a.uiLang, TextAIModelScaleDisplay, model, modelScale))
	}

	unknown := tr(a.uiLang, TextUnknownResolution)
	path := a.previewTargetFile()
	if path == "" {
		a.setResolutionLabels(unknown, unknown, unknown)
		return
	}

	width, height, err := a.detectMediaResolution(path)
	if err != nil || width <= 0 || height <= 0 {
		a.setResolutionLabels(unknown, unknown, unknown)
		return
	}

	inputPct := parsePositiveInt(a.inputScale.Text(), 100)
	outputPct := parsePositiveInt(a.outputScale.Text(), 100)

	inputW := scaleByPercent(width, inputPct)
	inputH := scaleByPercent(height, inputPct)
	aiW := inputW * modelScale
	aiH := inputH * modelScale
	finalW := scaleByPercent(aiW, outputPct)
	finalH := scaleByPercent(aiH, outputPct)

	a.setResolutionLabels(
		fmt.Sprintf("%dx%d", width, height),
		fmt.Sprintf("%dx%d", inputW, inputH),
		fmt.Sprintf("%dx%d", finalW, finalH),
	)
}

func (a *AppWindow) setResolutionLabels(source string, input string, output string) {
	if a.sourceResLabel != nil {
		a.sourceResLabel.SetText(tr(a.uiLang, TextSourceResolution, source))
	}
	if a.inputResLabel != nil {
		a.inputResLabel.SetText(tr(a.uiLang, TextInputResolution, input))
	}
	if a.finalResLabel != nil {
		a.finalResLabel.SetText(tr(a.uiLang, TextOutputResolution, output))
	}
}

func (a *AppWindow) previewTargetFile() string {
	if len(a.files) == 0 {
		return ""
	}
	if a.fileList != nil {
		idx := a.fileList.CurrentIndex()
		if idx >= 0 && idx < len(a.files) {
			return a.files[idx]
		}
	}
	return a.files[0]
}

func (a *AppWindow) detectMediaResolution(path string) (int, int, error) {
	if w, h, err := a.probeResolutionByFFmpeg(path); err == nil {
		return w, h, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()

	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}

func (a *AppWindow) probeResolutionByFFmpeg(path string) (int, int, error) {
	ffmpeg := strings.TrimSpace(a.ffmpegPath)
	if ffmpeg == "" {
		return 0, 0, errors.New("ffmpeg path is empty")
	}
	if _, err := os.Stat(ffmpeg); err != nil {
		return 0, 0, err
	}

	cmd := exec.Command(ffmpeg, "-hide_banner", "-i", path)
	hideWindow(cmd)
	out, err := cmd.CombinedOutput()
	if len(out) == 0 {
		if err != nil {
			return 0, 0, err
		}
		return 0, 0, errors.New("ffmpeg output is empty")
	}
	return parseResolutionFromText(string(out))
}

func parseResolutionFromText(text string) (int, int, error) {
	if match := ffmpegVideoResolutionRE.FindStringSubmatch(text); len(match) == 3 {
		w, errW := strconv.Atoi(match[1])
		h, errH := strconv.Atoi(match[2])
		if errW == nil && errH == nil && w > 0 && h > 0 {
			return w, h, nil
		}
	}
	if match := genericResolutionRE.FindStringSubmatch(text); len(match) == 3 {
		w, errW := strconv.Atoi(match[1])
		h, errH := strconv.Atoi(match[2])
		if errW == nil && errH == nil && w > 0 && h > 0 {
			return w, h, nil
		}
	}
	return 0, 0, errors.New("resolution not found")
}

func parsePositiveInt(raw string, fallback int) int {
	v, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func scaleByPercent(value int, percent int) int {
	scaled := int(math.Round(float64(value) * float64(percent) / 100.0))
	if scaled < 1 {
		return 1
	}
	return scaled
}

func (a *AppWindow) showHelp(topic HelpTopic) {
	content := helpContent(a.uiLang, topic)
	lines := ""
	for _, item := range content.Items {
		lines += "- " + item + "\n"
	}
	walk.MsgBox(a.mw, content.Title, content.Subtitle+"\n\n"+lines, walk.MsgBoxIconInformation)
}

func (a *AppWindow) capturePrefs() app.Preferences {
	return app.Preferences{
		UILanguage:        string(a.uiLang),
		UITheme:           themeFromIndex(a.currentIndex(a.theme)),
		UIFontScale:       fontScaleFromIndex(a.currentIndex(a.fontSize)),
		AIModel:           a.optionText(a.aiModel, app.AIModels, app.AIModels[0]),
		AIThreading:       a.optionText(a.aiThreads, app.AIThreads, app.AIThreads[0]),
		PerformanceMode:   a.optionText(a.perfMode, app.PerformanceModes, app.PerformanceModes[0]),
		GPU:               a.optionText(a.gpu, app.GPUs, app.GPUs[0]),
		KeepFrames:        a.optionText(a.keepFrames, app.KeepFrames, app.KeepFrames[0]),
		ImageExtension:    a.optionText(a.imageExt, app.ImageExts, app.ImageExts[0]),
		VideoExtension:    a.optionText(a.videoExt, app.VideoExts, app.VideoExts[0]),
		VideoCodec:        a.optionText(a.videoCodec, app.VideoCodecs, app.VideoCodecs[0]),
		Blending:          a.optionText(a.blending, app.Blending, app.Blending[1]),
		OutputPath:        func() string {
			outPath := a.outputPath.Text()
			if outPath == app.OutputPathCode {
				return ""
			}
			return outPath
		}(),
		InputScalePercent: a.inputScale.Text(),
		OutScalePercent:   a.outputScale.Text(),
		VRAMGB:            a.vramGB.Text(),
	}
}

func (a *AppWindow) currentText(cb *walk.ComboBox, fallback string) string {
	if cb == nil || cb.CurrentIndex() < 0 {
		return fallback
	}
	if t := cb.Text(); t != "" {
		return t
	}
	return fallback
}

func (a *AppWindow) currentIndex(cb *walk.ComboBox) int {
	if cb == nil {
		return -1
	}
	return cb.CurrentIndex()
}

func (a *AppWindow) optionText(cb *walk.ComboBox, options []string, fallback string) string {
	t := strings.TrimSpace(a.currentText(cb, fallback))
	for _, opt := range options {
		if strings.EqualFold(opt, t) {
			return opt
		}
	}
	if cb != nil {
		idx := cb.CurrentIndex()
		if idx >= 0 && idx < len(options) {
			return options[idx]
		}
	}
	return fallback
}

func (a *AppWindow) startProcessing() {
	if len(a.files) == 0 {
		a.setStatus(tr(a.uiLang, TextSelectOneFile))
		return
	}
	sortFilesNatural(a.files)
	a.refreshFileList()
	opts, err := a.readOptions()
	if err != nil {
		a.setStatus(err.Error())
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.mu.Lock()
	a.cancel = cancel
	a.mu.Unlock()
	a.upscaleButton.SetEnabled(false)
	a.stopButton.SetEnabled(true)
	a.lastProgress = -1
	a.setStatus(tr(a.uiLang, TextProcessing, core.AIBackend(), a.optionText(a.perfMode, app.PerformanceModes, app.PerformanceModes[0])))
	a.setProgress(0)
	a.setETAs(tr(a.uiLang, TextCalculating), tr(a.uiLang, TextCalculating), tr(a.uiLang, TextCalculating))
	go func() {
		err := core.ProcessBatch(ctx, a.files, opts, a.ffmpegPath,
			func(msg string) {
				a.mw.Synchronize(func() { a.setStatus(msg) })
			},
			func(progress float64, stageETA string, fileETA string, totalETA string) {
				a.mw.Synchronize(func() {
					a.setProgress(progress)
					a.setETAs(stageETA, fileETA, totalETA)
				})
			},
		)
		a.mw.Synchronize(func() {
			defer func() {
				a.upscaleButton.SetEnabled(true)
				a.stopButton.SetEnabled(false)
			}()
			if err == context.Canceled {
				a.setStatus(tr(a.uiLang, TextStopped))
				return
			}
			if err != nil {
				a.setStatus(tr(a.uiLang, TextErrorPrefix, err.Error()))
				walk.MsgBox(a.mw, tr(a.uiLang, TextTaskErrorTitle), err.Error(), walk.MsgBoxIconError)
				return
			}
			a.setStatus(tr(a.uiLang, TextCompleted))
		})
	}()
}

func (a *AppWindow) stopProcessing() {
	a.mu.Lock()
	cancel := a.cancel
	a.cancel = nil
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (a *AppWindow) readOptions() (core.Options, error) {
	inputScale, err := strconv.Atoi(strings.TrimSpace(a.inputScale.Text()))
	if err != nil || inputScale <= 0 {
		return core.Options{}, errors.New(tr(a.uiLang, TextInvalidInputScale))
	}
	outScale, err := strconv.Atoi(strings.TrimSpace(a.outputScale.Text()))
	if err != nil || outScale <= 0 {
		return core.Options{}, errors.New(tr(a.uiLang, TextInvalidOutputScale))
	}
	vram, err := strconv.Atoi(strings.TrimSpace(a.vramGB.Text()))
	if err != nil || vram <= 0 {
		return core.Options{}, errors.New(tr(a.uiLang, TextInvalidVRAM))
	}
	threading := 1
	if t := a.optionText(a.aiThreads, app.AIThreads, app.AIThreads[0]); strings.Contains(t, "threads") {
		n := strings.Fields(t)
		if len(n) > 0 {
			if parsed, err := strconv.Atoi(n[0]); err == nil {
				threading = parsed
			}
		}
	}
	blendText := a.optionText(a.blending, app.Blending, app.Blending[0])
	blend := 0.0
	switch blendText {
	case "Low":
		blend = 0.3
	case "Medium":
		blend = 0.5
	case "High":
		blend = 0.7
	}
	outPath := strings.TrimSpace(a.outputPath.Text())
	if outPath == "" {
		outPath = app.OutputPathCode
	}
	opts := core.Options{
		OutputPath:         outPath,
		AIModel:            a.optionText(a.aiModel, app.AIModels, app.AIModels[0]),
		AIThreads:          threading,
		InputScalePercent:  inputScale,
		OutputScalePercent: outScale,
		GPU:                a.optionText(a.gpu, app.GPUs, app.GPUs[0]),
		VRAMGB:             vram,
		BlendingFactor:     blend,
		KeepFrames:         a.optionText(a.keepFrames, app.KeepFrames, app.KeepFrames[0]) == "ON",
		ImageExtension:     a.optionText(a.imageExt, app.ImageExts, app.ImageExts[0]),
		VideoExtension:     a.optionText(a.videoExt, app.VideoExts, app.VideoExts[0]),
		VideoCodec:         a.optionText(a.videoCodec, app.VideoCodecs, app.VideoCodecs[0]),
		PerformanceMode:    a.optionText(a.perfMode, app.PerformanceModes, app.PerformanceModes[0]),
	}
	return core.ApplyPerformanceMode(opts), nil
}

func (a *AppWindow) setStatus(msg string) {
	if a.statusLabel != nil {
		a.statusLabel.SetText(msg)
	}
}

func (a *AppWindow) setProgress(value float64) {
	if a.progressBar != nil {
		v := int(math.Round(value))
		if v < 0 {
			v = 0
		}
		if v > 100 {
			v = 100
		}
		if v == a.lastProgress {
			return
		}
		a.lastProgress = v
		a.progressBar.SetValue(v)
	}
}

func (a *AppWindow) setETAs(stage string, file string, total string) {
	if a.stageEtaLabel != nil {
		a.stageEtaLabel.SetText(tr(a.uiLang, TextStageETA, stage))
	}
	if a.fileEtaLabel != nil {
		a.fileEtaLabel.SetText(tr(a.uiLang, TextFileETA, file))
	}
	if a.totalEtaLabel != nil {
		a.totalEtaLabel.SetText(tr(a.uiLang, TextTotalETA, total))
	}
}

func normalizeTheme(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "soft", "soft_gray", "soft-gray":
		return "soft"
	default:
		return "contrast"
	}
}

func themeIndex(v string) int {
	if normalizeTheme(v) == "soft" {
		return 1
	}
	return 0
}

func themeFromIndex(index int) string {
	if index == 1 {
		return "soft"
	}
	return "contrast"
}

func normalizeFontScale(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "compact":
		return "compact"
	case "large":
		return "large"
	default:
		return "standard"
	}
}

func fontScaleIndex(v string) int {
	switch normalizeFontScale(v) {
	case "compact":
		return 0
	case "large":
		return 2
	default:
		return 1
	}
}

func fontScaleFromIndex(index int) string {
	switch index {
	case 0:
		return "compact"
	case 2:
		return "large"
	default:
		return "standard"
	}
}

type appearancePalette struct {
	windowBg    walk.Color
	accent      walk.Color
	textPrimary walk.Color
	textMuted   walk.Color
	statusBg    walk.Color
	statusText  walk.Color
}

func paletteForTheme(v string) appearancePalette {
	if normalizeTheme(v) == "soft" {
		return appearancePalette{
			windowBg:    walk.RGB(244, 246, 250),
			accent:      walk.RGB(76, 99, 130),
			textPrimary: walk.RGB(36, 42, 54),
			textMuted:   walk.RGB(106, 116, 132),
			statusBg:    walk.RGB(76, 99, 130),
			statusText:  walk.RGB(255, 255, 255),
		}
	}
	return appearancePalette{
		windowBg:    walk.RGB(238, 242, 248),
		accent:      walk.RGB(32, 96, 255),
		textPrimary: walk.RGB(33, 37, 45),
		textMuted:   walk.RGB(92, 102, 120),
		statusBg:    walk.RGB(30, 88, 220),
		statusText:  walk.RGB(255, 255, 255),
	}
}

func baseFontPoint(scale string) int {
	switch normalizeFontScale(scale) {
	case "compact":
		return 9
	case "large":
		return 11
	default:
		return 10
	}
}

func setWidgetBackground(w interface{ SetBackground(walk.Brush) }, c walk.Color) {
	if w == nil {
		return
	}
	bg, err := walk.NewSolidColorBrush(c)
	if err != nil {
		return
	}
	w.SetBackground(bg)
}

func setWidgetFont(w interface{ SetFont(*walk.Font) }, family string, point int, style walk.FontStyle) {
	if w == nil {
		return
	}
	f, err := walk.NewFont(family, point, style)
	if err != nil {
		return
	}
	w.SetFont(f)
}

func (a *AppWindow) applyAppearance() {
	palette := paletteForTheme(a.prefs.UITheme)
	point := baseFontPoint(a.prefs.UIFontScale)

	if a.mw != nil {
		setWidgetBackground(a.mw, palette.windowBg)
	}

	for _, lbl := range []*walk.Label{a.languageLabel, a.themeLabel, a.fontSizeLabel, a.aiModelLabel, a.gpuLabel, a.perfModeLabel, a.aiThreadsLabel, a.vramLabel, a.blendingLabel, a.inputScaleLabel, a.outputScaleLabel, a.imageFormatLabel, a.videoFormatLabel, a.videoCodecLabel, a.keepFramesLabel, a.outputFolderLabel, a.modelScaleLabel} {
		if lbl != nil {
			lbl.SetTextColor(palette.textPrimary)
			setWidgetFont(lbl, a.uiFontFamily, point, 0)
		}
	}

	for _, lbl := range []*walk.Label{a.sourceResLabel, a.inputResLabel, a.stageEtaLabel, a.fileEtaLabel, a.totalEtaLabel} {
		if lbl != nil {
			lbl.SetTextColor(palette.textMuted)
			setWidgetFont(lbl, a.uiFontFamily, point, 0)
		}
	}

	for _, lbl := range []*walk.Label{a.inputFilesLabel, a.infoLabel, a.configLabel, a.resPreviewTitleLabel, a.finalResLabel} {
		if lbl != nil {
			lbl.SetTextColor(palette.accent)
			setWidgetFont(lbl, a.uiFontFamily, point+1, walk.FontBold)
		}
	}

	if a.statusLabel != nil {
		a.statusLabel.SetTextColor(palette.statusText)
		setWidgetBackground(a.statusLabel, palette.statusBg)
		setWidgetFont(a.statusLabel, a.uiFontFamily, point, walk.FontBold)
	}

	for _, cb := range []*walk.ComboBox{a.language, a.theme, a.fontSize, a.aiModel, a.aiThreads, a.perfMode, a.blending, a.gpu, a.keepFrames, a.imageExt, a.videoExt, a.videoCodec} {
		setWidgetFont(cb, a.uiFontFamily, point, 0)
	}

	for _, le := range []*walk.LineEdit{a.outputPath, a.inputScale, a.outputScale, a.vramGB} {
		setWidgetFont(le, a.uiFontFamily, point, 0)
	}

	for _, gb := range []*walk.GroupBox{a.systemInfoGroup, a.aiBackendGroup, a.outputSettingsGroup, a.statusGroup} {
		setWidgetFont(gb, a.uiFontFamily, point, walk.FontBold)
	}

	for _, btn := range []*walk.PushButton{a.addFilesBtn, a.clearFilesBtn, a.selectOutputBtn, a.upscaleButton, a.stopButton} {
		setWidgetFont(btn, a.uiFontFamily, point, walk.FontBold)
	}
}

func preferredUIFont() string {
	for _, family := range []string{"Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Segoe UI"} {
		f, err := walk.NewFont(family, 9, 0)
		if err == nil {
			f.Dispose()
			return family
		}
	}
	return "Segoe UI"
}

func sortFilesNatural(paths []string) {
	sort.SliceStable(paths, func(i, j int) bool {
		return naturalStringLess(strings.ToLower(paths[i]), strings.ToLower(paths[j]))
	})
}

func naturalStringLess(a, b string) bool {
	ia, ib := 0, 0
	for ia < len(a) && ib < len(b) {
		ca, cb := a[ia], b[ib]
		if ca >= '0' && ca <= '9' && cb >= '0' && cb <= '9' {
			ja, jb := ia, ib
			for ja < len(a) && a[ja] >= '0' && a[ja] <= '9' {
				ja++
			}
			for jb < len(b) && b[jb] >= '0' && b[jb] <= '9' {
				jb++
			}

			na := strings.TrimLeft(a[ia:ja], "0")
			nb := strings.TrimLeft(b[ib:jb], "0")
			if na == "" {
				na = "0"
			}
			if nb == "" {
				nb = "0"
			}

			if len(na) != len(nb) {
				return len(na) < len(nb)
			}
			if na != nb {
				return na < nb
			}

			ia, ib = ja, jb
			continue
		}

		if ca != cb {
			return ca < cb
		}
		ia++
		ib++
	}

	return len(a) < len(b)
}

func indexOf(items []string, target string) int {
	for i, v := range items {
		if strings.EqualFold(v, target) {
			return i
		}
	}
	return 0
}
