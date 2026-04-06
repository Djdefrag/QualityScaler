package gui_fyne

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	appcfg "qualityscaler-go/internal/app"
	"qualityscaler-go/internal/core"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"fyne.io/fyne/v2"
	fyneapp "fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

var ffmpegVideoResolutionRE = regexp.MustCompile(`(?m)Video:.*?(\d{2,5})x(\d{2,5})`)
var genericResolutionRE = regexp.MustCompile(`(?m)(\d{2,5})x(\d{2,5})`)

type appState struct {
	prefs      appcfg.Preferences
	ffmpegPath string

	files []string
	mu    sync.Mutex
	stop  context.CancelFunc
}

func Run() error {
	edition := appcfg.DetectBuildEdition()
	state := &appState{
		prefs:      appcfg.LoadPreferences(),
		ffmpegPath: appcfg.FindByRelativePath(filepath.Join("Assets", "ffmpeg.exe")),
	}

	// Log detected edition
	editionInfo := fmt.Sprintf("Build Edition: %s", edition)
	if edition != "" {
		fmt.Printf("[INFO] %s\n", editionInfo)
	}

	aiAvailable := core.InitializeAllBackends()
	model := strings.TrimSpace(state.prefs.AIModel)
	if model == "" {
		model = appcfg.AIModels[0]
	}

	// TensorRT 不需要预热 ONNX session；只有纯 ONNX 模式才需要
	if aiAvailable && !core.TensorRTAvailable() {
		_ = core.WarmupAISession(model)
	}

	a := fyneapp.NewWithID("io.github.qualityscaler.go")
	a.Settings().SetTheme(theme.DarkTheme())

	// Include edition in window title
	windowTitle := fmt.Sprintf("%s %s", appcfg.AppName, appcfg.AppVersion)
	if edition != "" {
		windowTitle += fmt.Sprintf(" [%s]", edition)
	}
	w := a.NewWindow(windowTitle)
	w.Resize(fyne.NewSize(1280, 820))

	statusBind := binding.NewString()
	_ = statusBind.Set("就绪")
	progressBind := binding.NewFloat()
	_ = progressBind.Set(0)

	fileList := widget.NewList(
		func() int {
			state.mu.Lock()
			defer state.mu.Unlock()
			return len(state.files)
		},
		func() fyne.CanvasObject {
			return widget.NewLabel("")
		},
		func(id widget.ListItemID, obj fyne.CanvasObject) {
			state.mu.Lock()
			defer state.mu.Unlock()
			if id < len(state.files) {
				// We don't need fyne.Do here as Fyne handles list callbacks on the main thread automatically
				obj.(*widget.Label).SetText(filepath.Base(state.files[id]))
			}
		},
	)

	sourceResBind := binding.NewString()
	inputResBind := binding.NewString()
	modelScaleBind := binding.NewString()
	outputResBind := binding.NewString()
	_ = sourceResBind.Set("源分辨率: 未选择文件")
	_ = inputResBind.Set("输入分辨率: 未选择文件")
	_ = modelScaleBind.Set("模型倍率: ×1")
	_ = outputResBind.Set("输出分辨率: 未选择文件")

	aiModelSel := widget.NewSelect(appcfg.AIModels, nil)
	gpuSel := widget.NewSelect(appcfg.GPUs, nil)
	perfSel := widget.NewSelect(appcfg.PerformanceModes, nil)
	threadsSel := widget.NewSelect(appcfg.AIThreads, nil)
	blendSel := widget.NewSelect(appcfg.Blending, nil)
	imgExtSel := widget.NewSelect(appcfg.ImageExts, nil)
	vidExtSel := widget.NewSelect(appcfg.VideoExts, nil)
	codecSel := widget.NewSelect(appcfg.VideoCodecs, nil)
	keepFramesSel := widget.NewSelect(appcfg.KeepFrames, nil)

	inputScaleEntry := widget.NewEntry()
	outputScaleEntry := widget.NewEntry()
	vramEntry := widget.NewEntry()
	outputPathBind := binding.NewString()
	outputPathEntry := widget.NewEntryWithData(outputPathBind)

	setSelectOrDefault := func(sel *widget.Select, value string, options []string) {
		for _, it := range options {
			if strings.EqualFold(it, value) {
				sel.SetSelected(it)
				return
			}
		}
		if len(options) > 0 {
			sel.SetSelected(options[0])
		}
	}

	setSelectOrDefault(aiModelSel, state.prefs.AIModel, appcfg.AIModels)
	setSelectOrDefault(gpuSel, state.prefs.GPU, appcfg.GPUs)
	setSelectOrDefault(perfSel, state.prefs.PerformanceMode, appcfg.PerformanceModes)
	setSelectOrDefault(threadsSel, state.prefs.AIThreading, appcfg.AIThreads)
	setSelectOrDefault(blendSel, state.prefs.Blending, appcfg.Blending)
	setSelectOrDefault(imgExtSel, state.prefs.ImageExtension, appcfg.ImageExts)
	setSelectOrDefault(vidExtSel, state.prefs.VideoExtension, appcfg.VideoExts)
	setSelectOrDefault(codecSel, state.prefs.VideoCodec, appcfg.VideoCodecs)
	setSelectOrDefault(keepFramesSel, state.prefs.KeepFrames, appcfg.KeepFrames)

	inputScaleEntry.SetText(defaultString(state.prefs.InputScalePercent, "25"))
	outputScaleEntry.SetText(defaultString(state.prefs.OutScalePercent, "100"))
	vramEntry.SetText(defaultString(state.prefs.VRAMGB, "12"))
	_ = outputPathBind.Set(defaultString(state.prefs.OutputPath, appcfg.OutputPathCode))

	previewTargetFile := func() string {
		state.mu.Lock()
		defer state.mu.Unlock()
		if len(state.files) == 0 {
			return ""
		}
		return state.files[0]
	}

	setResLabels := func(src, in, out string) {
		_ = sourceResBind.Set("源分辨率: " + src)
		_ = inputResBind.Set("输入分辨率(应用输入缩放): " + in)
		_ = outputResBind.Set("输出分辨率(模型+输出缩放): " + out)
	}

	updateResolutionPreview := func() {
		modelScale := core.ModelScale(aiModelSel.Selected)
		_ = modelScaleBind.Set(fmt.Sprintf("模型倍率(%s): ×%d", aiModelSel.Selected, modelScale))

		p := previewTargetFile()
		if p == "" {
			setResLabels("未选择文件", "未选择文件", "未选择文件")
			return
		}

		w0, h0, err := detectMediaResolution(state.ffmpegPath, p)
		if err != nil || w0 <= 0 || h0 <= 0 {
			setResLabels("未知", "未知", "未知")
			return
		}

		inputPct := parsePositiveInt(inputScaleEntry.Text, 100)
		outputPct := parsePositiveInt(outputScaleEntry.Text, 100)
		inW := scaleByPercent(w0, inputPct)
		inH := scaleByPercent(h0, inputPct)
		outW := scaleByPercent(inW*modelScale, outputPct)
		outH := scaleByPercent(inH*modelScale, outputPct)

		setResLabels(fmt.Sprintf("%dx%d", w0, h0), fmt.Sprintf("%dx%d", inW, inH), fmt.Sprintf("%dx%d", outW, outH))
	}

	aiModelSel.OnChanged = func(_ string) { updateResolutionPreview() }
	inputScaleEntry.OnChanged = func(_ string) { updateResolutionPreview() }
	outputScaleEntry.OnChanged = func(_ string) { updateResolutionPreview() }

	// addCandidates 将候选文件（已排序）去重后合并进队列
	addCandidates := func(newFiles []string) {
		state.mu.Lock()
		existing := make(map[string]struct{}, len(state.files))
		for _, f := range state.files {
			existing[strings.ToLower(filepath.Clean(f))] = struct{}{}
		}
		for _, f := range newFiles {
			key := strings.ToLower(filepath.Clean(f))
			if _, dup := existing[key]; !dup {
				state.files = append(state.files, f)
				existing[key] = struct{}{}
			}
		}
		sortFilesNatural(state.files)
		count := len(state.files)
		state.mu.Unlock()
		fileList.Refresh()
		updateResolutionPreview()
		_ = statusBind.Set(fmt.Sprintf("已导入文件，当前共 %d 个", count))
	}

	addFilesBtn := widget.NewButton("批量添加文件", func() {
		// 保存当前输出路径，防止 Fyne 文件对话框意外覆盖
		savedOutputPath, _ := outputPathBind.Get()
		fd := dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
			// 立即恢复输出路径，防止被对话框内部逻辑覆盖
			_ = outputPathBind.Set(savedOutputPath)
			if err != nil || reader == nil {
				return
			}
			_ = reader.Close()
			p := reader.URI().Path()
			// 单文件直接通过预览弹窗
			candidates := filterSupportedFiles([]string{p})
			showFilePickerDialog(w, candidates, addCandidates)
		}, w)
		fd.SetFilter(storage.NewExtensionFileFilter([]string{
			".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
			".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".gif",
		}))
		fd.SetConfirmText("选择文件")
		fd.Show()
	})

	addFolderBtn := widget.NewButton("添加文件夹", func() {
		// 保存当前输出路径，防止 Fyne 文件夹对话框意外覆盖
		savedOutputPath, _ := outputPathBind.Get()
		fd := dialog.NewFolderOpen(func(lu fyne.ListableURI, err error) {
			// 立即恢复输出路径，防止被对话框内部逻辑覆盖
			_ = outputPathBind.Set(savedOutputPath)
			if err != nil || lu == nil {
				return
			}
			candidates := scanSupportedFiles(lu.Path())
			showFilePickerDialog(w, candidates, addCandidates)
		}, w)
		fd.Show()
	})

	clearBtn := widget.NewButton("清空列表", func() {
		state.mu.Lock()
		state.files = nil
		state.mu.Unlock()
		fileList.Refresh()
		updateResolutionPreview()
		_ = statusBind.Set("文件列表已清空")
	})

	selectOutputBtn := widget.NewButton("选择输出目录", func() {
		fd := dialog.NewFolderOpen(func(lu fyne.ListableURI, err error) {
			if err != nil || lu == nil {
				return
			}
			_ = outputPathBind.Set(lu.Path())
		}, w)
		fd.Show()
	})

	readOptions := func() (core.Options, error) {
		inputScale, err := strconv.Atoi(strings.TrimSpace(inputScaleEntry.Text))
		if err != nil || inputScale <= 0 {
			return core.Options{}, errors.New("输入缩放 % 必须大于 0")
		}
		outScale, err := strconv.Atoi(strings.TrimSpace(outputScaleEntry.Text))
		if err != nil || outScale <= 0 {
			return core.Options{}, errors.New("输出缩放 % 必须大于 0")
		}
		vram, err := strconv.Atoi(strings.TrimSpace(vramEntry.Text))
		if err != nil || vram <= 0 {
			return core.Options{}, errors.New("GPU 显存(GB)必须大于 0")
		}

		threads := 1
		if strings.Contains(threadsSel.Selected, "threads") {
			parts := strings.Fields(threadsSel.Selected)
			if len(parts) > 0 {
				if n, err := strconv.Atoi(parts[0]); err == nil {
					threads = n
				}
			}
		}

		blend := 0.0
		switch blendSel.Selected {
		case "Low":
			blend = 0.3
		case "Medium":
			blend = 0.5
		case "High":
			blend = 0.7
		}

		outPathVal, _ := outputPathBind.Get()
		outPath := strings.TrimSpace(outPathVal)
		if outPath == "" {
			outPath = appcfg.OutputPathCode
		}

		opts := core.Options{
			OutputPath:         outPath,
			AIModel:            defaultString(aiModelSel.Selected, appcfg.AIModels[0]),
			AIThreads:          threads,
			InputScalePercent:  inputScale,
			OutputScalePercent: outScale,
			GPU:                defaultString(gpuSel.Selected, appcfg.GPUs[0]),
			VRAMGB:             vram,
			BlendingFactor:     blend,
			KeepFrames:         strings.EqualFold(keepFramesSel.Selected, "ON"),
			ImageExtension:     defaultString(imgExtSel.Selected, appcfg.ImageExts[0]),
			VideoExtension:     defaultString(vidExtSel.Selected, appcfg.VideoExts[0]),
			VideoCodec:         defaultString(codecSel.Selected, appcfg.VideoCodecs[0]),
			PerformanceMode:    defaultString(perfSel.Selected, appcfg.PerformanceModes[0]),
		}
		return core.ApplyPerformanceMode(opts), nil
	}

	startBtn := widget.NewButtonWithIcon("开始超分", theme.MediaPlayIcon(), func() {
		state.mu.Lock()
		busy := state.stop != nil
		state.mu.Unlock()
		if busy {
			_ = statusBind.Set("任务进行中，请先停止当前任务")
			return
		}

		state.mu.Lock()
		hasFiles := len(state.files) > 0
		sortFilesNatural(state.files)
		filesCopy := append([]string(nil), state.files...)
		state.mu.Unlock()
		fileList.Refresh()

		if !hasFiles {
			dialog.ShowInformation("提示", "请至少选择一个文件", w)
			return
		}

		opts, err := readOptions()
		if err != nil {
			dialog.ShowError(err, w)
			return
		}

		ctx, cancel := context.WithCancel(context.Background())
		state.mu.Lock()
		state.stop = cancel
		state.mu.Unlock()

		_ = progressBind.Set(0)
		_ = statusBind.Set(fmt.Sprintf("处理中... | AI后端: %s | 模式: %s", core.AIBackend(), opts.PerformanceMode))

		go func() {
			startTime := time.Now()
			currentMsg := "处理中..."
			err := core.ProcessBatchWithCallback(ctx, filesCopy, opts, state.ffmpegPath,
				func(msg string) {
					currentMsg = msg
					elapsed := time.Since(startTime).Round(time.Second)
					elapsedStr := fmt.Sprintf("%02d:%02d:%02d", int(elapsed.Hours()), int(elapsed.Minutes())%60, int(elapsed.Seconds())%60)
					_ = statusBind.Set(fmt.Sprintf("%s | 已耗时: %s | 剩余: 计算中...", currentMsg, elapsedStr))
				},
				func(progress float64, stageETA string, fileETA string, totalETA string) {
					elapsed := time.Since(startTime).Round(time.Second)
					elapsedStr := fmt.Sprintf("%02d:%02d:%02d", int(elapsed.Hours()), int(elapsed.Minutes())%60, int(elapsed.Seconds())%60)
					_ = progressBind.Set(progress / 100.0)
					if currentMsg != "" {
						_ = statusBind.Set(fmt.Sprintf("%s | 已耗时: %s | 总剩余: %s", currentMsg, elapsedStr, totalETA))
					}
				},
				func(completedFile string) {
					// 文件完成回调：从队列中移除（数据操作在 goroutine，UI 更新切到主线程）
					state.mu.Lock()
					var filtered []string
					for _, f := range state.files {
						normalizedF := filepath.Clean(f)
						normalizedCompleted := filepath.Clean(completedFile)
						if !strings.EqualFold(normalizedF, normalizedCompleted) {
							filtered = append(filtered, f)
						}
					}
					state.files = filtered
					state.mu.Unlock()
					// 必须在主线程刷新 UI
					fyne.Do(func() {
						fileList.Refresh()
					})
				},
			)

			state.mu.Lock()
			state.stop = nil
			state.mu.Unlock()

			elapsed := time.Since(startTime).Round(time.Second)
			elapsedStr := fmt.Sprintf("%02d:%02d:%02d", int(elapsed.Hours()), int(elapsed.Minutes())%60, int(elapsed.Seconds())%60)

			if err == context.Canceled {
				_ = statusBind.Set(fmt.Sprintf("已停止 | 耗时: %s", elapsedStr))
				return
			}
			if err != nil {
				errMsg := fmt.Sprintf("错误: %s | 耗时: %s", err.Error(), elapsedStr)
				_ = statusBind.Set(errMsg)
				fyne.Do(func() {
					dialog.ShowError(err, w)
				})
				return
			}
			_ = statusBind.Set(fmt.Sprintf("处理完成 | 总耗时: %s", elapsedStr))
		}()
	})

	stopBtn := widget.NewButtonWithIcon("停止", theme.MediaStopIcon(), func() {
		state.mu.Lock()
		cancel := state.stop
		state.stop = nil
		state.mu.Unlock()
		if cancel != nil {
			cancel()
		}
	})

	configForm := widget.NewForm(
		widget.NewFormItem("AI model", aiModelSel),
		widget.NewFormItem("AI blending", blendSel),
		widget.NewFormItem("AI multithreading", threadsSel),
		widget.NewFormItem("Input scale %", inputScaleEntry),
		widget.NewFormItem("Output scale %", outputScaleEntry),
		widget.NewFormItem("GPU", gpuSel),
		widget.NewFormItem("GPU VRAM (GB)", vramEntry),
		widget.NewFormItem("Performance", perfSel),
		widget.NewFormItem("Image output", imgExtSel),
		widget.NewFormItem("Video output", vidExtSel),
		widget.NewFormItem("Video codec", codecSel),
		widget.NewFormItem("Keep frames", keepFramesSel),
	)

	filePanel := container.NewBorder(
		container.NewVBox(
			widget.NewLabelWithStyle("SUPPORTED FILES", fyne.TextAlignCenter, fyne.TextStyle{Bold: true}),
			widget.NewLabelWithStyle("提示: 可拖拽文件/文件夹到窗口直接导入", fyne.TextAlignLeading, fyne.TextStyle{Italic: true}),
		),
		container.NewHBox(layout.NewSpacer(), addFilesBtn, addFolderBtn, clearBtn, layout.NewSpacer()),
		nil,
		nil,
		fileList,
	)

	resPanel := container.NewVBox(
		widget.NewLabelWithStyle("分辨率预览", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		widget.NewLabelWithData(sourceResBind),
		widget.NewLabelWithData(inputResBind),
		widget.NewLabelWithData(modelScaleBind),
		widget.NewLabelWithData(outputResBind),
	)

	outputPanel := container.NewBorder(nil, nil, nil, selectOutputBtn, outputPathEntry)
	statusPanel := container.NewVBox(
		widget.NewProgressBarWithData(progressBind),
		widget.NewLabelWithData(statusBind),
	)

	right := container.NewVBox(
		widget.NewLabelWithStyle("QualityScaler 2026.1", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		configForm,
		widget.NewSeparator(),
		resPanel,
		widget.NewSeparator(),
		container.NewBorder(nil, nil, widget.NewLabel("Output path"), nil, outputPanel),
		statusPanel,
		container.NewHBox(layout.NewSpacer(), startBtn, stopBtn),
	)

	root := container.NewHSplit(filePanel, container.NewVScroll(right))
	root.Offset = 0.34
	w.SetContent(root)

	// 注册窗口级文件拖拽（文件/文件夹均支持，拖入后进预览弹窗）
	setupWindowDrop(w, addCandidates)

	if !aiAvailable {
		detail := core.AIStatusDetail()
		missing := core.MissingRuntimeDLLs()
		if len(missing) > 0 {
			detail += "\nMissing: " + strings.Join(missing, ", ")
		}
		_ = statusBind.Set("就绪 | AI后端: CPU 回退")
		dialog.ShowInformation("AI 运行环境未就绪", detail, w)
	} else if !core.TensorRTAvailable() && core.CUDAUnavailable() {
		// 仅在确认 CUDA 不可用（且非 TensorRT 模式）时弹窗提示
		dialog.ShowInformation("AI 运行诊断", "ONNX 已加载，但 CUDA 未启用。\n后端: "+core.AIBackend()+"\n详情: "+core.AIStatusDetail(), w)
	}

	updateResolutionPreview()

	w.SetCloseIntercept(func() {
		state.mu.Lock()
		cancel := state.stop
		state.stop = nil
		state.mu.Unlock()
		if cancel != nil {
			cancel()
		}

		_ = appcfg.SavePreferences(appcfg.Preferences{
			UILanguage:        state.prefs.UILanguage,
			UITheme:           state.prefs.UITheme,
			UIFontScale:       state.prefs.UIFontScale,
			AIModel:           defaultString(aiModelSel.Selected, appcfg.AIModels[0]),
			AIThreading:       defaultString(threadsSel.Selected, appcfg.AIThreads[0]),
			PerformanceMode:   defaultString(perfSel.Selected, appcfg.PerformanceModes[0]),
			GPU:               defaultString(gpuSel.Selected, appcfg.GPUs[0]),
			KeepFrames:        defaultString(keepFramesSel.Selected, appcfg.KeepFrames[0]),
			ImageExtension:    defaultString(imgExtSel.Selected, appcfg.ImageExts[0]),
			VideoExtension:    defaultString(vidExtSel.Selected, appcfg.VideoExts[0]),
			VideoCodec:        defaultString(codecSel.Selected, appcfg.VideoCodecs[0]),
			Blending:          defaultString(blendSel.Selected, appcfg.Blending[0]),
			OutputPath:        func() string { v, _ := outputPathBind.Get(); return v }(),
			InputScalePercent: inputScaleEntry.Text,
			OutScalePercent:   outputScaleEntry.Text,
			VRAMGB:            vramEntry.Text,
		})
		w.Close()
	})

	w.ShowAndRun()
	return nil
}

func sortFilesNatural(paths []string) {
	sort.SliceStable(paths, func(i, j int) bool {
		ai := strings.ToLower(filepath.Base(paths[i]))
		aj := strings.ToLower(filepath.Base(paths[j]))
		if ai == aj {
			return naturalStringLess(strings.ToLower(paths[i]), strings.ToLower(paths[j]))
		}
		return naturalStringLess(ai, aj)
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

func defaultString(v string, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func detectMediaResolution(ffmpegPath string, path string) (int, int, error) {
	if w, h, err := probeResolutionByFFmpeg(ffmpegPath, path); err == nil {
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

func probeResolutionByFFmpeg(ffmpegPath string, path string) (int, int, error) {
	ffmpeg := strings.TrimSpace(ffmpegPath)
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
