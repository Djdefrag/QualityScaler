package gui_fyne

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	appcfg "qualityscaler-go/internal/app"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"
)

// scanSupportedFiles 递归扫描目录，返回所有支持的文件路径（自然排序）
func scanSupportedFiles(root string) []string {
	var found []string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := appcfg.SupportedExts[ext]; ok {
			found = append(found, path)
		}
		return nil
	})
	sortFilesNatural(found)
	return found
}

// filterSupportedFiles 从路径列表中过滤支持的文件（目录会被递归展开）
func filterSupportedFiles(paths []string) []string {
	var out []string
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		if info.IsDir() {
			out = append(out, scanSupportedFiles(p)...)
		} else {
			ext := strings.ToLower(filepath.Ext(p))
			if _, ok := appcfg.SupportedExts[ext]; ok {
				out = append(out, p)
			}
		}
	}
	sortFilesNatural(out)
	return out
}

// showFilePickerDialog 展示文件预览勾选弹窗。
// candidates 是候选文件列表（已排序），onConfirm 接收最终勾选的文件。
func showFilePickerDialog(w fyne.Window, candidates []string, onConfirm func(selected []string)) {
	if len(candidates) == 0 {
		dialog.ShowInformation("提示", "未找到支持的文件", w)
		return
	}

	n := len(candidates)
	checked := make([]bool, n)
	for i := range checked {
		checked[i] = true
	}

	// 范围选择输入框
	firstEntry := widget.NewEntry()
	firstEntry.SetPlaceHolder("起始序号")
	firstEntry.SetText("1")
	firstEntry.Resize(fyne.NewSize(60, firstEntry.MinSize().Height))

	lastEntry := widget.NewEntry()
	lastEntry.SetPlaceHolder("结束序号")
	lastEntry.SetText(fmt.Sprintf("%d", n))
	lastEntry.Resize(fyne.NewSize(60, lastEntry.MinSize().Height))

	// 每个文件对应一个 checkbox
	checks := make([]*widget.Check, n)
	for i, p := range candidates {
		idx := i
		label := fmt.Sprintf("%d.  %s", idx+1, filepath.Base(p))
		ch := widget.NewCheck(label, func(v bool) {
			checked[idx] = v
		})
		ch.SetChecked(true)
		checks[i] = ch
	}

	refreshChecks := func() {
		for i, ch := range checks {
			ch.SetChecked(checked[i])
		}
	}

	allBtn := widget.NewButton("全选", func() {
		for i := range checked {
			checked[i] = true
		}
		refreshChecks()
	})
	noneBtn := widget.NewButton("全不选", func() {
		for i := range checked {
			checked[i] = false
		}
		refreshChecks()
	})
	invertBtn := widget.NewButton("反选", func() {
		for i := range checked {
			checked[i] = !checked[i]
		}
		refreshChecks()
	})
	rangeBtn := widget.NewButton("选范围", func() {
		first := parsePositiveInt(firstEntry.Text, 1) - 1
		last := parsePositiveInt(lastEntry.Text, n) - 1
		if first < 0 {
			first = 0
		}
		if last >= n {
			last = n - 1
		}
		if first > last {
			first, last = last, first
		}
		for i := range checked {
			checked[i] = i >= first && i <= last
		}
		refreshChecks()
	})

	toolbar := container.NewHBox(
		allBtn, noneBtn, invertBtn,
		widget.NewSeparator(),
		widget.NewLabel("第"),
		firstEntry,
		widget.NewLabel("→"),
		lastEntry,
		widget.NewLabel("个"),
		rangeBtn,
	)

	// 文件列表（可滚动）
	listBox := container.NewVBox()
	for _, ch := range checks {
		listBox.Add(ch)
	}
	scroll := container.NewVScroll(listBox)
	scroll.SetMinSize(fyne.NewSize(520, 360))

	content := container.NewBorder(
		container.NewVBox(
			widget.NewLabelWithStyle(
				fmt.Sprintf("共 %d 个文件，勾选后点击[导入选中文件]", n),
				fyne.TextAlignLeading, fyne.TextStyle{Italic: true},
			),
			toolbar,
			widget.NewSeparator(),
		),
		nil, nil, nil,
		scroll,
	)

	dlg := dialog.NewCustomConfirm(
		"选择要导入的文件",
		"导入选中文件",
		"取消",
		content,
		func(ok bool) {
			if !ok {
				return
			}
			var sel []string
			for i, v := range checked {
				if v {
					sel = append(sel, candidates[i])
				}
			}
			if len(sel) == 0 {
				return
			}
			onConfirm(sel)
		},
		w,
	)
	dlg.Resize(fyne.NewSize(560, 520))
	dlg.Show()
}
