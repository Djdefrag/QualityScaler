package gui_fyne

import (
	"fyne.io/fyne/v2"
)

// setupWindowDrop 在窗口上注册文件拖拽。
// 拖入文件/文件夹后弹出预览勾选弹窗，用户确认后调用 onConfirm。
func setupWindowDrop(w fyne.Window, onConfirm func(selected []string)) {
	w.SetOnDropped(func(_ fyne.Position, uris []fyne.URI) {
		var rawPaths []string
		for _, u := range uris {
			rawPaths = append(rawPaths, u.Path())
		}
		if len(rawPaths) == 0 {
			return
		}
		candidates := filterSupportedFiles(rawPaths)
		if len(candidates) == 0 {
			return
		}
		fyne.Do(func() {
			showFilePickerDialog(w, candidates, onConfirm)
		})
	})
}
