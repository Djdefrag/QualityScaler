package core

import (
	"fmt"
	"strings"
	"sync"
)

// BatchLogger accumulates log messages and flushes them to stdout in batches.
// This prevents terminal flooding when processing many frames.
type BatchLogger struct {
	mu        sync.Mutex
	buf       []string
	batchSize int
}

// Add appends a message to the buffer. When the buffer reaches batchSize, it
// is automatically flushed. msg should NOT contain a trailing newline.
func (b *BatchLogger) Add(msg string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, msg)
	if len(b.buf) >= b.batchSize {
		b.flush()
	}
}

// Flush forces all buffered messages to be printed, even if the batch is not full.
// Call this after a processing batch completes to ensure no messages are lost.
func (b *BatchLogger) Flush() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.flush()
}

// flush is the internal (non-locking) implementation.
func (b *BatchLogger) flush() {
	if len(b.buf) == 0 {
		return
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("--- [batch %d msgs] ---\n", len(b.buf)))
	for _, msg := range b.buf {
		sb.WriteString(msg)
		sb.WriteByte('\n')
	}
	fmt.Print(sb.String())
	b.buf = b.buf[:0]
}

// TRTLog is the global batch logger for TensorRT per-frame inference messages.
var TRTLog = &BatchLogger{batchSize: 20}

// AILog is the global batch logger for ONNX Runtime per-frame inference messages.
var AILog = &BatchLogger{batchSize: 20}
