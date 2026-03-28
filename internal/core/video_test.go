package core

import "testing"

func TestParseGPUIndex(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want int
	}{
		{name: "auto", in: "Auto", want: -1},
		{name: "gpu1", in: "GPU 1", want: 0},
		{name: "gpu4", in: "GPU 4", want: 3},
		{name: "bad", in: "GPU x", want: -1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := parseGPUIndex(tc.in)
			if got != tc.want {
				t.Fatalf("parseGPUIndex(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}
