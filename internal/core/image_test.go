package core

import "testing"

func TestModelScale(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{model: "LVAx2", want: 2},
		{model: "RealESR_Gx4", want: 4},
		{model: "IRCNN_Mx1", want: 1},
		{model: "custom-x3", want: 3},
	}

	for _, tc := range tests {
		got := ModelScale(tc.model)
		if got != tc.want {
			t.Fatalf("ModelScale(%q) = %d, want %d", tc.model, got, tc.want)
		}
	}
}
