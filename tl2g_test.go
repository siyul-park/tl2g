package tl2g_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/siyul-park/minivm/optimize"
	"github.com/siyul-park/tl2g"
	"github.com/siyul-park/tl2g/compile"
	"github.com/siyul-park/tl2g/model"
	"github.com/siyul-park/tl2g/predict"
)

type golden struct {
	X         [][]float64 `json:"X"`
	Pred      []float64   `json:"pred"`
	Raw       []float64   `json:"raw"`
	Objective string      `json:"objective"`
}

func loadGolden(t testing.TB, name string) golden {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name+"_golden.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g golden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	return g
}

// TestPredictMatchesLightGBM verifies tl2g predictions match LightGBM's own
// output within a tight tolerance, for both regression and binary models.
func TestPredictMatchesLightGBM(t *testing.T) {
	const tol = 1e-6

	for _, name := range []string{"regression", "binary"} {
		t.Run(name, func(t *testing.T) {
			// Arrange
			g := loadGolden(t, name)
			p, err := tl2g.LoadPredictor(filepath.Join("testdata", name+".mdl"))
			if err != nil {
				t.Fatalf("LoadPredictor: %v", err)
			}
			defer p.Close()

			// Act
			out, err := p.Predict(tl2g.NewDMatrix(g.X))
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}

			// Assert
			if len(out) != len(g.Pred) {
				t.Fatalf("len mismatch: got %d want %d", len(out), len(g.Pred))
			}
			var maxDiff float64
			for i := range out {
				d := math.Abs(out[i] - g.Pred[i])
				if d > maxDiff {
					maxDiff = d
				}
				if d > tol {
					t.Errorf("row %d: got %.10f want %.10f (diff %.3g)", i, out[i], g.Pred[i], d)
				}
			}
			t.Logf("%s: %d rows, max abs diff %.3g", name, len(out), maxDiff)
		})
	}
}

func TestPredictWithOptLevel(t *testing.T) {
	const tol = 1e-6

	levels := []struct {
		name  string
		level optimize.Level
	}{
		{name: "O2", level: optimize.O2},
		{name: "O3", level: optimize.O3},
	}
	for _, name := range []string{"regression", "binary"} {
		for _, level := range levels {
			t.Run(name+"/"+level.name, func(t *testing.T) {
				g := loadGolden(t, name)
				m, err := model.Load(filepath.Join("testdata", name+".mdl"))
				if err != nil {
					t.Fatalf("Load: %v", err)
				}
				cm, err := compile.Compile(m, compile.WithOptLevel(level.level))
				if err != nil {
					t.Fatalf("Compile: %v", err)
				}
				p, err := predict.New(cm)
				if err != nil {
					t.Fatalf("NewPredictor: %v", err)
				}
				defer p.Close()

				out, err := p.Predict(tl2g.NewDMatrix(g.X))
				if err != nil {
					t.Fatalf("Predict: %v", err)
				}
				if len(out) != len(g.Pred) {
					t.Fatalf("len mismatch: got %d want %d", len(out), len(g.Pred))
				}
				for i := range out {
					if d := math.Abs(out[i] - g.Pred[i]); d > tol {
						t.Errorf("row %d: got %.10f want %.10f (diff %.3g)", i, out[i], g.Pred[i], d)
					}
				}
			})
		}
	}
}

func BenchmarkPredictSingle(b *testing.B) {
	for _, name := range []string{"regression", "binary"} {
		b.Run(name, func(b *testing.B) {
			g := loadGolden(b, name)
			p, err := tl2g.LoadPredictor(filepath.Join("testdata", name+".mdl"))
			if err != nil {
				b.Fatalf("LoadPredictor: %v", err)
			}
			defer p.Close()
			d := tl2g.NewDMatrix(g.X[:1])

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := p.Predict(d); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkPredictBatch(b *testing.B) {
	for _, name := range []string{"regression", "binary"} {
		b.Run(name, func(b *testing.B) {
			g := loadGolden(b, name)
			p, err := tl2g.LoadPredictor(filepath.Join("testdata", name+".mdl"))
			if err != nil {
				b.Fatalf("LoadPredictor: %v", err)
			}
			defer p.Close()
			d := tl2g.NewDMatrix(g.X)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := p.Predict(d); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
