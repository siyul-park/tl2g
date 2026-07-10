package tl2g_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/siyul-park/minivm/interp"
	"github.com/siyul-park/minivm/prof"
	"github.com/siyul-park/minivm/types"

	"github.com/siyul-park/tl2g/compile"
	"github.com/siyul-park/tl2g/model"
)

// Diagnostic: does tl2g's eval actually JIT, and does it help?
func TestDiagJIT(t *testing.T) {
	if os.Getenv("TL2G_DIAG") == "" {
		t.Skip("set TL2G_DIAG=1 to run JIT diagnostics")
	}

	g := loadGolden(t, "regression")
	mFull, err := model.Load(filepath.Join("testdata", "regression.mdl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model has %d trees", len(mFull.Trees))
	ctx := context.Background()

	run := func(cm *compile.CompiledModel, threshold int) (time.Duration, *prof.Profiler) {
		p := prof.New()
		vm := interp.New(cm.Program, interp.WithThreshold(threshold), interp.WithProfiler(p))
		const N = 200000
		start := time.Now()
		buf := make(types.TypedArray[float64], cm.NumFeature)
		featureConst, err := vm.Const(cm.FeatureIdx)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < N; i++ {
			vm.Reset()
			copy(buf, g.X[i%len(g.X)]) // vary branch paths each run
			if err := vm.Store(featureConst.Ref(), buf); err != nil {
				t.Fatal(err)
			}
			if err := vm.Run(ctx); err != nil {
				t.Fatal(err)
			}
			if _, err := vm.Pop(); err != nil {
				t.Fatal(err)
			}
		}
		d := time.Since(start) / N
		vm.Close() // flushes JIT metrics into the profiler
		return d, p
	}

	for _, k := range []int{1, 2, 4, 8, 16, 30} {
		if k > len(mFull.Trees) {
			break
		}
		m := *mFull
		m.Trees = mFull.Trees[:k]
		cm, err := compile.Compile(&m)
		if err != nil {
			t.Fatal(err)
		}
		d, p := run(cm, 1) // aggressive JIT
		emits, _ := p.Metric("vm_jit_emits_total")
		attempts, _ := p.Metric("vm_jit_attempts_total")
		errs, _ := p.Metric("vm_jit_errors_total")
		t.Logf("trees=%-3d  %5d ns/op   emits=%.0f attempts=%.0f errors=%.0f",
			k, d.Nanoseconds(), emits, attempts, errs)
	}
}
