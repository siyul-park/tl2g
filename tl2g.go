// Package tl2g compiles LightGBM `.mdl` models into minivm bytecode and runs
// prediction in pure Go. The API mirrors tl2cgen: Load a model, Compile it,
// then build a Predictor and Predict over a DMatrix.
//
//	p, _ := tl2g.LoadPredictor("model.mdl") // parse + compile + wrap
//	defer p.Close()
//	out, _ := p.Predict(tl2g.NewDMatrix(X))
//
// Or step by step: tl2g.Load -> tl2g.Compile -> tl2g.NewPredictor.
//
// Scope (POC): num_class=1, numerical splits, regression and binary objectives.
package tl2g

import (
	"github.com/siyul-park/minivm/optimize"
	"github.com/siyul-park/minivm/prof"

	"github.com/siyul-park/tl2g/compile"
	"github.com/siyul-park/tl2g/model"
	"github.com/siyul-park/tl2g/predict"
)

// Re-exported types for a flat tl2cgen-like surface.
type (
	Model         = model.Model
	CompiledModel = compile.CompiledModel
	Predictor     = predict.Predictor
	DMatrix       = predict.DMatrix
	PredictOption = predict.Option
	CompileOption = compile.Option
)

// Load parses a LightGBM text model from path.
func Load(path string) (*Model, error) { return model.Load(path) }

// Compile lowers a parsed model into a minivm bytecode program.
func Compile(m *Model, opts ...CompileOption) (*CompiledModel, error) {
	return compile.Compile(m, opts...)
}

// WithOptLevel sets the minivm optimizer level applied to the compiled
// program (O0 disables optimization).
func WithOptLevel(level optimize.Level) CompileOption { return compile.WithOptLevel(level) }

// NewPredictor builds a Predictor over a compiled model.
func NewPredictor(cm *CompiledModel, opts ...PredictOption) (*Predictor, error) {
	return predict.New(cm, opts...)
}

// WithPoolSize sets the maximum number of interpreters used by Predict.
func WithPoolSize(size int) PredictOption { return predict.WithPoolSize(size) }

// WithJITThreshold sets the minivm native JIT hotness threshold.
func WithJITThreshold(threshold int) PredictOption { return predict.WithJITThreshold(threshold) }

// WithProfiler attaches a minivm profiler to aggregate interpreter/JIT metrics.
func WithProfiler(p *prof.Profiler) PredictOption { return predict.WithProfiler(p) }

// NewDMatrix wraps row-major feature data for prediction.
func NewDMatrix(rows [][]float64) *DMatrix { return predict.NewDMatrix(rows) }

// LoadPredictor is a convenience that parses, compiles, and wraps a model in a
// Predictor in one call.
func LoadPredictor(path string, opts ...PredictOption) (*Predictor, error) {
	m, err := Load(path)
	if err != nil {
		return nil, err
	}
	cm, err := Compile(m)
	if err != nil {
		return nil, err
	}
	return NewPredictor(cm, opts...)
}
