// Package predict runs a compiled LightGBM model on the minivm interpreter and
// applies the objective's output transform.
package predict

import (
	"context"
	"fmt"
	"math"

	"github.com/siyul-park/minivm/interp"
	"github.com/siyul-park/minivm/types"

	"github.com/siyul-park/tl2g/compile"
)

// DMatrix is a dense row-major feature matrix, mirroring tl2cgen.DMatrix.
type DMatrix struct {
	rows [][]float64
}

// NewDMatrix wraps row-major feature data. Rows are not copied.
func NewDMatrix(rows [][]float64) *DMatrix { return &DMatrix{rows: rows} }

// Len returns the number of rows.
func (d *DMatrix) Len() int { return len(d.rows) }

// Predictor evaluates a CompiledModel. It is single-threaded: the underlying
// interpreter and shared feature buffer make one Predictor unsafe for
// concurrent Predict calls.
type Predictor struct {
	cm  *compile.CompiledModel
	vm  *interp.Interpreter
	ctx context.Context
}

// New builds a Predictor over a compiled model.
func New(cm *compile.CompiledModel) (*Predictor, error) {
	if cm == nil || cm.Program == nil {
		return nil, fmt.Errorf("predict: nil compiled model")
	}
	return &Predictor{
		cm:  cm,
		vm:  interp.New(cm.Program),
		ctx: context.Background(),
	}, nil
}

// Close releases interpreter resources.
func (p *Predictor) Close() error { return p.vm.Close() }

// Predict returns one prediction per row of d, with the objective transform
// applied (identity for regression, sigmoid for binary).
func (p *Predictor) Predict(d *DMatrix) ([]float64, error) {
	out := make([]float64, len(d.rows))
	for i, row := range d.rows {
		margin, err := p.margin(row)
		if err != nil {
			return nil, fmt.Errorf("predict: row %d: %w", i, err)
		}
		out[i] = p.transform(margin)
	}
	return out, nil
}

// margin runs the bytecode for a single row and returns the raw margin sum.
func (p *Predictor) margin(row []float64) (float64, error) {
	p.vm.Reset()

	fb := p.cm.FeatBuf
	for j := range fb {
		if j < len(row) {
			fb[j] = row[j]
		} else {
			fb[j] = 0
		}
	}

	if err := p.vm.Run(p.ctx); err != nil {
		return 0, err
	}
	v, err := p.vm.Pop()
	if err != nil {
		return 0, err
	}
	f, ok := v.(types.F64)
	if !ok {
		return 0, fmt.Errorf("unexpected result type %T", v)
	}
	return float64(f), nil
}

// transform maps a raw margin to the model's output space.
func (p *Predictor) transform(margin float64) float64 {
	switch p.cm.Objective.Name {
	case "binary":
		alpha := p.cm.Objective.Sigmoid
		if alpha == 0 {
			alpha = 1
		}
		return 1.0 / (1.0 + math.Exp(-alpha*margin))
	default:
		return margin
	}
}
