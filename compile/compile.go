// Package compile lowers a parsed LightGBM model into a minivm bytecode
// program. Each decision tree becomes a nested branch block; leaf
// contributions accumulate into a running sum kept on the operand stack, and
// the final margin is returned as the program result. Every tree is
// stack-neutral (it consumes its temporaries and adds one leaf value to the
// accumulator), so the accumulator stays at the bottom of the stack throughout
// and the code uses no locals — which keeps Interpreter.Reset (used to re-run
// per row) correct.
//
// Feature input is an F64 array stored into each interpreter's copy of the
// feature constant before Run. Splits read it with CONST_GET + ARRAY_GET. This
// keeps row state interpreter-local, so pooled interpreters can evaluate one
// shared Program concurrently.
package compile

import (
	"fmt"
	"math"

	"github.com/siyul-park/minivm/instr"
	"github.com/siyul-park/minivm/optimize"
	"github.com/siyul-park/minivm/program"
	"github.com/siyul-park/minivm/types"

	"github.com/siyul-park/tl2g/model"
)

// options controls compilation.
type options struct {
	optLevel optimize.Level
}

// Option configures Compile.
type Option func(*options)

// WithOptLevel sets the minivm optimizer level applied to the emitted program
// (O0 disables optimization). Defaults to O3.
func WithOptLevel(level optimize.Level) Option {
	return func(o *options) { o.optLevel = level }
}

// CompiledModel is a model lowered to minivm bytecode plus the metadata the
// Predictor needs to feed inputs and post-process outputs.
type CompiledModel struct {
	Program    *program.Program
	FeatureIdx int // constant-pool slot holding the feature row
	NumFeature int
	Objective  model.Objective

	// BatchProgram evaluates BatchCapacity rows per Run: it reads
	// BatchCapacity*NumFeature values from the constant at BatchInputIdx
	// (row-major) and writes BatchCapacity margins to the constant at
	// BatchOutputIdx. Callers must zero-pad any unused input rows; the
	// program always evaluates all BatchCapacity rows.
	BatchProgram   *program.Program
	BatchInputIdx  int
	BatchOutputIdx int
	BatchCapacity  int
}

// Compile lowers m into a top-level minivm program that leaves one raw margin
// f64 on the stack.
func Compile(m *model.Model, opts ...Option) (*CompiledModel, error) {
	if m.NumClass != 1 {
		return nil, fmt.Errorf("compile: num_class=%d unsupported (POC: 1)", m.NumClass)
	}

	opt := options{optLevel: optimize.O3}
	for _, o := range opts {
		o(&opt)
	}

	b := program.NewBuilder()
	featIdx := b.Const(types.TypedArray[float64](make([]float64, m.NumFeature())))

	b.Emit(instr.F64_CONST, math.Float64bits(0))
	for ti := range m.Trees {
		if err := emitTree(b, &m.Trees[ti], featIdx); err != nil {
			return nil, fmt.Errorf("compile: tree %d: %w", ti, err)
		}
	}
	b.Emit(instr.F64_CONST, 0)
	b.Emit(instr.DROP)

	prog, err := b.Build()
	if err != nil {
		return nil, fmt.Errorf("compile: build: %w", err)
	}

	if opt.optLevel > optimize.O0 {
		optimized, err := optimize.NewOptimizer(opt.optLevel).Optimize(prog)
		if err != nil {
			return nil, fmt.Errorf("compile: optimize: %w", err)
		}
		prog = optimized
	}

	batchProg, inputIdx, outputIdx, err := buildBatchProgram(m, opt.optLevel, DefaultBatchCapacity)
	if err != nil {
		return nil, fmt.Errorf("compile: batch: %w", err)
	}

	return &CompiledModel{
		Program:        prog,
		FeatureIdx:     featIdx,
		NumFeature:     m.NumFeature(),
		Objective:      m.Objective,
		BatchProgram:   batchProg,
		BatchInputIdx:  inputIdx,
		BatchOutputIdx: outputIdx,
		BatchCapacity:  DefaultBatchCapacity,
	}, nil
}

// DefaultBatchCapacity is the number of rows CompileBatch evaluates per Run.
const DefaultBatchCapacity = 32

// batch-loop local slot indices.
const (
	batchLocalRow = iota
	batchLocalBase
)

// buildBatchProgram lowers m into a program that loops over capacity rows in
// one Run: row-major input lives in a constant array at inputIdx sized
// capacity*NumFeature, and per-row margins are written to a constant array at
// outputIdx sized capacity. The loop counter (row) and each row's flattened
// base offset (row*NumFeature) live in locals so LOCAL_GET/LOCAL_SET can JIT.
func buildBatchProgram(m *model.Model, optLevel optimize.Level, capacity int) (*program.Program, int, int, error) {
	b := program.NewBuilder()
	b.Locals(types.TypeI32, types.TypeI32)

	inputIdx := b.Const(types.TypedArray[float64](make([]float64, capacity*m.NumFeature())))
	outputIdx := b.Const(types.TypedArray[float64](make([]float64, capacity)))

	b.Emit(instr.I32_CONST, 0)
	b.Emit(instr.LOCAL_SET, batchLocalRow)

	loopStart := b.Label()
	b.Bind(loopStart)

	// base = row * numFeature
	b.Emit(instr.LOCAL_GET, batchLocalRow)
	b.Emit(instr.I32_CONST, uint64(uint32(m.NumFeature())))
	b.Emit(instr.I32_MUL)
	b.Emit(instr.LOCAL_SET, batchLocalBase)

	// output[row] = sum(tree contributions), pushed ref/index before the
	// accumulator so the final F64 lands on top for ARRAY_SET's pop order.
	b.Emit(instr.CONST_GET, uint64(outputIdx))
	b.Emit(instr.LOCAL_GET, batchLocalRow)
	b.Emit(instr.F64_CONST, math.Float64bits(0))
	for ti := range m.Trees {
		if err := emitTreeBatch(b, &m.Trees[ti], inputIdx); err != nil {
			return nil, 0, 0, fmt.Errorf("tree %d: %w", ti, err)
		}
	}
	b.Emit(instr.ARRAY_SET)

	// row++; loop while row < capacity
	b.Emit(instr.LOCAL_GET, batchLocalRow)
	b.Emit(instr.I32_CONST, 1)
	b.Emit(instr.I32_ADD)
	b.Emit(instr.LOCAL_SET, batchLocalRow)
	b.Emit(instr.LOCAL_GET, batchLocalRow)
	b.Emit(instr.I32_CONST, uint64(uint32(capacity)))
	b.Emit(instr.I32_LT_S)
	b.BrIf(loopStart)

	prog, err := b.Build()
	if err != nil {
		return nil, 0, 0, fmt.Errorf("build: %w", err)
	}

	if optLevel > optimize.O0 {
		optimized, err := optimize.NewOptimizer(optLevel).Optimize(prog)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("optimize: %w", err)
		}
		prog = optimized
	}

	return prog, inputIdx, outputIdx, nil
}

// emitTreeBatch emits one tree with feature k read from input[base+k].
func emitTreeBatch(b *program.Builder, t *model.Tree, inputIdx int) error {
	return emitTreeFeatures(b, t, func(node int) {
		b.Emit(instr.CONST_GET, uint64(inputIdx))
		b.Emit(instr.LOCAL_GET, batchLocalBase)
		b.Emit(instr.I32_CONST, uint64(uint32(t.SplitFeature[node])))
		b.Emit(instr.I32_ADD)
		b.Emit(instr.ARRAY_GET)
	})
}

// emitTree emits one tree starting at its root (node 0). A degenerate
// single-leaf tree (no internal nodes) emits leaf 0 directly.
func emitTree(b *program.Builder, t *model.Tree, featIdx int) error {
	return emitTreeFeatures(b, t, func(node int) {
		b.Emit(instr.CONST_GET, uint64(featIdx))
		b.Emit(instr.I32_CONST, uint64(uint32(t.SplitFeature[node])))
		b.Emit(instr.ARRAY_GET)
	})
}

// emitTreeFeatures emits the shared tree control flow, using emitFeature for
// the single-row or flattened-batch feature-addressing scheme.
func emitTreeFeatures(b *program.Builder, t *model.Tree, emitFeature func(int)) error {
	// emitLeaf adds this leaf's contribution to the accumulator already on the
	// stack: [acc] -> [acc + leaf_value]. LightGBM bakes the learning rate into
	// the stored leaf_value, so the shrinkage field is not re-applied here.
	emitLeaf := func(leaf int) {
		b.Emit(instr.F64_CONST, math.Float64bits(t.LeafValue[leaf]))
		b.Emit(instr.F64_ADD)
	}

	if len(t.SplitFeature) == 0 {
		if len(t.LeafValue) == 0 {
			return fmt.Errorf("empty tree")
		}
		emitLeaf(0)
		return nil
	}

	var emitChild func(child int)
	var emitNode func(node int)
	emitNode = func(node int) {
		emitFeature(node)
		// compare: feature <= threshold
		b.Emit(instr.F64_CONST, math.Float64bits(t.Threshold[node]))
		b.Emit(instr.F64_LE)

		left := b.Label()
		end := b.Label()
		b.BrIf(left) // feature <= threshold -> left child
		emitChild(t.RightChild[node])
		b.Br(end)
		b.Bind(left)
		emitChild(t.LeftChild[node])
		b.Bind(end)
	}
	emitChild = func(child int) {
		if model.IsLeaf(child) {
			emitLeaf(model.LeafIndex(child))
			return
		}
		emitNode(child)
	}

	emitNode(0)
	return nil
}
