// Package compile lowers a parsed LightGBM model into a minivm bytecode
// program. Each decision tree becomes a nested branch block; leaf
// contributions accumulate into a running sum kept on the operand stack, and
// the final margin is returned as the program result. Every tree is
// stack-neutral (it consumes its temporaries and adds one leaf value to the
// accumulator), so the accumulator stays at the bottom of the stack throughout
// and the code uses no locals — which keeps Interpreter.Reset (used to re-run
// per row) correct.
//
// The tree-walk is emitted into a minivm *function* that the top-level code
// CALLs. minivm only traces and JIT-compiles code reached through a call
// (frame address > 0); top-level module code is never a JIT candidate. Putting
// the hot evaluation behind a CALL therefore makes it eligible for the trace
// JIT once it warms up.
//
// Feature input is a single mutable F64 array interned as a program constant.
// The Predictor writes the current row into CompiledModel.FeatBuf before each
// run; bytecode reads it via ARRAY_GET. Because the buffer backing is shared
// with the constant, a CompiledModel drives one Predictor and is not safe for
// concurrent prediction.
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
	FeatBuf    []float64 // shared backing of the feature-array constant
	NumFeature int
	Objective  model.Objective
}

// Compile lowers m into a minivm program. The optimizer runs over the
// top-level code (configurable via WithOptLevel); the eval function's body is
// left for the interpreter's threaded compiler and the trace JIT.
func Compile(m *model.Model, opts ...Option) (*CompiledModel, error) {
	if m.NumClass != 1 {
		return nil, fmt.Errorf("compile: num_class=%d unsupported (POC: 1)", m.NumClass)
	}

	opt := options{optLevel: optimize.O3}
	for _, o := range opts {
		o(&opt)
	}

	b := program.NewBuilder()

	featBuf := make([]float64, m.NumFeature())
	featArr := types.TypedArray[float64](featBuf)

	// Reserve the feature array's constant-pool slot up front so the eval
	// function body can reference it by index via CONST_GET.
	featIdx := b.Const(featArr)

	fn, err := buildEval(m, featIdx)
	if err != nil {
		return nil, err
	}

	// Top level: call eval() and leave its returned margin on the stack as the
	// program result.
	b.ConstGet(fn)
	b.Emit(instr.CALL)

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

	return &CompiledModel{
		Program:    prog,
		FeatBuf:    featBuf,
		NumFeature: m.NumFeature(),
		Objective:  m.Objective,
	}, nil
}

// buildEval emits the whole ensemble into a no-arg function returning one f64
// (the raw margin). featIdx is the feature array's index in the program's
// constant pool, referenced by CONST_GET in every split.
func buildEval(m *model.Model, featIdx int) (*types.Function, error) {
	fb := types.NewFunctionBuilder(nil).WithReturns(types.TypeF64)

	// accumulator (margin) starts at 0 on the operand stack
	fb.Emit(instr.New(instr.F64_CONST, math.Float64bits(0)))

	for ti := range m.Trees {
		if err := emitTree(fb, &m.Trees[ti], featIdx); err != nil {
			return nil, fmt.Errorf("compile: tree %d: %w", ti, err)
		}
	}

	// return the accumulated margin (top of stack)
	fb.Emit(instr.New(instr.RETURN))

	fn, err := fb.Build()
	if err != nil {
		return nil, fmt.Errorf("compile: build eval: %w", err)
	}
	return fn, nil
}

// emitTree emits one tree starting at its root (node 0). A degenerate
// single-leaf tree (no internal nodes) emits leaf 0 directly.
func emitTree(fb *types.FunctionBuilder, t *model.Tree, featIdx int) error {
	// emitLeaf adds this leaf's contribution to the accumulator already on the
	// stack: [acc] -> [acc + leaf_value]. LightGBM bakes the learning rate into
	// the stored leaf_value, so the shrinkage field is not re-applied here.
	emitLeaf := func(leaf int) {
		fb.Emit(instr.New(instr.F64_CONST, math.Float64bits(t.LeafValue[leaf])))
		fb.Emit(instr.New(instr.F64_ADD))
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
		// push feature value: featArr[split_feature[node]]
		fb.Emit(instr.New(instr.CONST_GET, uint64(featIdx)))
		fb.Emit(instr.New(instr.I32_CONST, uint64(uint32(t.SplitFeature[node]))))
		fb.Emit(instr.New(instr.ARRAY_GET))
		// compare: feature <= threshold
		fb.Emit(instr.New(instr.F64_CONST, math.Float64bits(t.Threshold[node])))
		fb.Emit(instr.New(instr.F64_LE))

		left := fb.Label()
		end := fb.Label()
		fb.BrIf(left) // feature <= threshold -> left child
		emitChild(t.RightChild[node])
		fb.Br(end)
		fb.Bind(left)
		emitChild(t.LeftChild[node])
		fb.Bind(end)
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
