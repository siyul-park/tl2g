// Package predict runs a compiled LightGBM model on the minivm interpreter and
// applies the objective's output transform.
package predict

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/siyul-park/minivm/interp"
	"github.com/siyul-park/minivm/prof"
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

type options struct {
	poolSize     int
	jitThreshold int
	profiler     *prof.Profiler
}

// Option configures Predictor construction.
type Option func(*options)

// WithPoolSize sets the maximum number of interpreters used by Predict.
// Values <= 0 are normalized to 1.
func WithPoolSize(size int) Option {
	return func(o *options) { o.poolSize = size }
}

// WithJITThreshold sets the minivm native JIT hotness threshold.
// The default is 0, which compiles on the first sample. Use -1 to disable JIT.
func WithJITThreshold(threshold int) Option {
	return func(o *options) { o.jitThreshold = threshold }
}

// WithProfiler attaches a minivm profiler to aggregate interpreter/JIT metrics.
func WithProfiler(p *prof.Profiler) Option {
	return func(o *options) { o.profiler = p }
}

// Predictor evaluates a CompiledModel with a pool of minivm interpreters.
// Predict is intended to be called by one goroutine at a time.
type Predictor struct {
	cm *compile.CompiledModel

	pool     *interp.Pool
	poolSize int

	mu   sync.Mutex
	bufs map[*interp.Interpreter]types.TypedArray[float64]

	// batchPool runs cm.BatchProgram, which evaluates cm.BatchCapacity rows
	// per Run instead of one and amortizes per-Run host overhead across a
	// whole chunk of rows.
	batchPool *interp.Pool
	batchMu   sync.Mutex
	batchBufs map[*interp.Interpreter]batchBuf

	ctx context.Context
}

// batchBuf holds one pooled interpreter's private input/output arrays for
// the batch program, decoupled from the shared constant backing array the
// same way loadFeatureBuf decouples the single-row feature buffer.
type batchBuf struct {
	in  types.TypedArray[float64]
	out types.TypedArray[float64]
}

// New builds a Predictor over a compiled model.
func New(cm *compile.CompiledModel, opts ...Option) (*Predictor, error) {
	if cm == nil || cm.Program == nil {
		return nil, fmt.Errorf("predict: nil compiled model")
	}

	opt := options{
		poolSize:     runtime.GOMAXPROCS(0),
		jitThreshold: 0,
	}
	for _, o := range opts {
		o(&opt)
	}
	if opt.poolSize <= 0 {
		opt.poolSize = 1
	}

	pool := interp.NewPool(
		cm.Program,
		opt.poolSize,
		interp.WithTick(1),
		interp.WithThreshold(opt.jitThreshold),
		interp.WithProfiler(opt.profiler),
	)

	p := &Predictor{
		cm:       cm,
		pool:     pool,
		poolSize: opt.poolSize,
		bufs:     make(map[*interp.Interpreter]types.TypedArray[float64], opt.poolSize),
		ctx:      context.Background(),
	}

	// Validate the feature constant resolves before handing out the Predictor.
	vm, err := pool.Get(p.ctx)
	if err != nil {
		_ = pool.Close()
		return nil, fmt.Errorf("predict: %w", err)
	}
	if _, err := featureRef(vm, cm.FeatureIdx); err != nil {
		pool.Put(vm)
		_ = pool.Close()
		return nil, fmt.Errorf("predict: %w", err)
	}
	pool.Put(vm)

	if cm.BatchProgram != nil {
		batchPool := interp.NewPool(
			cm.BatchProgram,
			opt.poolSize,
			interp.WithTick(1),
			interp.WithThreshold(opt.jitThreshold),
			interp.WithProfiler(opt.profiler),
		)
		p.batchPool = batchPool
		p.batchBufs = make(map[*interp.Interpreter]batchBuf, opt.poolSize)

		bvm, err := batchPool.Get(p.ctx)
		if err != nil {
			_ = pool.Close()
			_ = batchPool.Close()
			return nil, fmt.Errorf("predict: %w", err)
		}
		if _, _, err := batchRefs(bvm, cm); err != nil {
			batchPool.Put(bvm)
			_ = pool.Close()
			_ = batchPool.Close()
			return nil, fmt.Errorf("predict: %w", err)
		}
		batchPool.Put(bvm)
	}

	return p, nil
}

// Close releases interpreter resources.
func (p *Predictor) Close() error {
	err := p.pool.Close()
	if p.batchPool != nil {
		if berr := p.batchPool.Close(); err == nil {
			err = berr
		}
	}
	return err
}

// Predict returns one prediction per row of d, with the objective transform
// applied (identity for regression, sigmoid for binary).
func (p *Predictor) Predict(d *DMatrix) ([]float64, error) {
	out := make([]float64, len(d.rows))
	if len(d.rows) == 0 {
		return out, nil
	}
	// A single row has no batch to amortize over, so it always takes the
	// per-row path; rerunning the same small program over and over on one
	// VM is also what warms up and exercises minivm's native JIT (see
	// TestPredictorJITEmits and TestCompiledModelJITEmits).
	// Multi-row predicts go through the batch path whenever one is available.
	if len(d.rows) == 1 || p.batchPool == nil {
		if err := p.predictRange(out, d, 0, len(d.rows)); err != nil {
			return nil, fmt.Errorf("predict: %w", err)
		}
		return out, nil
	}
	// Cap batch workers so each chunk is at least one full BatchCapacity
	// Run: splitting fewer than BatchCapacity rows across many workers
	// would make every worker still pay for a full BatchCapacity-row loop
	// (the program always evaluates its whole fixed-capacity loop), wasting
	// most of that loop on zero-padded rows.
	batchWorkers := len(d.rows) / p.cm.BatchCapacity
	if batchWorkers < 1 {
		batchWorkers = 1
	}
	return p.predictBatchConcurrent(out, d, batchWorkers)
}

// predictBatchConcurrent splits d.rows into
// min(maxWorkers, poolSize, len(d.rows)) contiguous chunks and evaluates each
// chunk on its own
// goroutine, reusing one VM (and its buffers) per chunk instead of per row.
func (p *Predictor) predictBatchConcurrent(out []float64, d *DMatrix, maxWorkers int) ([]float64, error) {
	n := len(d.rows)
	workers := p.poolSize
	if workers > maxWorkers {
		workers = maxWorkers
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		if err := p.predictBatchRange(out, d, 0, n); err != nil {
			return nil, fmt.Errorf("predict: %w", err)
		}
		return out, nil
	}
	chunk := (n + workers - 1) / workers
	if capacity := p.cm.BatchCapacity; capacity > 0 {
		chunk = (chunk + capacity - 1) / capacity * capacity
	}

	errs := make(chan rowError, workers)
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			if err := p.predictBatchRange(out, d, start, end); err != nil {
				errs <- rowError{idx: start, err: err}
			}
		}(start, end)
	}
	wg.Wait()
	close(errs)

	var first rowError
	for err := range errs {
		if first.err == nil || err.idx < first.idx {
			first = err
		}
	}
	if first.err != nil {
		return nil, fmt.Errorf("predict: %w", first.err)
	}
	return out, nil
}

// predictRange evaluates rows [start, end) of d on one interpreter borrowed
// from the pool, loading the feature buffer once and reusing it in place for
// every row in the range.
func (p *Predictor) predictRange(out []float64, d *DMatrix, start, end int) error {
	vm, err := p.pool.Get(p.ctx)
	if err != nil {
		return err
	}
	defer p.pool.Put(vm)

	buf, err := p.loadFeatureBuf(vm)
	if err != nil {
		return err
	}
	for i := start; i < end; i++ {
		margin, err := p.margin(vm, buf, d.rows[i])
		if err != nil {
			return fmt.Errorf("row %d: %w", i, err)
		}
		out[i] = p.transform(margin)
	}
	return nil
}

// loadFeatureBuf gives vm a private feature buffer decoupled from the
// interpreter-shared backing array Program.Constants seeds every pooled
// interpreter's heap slot with. Interpreter.New copies the *same* constant
// slice into every VM's heap (interp/interp.go New, default constant case:
// i.keep(v) keeps v itself, not a copy), so reading that slot with Load and
// mutating it in place would race across VMs pulled from the same pool.
// Storing a freshly allocated buffer once replaces the heap slot with a
// slice this VM alone owns; every row after that can mutate it in place.
// The buffer is installed once per VM and cached: Pool.Put's Reset preserves
// base heap slots, so it stays installed across Get/Put cycles.
func (p *Predictor) loadFeatureBuf(vm *interp.Interpreter) (types.TypedArray[float64], error) {
	p.mu.Lock()
	buf, ok := p.bufs[vm]
	p.mu.Unlock()
	if ok {
		return buf, nil
	}

	ref, err := featureRef(vm, p.cm.FeatureIdx)
	if err != nil {
		return nil, err
	}
	buf = make(types.TypedArray[float64], p.cm.NumFeature)
	if err := vm.Store(ref, buf); err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.bufs[vm] = buf
	p.mu.Unlock()
	return buf, nil
}

// predictBatchRange evaluates rows [start, end) of d on one interpreter
// borrowed from the batch pool, processing cm.BatchCapacity rows per Run
// instead of one. This amortizes the Reset/Run/read-output host round trip
// across a whole chunk of rows.
func (p *Predictor) predictBatchRange(out []float64, d *DMatrix, start, end int) error {
	vm, err := p.batchPool.Get(p.ctx)
	if err != nil {
		return err
	}
	defer p.batchPool.Put(vm)

	buf, err := p.loadBatchBuf(vm)
	if err != nil {
		return err
	}
	capacity := p.cm.BatchCapacity
	for s := start; s < end; s += capacity {
		e := s + capacity
		if e > end {
			e = end
		}
		n := e - s
		fillBatchInput(buf.in, d.rows[s:e], p.cm.NumFeature)

		vm.Reset()
		if err := vm.Run(p.ctx); err != nil {
			return fmt.Errorf("rows %d-%d: %w", s, e, err)
		}
		for i := 0; i < n; i++ {
			out[s+i] = p.transform(buf.out[i])
		}
	}
	return nil
}

// loadBatchBuf gives vm private input/output arrays for the batch program,
// decoupled from the shared constant backing array the same way
// loadFeatureBuf decouples the single-row feature buffer.
func (p *Predictor) loadBatchBuf(vm *interp.Interpreter) (batchBuf, error) {
	p.batchMu.Lock()
	buf, ok := p.batchBufs[vm]
	p.batchMu.Unlock()
	if ok {
		return buf, nil
	}

	inRef, outRef, err := batchRefs(vm, p.cm)
	if err != nil {
		return batchBuf{}, err
	}
	buf = batchBuf{
		in:  make(types.TypedArray[float64], p.cm.BatchCapacity*p.cm.NumFeature),
		out: make(types.TypedArray[float64], p.cm.BatchCapacity),
	}
	if err := vm.Store(inRef, buf.in); err != nil {
		return batchBuf{}, err
	}
	if err := vm.Store(outRef, buf.out); err != nil {
		return batchBuf{}, err
	}
	p.batchMu.Lock()
	p.batchBufs[vm] = buf
	p.batchMu.Unlock()
	return buf, nil
}

// fillBatchInput flattens rows into in row-major, zero-padding any rows past
// len(rows) up to the batch program's fixed capacity so it always evaluates
// a full loop of well-defined input.
func fillBatchInput(in types.TypedArray[float64], rows [][]float64, numFeature int) {
	for i, row := range rows {
		dst := in[i*numFeature : (i+1)*numFeature]
		n := copy(dst, row)
		clear(dst[n:])
	}
	clear(in[len(rows)*numFeature:])
}

func batchRefs(vm *interp.Interpreter, cm *compile.CompiledModel) (int, int, error) {
	inRef, err := featureRef(vm, cm.BatchInputIdx)
	if err != nil {
		return 0, 0, err
	}
	outRef, err := featureRef(vm, cm.BatchOutputIdx)
	if err != nil {
		return 0, 0, err
	}
	return inRef, outRef, nil
}

// margin copies row into buf in place, runs the bytecode for that row, and
// returns the raw margin sum. Reset rewinds the VM's instruction pointer and
// stack for the next row; it does not touch buf, since Store above keeps the
// feature heap slot outside the dynamic (function-binding) tracking Reset
// sweeps.
func (p *Predictor) margin(vm *interp.Interpreter, buf types.TypedArray[float64], row []float64) (float64, error) {
	vm.Reset()
	copyRow(buf, row)

	if err := vm.Run(p.ctx); err != nil {
		return 0, err
	}
	b, err := vm.PopBoxed()
	if err != nil {
		return 0, err
	}
	return b.F64(), nil
}

func featureRef(vm *interp.Interpreter, featureIdx int) (int, error) {
	v, err := vm.Const(featureIdx)
	if err != nil {
		return 0, err
	}
	if v.Kind() != types.KindRef {
		return 0, fmt.Errorf("feature constant type %s", v.Kind())
	}
	return v.Ref(), nil
}

func copyRow(dst types.TypedArray[float64], src []float64) {
	n := copy(dst, src)
	clear(dst[n:])
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

type rowError struct {
	idx int
	err error
}
