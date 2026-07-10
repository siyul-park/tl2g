# tl2g

Compile [LightGBM](https://github.com/microsoft/LightGBM) `.mdl` models into
[minivm](https://github.com/siyul-park/minivm) bytecode and run prediction in
pure Go. The API mirrors [tl2cgen](https://tl2cgen.readthedocs.io/), but emits
portable bytecode for the minivm interpreter/JIT instead of C.

> **Status: POC.** Proves accuracy parity with LightGBM and measures throughput.

## How it works

```
.mdl text ──parse──► model.Model ──compile──► minivm program (bytecode)
                                                     │
                           DMatrix rows ──► Predictor.Predict ──► []float64
```

Each decision tree is lowered to a nested branch block: a feature compare
(`ARRAY_GET` + `F64_LE`) drives a `BR_IF`, and each leaf adds its `leaf_value`
to a margin accumulator carried on the operand stack. Each interpreter receives
its own feature-row buffer once and mutates that buffer in place between runs.
Binary objectives apply the sigmoid
transform in Go after the bytecode returns the raw margin.

## Usage

```go
p, _ := tl2g.LoadPredictor("model.mdl") // load + compile + wrap
defer p.Close()

out, _ := p.Predict(tl2g.NewDMatrix(X)) // X is [][]float64, row-major
```

Or step by step: `tl2g.Load` → `tl2g.Compile` → `tl2g.NewPredictor`.
Batch prediction uses an `interp.Pool`; set parallelism with
`tl2g.WithPoolSize(n)`. The default is `runtime.GOMAXPROCS(0)`. Predictors
enable minivm native JIT by default with an aggressive threshold. Batch traces
that exceed the safe register budget fall back to threaded dispatch instead of
spilling across their terminal `ARRAY_SET`. Use `tl2g.WithJITThreshold(n)` to
tune hotness, or `-1` to disable JIT.

## Scope (POC)

| Supported | Not yet |
|---|---|
| `num_class=1` | multiclass / multi-output |
| numerical splits (`<=`) | categorical splits |
| `regression`, `binary` | other objectives' transforms |

Categorical splits (`decision_type` bit 0, or `num_cat > 0`) are rejected with a
clear error. A `Predictor` owns a bounded interpreter pool. One `Predict` call
may evaluate rows in parallel; do not call `Predict` concurrently on the same
Predictor.

## Accuracy

Validated against LightGBM's own `predict` over 200 held-out rows
(`go test ./...`):

| Model | max abs diff vs LightGBM |
|---|---|
| regression | `0` (bit-exact) |
| binary | `2.8e-17` (machine epsilon) |

## Throughput

`go test -bench 'BenchmarkPredict(Single|Batch)' -benchmem ./...` — Apple M4
Pro, Go 1.26.2, 30-tree models, pure Go (no cgo):

| Benchmark | ns/op | allocs/op |
|---|---|---|
| single row, regression (10 feat) | ~1,590 | 1 |
| single row, binary (12 feat) | ~959 | 1 |
| batch 200 rows, regression | ~6,000 / row | 12 per batch |
| batch 200 rows, binary | ~5,150 / row | 12 per batch |
| batch 10,000 rows, regression | ~3,280 / row | 28-29 per batch |
| batch 10,000 rows, binary | ~2,890 / row | 28 per batch |

### vs LightGBM and TL2cgen

Batch prediction, ns/row, same fixtures (`python3 scripts/bench_compare.py`):

| Model | LightGBM (C) | TL2cgen (C) | **tl2g (Go bytecode)** |
|---|---|---|---|
| regression | not measured locally (`numpy` missing) | not measured locally | ~3,280 |
| binary | not measured locally (`numpy` missing) | not measured locally | ~2,890 |

### On the optimizer and the JIT

The emitted bytecode uses minivm optimizer **O3** by default
(`compile.WithOptLevel` overrides it). `TestCompiledModelJITEmits` verifies
native trace emission directly, and `TestPredictorJITEmits` verifies predictor
emission. The batch pool enables JIT, while minivm safely declines spilling for
high-pressure terminal-mutation traces and continues those paths in threaded
dispatch. Each VM run still amortizes host overhead over 32 rows.

## Regenerating fixtures

`testdata/*.mdl` and `*_golden.json` are committed. To rebuild them you need
Python with `lightgbm numpy scikit-learn` (and `libomp` on macOS):

```bash
python scripts/gen_model.py
```

The golden JSON stores `{X, pred, raw, objective}`, where `pred` is LightGBM's
own output — the accuracy oracle.

## Requirements

- Go 1.26.2+ (minivm requirement)
