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
to a margin accumulator carried on the operand stack. The feature row is a
single mutable `F64` array constant the predictor rewrites per row (no
per-prediction allocation of the input). Binary objectives apply the sigmoid
transform in Go after the bytecode returns the raw margin.

## Usage

```go
p, _ := tl2g.LoadPredictor("model.mdl") // load + compile + wrap
defer p.Close()

out, _ := p.Predict(tl2g.NewDMatrix(X)) // X is [][]float64, row-major
```

Or step by step: `tl2g.Load` → `tl2g.Compile` → `tl2g.NewPredictor`.

## Scope (POC)

| Supported | Not yet |
|---|---|
| `num_class=1` | multiclass / multi-output |
| numerical splits (`<=`) | categorical splits |
| `regression`, `binary` | other objectives' transforms |

Categorical splits (`decision_type` bit 0, or `num_cat > 0`) are rejected with a
clear error. A `Predictor` is single-threaded (shared feature buffer).

## Accuracy

Validated against LightGBM's own `predict` over 200 held-out rows
(`go test ./...`):

| Model | max abs diff vs LightGBM |
|---|---|
| regression | `0` (bit-exact) |
| binary | `2.8e-17` (machine epsilon) |

## Throughput

`go test -bench . -benchmem ./...` — Apple M4 Pro, Go 1.26.2, 30-tree models,
pure Go (no cgo):

| Benchmark | ns/op | allocs/op |
|---|---|---|
| single row, regression (10 feat) | ~2,617 | 2 |
| single row, binary (12 feat) | ~1,733 | 2 |
| batch 200 rows, regression | ~3,672 / row | 202 |
| batch 200 rows, binary | ~2,890 / row | 202 |

### vs LightGBM and TL2cgen

Batch prediction, ns/row, same fixtures (`python3 scripts/bench_compare.py`):

| Model | LightGBM (C) | TL2cgen (C) | **tl2g (Go bytecode)** |
|---|---|---|---|
| regression | ~282 | ~310 | ~3,672 |
| binary | ~384 | ~354 | ~2,890 |

### On the optimizer and the JIT

The emitted bytecode uses minivm optimizer **O3** by default
(`compile.WithOptLevel` overrides it). The tree-walk lives in a called function,
so it is JIT-eligible, but branch-heavy tree inference currently relies on the
threaded interpreter path for best throughput.

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
