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

`go test -bench . -benchmem` — Apple M4 Pro, Go 1.26.2, 30-tree models,
pure Go (no cgo):

| Benchmark | ns/op | allocs/op |
|---|---|---|
| single row, regression (10 feat) | ~2,600 | 2 |
| single row, binary (12 feat) | ~1,700 | 2 |
| batch 200 rows, regression | ~3,600 / row | — |
| batch 200 rows, binary | ~2,800 / row | — |

### vs LightGBM and TL2cgen

Batch prediction, ns/row, same fixtures (`python scripts/bench_compare.py`):

| Model | LightGBM (C) | TL2cgen (C) | **tl2g (Go bytecode)** |
|---|---|---|---|
| regression | ~282 | ~310 | ~3,640 |
| binary | ~384 | ~354 | ~2,846 |

tl2g currently trails the native-C runtimes by ~9–13×, as expected for a pure-Go
bytecode VM with no cgo. The POC goal — bit-exact accuracy with a tl2cgen-style
API and a portable, dependency-free Go artifact — is met.

### On the optimizer and the JIT

The emitted bytecode is run through the minivm optimizer at **O1** (cheap local
rewrites). O2/O3 add CFG passes that currently reject these deeply branched tree
programs with an `invalid jump` error, so O1 is the safe default
(`compile.WithOptLevel` overrides it).

The tree-walk is emitted into a minivm **function** that the top-level code
`CALL`s, specifically so it is **JIT-eligible** — minivm only traces code
reached through a call (frame address > 0), never top-level module code. With
that change the trace JIT does engage, but it does **not** speed up tree
inference: measured pure-interpreter vs aggressive-JIT is ~3,740 vs ~3,830 ns/row
(JIT marginally *slower*). Decision-tree evaluation is the canonical worst case
for a trace JIT — every input row walks a different root-to-leaf path, so a
recorded trace's branch guards fail on most rows and force constant side-exit
deopts. Closing the gap needs a tree-shaped strategy (e.g. compiling each split
to native straight-line code, batching the row loop into bytecode), not the
trace JIT. `predict` keeps the JIT at its default threshold; `interp.WithThreshold(-1)`
would pin a pure interpreter.

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
- Prediction runs on minivm's threaded interpreter. The trace JIT is eligible
  (the eval body is a called function) but does not benefit tree inference — see
  [On the optimizer and the JIT](#on-the-optimizer-and-the-jit).
