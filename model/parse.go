package model

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Load reads and parses a LightGBM `.mdl` text model from path.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model: open %q: %w", path, err)
	}
	defer f.Close()
	return Parse(f)
}

// Parse reads a LightGBM text model from r.
//
// The format is line-oriented `key=value`; list values are space-separated.
// A header block precedes one or more `Tree=N` blocks, terminated by
// "end of trees". Only numerical splits are supported; categorical splits
// (decision_type bit 0 set, or num_cat > 0) produce an error.
func Parse(r io.Reader) (*Model, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	m := &Model{NumClass: 1}
	var (
		inTree  bool
		cur     map[string]string
		flushed bool
	)

	flush := func() error {
		if cur == nil {
			return nil
		}
		t, err := parseTree(cur)
		if err != nil {
			return err
		}
		m.Trees = append(m.Trees, *t)
		cur = nil
		return nil
	}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "end of trees" {
			if err := flush(); err != nil {
				return nil, err
			}
			flushed = true
			break
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue // skip non key=value lines (e.g. "parameters:")
		}

		if key == "Tree" {
			if err := flush(); err != nil {
				return nil, err
			}
			inTree = true
			cur = map[string]string{}
			continue
		}

		if inTree {
			cur[key] = val
			continue
		}

		switch key {
		case "num_class":
			m.NumClass, _ = strconv.Atoi(val)
		case "max_feature_idx":
			m.MaxFeatureIdx, _ = strconv.Atoi(val)
		case "objective":
			m.Objective = parseObjective(val)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("model: read: %w", err)
	}
	if !flushed {
		if err := flush(); err != nil {
			return nil, err
		}
	}
	if len(m.Trees) == 0 {
		return nil, fmt.Errorf("model: no trees parsed")
	}
	if m.NumClass != 1 {
		return nil, fmt.Errorf("model: num_class=%d unsupported (POC: num_class=1 only)", m.NumClass)
	}
	return m, nil
}

// parseObjective splits a LightGBM objective line such as "binary sigmoid:1"
// into a name and optional sigmoid coefficient.
func parseObjective(val string) Objective {
	fields := strings.Fields(val)
	obj := Objective{Sigmoid: 1.0}
	if len(fields) > 0 {
		obj.Name = fields[0]
	}
	for _, f := range fields[1:] {
		if k, v, ok := strings.Cut(f, ":"); ok && k == "sigmoid" {
			if a, err := strconv.ParseFloat(v, 64); err == nil {
				obj.Sigmoid = a
			}
		}
	}
	return obj
}

func parseTree(kv map[string]string) (*Tree, error) {
	if nc := kv["num_cat"]; nc != "" && nc != "0" {
		return nil, fmt.Errorf("model: categorical splits unsupported (num_cat=%s)", nc)
	}

	dt, err := parseInts(kv["decision_type"])
	if err != nil {
		return nil, fmt.Errorf("model: decision_type: %w", err)
	}
	for i, d := range dt {
		if d&1 == 1 {
			return nil, fmt.Errorf("model: categorical split at node %d (decision_type=%d) unsupported", i, d)
		}
	}

	sf, err := parseInts(kv["split_feature"])
	if err != nil {
		return nil, fmt.Errorf("model: split_feature: %w", err)
	}
	th, err := parseFloats(kv["threshold"])
	if err != nil {
		return nil, fmt.Errorf("model: threshold: %w", err)
	}
	lc, err := parseInts(kv["left_child"])
	if err != nil {
		return nil, fmt.Errorf("model: left_child: %w", err)
	}
	rc, err := parseInts(kv["right_child"])
	if err != nil {
		return nil, fmt.Errorf("model: right_child: %w", err)
	}
	lv, err := parseFloats(kv["leaf_value"])
	if err != nil {
		return nil, fmt.Errorf("model: leaf_value: %w", err)
	}

	shrink := 1.0
	if s := kv["shrinkage"]; s != "" {
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			shrink = v
		}
	}

	if len(sf) != len(th) || len(sf) != len(lc) || len(sf) != len(rc) {
		return nil, fmt.Errorf("model: inconsistent node array lengths")
	}

	return &Tree{
		SplitFeature: sf,
		Threshold:    th,
		DecisionType: dt,
		LeftChild:    lc,
		RightChild:   rc,
		LeafValue:    lv,
		Shrinkage:    shrink,
	}, nil
}

func parseInts(s string) ([]int, error) {
	if s == "" {
		return nil, nil
	}
	fields := strings.Fields(s)
	out := make([]int, len(fields))
	for i, f := range fields {
		v, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", f, err)
		}
		out[i] = v
	}
	return out, nil
}

func parseFloats(s string) ([]float64, error) {
	if s == "" {
		return nil, nil
	}
	fields := strings.Fields(s)
	out := make([]float64, len(fields))
	for i, f := range fields {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", f, err)
		}
		out[i] = v
	}
	return out, nil
}
