// Package model holds the in-memory representation of a LightGBM gradient
// boosted tree model parsed from its `.mdl` text format, plus the parser.
//
// Scope (POC): numerical splits only; objectives regression and binary.
package model

// Objective describes the LightGBM training objective and the post-processing
// applied to the raw margin sum.
type Objective struct {
	// Name is the objective family, e.g. "regression" or "binary".
	Name string
	// Sigmoid is the binary sigmoid coefficient (alpha) from "sigmoid:<a>";
	// 1.0 by default. Unused for regression.
	Sigmoid float64
}

// Tree is a single decision tree. Internal nodes are indexed 0..len-1 in the
// per-node slices; a child value < 0 denotes leaf index ^child (bitwise NOT)
// into LeafValue, matching LightGBM's text encoding.
type Tree struct {
	SplitFeature []int     // feature index tested at each internal node
	Threshold    []float64 // split threshold; feature <= threshold => left
	DecisionType []int     // LightGBM decision flags; bit0 set => categorical
	LeftChild    []int     // left child (node idx >=0, or leaf ^child <0)
	RightChild   []int     // right child (node idx >=0, or leaf ^child <0)
	LeafValue    []float64 // raw leaf output, indexed by leaf id
	Shrinkage    float64   // per-tree contribution multiplier
}

// Model is a parsed LightGBM ensemble.
type Model struct {
	NumClass      int
	MaxFeatureIdx int // highest feature index used; NumFeature = MaxFeatureIdx+1
	Objective     Objective
	Trees         []Tree
}

// NumFeature returns the dense feature-vector width the model expects.
func (m *Model) NumFeature() int { return m.MaxFeatureIdx + 1 }

// IsLeaf reports whether a child reference points to a leaf.
func IsLeaf(child int) bool { return child < 0 }

// LeafIndex converts a negative child reference to its leaf id.
func LeafIndex(child int) int { return ^child }
