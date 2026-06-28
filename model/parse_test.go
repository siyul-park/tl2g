package model

import (
	"strings"
	"testing"
)

const miniNumerical = `tree
version=v4
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=binary sigmoid:2

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
threshold=0.5 1.5
decision_type=2 2
left_child=-1 -3
right_child=1 -2
leaf_value=10 20 30
shrinkage=1

end of trees
`

func TestParseNumerical(t *testing.T) {
	// Arrange + Act
	m, err := Parse(strings.NewReader(miniNumerical))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	// Assert
	if m.NumFeature() != 2 {
		t.Errorf("NumFeature = %d, want 2", m.NumFeature())
	}
	if m.Objective.Name != "binary" || m.Objective.Sigmoid != 2 {
		t.Errorf("objective = %+v, want {binary 2}", m.Objective)
	}
	if len(m.Trees) != 1 {
		t.Fatalf("trees = %d, want 1", len(m.Trees))
	}
	tr := m.Trees[0]
	if len(tr.SplitFeature) != 2 || len(tr.LeafValue) != 3 {
		t.Fatalf("node arrays wrong: split=%d leaf=%d", len(tr.SplitFeature), len(tr.LeafValue))
	}
	// left_child[0] = -1 => leaf ^(-1) = 0
	if !IsLeaf(tr.LeftChild[0]) || LeafIndex(tr.LeftChild[0]) != 0 {
		t.Errorf("left_child[0]=%d -> leaf %d, want leaf 0", tr.LeftChild[0], LeafIndex(tr.LeftChild[0]))
	}
	// right_child[0] = 1 => internal node 1
	if IsLeaf(tr.RightChild[0]) {
		t.Errorf("right_child[0]=%d, want internal node", tr.RightChild[0])
	}
	// right_child[1] = -2 => leaf ^(-2) = 1
	if !IsLeaf(tr.RightChild[1]) || LeafIndex(tr.RightChild[1]) != 1 {
		t.Errorf("right_child[1]=%d -> leaf %d, want leaf 1", tr.RightChild[1], LeafIndex(tr.RightChild[1]))
	}
}

func TestParseRejectsCategorical(t *testing.T) {
	// decision_type with bit 0 set => categorical split
	src := strings.Replace(miniNumerical, "decision_type=2 2", "decision_type=1 2", 1)
	_, err := Parse(strings.NewReader(src))
	if err == nil {
		t.Fatal("expected error for categorical split, got nil")
	}
	if !strings.Contains(err.Error(), "categorical") {
		t.Errorf("error = %q, want mention of categorical", err)
	}
}

func TestParseRejectsNumCat(t *testing.T) {
	src := strings.Replace(miniNumerical, "num_cat=0", "num_cat=1", 1)
	_, err := Parse(strings.NewReader(src))
	if err == nil || !strings.Contains(err.Error(), "categorical") {
		t.Fatalf("expected categorical error, got %v", err)
	}
}
