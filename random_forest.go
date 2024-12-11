package main

import (
	"math/rand"
	"sort"
	"sync"
)

type DecisionTree struct {
	feature    int
	threshold  float64
	left       *DecisionTree
	right      *DecisionTree
	prediction int
}

func buildTree(data []Point, maxDepth, minSamples int) *DecisionTree {
	if len(data) < minSamples || maxDepth == 0 {
		prediction := getMajorityLabel(data)
		return &DecisionTree{prediction: prediction}
	}

	bestFeature, bestThreshold := findBestSplit(data)
	leftData, rightData := splitData(data, bestFeature, bestThreshold)

	if len(leftData) == 0 || len(rightData) == 0 {
		prediction := getMajorityLabel(data)
		return &DecisionTree{prediction: prediction}
	}

	leftSubtree := buildTree(leftData, maxDepth-1, minSamples)
	rightSubtree := buildTree(rightData, maxDepth-1, minSamples)

	return &DecisionTree{
		feature:   bestFeature,
		threshold: bestThreshold,
		left:      leftSubtree,
		right:     rightSubtree,
	}
}

func findBestSplit(data []Point) (int, float64) {
	// Implement Gini impurity-based split
	// This is a simplified version
	bestFeature, bestThreshold := 0, 0.0
	bestGini := math.Inf(1)

	for feature := 0; feature < len(data[0].features); feature++ {
		values := make([]float64, len(data))
		for i, point := range data {
			values[i] = point.features[feature]
		}
		sort.Float64s(values)

		for i := 1; i < len(values); i++ {
			threshold := (values[i-1] + values[i]) / 2
			gini := calculateGiniImpurity(data, feature, threshold)
			if gini < bestGini {
				bestGini = gini
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold
}

func calculateGiniImpurity(data []Point, feature int, threshold float64) float64 {
	// Implement Gini impurity calculation
	// This is a placeholder implementation
	return rand.Float64()
}

func splitData(data []Point, feature int, threshold float64) ([]Point, []Point) {
	var left, right []Point
	for _, point := range data {
		if point.features[feature] <= threshold {
			left = append(left, point)
		} else {
			right = append(right, point)
		}
	}
	return left, right
}

func getMajorityLabel(data []Point) int {
	labelCounts := make(map[int]int)
	for _, point := range data {
		labelCounts[point.label]++
	}
	maxCount, maxLabel := 0, 0
	for label, count := range labelCounts {
		if count > maxCount {
			maxCount = count
			maxLabel = label
		}
	}
	return maxLabel
}

type RandomForest struct {
	trees []*DecisionTree
}

func (rf *RandomForest) Train(data []Point, numTrees, maxDepth, minSamples int) {
	var wg sync.WaitGroup
	rf.trees = make([]*DecisionTree, numTrees)

	for i := 0; i < numTrees; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			bootstrapData := bootstrapSample(data)
			rf.trees[index] = buildTree(bootstrapData, maxDepth, minSamples)
		}(i)
	}
	wg.Wait()
}

func (rf *RandomForest) Predict(point Point) int {
	predictions := make([]int, len(rf.trees))
	var wg sync.WaitGroup

	for i, tree := range rf.trees {
		wg.Add(1)
		go func(index int, t *DecisionTree) {
			defer wg.Done()
			predictions[index] = predictTree(t, point)
		}(i, tree)
	}
	wg.Wait()

	return getMajorityVote(predictions)
}

func predictTree(tree *DecisionTree, point Point) int {
	if tree.left == nil && tree.right == nil {
		return tree.prediction
	}
	if point.features[tree.feature] <= tree.threshold {
		return predictTree(tree.left, point)
	}
	return predictTree(tree.right, point)
}

func bootstrapSample(data []Point) []Point {
	sample := make([]Point, len(data))
	for i := range sample {
		sample[i] = data[rand.Intn(len(data))]
	}
	return sample
}

func getMajorityVote(predictions []int) int {
	counts := make(map[int]int)
	for _, pred := range predictions {
		counts[pred]++
	}
	maxCount, maxLabel := 0, 0
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			maxLabel = label
		}
	}
	return maxLabel
}
