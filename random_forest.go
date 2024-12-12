package main

import (
    "sync"
    "time"
    "math/rand"
)

type DecisionTree struct {
    prediction int // Placeholder for simplicity; implement tree structure as needed.
}

func buildTree(data []Point) *DecisionTree {
    prediction := getMajorityLabel(data)
    return &DecisionTree{prediction: prediction}
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

func (rf *RandomForest) Train(data []Point, numTrees int) {
    var wg sync.WaitGroup
    rf.trees = make([]*DecisionTree, numTrees)

    for i := 0; i < numTrees; i++ {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            rf.trees[index] = buildTree(bootstrapSample(data))
        }(i)
    }
    wg.Wait()
}

func bootstrapSample(data []Point) []Point {
    sample := make([]Point, len(data))
    for i := range sample {
        sample[i] = data[rand.Intn(len(data))]
    }
    return sample
}

func (rf *RandomForest) Predict(point Point) int {
    predictions := make([]int, len(rf.trees))
    var wg sync.WaitGroup

    for i, tree := range rf.trees {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            predictions[index] = tree.prediction // Simplified; implement prediction logic.
        }(i)
    }
    wg.Wait()

    return getMajorityVote(predictions)
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

// Benchmarking function for Random Forest
func benchmarkRandomForest(trainData, testData []Point, numTrees int) time.Duration {
	start := time.Now()
	rf := &RandomForest{}
	rf.Train(trainData, numTrees)

	for _, testPoint := range testData {
	    rf.Predict(testPoint)
	}
	
	return time.Since(start)
}
