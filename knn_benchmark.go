package main

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

type Point struct {
	features []float64
	label    int
}

// Function to compute Euclidean distance between two points
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// KNN classification function
func knnClassify(trainData []Point, testPoint Point, k int) int {
	distances := make([]struct {
		distance float64
		index    int
	}, len(trainData))

	var wg sync.WaitGroup

	for i := range trainData {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			distances[i] = struct {
				distance float64
				index    int
			}{euclideanDistance(trainData[i].features, testPoint.features), i}
		}(i)
	}
	wg.Wait()

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	labelCounts := make(map[int]int)
	for i := 0; i < k; i++ {
		label := trainData[distances[i].index].label
		labelCounts[label]++
	}

	maxCount := 0
	maxLabel := -1
	for label, count := range labelCounts {
		if count > maxCount {
			maxCount = count
			maxLabel = label
		}
	}

	return maxLabel
}

// Benchmarking function for KNN
func benchmarkKNN(trainData, testData []Point, k int) time.Duration {
	start := time.Now()
	for _, testPoint := range testData {
		knnClassify(trainData, testPoint, k)
	}
	return time.Since(start)
}
