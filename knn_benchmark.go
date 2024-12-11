package main

import (
	"math"
	"sort"
	"sync"
)

type Point struct {
	features []float64
	label    int
}

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func knnClassify(trainData []Point, testPoint Point, k int, numGoroutines int) int {
	distances := make([]float64, len(trainData))
	var wg sync.WaitGroup
	chunkSize := len(trainData) / numGoroutines

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				distances[j] = euclideanDistance(trainData[j].features, testPoint.features)
			}
		}(i*chunkSize, (i+1)*chunkSize)
	}
	wg.Wait()

	type distanceIndex struct {
		distance float64
		index    int
	}
	distanceIndices := make([]distanceIndex, len(distances))
	for i, d := range distances {
		distanceIndices[i] = distanceIndex{d, i}
	}

	sort.Slice(distanceIndices, func(i, j int) bool {
		return distanceIndices[i].distance < distanceIndices[j].distance
	})

	labelCounts := make(map[int]int)
	for i := 0; i < k; i++ {
		label := trainData[distanceIndices[i].index].label
		labelCounts[label]++
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
