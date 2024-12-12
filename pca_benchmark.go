package main

import (
	"math/rand"
	"time"
	"github.com/sjwhitworth/golearn/pca"
	"github.com/sjwhitworth/golearn/base"
	"gonum.org/v1/gonum/mat"
)

func generateDataset(nSamples, nFeatures int) base.FixedDataGrid {
	data := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			data.Set(i, j, rand.Float64())
		}
	}
	
	// Create attributes
	attributes := make([]base.Attribute, nFeatures)
	for i := range attributes {
		attributes[i] = base.NewFloatAttribute(fmt.Sprintf("attr%d", i))
	}

	// Create FixedDataGrid
	instances := base.NewDenseInstances()
	instances.SetAttributes(attributes)
	for i := 0; i < nSamples; i++ {
		row := data.RawRowView(i)
		instances.AddClassifierAttribute(attributes[0])
		instances.AddSampleFromRow(row)
	}

	return instances
}

func runPCABenchmark(datasetConfigs [][2]int, threadCounts []int) map[[2]int]map[string][]float64 {
	results := make(map[[2]int]map[string][]float64)

	for _, config := range datasetConfigs {
		nSamples, nFeatures := config[0], config[1]
		rawData := generateDataset(nSamples, nFeatures)

		executionTimes := make([]float64, len(threadCounts))
		speedups := make([]float64, len(threadCounts))

		for i, threads := range threadCounts {
			start := time.Now()
			pca := pca.NewPCA(threads)
			pca.Fit(rawData)
			executionTime := time.Since(start).Seconds()
			executionTimes[i] = executionTime

			if i == 0 {
				speedups[i] = 1.0
			} else {
				speedups[i] = executionTimes[0] / executionTime
			}
		}

		results[config] = map[string][]float64{
			"execution_times": executionTimes,
			"speedups":        speedups,
		}
	}

	return results
}
