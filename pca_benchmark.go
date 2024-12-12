package main

import (
    "math"
    "math/rand"
    "fmt"
    "time"
    "github.com/sjwhitworth/golearn/pca"
    "github.com/sjwhitworth/golearn/base"
    "gonum.org/v1/gonum/mat"
)

func generateDataset(nSamples, nFeatures int) *mat.Dense {
    data := mat.NewDense(nSamples, nFeatures, nil)
    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            data.Set(i, j, rand.Float64())
        }
    }
    return data
}

func runPCABenchmark(datasetConfigs [][2]int, threadCounts []int) map[[2]int]map[string][]float64 {
    results := make(map[[2]int]map[string][]float64)

    for _, config := range datasetConfigs {
        nSamples, nFeatures := config[0], config[1]
        fmt.Printf("PCA: Processing dataset with %d samples and %d features...\n", nSamples, nFeatures)

        data := generateDataset(nSamples, nFeatures)
        rawData := base.NewDenseInstances(data, nil)

        executionTimes := make([]float64, len(threadCounts))
        speedups := make([]float64, len(threadCounts))

        for i, threads := range threadCounts {
            start := time.Now()
            pca.NewPCA(threads).Fit(rawData)
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
