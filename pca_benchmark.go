package main

import (
     "math/rand"
     "time"
     "gonum.org/v1/gonum/mat"
     "gonum.org/v1/gonum/stat"
)

// Generate a dataset with random values.
 func generateDataset(nSamples,nFeatures int)*mat.Dense{
     data:=mat.NewDense(nSamples,nFeatures,nil )
     for i:=0;i<nSamples;i++{
         for j:=0;j<nFeatures;j++{
             data.Set(i,j,rand.Float64())
         }
     }
     return data 
 }

func performPCA(data *mat.Dense, k int) *mat.Dense {
    var pc stat.PC
    ok := pc.PrincipalComponents(data, nil)
    if !ok {
        panic("PCA failed")
    }

    _, cols := data.Dims()
    var vectors mat.Dense
    pc.VectorsTo(&vectors)
    
    var proj mat.Dense
    proj.Mul(data, vectors.Slice(0, cols, 0, k))
    return &proj
}



// Run PCA benchmark using threading.

func runPCABenchmark(datasetConfigs [][2]int, threadCounts []int) map[[2]int]map[string][]float64 {
    results := make(map[[2]int]map[string][]float64)

    for _, config := range datasetConfigs {
        nSamples, nFeatures := config[0], config[1]
        data := generateDataset(nSamples, nFeatures)

        executionTimes := make([]float64, len(threadCounts))
        speedups := make([]float64, len(threadCounts))

        for i, threads := range threadCounts {
            start := time.Now()
            performPCA(data, threads)
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



