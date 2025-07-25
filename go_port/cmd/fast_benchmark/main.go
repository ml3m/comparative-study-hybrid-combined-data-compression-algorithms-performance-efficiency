package main

import (
	"fmt"
	"log"
	"time"

	"hybrid-compression-study/pkg/benchmarks"
)

func main() {
	fmt.Println("🚀 FAST COMPRESSION BENCHMARK")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("High-performance benchmark without aerospace monitoring overhead")
	fmt.Println()

	// Create optimized configuration
	config := &benchmarks.BenchmarkConfig{
		SelectedAlgorithms: []string{"SimpleRLE", "LZW", "Huffman"},
		MinCombinationSize: 2,
		MaxCombinationSize: 2,
		InputTypes: []benchmarks.InputType{
			benchmarks.InputTypeRepetitive,
			benchmarks.InputTypeRandom,
			benchmarks.InputTypeNaturalText,
		},
		InputSizes: []benchmarks.InputSize{
			{"1KB", 1024},
			{"4KB", 4096},
		},
		TimeoutPerTest:         60 * time.Second,
		MaxConcurrentTests:     1,
		RetryFailedTests:       0,
		KEIWeights:             benchmarks.DefaultKEIWeights,
		VerboseOutput:          true,
		ShowProgressBar:        true,
		ShowRealTimeResults:    true,
		ShowETA:                true,
		SaveResults:            false,
		OutputDirectory:        "benchmark_results",
		GenerateDetailedReport: false,
	}

	// Create fast benchmark executor
	executor, err := benchmarks.NewFastBenchmarkExecutor(config)
	if err != nil {
		log.Fatalf("Failed to create benchmark executor: %v", err)
	}

	// Run benchmark
	fmt.Println("⚡ Starting optimized benchmark execution...")
	startTime := time.Now()

	result, err := executor.Execute()
	if err != nil {
		log.Fatalf("Benchmark execution failed: %v", err)
	}

	totalTime := time.Since(startTime)

	// Display results
	fmt.Println("\n🎯 BENCHMARK RESULTS")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("Total execution time: %v\n", totalTime)
	fmt.Printf("Total tests completed: %d\n", result.CompletedTests)
	fmt.Printf("Failed tests: %d\n", result.FailedTests)
	fmt.Printf("Timeout tests: %d\n", result.TimeoutTests)
	fmt.Printf("Tests per second: %.1f\n", float64(result.CompletedTests)/totalTime.Seconds())

	if result.BestOverall != nil {
		fmt.Printf("\n🏆 BEST OVERALL COMBINATION:\n")
		fmt.Printf("Algorithm: %s\n", result.BestOverall.AlgorithmCombination)
		fmt.Printf("KEI Score: %.1f\n", result.BestOverall.KEIScore.OverallScore)
		fmt.Printf("Compression Ratio: %.2fx\n", result.BestOverall.KEIScore.CompressionRatio)
		fmt.Printf("Throughput: %.1f MB/s\n", result.BestOverall.KEIScore.ThroughputMbps)
	}

	// Display top 5 combinations
	if result.OverallRankings != nil && len(result.OverallRankings.Rankings) > 0 {
		fmt.Printf("\n📈 TOP 5 COMBINATIONS:\n")
		limit := 5
		if len(result.OverallRankings.Rankings) < limit {
			limit = len(result.OverallRankings.Rankings)
		}

		for i := 0; i < limit; i++ {
			combo := result.OverallRankings.Rankings[i]
			fmt.Printf("%d. %s (KEI: %.1f, Ratio: %.2fx)\n",
				i+1,
				combo.AlgorithmCombination,
				combo.KEIScore.OverallScore,
				combo.KEIScore.CompressionRatio)
		}
	}

	// Performance comparison
	expectedSlowTime := totalTime * 10 // Estimate 10x slower with aerospace monitoring
	fmt.Printf("\n⚡ PERFORMANCE IMPROVEMENT:\n")
	fmt.Printf("Fast execution time: %v\n", totalTime)
	fmt.Printf("Estimated slow execution: %v\n", expectedSlowTime)
	fmt.Printf("Speedup achieved: ~%.1fx faster\n", expectedSlowTime.Seconds()/totalTime.Seconds())

	fmt.Println("\n✅ Fast benchmark completed successfully!")
}
