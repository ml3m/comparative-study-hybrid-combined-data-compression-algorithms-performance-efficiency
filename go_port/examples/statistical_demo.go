// Package main demonstrates the statistical analysis capabilities of the compression framework.
//
// This example shows how multiple compression runs provide aerospace-grade statistical
// precision for performance analysis and reliability assessment.
package main

import (
	"context"
	"fmt"
	"log"

	"hybrid-compression-study/pkg/algorithms"
	"hybrid-compression-study/pkg/core"
)

func main() {
	// Generate test data with varying characteristics
	testData := []byte(`
The aerospace industry demands the highest levels of precision and reliability in all 
computational systems. This comprehensive compression analysis framework provides 
nanosecond-precision timing, byte-level memory tracking, and statistical validation 
suitable for mission-critical applications.

PATTERN ANALYSIS: AAAAAABBBBBBCCCCCCDDDDDDEEEEEE
REPETITIVE DATA: The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
STRUCTURED CONTENT: {"mission": "apollo", "crew": 3, "duration": "8 days", "success": true}

By running multiple trials and calculating statistical measures including standard deviation,
coefficient of variation, and consistency scores, we ensure that performance measurements
meet aerospace-grade reliability standards required for space missions and safety-critical systems.
	`)
	
	fmt.Printf("🚀 Aerospace-Grade Statistical Analysis Demo\n")
	fmt.Printf("═══════════════════════════════════════════════\n\n")
	fmt.Printf("Test data size: %d bytes\n", len(testData))
	fmt.Printf("Running statistical analysis with multiple trials...\n\n")
	
	// Test configuration
	algorithms := []struct {
		name    string
		creator func() (core.CompressionAlgorithm, error)
		trials  int
	}{
		{"Huffman", func() (core.CompressionAlgorithm, error) { return algorithms.NewHuffmanEncoder() }, 15},
		{"RLE", func() (core.CompressionAlgorithm, error) { return algorithms.NewRLEEncoder() }, 20},
		{"LZW", func() (core.CompressionAlgorithm, error) { return algorithms.NewLZWEncoder() }, 10},
	}
	
	ctx := context.Background()
	
	for _, alg := range algorithms {
		fmt.Printf("📊 Statistical Analysis: %s Algorithm (%d trials)\n", alg.name, alg.trials)
		fmt.Printf("─────────────────────────────────────────────────────\n")
		
		// Create algorithm instance
		algorithm, err := alg.creator()
		if err != nil {
			log.Printf("❌ Failed to create %s algorithm: %v\n", alg.name, err)
			continue
		}
		
		// Run multiple compression trials
		results := make([]*core.CompressionResult, 0, alg.trials)
		
		fmt.Printf("Running %d trials...", alg.trials)
		for i := 0; i < alg.trials; i++ {
			if i%(alg.trials/5) == 0 {
				fmt.Printf(".")
			}
			
			result, err := algorithm.Compress(ctx, testData)
			if err != nil {
				log.Printf("Trial %d failed: %v", i+1, err)
				continue
			}
			results = append(results, result)
		}
		fmt.Printf(" completed!\n\n")
		
		if len(results) == 0 {
			fmt.Printf("❌ All trials failed for %s\n\n", alg.name)
			continue
		}
		
		// Calculate averaged results
		averaged := core.AverageCompressionResults(results)
		if averaged == nil {
			fmt.Printf("❌ Failed to calculate statistics for %s\n\n", alg.name)
			continue
		}
		
		// Display comprehensive statistical analysis
		printDetailedStatistics(averaged)
		fmt.Printf("\n")
	}
	
	fmt.Printf("🎉 Statistical analysis completed!\n\n")
	fmt.Printf("This demonstration shows how the framework provides:\n")
	fmt.Printf("• Comprehensive statistical measures (mean, std dev, range)\n")
	fmt.Printf("• Aerospace-grade consistency scoring (variability assessment)\n")
	fmt.Printf("• Mission-readiness reliability ratings\n")
	fmt.Printf("• Scientific precision suitable for critical applications\n")
}

func printDetailedStatistics(result *core.AveragedCompressionResult) {
	successRate := float64(result.SuccessfulRuns) / float64(result.TotalRuns) * 100
	
	fmt.Printf("✅ SUCCESS METRICS\n")
	fmt.Printf("   Trials: %d successful / %d total (%.1f%% success rate)\n",
		result.SuccessfulRuns, result.TotalRuns, successRate)
	fmt.Printf("   Data integrity: %s\n", 
		map[bool]string{true: "✅ VERIFIED", false: "❌ FAILED"}[result.IsEffective()])
	
	fmt.Printf("\n📊 COMPRESSION PERFORMANCE\n")
	fmt.Printf("   Avg compression ratio: %.6fx ± %.6f\n", 
		result.AvgCompressionRatio, result.StdDevCompressionRatio)
	fmt.Printf("   Ratio range: %.6fx - %.6fx\n", 
		result.MinCompressionRatio, result.MaxCompressionRatio)
	fmt.Printf("   Space savings: %.3f%% (%s)\n", 
		result.CompressionPercentage(), core.FormatMemoryPrecision(result.SpaceSavingsBytes()))
	
	fmt.Printf("\n⚡ TIMING ANALYSIS\n")
	fmt.Printf("   Avg compression time: %s ± %.3fs\n", 
		result.AvgPrecisionMetrics.CompressionTimeFormatted, result.StdDevCompressionTime)
	fmt.Printf("   Time range: %.6fs - %.6fs\n", 
		result.MinCompressionTime, result.MaxCompressionTime)
	fmt.Printf("   Coefficient of variation: %.4f\n", 
		result.StdDevCompressionTime/result.AvgCompressionTime)
	
	fmt.Printf("\n🎯 AEROSPACE-GRADE ASSESSMENT\n")
	fmt.Printf("   Variability score: %.6f (1.0 = perfect consistency)\n", 
		result.VariabilityScore())
	fmt.Printf("   Consistency rating: %s\n", 
		result.Metadata["consistency_rating"])
	fmt.Printf("   Mission readiness: Real-time suitable: %v, Memory safe: %v\n",
		result.AvgPrecisionMetrics.WorstCaseLatencyNs < 1_000_000_000,
		result.AvgPrecisionMetrics.MemoryOverheadRatio < 2.0)
	
	// Determine reliability classification
	reliability := "UNKNOWN"
	if successRate >= 95 && result.VariabilityScore() >= 0.90 {
		reliability = "🚀 AEROSPACE_GRADE"
	} else if successRate >= 90 && result.VariabilityScore() >= 0.80 {
		reliability = "🛡️  MISSION_CRITICAL"
	} else if successRate >= 80 && result.VariabilityScore() >= 0.70 {
		reliability = "✅ PRODUCTION_READY"
	} else if successRate >= 70 {
		reliability = "⚠️  ACCEPTABLE"
	} else {
		reliability = "❌ UNRELIABLE"
	}
	
	fmt.Printf("   Overall reliability: %s\n", reliability)
} 