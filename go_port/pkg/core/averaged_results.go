// Package core provides averaged result structures for multiple compression runs.
package core

import (
	"fmt"
	"math"
	"sort"
)

// AveragedCompressionResult represents averaged results from multiple compression runs
type AveragedCompressionResult struct {
	AlgorithmName  string `json:"algorithm_name"`
	OriginalSize   int64  `json:"original_size"`
	CompressedSize int64  `json:"compressed_size"`

	// Averaged metrics
	AvgCompressionRatio float64 `json:"avg_compression_ratio"`
	AvgCompressionTime  float64 `json:"avg_compression_time"`
	AvgThroughputMBPS   float64 `json:"avg_throughput_mbps"`

	// Statistical measures
	StdDevCompressionRatio float64 `json:"stddev_compression_ratio"`
	StdDevCompressionTime  float64 `json:"stddev_compression_time"`
	StdDevThroughputMBPS   float64 `json:"stddev_throughput_mbps"`

	// Min/Max values for range
	MinCompressionRatio float64 `json:"min_compression_ratio"`
	MaxCompressionRatio float64 `json:"max_compression_ratio"`
	MinCompressionTime  float64 `json:"min_compression_time"`
	MaxCompressionTime  float64 `json:"max_compression_time"`
	MinThroughputMBPS   float64 `json:"min_throughput_mbps"`
	MaxThroughputMBPS   float64 `json:"max_throughput_mbps"`

	// Averaged precision metrics
	AvgPrecisionMetrics StatisticalPrecisionMetrics `json:"avg_precision_metrics"`

	// Run information
	TotalRuns      int                    `json:"total_runs"`
	SuccessfulRuns int                    `json:"successful_runs"`
	Metadata       map[string]interface{} `json:"metadata"`

	// Individual results for detailed analysis
	IndividualResults []*CompressionResult `json:"individual_results,omitempty"`
}

// CompressionPercentage calculates averaged compression percentage
func (ar *AveragedCompressionResult) CompressionPercentage() float64 {
	return SafePercent(float64(ar.OriginalSize-ar.CompressedSize), float64(ar.OriginalSize))
}

// SpaceSavingsBytes calculates space savings in bytes
func (ar *AveragedCompressionResult) SpaceSavingsBytes() int64 {
	if ar.CompressedSize > ar.OriginalSize {
		return 0
	}
	return ar.OriginalSize - ar.CompressedSize
}

// IsEffective checks if compression was effective on average
func (ar *AveragedCompressionResult) IsEffective() bool {
	return ar.AvgCompressionRatio > 1.0
}

// VariabilityScore calculates a score (0-1) indicating result consistency
// 0 = highly variable, 1 = perfectly consistent
func (ar *AveragedCompressionResult) VariabilityScore() float64 {
	if ar.AvgCompressionTime == 0 || ar.StdDevCompressionTime == 0 {
		return 1.0
	}

	// Calculate coefficient of variation for time (lower is better)
	timeCV := ar.StdDevCompressionTime / ar.AvgCompressionTime

	// Check for infinite or NaN values
	if math.IsInf(timeCV, 0) || math.IsNaN(timeCV) {
		return 0.0
	}

	// Convert to consistency score (higher is better)
	consistencyScore := math.Max(0, 1.0-timeCV)
	return math.Min(1.0, consistencyScore)
}

// AveragedDecompressionResult represents averaged results from multiple decompression runs
type AveragedDecompressionResult struct {
	AlgorithmName          string `json:"algorithm_name"`
	OriginalCompressedSize int64  `json:"original_compressed_size"`
	DecompressedSize       int64  `json:"decompressed_size"`

	// Averaged metrics
	AvgDecompressionTime float64 `json:"avg_decompression_time"`
	AvgThroughputMBPS    float64 `json:"avg_throughput_mbps"`

	// Statistical measures
	StdDevDecompressionTime float64 `json:"stddev_decompression_time"`
	StdDevThroughputMBPS    float64 `json:"stddev_throughput_mbps"`

	// Min/Max values
	MinDecompressionTime float64 `json:"min_decompression_time"`
	MaxDecompressionTime float64 `json:"max_decompression_time"`
	MinThroughputMBPS    float64 `json:"min_throughput_mbps"`
	MaxThroughputMBPS    float64 `json:"max_throughput_mbps"`

	// Averaged precision metrics
	AvgPrecisionMetrics StatisticalPrecisionMetrics `json:"avg_precision_metrics"`

	// Run information
	TotalRuns      int                    `json:"total_runs"`
	SuccessfulRuns int                    `json:"successful_runs"`
	Metadata       map[string]interface{} `json:"metadata"`

	// Individual results
	IndividualResults []*DecompressionResult `json:"individual_results,omitempty"`
}

// ExpansionRatio calculates averaged expansion ratio
func (ar *AveragedDecompressionResult) ExpansionRatio() float64 {
	return SafeDiv(float64(ar.DecompressedSize), float64(ar.OriginalCompressedSize))
}

// AverageCompressionResults calculates averaged results from multiple compression runs
func AverageCompressionResults(results []*CompressionResult) *AveragedCompressionResult {
	if len(results) == 0 {
		return nil
	}

	// Filter successful results
	successfulResults := make([]*CompressionResult, 0)
	for _, result := range results {
		if result != nil {
			successfulResults = append(successfulResults, result)
		}
	}

	if len(successfulResults) == 0 {
		return &AveragedCompressionResult{
			TotalRuns:      len(results),
			SuccessfulRuns: 0,
		}
	}

	// Initialize result with first successful result's static data
	first := successfulResults[0]
	averaged := &AveragedCompressionResult{
		AlgorithmName:     first.AlgorithmName,
		OriginalSize:      first.OriginalSize,
		CompressedSize:    first.CompressedSize, // This should be consistent
		TotalRuns:         len(results),
		SuccessfulRuns:    len(successfulResults),
		IndividualResults: successfulResults,
		Metadata:          make(map[string]interface{}),
	}

	// Calculate averages and statistics
	var compressionRatios []float64
	var compressionTimes []float64
	var throughputs []float64

	// Precision metrics accumulator
	var totalCompressionTimeNs int64
	var totalMemoryPeakBytes int64
	var totalCPUPercentAvg float64
	var totalThroughputMBPS float64

	for _, result := range successfulResults {
		compressionRatios = append(compressionRatios, result.CompressionRatio)
		compressionTimes = append(compressionTimes, result.CompressionTime)
		throughputs = append(throughputs, result.PrecisionMetrics.ThroughputMBPS)

		// Accumulate precision metrics
		totalCompressionTimeNs += result.PrecisionMetrics.CompressionTimeNs
		totalMemoryPeakBytes += result.PrecisionMetrics.MemoryPeakBytes
		totalCPUPercentAvg += result.PrecisionMetrics.CPUPercentAvg
		totalThroughputMBPS += result.PrecisionMetrics.ThroughputMBPS
	}

	n := float64(len(successfulResults))

	// Calculate averages
	averaged.AvgCompressionRatio = average(compressionRatios)
	averaged.AvgCompressionTime = average(compressionTimes)
	averaged.AvgThroughputMBPS = average(throughputs)

	// Calculate standard deviations
	averaged.StdDevCompressionRatio = standardDeviation(compressionRatios, averaged.AvgCompressionRatio)
	averaged.StdDevCompressionTime = standardDeviation(compressionTimes, averaged.AvgCompressionTime)
	averaged.StdDevThroughputMBPS = standardDeviation(throughputs, averaged.AvgThroughputMBPS)

	// Calculate min/max
	sort.Float64s(compressionRatios)
	sort.Float64s(compressionTimes)
	sort.Float64s(throughputs)

	averaged.MinCompressionRatio = compressionRatios[0]
	averaged.MaxCompressionRatio = compressionRatios[len(compressionRatios)-1]
	averaged.MinCompressionTime = compressionTimes[0]
	averaged.MaxCompressionTime = compressionTimes[len(compressionTimes)-1]
	averaged.MinThroughputMBPS = throughputs[0]
	averaged.MaxThroughputMBPS = throughputs[len(throughputs)-1]

	// Calculate averaged precision metrics with safe division
	avgCompressionTimeNs := totalCompressionTimeNs / int64(n)
	avgMemoryPeakBytes := totalMemoryPeakBytes / int64(n)
	avgThroughputMBPS := totalThroughputMBPS / n

	// Safe division functions to prevent infinite values
	safeDiv := func(numerator, denominator float64) float64 {
		if denominator == 0 || math.IsInf(denominator, 0) || math.IsNaN(denominator) {
			return 0.0
		}
		result := numerator / denominator
		if math.IsInf(result, 0) || math.IsNaN(result) {
			return 0.0
		}
		return result
	}

	safeDivInt := func(numerator float64, denominator int64) float64 {
		if denominator == 0 {
			return 0.0
		}
		return safeDiv(numerator, float64(denominator))
	}

	averaged.AvgPrecisionMetrics = StatisticalPrecisionMetrics{
		CompressionTimeNs:          avgCompressionTimeNs,
		TotalTimeNs:                avgCompressionTimeNs,
		MemoryPeakBytes:            avgMemoryPeakBytes,
		CPUPercentAvg:              totalCPUPercentAvg / n,
		ThroughputMBPS:             avgThroughputMBPS,
		ThroughputBytesPerSecond:   avgThroughputMBPS * 1024 * 1024,
		TimePerByteNs:              safeDivInt(float64(avgCompressionTimeNs), first.OriginalSize),
		DeterminismScore:           1.0, // Deterministic algorithms
		MemoryOverheadRatio:        safeDivInt(float64(avgMemoryPeakBytes), first.OriginalSize),
		BitsPerByte:                safeDiv(float64(first.CompressedSize)*8, float64(first.OriginalSize)),
		EntropyEfficiency:          safeDiv(float64(first.OriginalSize), float64(first.CompressedSize)*8.0),
		EnergyEfficiencyBytesPerNs: safeDivInt(float64(first.OriginalSize), avgCompressionTimeNs),
		WorstCaseLatencyNs:         avgCompressionTimeNs,
	}

	averaged.AvgPrecisionMetrics.UpdateFormattedTimes()

	// Add statistical metadata with safe calculations
	variabilityScore := ValidateFloat(averaged.VariabilityScore())
	coefficientOfVariation := SafeCoefficientOfVariation(averaged.StdDevCompressionTime, averaged.AvgCompressionTime)

	averaged.Metadata["variability_score"] = variabilityScore
	averaged.Metadata["consistency_rating"] = getConsistencyRating(variabilityScore)
	averaged.Metadata["coefficient_of_variation_time"] = coefficientOfVariation

	return averaged
}

// AverageDecompressionResults calculates averaged results from multiple decompression runs
func AverageDecompressionResults(results []*DecompressionResult) *AveragedDecompressionResult {
	if len(results) == 0 {
		return nil
	}

	// Filter successful results
	successfulResults := make([]*DecompressionResult, 0)
	for _, result := range results {
		if result != nil {
			successfulResults = append(successfulResults, result)
		}
	}

	if len(successfulResults) == 0 {
		return &AveragedDecompressionResult{
			TotalRuns:      len(results),
			SuccessfulRuns: 0,
		}
	}

	// Initialize result
	first := successfulResults[0]
	averaged := &AveragedDecompressionResult{
		AlgorithmName:          first.AlgorithmName,
		OriginalCompressedSize: first.OriginalCompressedSize,
		DecompressedSize:       first.DecompressedSize,
		TotalRuns:              len(results),
		SuccessfulRuns:         len(successfulResults),
		IndividualResults:      successfulResults,
		Metadata:               make(map[string]interface{}),
	}

	// Calculate statistics
	var decompressionTimes []float64
	var throughputs []float64

	for _, result := range successfulResults {
		decompressionTimes = append(decompressionTimes, result.DecompressionTime)
		throughputs = append(throughputs, result.PrecisionMetrics.ThroughputMBPS)
	}

	// Calculate averages
	averaged.AvgDecompressionTime = average(decompressionTimes)
	averaged.AvgThroughputMBPS = average(throughputs)

	// Calculate standard deviations
	averaged.StdDevDecompressionTime = standardDeviation(decompressionTimes, averaged.AvgDecompressionTime)
	averaged.StdDevThroughputMBPS = standardDeviation(throughputs, averaged.AvgThroughputMBPS)

	// Calculate min/max
	sort.Float64s(decompressionTimes)
	sort.Float64s(throughputs)

	averaged.MinDecompressionTime = decompressionTimes[0]
	averaged.MaxDecompressionTime = decompressionTimes[len(decompressionTimes)-1]
	averaged.MinThroughputMBPS = throughputs[0]
	averaged.MaxThroughputMBPS = throughputs[len(throughputs)-1]

	return averaged
}

// Helper functions
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func standardDeviation(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}

	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}

	variance := sumSquares / float64(len(values)-1) // Sample standard deviation
	return math.Sqrt(variance)
}

func getConsistencyRating(variabilityScore float64) string {
	if variabilityScore >= 0.95 {
		return "EXCELLENT"
	} else if variabilityScore >= 0.90 {
		return "VERY_GOOD"
	} else if variabilityScore >= 0.80 {
		return "GOOD"
	} else if variabilityScore >= 0.70 {
		return "FAIR"
	} else {
		return "POOR"
	}
}

// ToMissionReport generates a comprehensive mission-critical report for averaged results
func (ar *AveragedCompressionResult) ToMissionReport() map[string]interface{} {
	effectiveness := "NEGATIVE"
	if ar.IsEffective() {
		effectiveness = "POSITIVE"
	}

	suitableForRealtime := ar.AvgPrecisionMetrics.WorstCaseLatencyNs < 1_000_000_000
	memoryConstrainedSafe := ar.AvgPrecisionMetrics.MemoryOverheadRatio < 2.0

	return map[string]interface{}{
		"algorithm": ar.AlgorithmName,
		"statistical_summary": map[string]interface{}{
			"total_runs":         ar.TotalRuns,
			"successful_runs":    ar.SuccessfulRuns,
			"success_rate":       fmt.Sprintf("%.1f%%", SafePercent(float64(ar.SuccessfulRuns), float64(ar.TotalRuns))),
			"variability_score":  ValidateFloat(ar.VariabilityScore()),
			"consistency_rating": getConsistencyRating(ValidateFloat(ar.VariabilityScore())),
		},
		"data_integrity": map[string]interface{}{
			"original_size_bytes":   ar.OriginalSize,
			"compressed_size_bytes": ar.CompressedSize,
			"avg_compression_ratio": fmt.Sprintf("%.6fx", ValidateFloat(ar.AvgCompressionRatio)),
			"ratio_std_dev":         fmt.Sprintf("±%.6f", ValidateFloat(ar.StdDevCompressionRatio)),
			"space_savings_bytes":   ar.SpaceSavingsBytes(),
			"space_savings_percent": fmt.Sprintf("%.3f%%", ValidateFloat(ar.CompressionPercentage())),
			"effectiveness":         effectiveness,
		},
		"performance_profile": map[string]interface{}{
			"avg_compression_time": ar.AvgPrecisionMetrics.CompressionTimeFormatted,
			"time_std_dev":         fmt.Sprintf("±%.3fs", ValidateFloat(ar.StdDevCompressionTime)),
			"time_range":           fmt.Sprintf("%.3fs - %.3fs", ValidateFloat(ar.MinCompressionTime), ValidateFloat(ar.MaxCompressionTime)),
			"avg_throughput_mbps":  fmt.Sprintf("%.6f", ValidateFloat(ar.AvgThroughputMBPS)),
			"throughput_range":     fmt.Sprintf("%.3f - %.3f MB/s", ValidateFloat(ar.MinThroughputMBPS), ValidateFloat(ar.MaxThroughputMBPS)),
			"time_per_byte":        fmt.Sprintf("%.2f ns/byte", ValidateFloat(ar.AvgPrecisionMetrics.TimePerByteNs)),
		},
		"resource_utilization": map[string]interface{}{
			"avg_peak_memory":       fmt.Sprintf("%d bytes", ar.AvgPrecisionMetrics.MemoryPeakBytes),
			"memory_overhead_ratio": fmt.Sprintf("%.4f", ValidateFloat(ar.AvgPrecisionMetrics.MemoryOverheadRatio)),
			"avg_cpu_percent":       fmt.Sprintf("%.2f%%", ValidateFloat(ar.AvgPrecisionMetrics.CPUPercentAvg)),
			"determinism_score":     fmt.Sprintf("%.6f", ValidateFloat(ar.AvgPrecisionMetrics.DeterminismScore)),
		},
		"mission_readiness": map[string]interface{}{
			"avg_worst_case_latency":  ar.AvgPrecisionMetrics.WorstCaseLatencyNs,
			"energy_efficiency":       fmt.Sprintf("%.2e bytes/ns", ValidateFloat(ar.AvgPrecisionMetrics.EnergyEfficiencyBytesPerNs)),
			"entropy_efficiency":      fmt.Sprintf("%.4f", ValidateFloat(ar.AvgPrecisionMetrics.EntropyEfficiency)),
			"suitable_for_realtime":   suitableForRealtime,
			"memory_constrained_safe": memoryConstrainedSafe,
			"result_consistency":      getConsistencyRating(ar.VariabilityScore()),
		},
	}
}
