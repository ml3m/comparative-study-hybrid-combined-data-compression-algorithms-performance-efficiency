package benchmarks

import (
	"hybrid-compression-study/pkg/core"
	"math"
)

// KEIWeights defines the weights for different metrics in the KEI calculation
type KEIWeights struct {
	CompressionRatio float64 `json:"compression_ratio" yaml:"compression_ratio"` // Weight for compression effectiveness
	Speed            float64 `json:"speed" yaml:"speed"`                         // Weight for processing speed
	Memory           float64 `json:"memory" yaml:"memory"`                       // Weight for memory efficiency
	Stability        float64 `json:"stability" yaml:"stability"`                 // Weight for performance consistency
	Energy           float64 `json:"energy" yaml:"energy"`                       // Weight for energy efficiency
}

// DefaultKEIWeights provides balanced weights for KEI calculation
var DefaultKEIWeights = KEIWeights{
	CompressionRatio: 0.3,  // 30% - Most important for compression
	Speed:            0.25, // 25% - Important for practical use
	Memory:           0.2,  // 20% - Memory consumption matters
	Stability:        0.15, // 15% - Consistency is valuable
	Energy:           0.1,  // 10% - Energy efficiency (future-focused)
}

// PerformanceWeights provides speed-focused weights
var PerformanceWeights = KEIWeights{
	CompressionRatio: 0.2,
	Speed:            0.4,
	Memory:           0.25,
	Stability:        0.1,
	Energy:           0.05,
}

// CompressionWeights provides compression-ratio-focused weights
var CompressionWeights = KEIWeights{
	CompressionRatio: 0.5,
	Speed:            0.15,
	Memory:           0.2,
	Stability:        0.1,
	Energy:           0.05,
}

// MemoryWeights provides memory-efficiency-focused weights
var MemoryWeights = KEIWeights{
	CompressionRatio: 0.25,
	Speed:            0.2,
	Memory:           0.4,
	Stability:        0.1,
	Energy:           0.05,
}

// KEIScore represents a comprehensive efficiency score for a compression result
type KEIScore struct {
	// Overall KEI score (0-100, higher is better)
	OverallScore float64 `json:"overall_score"`

	// Individual dimension scores (0-100, higher is better)
	CompressionScore float64 `json:"compression_score"`
	SpeedScore       float64 `json:"speed_score"`
	MemoryScore      float64 `json:"memory_score"`
	StabilityScore   float64 `json:"stability_score"`
	EnergyScore      float64 `json:"energy_score"`

	// Raw metrics used for calculation
	CompressionRatio    float64 `json:"compression_ratio"`
	ThroughputMbps      float64 `json:"throughput_mbps"`
	MemoryEfficiency    float64 `json:"memory_efficiency"`
	PerformanceVariance float64 `json:"performance_variance"`
	EnergyEfficiency    float64 `json:"energy_efficiency"`

	// Context information
	InputType            string     `json:"input_type"`
	InputSize            int64      `json:"input_size"`
	AlgorithmCombination string     `json:"algorithm_combination"`
	Weights              KEIWeights `json:"weights"`
}

// KEICalculator computes Key Efficiency Index scores
type KEICalculator struct {
	weights KEIWeights

	// Reference values for normalization (updated as we see results)
	maxCompressionRatio    float64
	maxThroughputMbps      float64
	maxMemoryEfficiency    float64
	minPerformanceVariance float64
	maxEnergyEfficiency    float64

	// Statistical tracking for dynamic normalization
	compressionRatios    []float64
	throughputValues     []float64
	memoryEfficiencies   []float64
	performanceVariances []float64
	energyEfficiencies   []float64
}

// NewKEICalculator creates a new KEI calculator with specified weights
func NewKEICalculator(weights KEIWeights) *KEICalculator {
	return &KEICalculator{
		weights:                weights,
		maxCompressionRatio:    0.0,
		maxThroughputMbps:      0.0,
		maxMemoryEfficiency:    0.0,
		minPerformanceVariance: math.MaxFloat64,
		maxEnergyEfficiency:    0.0,
		compressionRatios:      make([]float64, 0),
		throughputValues:       make([]float64, 0),
		memoryEfficiencies:     make([]float64, 0),
		performanceVariances:   make([]float64, 0),
		energyEfficiencies:     make([]float64, 0),
	}
}

// NewDefaultKEICalculator creates a KEI calculator with default weights
func NewDefaultKEICalculator() *KEICalculator {
	return NewKEICalculator(DefaultKEIWeights)
}

// CalculateKEI computes the KEI score for a compression result
func (kei *KEICalculator) CalculateKEI(result *core.CompressionResult, inputType string, algorithmCombination string) *KEIScore {
	// Extract metrics from the result
	compressionRatio := kei.calculateCompressionRatio(result)
	throughputMbps := result.PrecisionMetrics.ThroughputMBPS
	memoryEfficiency := result.PrecisionMetrics.MemoryEfficiencyRatio
	performanceVariance := kei.calculatePerformanceVariance(result)
	energyEfficiency := result.PrecisionMetrics.EnergyEfficiencyBytesPerNs

	// Update reference values for normalization
	kei.updateReferenceValues(compressionRatio, throughputMbps, memoryEfficiency, performanceVariance, energyEfficiency)

	// Calculate individual dimension scores (0-100)
	compressionScore := kei.normalizeCompressionScore(compressionRatio)
	speedScore := kei.normalizeSpeedScore(throughputMbps)
	memoryScore := kei.normalizeMemoryScore(memoryEfficiency)
	stabilityScore := kei.normalizeStabilityScore(performanceVariance)
	energyScore := kei.normalizeEnergyScore(energyEfficiency)

	// Calculate weighted overall score
	overallScore := kei.weights.CompressionRatio*compressionScore +
		kei.weights.Speed*speedScore +
		kei.weights.Memory*memoryScore +
		kei.weights.Stability*stabilityScore +
		kei.weights.Energy*energyScore

	return &KEIScore{
		OverallScore:         overallScore,
		CompressionScore:     compressionScore,
		SpeedScore:           speedScore,
		MemoryScore:          memoryScore,
		StabilityScore:       stabilityScore,
		EnergyScore:          energyScore,
		CompressionRatio:     compressionRatio,
		ThroughputMbps:       throughputMbps,
		MemoryEfficiency:     memoryEfficiency,
		PerformanceVariance:  performanceVariance,
		EnergyEfficiency:     energyEfficiency,
		InputType:            inputType,
		InputSize:            result.OriginalSize,
		AlgorithmCombination: algorithmCombination,
		Weights:              kei.weights,
	}
}

// calculateCompressionRatio computes the compression ratio (higher is better)
func (kei *KEICalculator) calculateCompressionRatio(result *core.CompressionResult) float64 {
	if result.CompressedSize == 0 {
		return 0.0
	}
	return float64(result.OriginalSize) / float64(result.CompressedSize)
}

// calculatePerformanceVariance estimates performance variance (lower is better)
func (kei *KEICalculator) calculatePerformanceVariance(result *core.CompressionResult) float64 {
	// Use determinism score as an inverse measure of variance
	// Higher determinism = lower variance
	if result.PrecisionMetrics.DeterminismScore > 0 {
		return 1.0 - result.PrecisionMetrics.DeterminismScore
	}

	// If no determinism score, use CPU consistency as proxy
	avgCpu := result.PrecisionMetrics.CPUPercentAvg
	peakCpu := result.PrecisionMetrics.CPUPercentPeak

	if avgCpu == 0 {
		return 0.5 // Medium variance assumption
	}

	variance := math.Abs(peakCpu-avgCpu) / avgCpu
	return math.Min(variance, 1.0) // Cap at 1.0
}

// updateReferenceValues updates the maximum/minimum values for normalization
func (kei *KEICalculator) updateReferenceValues(compressionRatio, throughputMbps, memoryEfficiency, performanceVariance, energyEfficiency float64) {
	// Update maxima for "higher is better" metrics
	if compressionRatio > kei.maxCompressionRatio {
		kei.maxCompressionRatio = compressionRatio
	}
	if throughputMbps > kei.maxThroughputMbps {
		kei.maxThroughputMbps = throughputMbps
	}
	if memoryEfficiency > kei.maxMemoryEfficiency {
		kei.maxMemoryEfficiency = memoryEfficiency
	}
	if energyEfficiency > kei.maxEnergyEfficiency {
		kei.maxEnergyEfficiency = energyEfficiency
	}

	// Update minimum for "lower is better" metrics
	if performanceVariance < kei.minPerformanceVariance {
		kei.minPerformanceVariance = performanceVariance
	}

	// Store values for statistical analysis
	kei.compressionRatios = append(kei.compressionRatios, compressionRatio)
	kei.throughputValues = append(kei.throughputValues, throughputMbps)
	kei.memoryEfficiencies = append(kei.memoryEfficiencies, memoryEfficiency)
	kei.performanceVariances = append(kei.performanceVariances, performanceVariance)
	kei.energyEfficiencies = append(kei.energyEfficiencies, energyEfficiency)
}

// Normalization functions (all return 0-100 scores)

func (kei *KEICalculator) normalizeCompressionScore(ratio float64) float64 {
	if kei.maxCompressionRatio == 0 {
		return 50.0 // Default middle score
	}

	// Use logarithmic scale for compression ratios to better distinguish small improvements
	if ratio <= 1.0 {
		return 0.0 // No compression or expansion
	}

	maxLogRatio := math.Log(kei.maxCompressionRatio)
	currentLogRatio := math.Log(ratio)

	score := (currentLogRatio / maxLogRatio) * 100.0
	return math.Min(score, 100.0)
}

func (kei *KEICalculator) normalizeSpeedScore(throughputMbps float64) float64 {
	if kei.maxThroughputMbps == 0 {
		return 50.0
	}

	score := (throughputMbps / kei.maxThroughputMbps) * 100.0
	return math.Min(score, 100.0)
}

func (kei *KEICalculator) normalizeMemoryScore(efficiency float64) float64 {
	if kei.maxMemoryEfficiency == 0 {
		kei.maxMemoryEfficiency = 1.0 // Default assumption
	}

	// Memory efficiency ratio where lower is better (less memory per byte)
	// We need to invert this for scoring
	score := (1.0 / (efficiency + 0.01)) * 20.0 // +0.01 to avoid division by zero
	return math.Min(score, 100.0)
}

func (kei *KEICalculator) normalizeStabilityScore(variance float64) float64 {
	// Lower variance is better, so invert the score
	if kei.minPerformanceVariance == math.MaxFloat64 {
		kei.minPerformanceVariance = 0.0
	}

	// Score based on how close to minimum variance we are
	maxVariance := 1.0 // Assume maximum possible variance is 1.0
	normalizedVariance := 1.0 - (variance / maxVariance)

	return math.Max(normalizedVariance*100.0, 0.0)
}

func (kei *KEICalculator) normalizeEnergyScore(efficiency float64) float64 {
	if kei.maxEnergyEfficiency == 0 {
		return 50.0
	}

	score := (efficiency / kei.maxEnergyEfficiency) * 100.0
	return math.Min(score, 100.0)
}

// KEIRanking represents a ranked list of algorithm combinations
type KEIRanking struct {
	Rankings    []*RankedCombination `json:"rankings"`
	InputType   string               `json:"input_type"`
	InputSize   int64                `json:"input_size"`
	TotalTested int                  `json:"total_tested"`
	Weights     KEIWeights           `json:"weights"`
}

// RankedCombination represents a single ranked algorithm combination
type RankedCombination struct {
	Rank                 int       `json:"rank"`
	AlgorithmCombination string    `json:"algorithm_combination"`
	KEIScore             *KEIScore `json:"kei_score"`
	PerformanceCategory  string    `json:"performance_category"`
	Strengths            []string  `json:"strengths"`
	Weaknesses           []string  `json:"weaknesses"`
	RecommendedFor       []string  `json:"recommended_for"`
}

// CreateRanking creates a KEI-based ranking of algorithm combinations
func (kei *KEICalculator) CreateRanking(scores []*KEIScore, inputType string, inputSize int64) *KEIRanking {
	// Sort scores by overall KEI score (descending)
	rankedCombinations := make([]*RankedCombination, len(scores))

	// Create ranked combinations
	for i, score := range scores {
		ranked := &RankedCombination{
			Rank:                 i + 1,
			AlgorithmCombination: score.AlgorithmCombination,
			KEIScore:             score,
			PerformanceCategory:  kei.categorizePerformance(score),
			Strengths:            kei.identifyStrengths(score),
			Weaknesses:           kei.identifyWeaknesses(score),
			RecommendedFor:       kei.generateRecommendations(score),
		}
		rankedCombinations[i] = ranked
	}

	return &KEIRanking{
		Rankings:    rankedCombinations,
		InputType:   inputType,
		InputSize:   inputSize,
		TotalTested: len(scores),
		Weights:     kei.weights,
	}
}

// categorizePerformance assigns a performance category based on KEI score
func (kei *KEICalculator) categorizePerformance(score *KEIScore) string {
	switch {
	case score.OverallScore >= 90:
		return "Exceptional"
	case score.OverallScore >= 80:
		return "Excellent"
	case score.OverallScore >= 70:
		return "Very Good"
	case score.OverallScore >= 60:
		return "Good"
	case score.OverallScore >= 50:
		return "Average"
	case score.OverallScore >= 40:
		return "Below Average"
	case score.OverallScore >= 30:
		return "Poor"
	default:
		return "Very Poor"
	}
}

// identifyStrengths identifies the strongest aspects of a combination
func (kei *KEICalculator) identifyStrengths(score *KEIScore) []string {
	var strengths []string
	threshold := 70.0 // Consider scores above 70 as strengths

	if score.CompressionScore >= threshold {
		strengths = append(strengths, "Excellent compression ratio")
	}
	if score.SpeedScore >= threshold {
		strengths = append(strengths, "High processing speed")
	}
	if score.MemoryScore >= threshold {
		strengths = append(strengths, "Memory efficient")
	}
	if score.StabilityScore >= threshold {
		strengths = append(strengths, "Consistent performance")
	}
	if score.EnergyScore >= threshold {
		strengths = append(strengths, "Energy efficient")
	}

	return strengths
}

// identifyWeaknesses identifies the weakest aspects of a combination
func (kei *KEICalculator) identifyWeaknesses(score *KEIScore) []string {
	var weaknesses []string
	threshold := 40.0 // Consider scores below 40 as weaknesses

	if score.CompressionScore < threshold {
		weaknesses = append(weaknesses, "Poor compression ratio")
	}
	if score.SpeedScore < threshold {
		weaknesses = append(weaknesses, "Slow processing")
	}
	if score.MemoryScore < threshold {
		weaknesses = append(weaknesses, "High memory usage")
	}
	if score.StabilityScore < threshold {
		weaknesses = append(weaknesses, "Inconsistent performance")
	}
	if score.EnergyScore < threshold {
		weaknesses = append(weaknesses, "High energy consumption")
	}

	return weaknesses
}

// generateRecommendations suggests use cases for the combination
func (kei *KEICalculator) generateRecommendations(score *KEIScore) []string {
	var recommendations []string

	// Based on strongest dimension
	maxScore := math.Max(score.CompressionScore,
		math.Max(score.SpeedScore,
			math.Max(score.MemoryScore,
				math.Max(score.StabilityScore, score.EnergyScore))))

	switch {
	case score.CompressionScore == maxScore && score.CompressionScore >= 70:
		recommendations = append(recommendations, "Archival storage", "Bandwidth-limited transmission")
	case score.SpeedScore == maxScore && score.SpeedScore >= 70:
		recommendations = append(recommendations, "Real-time processing", "Interactive applications")
	case score.MemoryScore == maxScore && score.MemoryScore >= 70:
		recommendations = append(recommendations, "Embedded systems", "Memory-constrained environments")
	case score.StabilityScore == maxScore && score.StabilityScore >= 70:
		recommendations = append(recommendations, "Mission-critical systems", "Predictable workloads")
	case score.EnergyScore == maxScore && score.EnergyScore >= 70:
		recommendations = append(recommendations, "Mobile devices", "Green computing")
	}

	// Based on input type affinity
	switch score.InputType {
	case "repetitive":
		if score.CompressionScore >= 60 {
			recommendations = append(recommendations, "Log file compression", "Database backups")
		}
	case "text_patterns":
		if score.CompressionScore >= 60 {
			recommendations = append(recommendations, "Document archival", "Text processing")
		}
	case "random":
		if score.SpeedScore >= 60 {
			recommendations = append(recommendations, "Encrypted data", "Multimedia processing")
		}
	}

	return recommendations
}

// GetStatistics returns comprehensive statistics about the KEI calculations
func (kei *KEICalculator) GetStatistics() map[string]interface{} {
	stats := map[string]interface{}{
		"samples_processed": len(kei.compressionRatios),
		"reference_values": map[string]interface{}{
			"max_compression_ratio":    kei.maxCompressionRatio,
			"max_throughput_mbps":      kei.maxThroughputMbps,
			"max_memory_efficiency":    kei.maxMemoryEfficiency,
			"min_performance_variance": kei.minPerformanceVariance,
			"max_energy_efficiency":    kei.maxEnergyEfficiency,
		},
		"weights": kei.weights,
	}

	// Add statistical summaries if we have data
	if len(kei.compressionRatios) > 0 {
		stats["compression_stats"] = kei.calculateStats(kei.compressionRatios)
		stats["throughput_stats"] = kei.calculateStats(kei.throughputValues)
		stats["memory_stats"] = kei.calculateStats(kei.memoryEfficiencies)
		stats["variance_stats"] = kei.calculateStats(kei.performanceVariances)
		stats["energy_stats"] = kei.calculateStats(kei.energyEfficiencies)
	}

	return stats
}

// calculateStats computes basic statistics for a slice of values
func (kei *KEICalculator) calculateStats(values []float64) map[string]float64 {
	if len(values) == 0 {
		return map[string]float64{}
	}

	sum := 0.0
	min := values[0]
	max := values[0]

	for _, v := range values {
		sum += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	mean := sum / float64(len(values))

	// Calculate standard deviation
	sumSquaredDiffs := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiffs += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiffs / float64(len(values)))

	return map[string]float64{
		"min":    min,
		"max":    max,
		"mean":   mean,
		"stddev": stdDev,
		"count":  float64(len(values)),
	}
}
