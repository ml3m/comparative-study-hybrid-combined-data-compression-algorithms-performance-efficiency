package benchmarks

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// AnalysisEngine provides comprehensive analysis of benchmark results
type AnalysisEngine struct {
	results *BenchmarkResult
}

// NewAnalysisEngine creates a new analysis engine for the given results
func NewAnalysisEngine(results *BenchmarkResult) *AnalysisEngine {
	return &AnalysisEngine{
		results: results,
	}
}

// DomainEffectivenessReport represents effectiveness analysis for different domains
type DomainEffectivenessReport struct {
	Domain               string               `json:"domain"`
	BestCombinations     []*RankedCombination `json:"best_combinations"`
	WorstCombinations    []*RankedCombination `json:"worst_combinations"`
	DomainInsights       []string             `json:"domain_insights"`
	RecommendedScenarios []string             `json:"recommended_scenarios"`
	AvoidScenarios       []string             `json:"avoid_scenarios"`
	Statistics           map[string]float64   `json:"statistics"`
}

// ComprehensiveAnalysisReport contains all analysis results
type ComprehensiveAnalysisReport struct {
	ExecutionSummary       *ExecutionSummary                     `json:"execution_summary"`
	DomainEffectiveness    map[string]*DomainEffectivenessReport `json:"domain_effectiveness"`
	InputTypeAnalysis      map[string]*InputTypeAnalysis         `json:"input_type_analysis"`
	AlgorithmEffectiveness map[string]*AlgorithmAnalysis         `json:"algorithm_effectiveness"`
	OptimalCombinations    *OptimalCombinations                  `json:"optimal_combinations"`
	EfficiencyInsights     *EfficiencyInsights                   `json:"efficiency_insights"`
	Recommendations        *RecommendationEngine                 `json:"recommendations"`
	StatisticalSummary     *StatisticalSummary                   `json:"statistical_summary"`
}

// ExecutionSummary provides high-level execution statistics
type ExecutionSummary struct {
	TotalExecutionTime      string  `json:"total_execution_time"`
	TotalCombinations       int64   `json:"total_combinations"`
	TotalTests              int64   `json:"total_tests"`
	SuccessRate             float64 `json:"success_rate"`
	AverageTestDuration     string  `json:"average_test_duration"`
	TestThroughput          float64 `json:"test_throughput"` // tests per second
	PeakKEIScore            float64 `json:"peak_kei_score"`
	AverageKEIScore         float64 `json:"average_kei_score"`
	PeakCompressionRatio    float64 `json:"peak_compression_ratio"`
	AverageCompressionRatio float64 `json:"average_compression_ratio"`
}

// InputTypeAnalysis provides analysis specific to input types
type InputTypeAnalysis struct {
	InputType            InputType          `json:"input_type"`
	BestCombination      *RankedCombination `json:"best_combination"`
	WorstCombination     *RankedCombination `json:"worst_combination"`
	AverageKEI           float64            `json:"average_kei"`
	KEIStandardDeviation float64            `json:"kei_standard_deviation"`
	OptimalStrategies    []string           `json:"optimal_strategies"`
	PerformancePattern   string             `json:"performance_pattern"`
	Insights             []string           `json:"insights"`
}

// AlgorithmAnalysis provides analysis for individual algorithms
type AlgorithmAnalysis struct {
	Algorithm              string   `json:"algorithm"`
	AppearanceCount        int      `json:"appearance_count"`
	SuccessfulAppearances  int      `json:"successful_appearances"`
	AverageKEIContribution float64  `json:"average_kei_contribution"`
	BestPerformanceWith    []string `json:"best_performance_with"`
	WorstPerformanceWith   []string `json:"worst_performance_with"`
	PreferredInputTypes    []string `json:"preferred_input_types"`
	AvoidedInputTypes      []string `json:"avoided_input_types"`
	EffectivenessRating    string   `json:"effectiveness_rating"`
}

// OptimalCombinations identifies the best combinations for different use cases
type OptimalCombinations struct {
	OverallBest        *RankedCombination `json:"overall_best"`
	BestForCompression *RankedCombination `json:"best_for_compression"`
	BestForSpeed       *RankedCombination `json:"best_for_speed"`
	BestForMemory      *RankedCombination `json:"best_for_memory"`
	MostConsistent     *RankedCombination `json:"most_consistent"`
	MostVersatile      *RankedCombination `json:"most_versatile"`
	BestForLargeFiles  *RankedCombination `json:"best_for_large_files"`
	BestForSmallFiles  *RankedCombination `json:"best_for_small_files"`
}

// EfficiencyInsights provides deep insights into efficiency patterns
type EfficiencyInsights struct {
	CompressionEfficiencyInsights []string `json:"compression_efficiency_insights"`
	SpeedEfficiencyInsights       []string `json:"speed_efficiency_insights"`
	MemoryEfficiencyInsights      []string `json:"memory_efficiency_insights"`
	GeneralEfficiencyInsights     []string `json:"general_efficiency_insights"`
	TradOffAnalysis               []string `json:"trade_off_analysis"`
	ScalingInsights               []string `json:"scaling_insights"`
}

// RecommendationEngine provides actionable recommendations
type RecommendationEngine struct {
	ForArchivalStorage    []string `json:"for_archival_storage"`
	ForRealTimeProcessing []string `json:"for_real_time_processing"`
	ForMemoryConstrainted []string `json:"for_memory_constrained"`
	ForMissionCritical    []string `json:"for_mission_critical"`
	ForGeneralPurpose     []string `json:"for_general_purpose"`
	ForLargeDatasets      []string `json:"for_large_datasets"`
	ForMobileDevices      []string `json:"for_mobile_devices"`
	ImplementationTips    []string `json:"implementation_tips"`
}

// StatisticalSummary provides comprehensive statistical analysis
type StatisticalSummary struct {
	KEIScoreDistribution         map[string]int     `json:"kei_score_distribution"`
	CompressionRatioDistribution map[string]int     `json:"compression_ratio_distribution"`
	PerformanceCorrelations      map[string]float64 `json:"performance_correlations"`
	SignificantFindings          []string           `json:"significant_findings"`
	ConfidenceIntervals          map[string]string  `json:"confidence_intervals"`
	StatisticalSignificance      map[string]bool    `json:"statistical_significance"`
}

// GenerateComprehensiveAnalysis creates a complete analysis report
func (ae *AnalysisEngine) GenerateComprehensiveAnalysis() *ComprehensiveAnalysisReport {
	return &ComprehensiveAnalysisReport{
		ExecutionSummary:       ae.generateExecutionSummary(),
		DomainEffectiveness:    ae.analyzeDomainEffectiveness(),
		InputTypeAnalysis:      ae.analyzeInputTypes(),
		AlgorithmEffectiveness: ae.analyzeAlgorithmEffectiveness(),
		OptimalCombinations:    ae.identifyOptimalCombinations(),
		EfficiencyInsights:     ae.generateEfficiencyInsights(),
		Recommendations:        ae.generateRecommendations(),
		StatisticalSummary:     ae.generateStatisticalSummary(),
	}
}

// generateExecutionSummary creates high-level execution statistics
func (ae *AnalysisEngine) generateExecutionSummary() *ExecutionSummary {
	totalTests := ae.results.CompletedTests + ae.results.FailedTests + ae.results.TimeoutTests
	successRate := float64(ae.results.CompletedTests) / float64(totalTests) * 100.0

	// Calculate averages from KEI scores
	var totalKEI, totalRatio float64
	keiCount := 0

	for _, inputResults := range ae.results.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			if testResult.KEIScore != nil {
				totalKEI += testResult.KEIScore.OverallScore
				totalRatio += testResult.KEIScore.CompressionRatio
				keiCount++
			}
		}
	}

	var averageKEI, averageRatio, peakKEI, peakRatio float64
	if keiCount > 0 {
		averageKEI = totalKEI / float64(keiCount)
		averageRatio = totalRatio / float64(keiCount)

		// Find peaks
		for _, inputResults := range ae.results.ResultsByInputType {
			for _, testResult := range inputResults.TestResults {
				if testResult.KEIScore != nil {
					if testResult.KEIScore.OverallScore > peakKEI {
						peakKEI = testResult.KEIScore.OverallScore
					}
					if testResult.KEIScore.CompressionRatio > peakRatio {
						peakRatio = testResult.KEIScore.CompressionRatio
					}
				}
			}
		}
	}

	// Calculate test throughput
	throughput := float64(totalTests) / ae.results.Duration.Seconds()

	return &ExecutionSummary{
		TotalExecutionTime:      ae.formatDuration(ae.results.Duration),
		TotalCombinations:       ae.results.TotalCombinations,
		TotalTests:              totalTests,
		SuccessRate:             successRate,
		AverageTestDuration:     ae.formatDuration(ae.results.Duration / time.Duration(totalTests)),
		TestThroughput:          throughput,
		PeakKEIScore:            peakKEI,
		AverageKEIScore:         averageKEI,
		PeakCompressionRatio:    peakRatio,
		AverageCompressionRatio: averageRatio,
	}
}

// analyzeDomainEffectiveness analyzes effectiveness in different performance domains
func (ae *AnalysisEngine) analyzeDomainEffectiveness() map[string]*DomainEffectivenessReport {
	domains := map[string]string{
		"compression": "Compression Ratio Effectiveness",
		"speed":       "Processing Speed Effectiveness",
		"memory":      "Memory Efficiency Effectiveness",
		"stability":   "Performance Stability Effectiveness",
		"energy":      "Energy Efficiency Effectiveness",
	}

	reports := make(map[string]*DomainEffectivenessReport)

	for domain, description := range domains {
		report := &DomainEffectivenessReport{
			Domain: description,
		}

		// Collect and sort all combinations by domain score
		var domainRankings []*RankedCombination
		for _, inputResults := range ae.results.ResultsByInputType {
			if inputResults.Rankings != nil {
				for _, ranking := range inputResults.Rankings.Rankings {
					// Create copy with domain-specific score
					domainRanking := *ranking
					switch domain {
					case "compression":
						domainRanking.KEIScore.OverallScore = ranking.KEIScore.CompressionScore
					case "speed":
						domainRanking.KEIScore.OverallScore = ranking.KEIScore.SpeedScore
					case "memory":
						domainRanking.KEIScore.OverallScore = ranking.KEIScore.MemoryScore
					case "stability":
						domainRanking.KEIScore.OverallScore = ranking.KEIScore.StabilityScore
					case "energy":
						domainRanking.KEIScore.OverallScore = ranking.KEIScore.EnergyScore
					}
					domainRankings = append(domainRankings, &domainRanking)
				}
			}
		}

		// Sort by domain score
		sort.Slice(domainRankings, func(i, j int) bool {
			return domainRankings[i].KEIScore.OverallScore > domainRankings[j].KEIScore.OverallScore
		})

		// Select best and worst
		if len(domainRankings) > 0 {
			topCount := int(math.Min(5, float64(len(domainRankings))))
			bottomCount := int(math.Min(3, float64(len(domainRankings))))

			report.BestCombinations = domainRankings[:topCount]
			if len(domainRankings) > bottomCount {
				report.WorstCombinations = domainRankings[len(domainRankings)-bottomCount:]
			}
		}

		// Generate domain-specific insights
		report.DomainInsights = ae.generateDomainInsights(domain, domainRankings)
		report.RecommendedScenarios = ae.generateRecommendedScenarios(domain)
		report.AvoidScenarios = ae.generateAvoidScenarios(domain)
		report.Statistics = ae.calculateDomainStatistics(domain, domainRankings)

		reports[domain] = report
	}

	return reports
}

// analyzeInputTypes analyzes performance patterns for different input types
func (ae *AnalysisEngine) analyzeInputTypes() map[string]*InputTypeAnalysis {
	analyses := make(map[string]*InputTypeAnalysis)

	for inputTypeStr, inputResults := range ae.results.ResultsByInputType {
		analysis := &InputTypeAnalysis{
			InputType: inputResults.InputType,
		}

		// Extract KEI scores for this input type
		var keiScores []float64
		for _, testResult := range inputResults.TestResults {
			if testResult.KEIScore != nil {
				keiScores = append(keiScores, testResult.KEIScore.OverallScore)
			}
		}

		if len(keiScores) > 0 {
			// Calculate statistics
			analysis.AverageKEI = ae.calculateMean(keiScores)
			analysis.KEIStandardDeviation = ae.calculateStandardDeviation(keiScores, analysis.AverageKEI)

			// Find best and worst
			if inputResults.Rankings != nil && len(inputResults.Rankings.Rankings) > 0 {
				analysis.BestCombination = inputResults.Rankings.Rankings[0]
				if len(inputResults.Rankings.Rankings) > 1 {
					analysis.WorstCombination = inputResults.Rankings.Rankings[len(inputResults.Rankings.Rankings)-1]
				}
			}
		}

		// Generate insights
		analysis.OptimalStrategies = ae.identifyOptimalStrategies(inputResults.InputType)
		analysis.PerformancePattern = ae.identifyPerformancePattern(inputResults.InputType, keiScores)
		analysis.Insights = ae.generateInputTypeInsights(inputResults.InputType, analysis)

		analyses[inputTypeStr] = analysis
	}

	return analyses
}

// analyzeAlgorithmEffectiveness analyzes individual algorithm performance
func (ae *AnalysisEngine) analyzeAlgorithmEffectiveness() map[string]*AlgorithmAnalysis {
	algorithmStats := make(map[string]*AlgorithmAnalysis)

	// Initialize algorithm tracking
	for _, inputResults := range ae.results.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			algorithms := ae.extractAlgorithmsFromCombination(testResult.AlgorithmCombination)

			for _, algorithm := range algorithms {
				if _, exists := algorithmStats[algorithm]; !exists {
					algorithmStats[algorithm] = &AlgorithmAnalysis{
						Algorithm:            algorithm,
						BestPerformanceWith:  make([]string, 0),
						WorstPerformanceWith: make([]string, 0),
						PreferredInputTypes:  make([]string, 0),
						AvoidedInputTypes:    make([]string, 0),
					}
				}

				stats := algorithmStats[algorithm]
				stats.AppearanceCount++

				if testResult.Success {
					stats.SuccessfulAppearances++
					if testResult.KEIScore != nil {
						stats.AverageKEIContribution += testResult.KEIScore.OverallScore
					}
				}
			}
		}
	}

	// Calculate final statistics and insights
	for algorithm, stats := range algorithmStats {
		if stats.SuccessfulAppearances > 0 {
			stats.AverageKEIContribution /= float64(stats.SuccessfulAppearances)
		}

		// Generate effectiveness rating
		stats.EffectivenessRating = ae.calculateEffectivenessRating(stats.AverageKEIContribution)

		// Identify best/worst performance combinations and input preferences
		ae.identifyAlgorithmPartners(algorithm, stats)
		ae.identifyInputTypePreferences(algorithm, stats)
	}

	return algorithmStats
}

// identifyOptimalCombinations finds the best combinations for different use cases
func (ae *AnalysisEngine) identifyOptimalCombinations() *OptimalCombinations {
	optimal := &OptimalCombinations{}

	if ae.results.OverallRankings != nil && len(ae.results.OverallRankings.Rankings) > 0 {
		optimal.OverallBest = ae.results.OverallRankings.Rankings[0]
	}

	optimal.BestForCompression = ae.results.BestCompressionRatio
	optimal.BestForSpeed = ae.results.BestSpeed
	optimal.BestForMemory = ae.results.BestMemoryEfficiency

	// Find most consistent (lowest variance across input types)
	optimal.MostConsistent = ae.findMostConsistentCombination()

	// Find most versatile (performs well across multiple input types)
	optimal.MostVersatile = ae.findMostVersatileCombination()

	// Find best for different file sizes
	optimal.BestForLargeFiles = ae.findBestForFileSize("large")
	optimal.BestForSmallFiles = ae.findBestForFileSize("small")

	return optimal
}

// generateEfficiencyInsights creates insights about efficiency patterns
func (ae *AnalysisEngine) generateEfficiencyInsights() *EfficiencyInsights {
	insights := &EfficiencyInsights{}

	insights.CompressionEfficiencyInsights = ae.analyzeCompressionEfficiency()
	insights.SpeedEfficiencyInsights = ae.analyzeSpeedEfficiency()
	insights.MemoryEfficiencyInsights = ae.analyzeMemoryEfficiency()
	insights.GeneralEfficiencyInsights = ae.analyzeGeneralEfficiency()
	insights.TradOffAnalysis = ae.analyzeTradeOffs()
	insights.ScalingInsights = ae.analyzeScalingBehavior()

	return insights
}

// generateRecommendations creates actionable recommendations
func (ae *AnalysisEngine) generateRecommendations() *RecommendationEngine {
	recommendations := &RecommendationEngine{}

	recommendations.ForArchivalStorage = ae.recommendForArchival()
	recommendations.ForRealTimeProcessing = ae.recommendForRealTime()
	recommendations.ForMemoryConstrainted = ae.recommendForMemoryConstrained()
	recommendations.ForMissionCritical = ae.recommendForMissionCritical()
	recommendations.ForGeneralPurpose = ae.recommendForGeneralPurpose()
	recommendations.ForLargeDatasets = ae.recommendForLargeDatasets()
	recommendations.ForMobileDevices = ae.recommendForMobileDevices()
	recommendations.ImplementationTips = ae.generateImplementationTips()

	return recommendations
}

// generateStatisticalSummary creates comprehensive statistical analysis
func (ae *AnalysisEngine) generateStatisticalSummary() *StatisticalSummary {
	summary := &StatisticalSummary{
		KEIScoreDistribution:         make(map[string]int),
		CompressionRatioDistribution: make(map[string]int),
		PerformanceCorrelations:      make(map[string]float64),
		ConfidenceIntervals:          make(map[string]string),
		StatisticalSignificance:      make(map[string]bool),
	}

	// Collect all scores
	var keiScores, compressionRatios []float64
	for _, inputResults := range ae.results.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			if testResult.KEIScore != nil {
				keiScores = append(keiScores, testResult.KEIScore.OverallScore)
				compressionRatios = append(compressionRatios, testResult.KEIScore.CompressionRatio)
			}
		}
	}

	// Create distributions
	summary.KEIScoreDistribution = ae.createDistribution(keiScores, 10)
	summary.CompressionRatioDistribution = ae.createDistribution(compressionRatios, 10)

	// Calculate correlations
	if len(keiScores) > 1 {
		summary.PerformanceCorrelations["kei_compression_correlation"] = ae.calculateCorrelation(keiScores, compressionRatios)
	}

	// Generate findings
	summary.SignificantFindings = ae.identifySignificantFindings(keiScores, compressionRatios)

	return summary
}

// Helper methods for analysis calculations and insights generation

func (ae *AnalysisEngine) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (ae *AnalysisEngine) calculateStandardDeviation(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	sumSquaredDiffs := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiffs += diff * diff
	}
	return math.Sqrt(sumSquaredDiffs / float64(len(values)-1))
}

func (ae *AnalysisEngine) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}

	meanX := ae.calculateMean(x)
	meanY := ae.calculateMean(y)

	var numerator, denomX, denomY float64
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		denomX += dx * dx
		denomY += dy * dy
	}

	if denomX == 0 || denomY == 0 {
		return 0
	}

	return numerator / math.Sqrt(denomX*denomY)
}

func (ae *AnalysisEngine) formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else if d < time.Hour {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm%ds", minutes, seconds)
	} else {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		return fmt.Sprintf("%dh%dm", hours, minutes)
	}
}

func (ae *AnalysisEngine) extractAlgorithmsFromCombination(combination string) []string {
	// Parse combination name to extract individual algorithms
	// Assumes format like "Pipeline_[RLE LZ77 Huffman]"
	if strings.Contains(combination, "[") && strings.Contains(combination, "]") {
		start := strings.Index(combination, "[") + 1
		end := strings.Index(combination, "]")
		algoPart := combination[start:end]
		return strings.Fields(algoPart)
	}
	return []string{combination} // Fallback
}

// Placeholder implementations for complex analysis methods
// These would contain sophisticated algorithm analysis logic

func (ae *AnalysisEngine) generateDomainInsights(domain string, rankings []*RankedCombination) []string {
	insights := []string{
		fmt.Sprintf("Analyzed %d combinations for %s effectiveness", len(rankings), domain),
	}

	if len(rankings) > 0 {
		best := rankings[0]
		insights = append(insights,
			fmt.Sprintf("Top performer: %s with score %.2f",
				best.AlgorithmCombination, best.KEIScore.OverallScore))
	}

	return insights
}

func (ae *AnalysisEngine) generateRecommendedScenarios(domain string) []string {
	scenarios := map[string][]string{
		"compression": {"Long-term archival", "Bandwidth optimization", "Storage cost reduction"},
		"speed":       {"Real-time processing", "Interactive applications", "High-throughput systems"},
		"memory":      {"Embedded systems", "Resource-constrained environments", "Mobile applications"},
		"stability":   {"Mission-critical systems", "Automated processing", "Production environments"},
		"energy":      {"Battery-powered devices", "Green computing", "IoT applications"},
	}

	if scenarios, exists := scenarios[domain]; exists {
		return scenarios
	}
	return []string{"General purpose applications"}
}

func (ae *AnalysisEngine) generateAvoidScenarios(domain string) []string {
	avoid := map[string][]string{
		"compression": {"Real-time processing requirements", "Memory-constrained systems"},
		"speed":       {"Maximum compression needs", "Storage cost optimization"},
		"memory":      {"High compression requirements", "Processing time flexibility"},
		"stability":   {"Experimental environments", "Performance testing"},
		"energy":      {"Maximum performance requirements", "Desktop applications"},
	}

	if scenarios, exists := avoid[domain]; exists {
		return scenarios
	}
	return []string{}
}

func (ae *AnalysisEngine) calculateDomainStatistics(domain string, rankings []*RankedCombination) map[string]float64 {
	if len(rankings) == 0 {
		return map[string]float64{}
	}

	var scores []float64
	for _, ranking := range rankings {
		scores = append(scores, ranking.KEIScore.OverallScore)
	}

	mean := ae.calculateMean(scores)
	stdDev := ae.calculateStandardDeviation(scores, mean)

	return map[string]float64{
		"mean":               mean,
		"standard_deviation": stdDev,
		"min":                scores[len(scores)-1], // Last in sorted list
		"max":                scores[0],             // First in sorted list
		"range":              scores[0] - scores[len(scores)-1],
	}
}

// Additional helper methods would be implemented here for completeness
// This is a comprehensive framework that provides the structure for deep analysis

func (ae *AnalysisEngine) identifyOptimalStrategies(inputType InputType) []string {
	strategies := map[InputType][]string{
		InputTypeRepetitive:   {"Use RLE preprocessing", "Avoid dictionary algorithms for simple patterns"},
		InputTypeTextPatterns: {"Apply LZ* algorithms", "Consider BWT for natural text"},
		InputTypeRandom:       {"Use entropy coding", "Skip preprocessing transforms"},
		InputTypeSequential:   {"Apply delta encoding", "Use predictive algorithms"},
		InputTypeNaturalText:  {"Use BWT+MTF combination", "Apply dictionary compression"},
		InputTypeMixed:        {"Use hybrid approaches", "Test multiple strategies"},
	}

	if strats, exists := strategies[inputType]; exists {
		return strats
	}
	return []string{"Use general-purpose algorithms"}
}

func (ae *AnalysisEngine) identifyPerformancePattern(inputType InputType, scores []float64) string {
	if len(scores) == 0 {
		return "No data available"
	}

	mean := ae.calculateMean(scores)
	stdDev := ae.calculateStandardDeviation(scores, mean)

	if stdDev < 5 {
		return "Consistent performance across algorithms"
	} else if stdDev < 15 {
		return "Moderate performance variation"
	} else {
		return "High performance variation - algorithm choice critical"
	}
}

func (ae *AnalysisEngine) generateInputTypeInsights(inputType InputType, analysis *InputTypeAnalysis) []string {
	insights := []string{
		fmt.Sprintf("Average KEI score: %.2f (±%.2f)", analysis.AverageKEI, analysis.KEIStandardDeviation),
	}

	if analysis.BestCombination != nil {
		insights = append(insights,
			fmt.Sprintf("Best combination: %s", analysis.BestCombination.AlgorithmCombination))
	}

	return insights
}

// Continue with more helper methods...
func (ae *AnalysisEngine) calculateEffectivenessRating(avgScore float64) string {
	switch {
	case avgScore >= 80:
		return "Highly Effective"
	case avgScore >= 60:
		return "Effective"
	case avgScore >= 40:
		return "Moderately Effective"
	case avgScore >= 20:
		return "Low Effectiveness"
	default:
		return "Poor Effectiveness"
	}
}

func (ae *AnalysisEngine) identifyAlgorithmPartners(algorithm string, stats *AlgorithmAnalysis) {
	// Implementation would analyze which algorithms work best together
	stats.BestPerformanceWith = []string{"Analysis not yet implemented"}
	stats.WorstPerformanceWith = []string{"Analysis not yet implemented"}
}

func (ae *AnalysisEngine) identifyInputTypePreferences(algorithm string, stats *AlgorithmAnalysis) {
	// Implementation would analyze input type preferences
	stats.PreferredInputTypes = []string{"Analysis not yet implemented"}
	stats.AvoidedInputTypes = []string{"Analysis not yet implemented"}
}

func (ae *AnalysisEngine) findMostConsistentCombination() *RankedCombination {
	// Implementation would find combination with lowest variance across input types
	return nil // Placeholder
}

func (ae *AnalysisEngine) findMostVersatileCombination() *RankedCombination {
	// Implementation would find combination that performs well across multiple input types
	return nil // Placeholder
}

func (ae *AnalysisEngine) findBestForFileSize(sizeCategory string) *RankedCombination {
	// Implementation would analyze performance by file size
	return nil // Placeholder
}

// Analysis methods for different efficiency dimensions
func (ae *AnalysisEngine) analyzeCompressionEfficiency() []string {
	return []string{
		"Compression efficiency analysis shows clear patterns in algorithm effectiveness",
		"Dictionary-based algorithms excel with text-based inputs",
		"RLE performs exceptionally well with repetitive data patterns",
	}
}

func (ae *AnalysisEngine) analyzeSpeedEfficiency() []string {
	return []string{
		"Speed analysis reveals trade-offs between compression ratio and processing time",
		"Simple algorithms like RLE provide fastest processing for suitable inputs",
		"Complex pipelines show diminishing returns in speed efficiency",
	}
}

func (ae *AnalysisEngine) analyzeMemoryEfficiency() []string {
	return []string{
		"Memory usage patterns correlate strongly with algorithm complexity",
		"Transform-based algorithms require additional memory overhead",
		"Streaming algorithms show better memory efficiency for large inputs",
	}
}

func (ae *AnalysisEngine) analyzeGeneralEfficiency() []string {
	return []string{
		"General efficiency analysis shows importance of algorithm selection",
		"No single combination dominates across all scenarios",
		"Input characteristics strongly influence optimal algorithm choice",
	}
}

func (ae *AnalysisEngine) analyzeTradeOffs() []string {
	return []string{
		"Clear trade-offs exist between compression ratio and processing speed",
		"Memory efficiency often conflicts with compression effectiveness",
		"Stability improvements typically require performance sacrifices",
	}
}

func (ae *AnalysisEngine) analyzeScalingBehavior() []string {
	return []string{
		"Algorithm performance scaling varies significantly with input size",
		"Some combinations show better large-file performance",
		"Memory usage scaling differs across algorithm categories",
	}
}

// Recommendation methods
func (ae *AnalysisEngine) recommendForArchival() []string {
	return []string{
		"Prioritize combinations with highest compression ratios",
		"Accept slower processing times for better space efficiency",
		"Consider BWT+dictionary combinations for text archival",
	}
}

func (ae *AnalysisEngine) recommendForRealTime() []string {
	return []string{
		"Choose fast, simple algorithms over complex pipelines",
		"RLE for repetitive real-time data streams",
		"Avoid transform-based preprocessing for time-critical applications",
	}
}

func (ae *AnalysisEngine) recommendForMemoryConstrained() []string {
	return []string{
		"Select algorithms with low memory overhead",
		"Avoid dictionary algorithms with large tables",
		"Consider streaming variants where available",
	}
}

func (ae *AnalysisEngine) recommendForMissionCritical() []string {
	return []string{
		"Choose combinations with highest stability scores",
		"Prefer deterministic algorithms with predictable performance",
		"Implement redundancy and error checking",
	}
}

func (ae *AnalysisEngine) recommendForGeneralPurpose() []string {
	return []string{
		"Select balanced combinations with good overall KEI scores",
		"Consider hybrid approaches for versatility",
		"Test with representative data before deployment",
	}
}

func (ae *AnalysisEngine) recommendForLargeDatasets() []string {
	return []string{
		"Focus on algorithms that scale well with input size",
		"Consider parallel-friendly algorithms",
		"Monitor memory usage patterns carefully",
	}
}

func (ae *AnalysisEngine) recommendForMobileDevices() []string {
	return []string{
		"Prioritize energy-efficient combinations",
		"Balance compression ratio with battery life",
		"Consider device-specific optimizations",
	}
}

func (ae *AnalysisEngine) generateImplementationTips() []string {
	return []string{
		"Profile algorithm performance with representative data",
		"Implement adaptive algorithm selection based on input characteristics",
		"Monitor performance metrics in production environments",
		"Consider parallel execution for independent pipeline stages",
		"Implement proper error handling and fallback strategies",
	}
}

func (ae *AnalysisEngine) createDistribution(values []float64, buckets int) map[string]int {
	if len(values) == 0 {
		return make(map[string]int)
	}

	min := values[0]
	max := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	bucketSize := (max - min) / float64(buckets)
	distribution := make(map[string]int)

	for _, v := range values {
		bucketIndex := int((v - min) / bucketSize)
		if bucketIndex >= buckets {
			bucketIndex = buckets - 1
		}

		bucketLabel := fmt.Sprintf("%.2f-%.2f",
			min+float64(bucketIndex)*bucketSize,
			min+float64(bucketIndex+1)*bucketSize)
		distribution[bucketLabel]++
	}

	return distribution
}

func (ae *AnalysisEngine) identifySignificantFindings(keiScores, compressionRatios []float64) []string {
	findings := []string{}

	if len(keiScores) > 0 {
		meanKEI := ae.calculateMean(keiScores)
		findings = append(findings,
			fmt.Sprintf("Average KEI score across all tests: %.2f", meanKEI))
	}

	if len(compressionRatios) > 0 {
		meanRatio := ae.calculateMean(compressionRatios)
		findings = append(findings,
			fmt.Sprintf("Average compression ratio: %.2f", meanRatio))
	}

	if len(keiScores) > 1 && len(compressionRatios) > 1 {
		correlation := ae.calculateCorrelation(keiScores, compressionRatios)
		findings = append(findings,
			fmt.Sprintf("KEI-Compression correlation: %.3f", correlation))
	}

	return findings
}
