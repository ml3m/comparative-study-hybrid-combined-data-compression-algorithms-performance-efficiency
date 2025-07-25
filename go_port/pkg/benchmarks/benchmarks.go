// Package benchmarks provides comprehensive benchmarking infrastructure for compression algorithms.
//
// This package contains sophisticated test data generators, algorithm combination frameworks,
// Key Efficiency Index (KEI) calculations, and comprehensive analysis engines designed to
// thoroughly evaluate compression algorithm combinations across multiple dimensions.
//
// The benchmarking system implements a brute force approach to test all meaningful
// combinations of compression algorithms against engineered test inputs, providing
// detailed efficiency analysis and actionable recommendations.
package benchmarks

import (
	"fmt"
	"time"
)

// BenchmarkSuite represents the main benchmarking suite
type BenchmarkSuite struct {
	inputGenerator   *InputGenerator
	combinationGen   *CombinationGenerator
	keiCalculator    *KEICalculator
	analysisEngine   *AnalysisEngine
	progressReporter *ProgressReporter
	executor         *BenchmarkExecutor
}

// NewBenchmarkSuite creates a new benchmark suite with default configuration
func NewBenchmarkSuite() (*BenchmarkSuite, error) {
	config := DefaultBenchmarkConfig()

	// Set some reasonable defaults for a quick start
	config.SelectedAlgorithms = []string{"RLE", "LZ77", "Huffman"}
	config.MinCombinationSize = 2
	config.MaxCombinationSize = 3
	config.InputTypes = []InputType{
		InputTypeRepetitive,
		InputTypeTextPatterns,
		InputTypeRandom,
	}
	config.InputSizes = []InputSize{
		{"1KB", 1024},
		{"10KB", 10 * 1024},
	}

	executor, err := NewBenchmarkExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create benchmark executor: %w", err)
	}

	return &BenchmarkSuite{
		inputGenerator: NewInputGenerator(),
		combinationGen: NewCombinationGenerator(),
		keiCalculator:  NewDefaultKEICalculator(),
		executor:       executor,
	}, nil
}

// NewBenchmarkSuiteWithConfig creates a benchmark suite with custom configuration
func NewBenchmarkSuiteWithConfig(config *BenchmarkConfig) (*BenchmarkSuite, error) {
	executor, err := NewBenchmarkExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create benchmark executor: %w", err)
	}

	return &BenchmarkSuite{
		inputGenerator: NewInputGenerator(),
		combinationGen: NewCombinationGenerator(),
		keiCalculator:  NewKEICalculator(config.KEIWeights),
		executor:       executor,
	}, nil
}

// RunQuickBenchmark runs a quick benchmark with minimal configuration
func (bs *BenchmarkSuite) RunQuickBenchmark() (*BenchmarkResult, error) {
	return bs.executor.Execute()
}

// RunFullBenchmark runs a comprehensive benchmark with all available options
func (bs *BenchmarkSuite) RunFullBenchmark(algorithms []string) (*BenchmarkResult, error) {
	// Update configuration for full benchmark
	config := bs.executor.config
	config.SelectedAlgorithms = algorithms
	config.MinCombinationSize = 2
	config.MaxCombinationSize = 4
	config.InputTypes = []InputType{
		InputTypeRepetitive,
		InputTypeTextPatterns,
		InputTypeRandom,
		InputTypeSequential,
		InputTypeNaturalText,
		InputTypeStructuredBin,
		InputTypeMixed,
		InputTypeSparse,
		InputTypeAlternating,
		InputTypeLog,
	}

	// Create new executor with updated config
	executor, err := NewBenchmarkExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create full benchmark executor: %w", err)
	}

	return executor.Execute()
}

// GenerateTestInputs generates all standard test inputs
func (bs *BenchmarkSuite) GenerateTestInputs() ([]*TestInput, error) {
	return bs.inputGenerator.GenerateAllInputs()
}

// GenerateCustomInput creates a custom test input
func (bs *BenchmarkSuite) GenerateCustomInput(inputType InputType, size int64) (*TestInput, error) {
	return bs.inputGenerator.GenerateInput(inputType, size)
}

// GetAvailableAlgorithms returns information about available algorithms
func (bs *BenchmarkSuite) GetAvailableAlgorithms() map[string]*AlgorithmInfo {
	return bs.combinationGen.GetAvailableAlgorithms()
}

// EstimateCombinations estimates the number of combinations for given parameters
func (bs *BenchmarkSuite) EstimateCombinations(algorithms []string, minSize, maxSize int) (int64, error) {
	combGen := NewCombinationGenerator()
	err := combGen.SetSelectedAlgorithms(algorithms)
	if err != nil {
		return 0, err
	}

	err = combGen.SetCombinationSizeRange(minSize, maxSize)
	if err != nil {
		return 0, err
	}

	return combGen.EstimateTotalCombinations(), nil
}

// AnalyzeResults performs comprehensive analysis on benchmark results
func (bs *BenchmarkSuite) AnalyzeResults(results *BenchmarkResult) *ComprehensiveAnalysisReport {
	analysisEngine := NewAnalysisEngine(results)
	return analysisEngine.GenerateComprehensiveAnalysis()
}

// CreateKEIRanking creates a KEI-based ranking from scores
func (bs *BenchmarkSuite) CreateKEIRanking(scores []*KEIScore, inputType string, inputSize int64) *KEIRanking {
	return bs.keiCalculator.CreateRanking(scores, inputType, inputSize)
}

// BenchmarkSingleCombination benchmarks a specific algorithm combination
func (bs *BenchmarkSuite) BenchmarkSingleCombination(algorithms []string, input *TestInput) (*TestResult, error) {
	// Create a minimal configuration for single combination test
	config := &BenchmarkConfig{
		SelectedAlgorithms:     algorithms,
		MinCombinationSize:     len(algorithms),
		MaxCombinationSize:     len(algorithms),
		InputTypes:             []InputType{input.Type},
		InputSizes:             []InputSize{{"custom", input.Size}},
		TimeoutPerTest:         30 * time.Second,
		MaxConcurrentTests:     1,
		RetryFailedTests:       0,
		KEIWeights:             DefaultKEIWeights,
		VerboseOutput:          false,
		ShowProgressBar:        false,
		ShowRealTimeResults:    false,
		ShowETA:                false,
		SaveResults:            false,
		GenerateDetailedReport: false,
	}

	executor, err := NewBenchmarkExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create executor: %w", err)
	}

	// Generate combination
	combGen := NewCombinationGenerator()
	combGen.SetSelectedAlgorithms(algorithms)
	combGen.SetCombinationSizeRange(len(algorithms), len(algorithms))

	combinations, err := combGen.GenerateAllCombinations()
	if err != nil {
		return nil, fmt.Errorf("failed to generate combination: %w", err)
	}

	if len(combinations) == 0 {
		return nil, fmt.Errorf("no valid combinations generated")
	}

	// Execute single test
	return executor.executeTest(combinations[0], input, 0), nil
}

// GetKEIWeightPresets returns predefined KEI weight configurations
func GetKEIWeightPresets() map[string]KEIWeights {
	return map[string]KEIWeights{
		"default":     DefaultKEIWeights,
		"performance": PerformanceWeights,
		"compression": CompressionWeights,
		"memory":      MemoryWeights,
	}
}

// ValidateConfiguration validates a benchmark configuration
func ValidateConfiguration(config *BenchmarkConfig) error {
	if len(config.SelectedAlgorithms) == 0 {
		return fmt.Errorf("no algorithms selected")
	}

	if config.MinCombinationSize < 1 {
		return fmt.Errorf("minimum combination size must be at least 1")
	}

	if config.MaxCombinationSize < config.MinCombinationSize {
		return fmt.Errorf("maximum combination size must be >= minimum")
	}

	if config.MaxCombinationSize > len(config.SelectedAlgorithms) {
		return fmt.Errorf("maximum combination size cannot exceed number of selected algorithms")
	}

	if len(config.InputTypes) == 0 {
		return fmt.Errorf("no input types selected")
	}

	if len(config.InputSizes) == 0 {
		return fmt.Errorf("no input sizes selected")
	}

	if config.TimeoutPerTest <= 0 {
		return fmt.Errorf("timeout per test must be positive")
	}

	if config.MaxConcurrentTests < 1 {
		return fmt.Errorf("max concurrent tests must be at least 1")
	}

	// Validate KEI weights sum to approximately 1.0
	total := config.KEIWeights.CompressionRatio +
		config.KEIWeights.Speed +
		config.KEIWeights.Memory +
		config.KEIWeights.Stability +
		config.KEIWeights.Energy

	if total < 0.99 || total > 1.01 {
		return fmt.Errorf("KEI weights must sum to 1.0 (current sum: %.3f)", total)
	}

	return nil
}

// PrintSystemInfo prints information about the benchmarking system
func PrintSystemInfo() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("         HYBRID COMPRESSION ALGORITHM BENCHMARK SYSTEM")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("Version: %s\n", "1.0.0")
	fmt.Printf("Built: %s\n", time.Now().Format("2006-01-02"))
	fmt.Println()
	fmt.Println("This system provides comprehensive benchmarking capabilities for")
	fmt.Println("compression algorithm combinations using brute force analysis.")
	fmt.Println()
	fmt.Println("Features:")
	fmt.Println("  • 10 engineered input types for comprehensive testing")
	fmt.Println("  • Brute force algorithm combination generation")
	fmt.Println("  • Key Efficiency Index (KEI) scoring system")
	fmt.Println("  • Real-time progress reporting with ETA")
	fmt.Println("  • Comprehensive analysis and reporting")
	fmt.Println("  • Configurable execution parameters")
	fmt.Println("  • Multi-dimensional efficiency analysis")
	fmt.Println("═══════════════════════════════════════════════════════════════")
}

// Example usage functions for documentation

// ExampleQuickBenchmark demonstrates basic usage
func ExampleQuickBenchmark() error {
	// Create benchmark suite
	suite, err := NewBenchmarkSuite()
	if err != nil {
		return fmt.Errorf("failed to create benchmark suite: %w", err)
	}

	// Run quick benchmark
	results, err := suite.RunQuickBenchmark()
	if err != nil {
		return fmt.Errorf("benchmark failed: %w", err)
	}

	// Display basic results
	fmt.Printf("Benchmark completed in %s\n", results.Duration)
	if results.BestOverall != nil {
		fmt.Printf("Best combination: %s (KEI: %.2f)\n",
			results.BestOverall.AlgorithmCombination,
			results.BestOverall.KEIScore.OverallScore)
	}

	return nil
}

// ExampleFullAnalysis demonstrates comprehensive analysis
func ExampleFullAnalysis() error {
	// Create benchmark suite with custom config
	config := DefaultBenchmarkConfig()
	config.SelectedAlgorithms = []string{"RLE", "BWT", "LZ77", "Huffman"}
	config.GenerateDetailedReport = true

	suite, err := NewBenchmarkSuiteWithConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create benchmark suite: %w", err)
	}

	// Run benchmark
	results, err := suite.RunFullBenchmark(config.SelectedAlgorithms)
	if err != nil {
		return fmt.Errorf("benchmark failed: %w", err)
	}

	// Perform comprehensive analysis
	analysis := suite.AnalyzeResults(results)

	// Display analysis highlights
	fmt.Printf("Analysis Summary:\n")
	fmt.Printf("  Total Tests: %d\n", analysis.ExecutionSummary.TotalTests)
	fmt.Printf("  Success Rate: %.1f%%\n", analysis.ExecutionSummary.SuccessRate)
	fmt.Printf("  Average KEI: %.2f\n", analysis.ExecutionSummary.AverageKEIScore)

	return nil
}

// ExampleCustomInput demonstrates custom input generation
func ExampleCustomInput() error {
	suite, err := NewBenchmarkSuite()
	if err != nil {
		return err
	}

	// Generate custom input
	customInput, err := suite.GenerateCustomInput(InputTypeRepetitive, 50*1024) // 50KB
	if err != nil {
		return fmt.Errorf("failed to generate input: %w", err)
	}

	// Test single combination
	result, err := suite.BenchmarkSingleCombination(
		[]string{"RLE", "Huffman"},
		customInput,
	)
	if err != nil {
		return fmt.Errorf("benchmark failed: %w", err)
	}

	if result.Success && result.KEIScore != nil {
		fmt.Printf("Custom test result:\n")
		fmt.Printf("  KEI Score: %.2f\n", result.KEIScore.OverallScore)
		fmt.Printf("  Compression Ratio: %.2f\n", result.KEIScore.CompressionRatio)
		fmt.Printf("  Duration: %s\n", result.Duration)
	}

	return nil
}

// Package initialization and version info
var (
	// Version information
	Version   = "1.0.0"
	BuildTime = time.Now().Format("2006-01-02 15:04:05")

	// Default configurations that can be modified
	DefaultTimeout     = 30 * time.Second
	DefaultConcurrency = 4
	DefaultRetries     = 1
	DefaultOutputDir   = "benchmark_results"
)

// GetVersion returns version information
func GetVersion() string {
	return fmt.Sprintf("Hybrid Compression Benchmark v%s (built %s)", Version, BuildTime)
}
