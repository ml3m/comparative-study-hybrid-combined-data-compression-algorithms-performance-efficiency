package benchmarks

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"hybrid-compression-study/pkg/core"
	"hybrid-compression-study/pkg/pipeline"
	"runtime"
)

// BenchmarkConfig holds configuration for benchmark execution
type BenchmarkConfig struct {
	// Algorithm selection
	SelectedAlgorithms []string `json:"selected_algorithms"`
	MinCombinationSize int      `json:"min_combination_size"`
	MaxCombinationSize int      `json:"max_combination_size"`

	// Input configuration
	InputTypes   []InputType  `json:"input_types"`
	InputSizes   []InputSize  `json:"input_sizes"`
	CustomInputs []*TestInput `json:"custom_inputs"`

	// Execution configuration
	TimeoutPerTest     time.Duration `json:"timeout_per_test"`
	MaxConcurrentTests int           `json:"max_concurrent_tests"`
	RetryFailedTests   int           `json:"retry_failed_tests"`

	// KEI configuration
	KEIWeights KEIWeights `json:"kei_weights"`

	// Progress reporting
	VerboseOutput       bool `json:"verbose_output"`
	ShowProgressBar     bool `json:"show_progress_bar"`
	ShowRealTimeResults bool `json:"show_real_time_results"`
	ShowETA             bool `json:"show_eta"`

	// Output configuration
	SaveResults            bool   `json:"save_results"`
	OutputDirectory        string `json:"output_directory"`
	GenerateDetailedReport bool   `json:"generate_detailed_report"`
}

// DefaultBenchmarkConfig provides sensible defaults
func DefaultBenchmarkConfig() *BenchmarkConfig {
	return &BenchmarkConfig{
		SelectedAlgorithms:     []string{}, // Must be set by user
		MinCombinationSize:     2,
		MaxCombinationSize:     3,
		InputTypes:             []InputType{InputTypeRepetitive, InputTypeTextPatterns, InputTypeRandom, InputTypeSequential},
		InputSizes:             []InputSize{{"1KB", 1024}, {"10KB", 10 * 1024}, {"100KB", 100 * 1024}},
		TimeoutPerTest:         30 * time.Second,
		MaxConcurrentTests:     4,
		RetryFailedTests:       1,
		KEIWeights:             DefaultKEIWeights,
		VerboseOutput:          true,
		ShowProgressBar:        true,
		ShowRealTimeResults:    true,
		ShowETA:                true,
		SaveResults:            true,
		OutputDirectory:        "benchmark_results",
		GenerateDetailedReport: true,
	}
}

// BenchmarkResult represents the complete results of a benchmark execution
type BenchmarkResult struct {
	Config            *BenchmarkConfig `json:"config"`
	StartTime         time.Time        `json:"start_time"`
	EndTime           time.Time        `json:"end_time"`
	Duration          time.Duration    `json:"duration"`
	TotalCombinations int64            `json:"total_combinations"`
	CompletedTests    int64            `json:"completed_tests"`
	FailedTests       int64            `json:"failed_tests"`
	TimeoutTests      int64            `json:"timeout_tests"`

	// Results by input type
	ResultsByInputType map[string]*InputTypeResults `json:"results_by_input_type"`

	// Overall rankings
	OverallRankings *KEIRanking `json:"overall_rankings"`

	// Best combinations for each metric
	BestCompressionRatio *RankedCombination `json:"best_compression_ratio"`
	BestSpeed            *RankedCombination `json:"best_speed"`
	BestMemoryEfficiency *RankedCombination `json:"best_memory_efficiency"`
	BestOverall          *RankedCombination `json:"best_overall"`

	// Statistics
	Statistics map[string]interface{} `json:"statistics"`
}

// InputTypeResults holds results for a specific input type
type InputTypeResults struct {
	InputType   InputType              `json:"input_type"`
	TestResults []*TestResult          `json:"test_results"`
	Rankings    *KEIRanking            `json:"rankings"`
	BestOverall *RankedCombination     `json:"best_overall"`
	Statistics  map[string]interface{} `json:"statistics"`
}

// TestResult represents the result of testing one combination on one input
type TestResult struct {
	AlgorithmCombination string                  `json:"algorithm_combination"`
	InputName            string                  `json:"input_name"`
	InputType            InputType               `json:"input_type"`
	InputSize            int64                   `json:"input_size"`
	Success              bool                    `json:"success"`
	Error                string                  `json:"error,omitempty"`
	Timeout              bool                    `json:"timeout"`
	Duration             time.Duration           `json:"duration"`
	CompressionResult    *core.CompressionResult `json:"compression_result,omitempty"`
	KEIScore             *KEIScore               `json:"kei_score,omitempty"`
}

// BenchmarkExecutor orchestrates the entire benchmark execution
type BenchmarkExecutor struct {
	config           *BenchmarkConfig
	inputGenerator   *InputGenerator
	combinationGen   *CombinationGenerator
	keiCalculator    *KEICalculator
	progressReporter *ProgressReporter

	// Execution state
	mutex      sync.RWMutex
	isRunning  bool
	isPaused   bool
	cancelCtx  context.Context
	cancelFunc context.CancelFunc

	// Results accumulation - FIXED for final analysis
	allTestResults []*TestResult
	allKEIScores   []*KEIScore

	// MEMORY LEAK FIX: Reuse algorithm instances
	algorithmCache map[string]core.CompressionAlgorithm
	algorithmMutex sync.Mutex

	// RESULTS FIX: Store final results separately for analysis
	finalResults   []*TestResult
	finalKEIScores []*KEIScore
}

// NewBenchmarkExecutor creates a new benchmark executor
func NewBenchmarkExecutor(config *BenchmarkConfig) (*BenchmarkExecutor, error) {
	if len(config.SelectedAlgorithms) == 0 {
		return nil, fmt.Errorf("no algorithms selected")
	}

	// Create components
	inputGenerator := NewInputGenerator()
	combinationGen := NewCombinationGenerator()
	keiCalculator := NewKEICalculator(config.KEIWeights)

	// Configure combination generator
	err := combinationGen.SetSelectedAlgorithms(config.SelectedAlgorithms)
	if err != nil {
		return nil, fmt.Errorf("failed to set selected algorithms: %w", err)
	}

	err = combinationGen.SetCombinationSizeRange(config.MinCombinationSize, config.MaxCombinationSize)
	if err != nil {
		return nil, fmt.Errorf("failed to set combination size range: %w", err)
	}

	// Estimate total tasks for progress reporting
	totalCombinations := combinationGen.EstimateTotalCombinations()
	totalInputs := int64(len(config.InputTypes)*len(config.InputSizes)) + int64(len(config.CustomInputs))
	totalTasks := totalCombinations * totalInputs

	progressReporter := NewProgressReporter(totalTasks)
	progressReporter.SetConfiguration(
		100*time.Millisecond,
		config.ShowETA,
		config.ShowProgressBar,
		config.ShowRealTimeResults,
	)

	return &BenchmarkExecutor{
		config:           config,
		inputGenerator:   inputGenerator,
		combinationGen:   combinationGen,
		keiCalculator:    keiCalculator,
		progressReporter: progressReporter,
		allTestResults:   make([]*TestResult, 0),
		allKEIScores:     make([]*KEIScore, 0),
		algorithmCache:   make(map[string]core.CompressionAlgorithm),
		finalResults:     make([]*TestResult, 0),
		finalKEIScores:   make([]*KEIScore, 0),
	}, nil
}

// Execute runs the complete benchmark suite
func (be *BenchmarkExecutor) Execute() (*BenchmarkResult, error) {
	be.mutex.Lock()
	if be.isRunning {
		be.mutex.Unlock()
		return nil, fmt.Errorf("benchmark is already running")
	}
	be.isRunning = true
	be.cancelCtx, be.cancelFunc = context.WithCancel(context.Background())
	be.mutex.Unlock()

	defer func() {
		be.mutex.Lock()
		be.isRunning = false
		be.mutex.Unlock()
	}()

	startTime := time.Now()

	// Print initial information
	if be.config.VerboseOutput {
		be.printExecutionPlan()
	}

	// Generate test inputs
	be.progressReporter.PrintTaskStarted("Generating test inputs")
	testInputs, err := be.generateTestInputs()
	if err != nil {
		return nil, fmt.Errorf("failed to generate test inputs: %w", err)
	}
	be.progressReporter.PrintTaskCompleted("Generated test inputs", time.Since(startTime))

	// Generate algorithm combinations
	be.progressReporter.PrintTaskStarted("Generating algorithm combinations")
	combinations, err := be.combinationGen.GenerateAllCombinations()
	if err != nil {
		return nil, fmt.Errorf("failed to generate algorithm combinations: %w", err)
	}
	be.progressReporter.PrintTaskCompleted("Generated algorithm combinations", time.Since(startTime))

	// Update total tasks with actual numbers
	totalTasks := int64(len(combinations) * len(testInputs))
	be.progressReporter.SetTotalTasks(totalTasks)

	// Execute all tests
	be.progressReporter.PrintTaskStarted("Executing benchmark tests")
	err = be.executeAllTests(combinations, testInputs)
	if err != nil {
		return nil, fmt.Errorf("failed to execute tests: %w", err)
	}

	// Generate results
	be.progressReporter.PrintTaskStarted("Analyzing results and generating report")
	result := be.generateBenchmarkResult(startTime, time.Now())
	be.progressReporter.PrintTaskCompleted("Analysis complete", time.Since(startTime))

	// Print final summary
	be.progressReporter.PrintFinalSummary()

	return result, nil
}

// generateTestInputs creates all test inputs based on configuration
func (be *BenchmarkExecutor) generateTestInputs() ([]*TestInput, error) {
	var inputs []*TestInput

	// Generate standard inputs
	for _, inputType := range be.config.InputTypes {
		for _, size := range be.config.InputSizes {
			input, err := be.inputGenerator.GenerateInput(inputType, size.Bytes)
			if err != nil {
				return nil, fmt.Errorf("failed to generate %s input of size %s: %w", inputType, size.Name, err)
			}
			input.Name = fmt.Sprintf("%s_%s", inputType, size.Name)
			inputs = append(inputs, input)
		}
	}

	// Add custom inputs
	inputs = append(inputs, be.config.CustomInputs...)

	return inputs, nil
}

// executeAllTests runs all combination/input pairs
func (be *BenchmarkExecutor) executeAllTests(combinations []*AlgorithmCombination, inputs []*TestInput) error {
	// Create semaphore for concurrency control
	semaphore := make(chan struct{}, be.config.MaxConcurrentTests)

	var wg sync.WaitGroup
	var mutex sync.Mutex

	// Execute tests
	for _, combination := range combinations {
		for _, input := range inputs {
			select {
			case <-be.cancelCtx.Done():
				return fmt.Errorf("benchmark execution cancelled")
			default:
			}

			wg.Add(1)
			go func(combo *AlgorithmCombination, testInput *TestInput) {
				defer wg.Done()

				// Acquire semaphore
				semaphore <- struct{}{}
				defer func() { <-semaphore }()

				// Execute test with retries
				var result *TestResult
				for attempt := 0; attempt <= be.config.RetryFailedTests; attempt++ {
					result = be.executeTest(combo, testInput, attempt)
					if result.Success {
						break
					}
				}

				// Update progress with memory management
				mutex.Lock()

				// Store in final results for analysis (lightweight copy)
				finalResult := &TestResult{
					AlgorithmCombination: result.AlgorithmCombination,
					InputName:            result.InputName,
					InputType:            result.InputType,
					InputSize:            result.InputSize,
					Success:              result.Success,
					Error:                result.Error,
					Timeout:              result.Timeout,
					Duration:             result.Duration,
					KEIScore:             result.KEIScore,
					// Don't store CompressionResult to save memory
				}
				be.finalResults = append(be.finalResults, finalResult)
				if result.KEIScore != nil {
					be.finalKEIScores = append(be.finalKEIScores, result.KEIScore)
				}

				// Keep only last 5 in runtime memory for progress reporting
				be.allTestResults = append(be.allTestResults, result)
				if result.KEIScore != nil {
					be.allKEIScores = append(be.allKEIScores, result.KEIScore)
				}

				// More aggressive cleanup of runtime results - keep only 5 results
				const maxResultsInMemory = 5
				if len(be.allTestResults) > maxResultsInMemory {
					// Explicitly clear old data
					for i := 0; i < len(be.allTestResults)-maxResultsInMemory; i++ {
						oldResult := be.allTestResults[i]
						if oldResult.CompressionResult != nil {
							oldResult.CompressionResult.CompressedData = nil
							oldResult.CompressionResult.Metadata = nil
							oldResult.CompressionResult = nil
						}
						oldResult.KEIScore = nil
					}
					be.allTestResults = be.allTestResults[len(be.allTestResults)-maxResultsInMemory:]
				}
				if len(be.allKEIScores) > maxResultsInMemory {
					be.allKEIScores = be.allKEIScores[len(be.allKEIScores)-maxResultsInMemory:]
				}

				mutex.Unlock()

				// More frequent garbage collection
				if len(be.allTestResults)%2 == 0 {
					runtime.GC()
				}

				// Report progress
				progressResult := &ProgressResult{
					AlgorithmCombination: combo.Name,
					InputType:            string(testInput.Type),
					InputSize:            testInput.Size,
					Status:               "success",
				}

				if result.Success && result.KEIScore != nil {
					progressResult.CompressionRatio = result.KEIScore.CompressionRatio
					progressResult.ProcessingTimeMs = float64(result.Duration.Nanoseconds()) / 1e6
					progressResult.KEIScore = result.KEIScore.OverallScore
				} else {
					progressResult.Status = "failed"
					if result.Timeout {
						progressResult.Status = "timeout"
					}
				}

				be.progressReporter.CompleteTask(progressResult)

			}(combination, input)
		}
	}

	wg.Wait()
	return nil
}

// cleanupMemory aggressively cleans up memory during execution
func (be *BenchmarkExecutor) cleanupMemory() {
	be.mutex.Lock()
	defer be.mutex.Unlock()

	// Clear old results more aggressively
	const maxResults = 10 // Keep only last 10 results
	if len(be.allTestResults) > maxResults {
		// Clear old results explicitly
		for i := 0; i < len(be.allTestResults)-maxResults; i++ {
			result := be.allTestResults[i]
			if result.CompressionResult != nil {
				result.CompressionResult.CompressedData = nil
				result.CompressionResult.Metadata = nil
				result.CompressionResult = nil
			}
			result.KEIScore = nil
		}
		be.allTestResults = be.allTestResults[len(be.allTestResults)-maxResults:]
	}

	if len(be.allKEIScores) > maxResults {
		be.allKEIScores = be.allKEIScores[len(be.allKEIScores)-maxResults:]
	}

	// Force multiple GC cycles
	runtime.GC()
	runtime.GC()
	runtime.GC()
}

// executeTest runs a single combination/input test
func (be *BenchmarkExecutor) executeTest(combination *AlgorithmCombination, input *TestInput, attempt int) *TestResult {
	taskName := fmt.Sprintf("%s on %s", combination.Name, input.Name)
	be.progressReporter.StartTask(taskName)

	result := &TestResult{
		AlgorithmCombination: combination.Name,
		InputName:            input.Name,
		InputType:            input.Type,
		InputSize:            input.Size,
		Success:              false,
	}

	startTime := time.Now()

	// Create timeout context
	ctx, cancel := context.WithTimeout(be.cancelCtx, be.config.TimeoutPerTest)
	defer cancel()

	// Create pipeline from combination
	pipelineCtx, pipelineCancel := context.WithCancel(ctx)
	defer pipelineCancel()

	compressionResult, err := be.executePipeline(pipelineCtx, combination, input)
	result.Duration = time.Since(startTime)

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			result.Timeout = true
			result.Error = "timeout"
		} else {
			result.Error = err.Error()
		}
		return result
	}

	result.Success = true
	result.CompressionResult = compressionResult

	// Calculate KEI score
	keiScore := be.keiCalculator.CalculateKEI(compressionResult, string(input.Type), combination.Name)
	result.KEIScore = keiScore

	return result
}

// getOrCreateAlgorithm gets a cached algorithm instance or creates one if needed
func (be *BenchmarkExecutor) getOrCreateAlgorithm(algorithmName string) (core.CompressionAlgorithm, error) {
	be.algorithmMutex.Lock()
	defer be.algorithmMutex.Unlock()

	// Check cache first
	if algorithm, exists := be.algorithmCache[algorithmName]; exists {
		return algorithm, nil
	}

	// Create new instance only if not cached
	algInfo, exists := be.combinationGen.availableAlgorithms[algorithmName]
	if !exists {
		return nil, fmt.Errorf("algorithm %s not found", algorithmName)
	}

	algorithm, err := algInfo.Creator()
	if err != nil {
		return nil, fmt.Errorf("failed to create algorithm %s: %w", algorithmName, err)
	}

	// Cache for reuse
	be.algorithmCache[algorithmName] = algorithm
	return algorithm, nil
}

// executePipeline creates and executes a compression pipeline
func (be *BenchmarkExecutor) executePipeline(ctx context.Context, combination *AlgorithmCombination, input *TestInput) (*core.CompressionResult, error) {
	// Create pipeline
	pipelineName := fmt.Sprintf("Benchmark_%s", combination.Name)
	compressionPipeline, err := pipeline.NewCompressionPipeline(pipelineName)
	if err != nil {
		return nil, fmt.Errorf("failed to create pipeline: %w", err)
	}

	// Add stages using cached algorithms
	for _, algorithmName := range combination.Algorithms {
		algorithm, err := be.getOrCreateAlgorithm(algorithmName)
		if err != nil {
			return nil, err
		}

		compressionPipeline.AddStage(algorithm, algorithmName, nil)
	}

	// Execute pipeline
	pipelineResult, err := compressionPipeline.Compress(ctx, input.Data)
	if err != nil {
		return nil, fmt.Errorf("pipeline execution failed: %w", err)
	}

	// Convert PipelineResult to CompressionResult
	result := be.convertPipelineResultToCompressionResult(pipelineResult, combination.Name)

	return result, nil
}

// generateBenchmarkResult creates the final benchmark result
func (be *BenchmarkExecutor) generateBenchmarkResult(startTime, endTime time.Time) *BenchmarkResult {
	be.mutex.RLock()
	defer be.mutex.RUnlock()

	// Count test outcomes using final results
	var completed, failed, timeout int64
	for _, result := range be.finalResults {
		if result.Success {
			completed++
		} else if result.Timeout {
			timeout++
		} else {
			failed++
		}
	}

	// Group results by input type using final results
	resultsByInputType := make(map[string]*InputTypeResults)
	for _, result := range be.finalResults {
		inputTypeStr := string(result.InputType)
		if _, exists := resultsByInputType[inputTypeStr]; !exists {
			resultsByInputType[inputTypeStr] = &InputTypeResults{
				InputType:   result.InputType,
				TestResults: make([]*TestResult, 0),
			}
		}
		resultsByInputType[inputTypeStr].TestResults = append(resultsByInputType[inputTypeStr].TestResults, result)
	}

	// Generate rankings for each input type
	for inputType, inputResults := range resultsByInputType {
		var inputKEIScores []*KEIScore
		for _, testResult := range inputResults.TestResults {
			if testResult.KEIScore != nil {
				inputKEIScores = append(inputKEIScores, testResult.KEIScore)
			}
		}

		if len(inputKEIScores) > 0 {
			// Sort by KEI score
			sort.Slice(inputKEIScores, func(i, j int) bool {
				return inputKEIScores[i].OverallScore > inputKEIScores[j].OverallScore
			})

			inputResults.Rankings = be.keiCalculator.CreateRanking(inputKEIScores, inputType, 0)
			if len(inputResults.Rankings.Rankings) > 0 {
				inputResults.BestOverall = inputResults.Rankings.Rankings[0]
			}
		}
	}

	// Generate overall rankings using final KEI scores
	var overallKEIScores []*KEIScore
	overallKEIScores = append(overallKEIScores, be.finalKEIScores...)
	sort.Slice(overallKEIScores, func(i, j int) bool {
		return overallKEIScores[i].OverallScore > overallKEIScores[j].OverallScore
	})

	overallRankings := be.keiCalculator.CreateRanking(overallKEIScores, "all", 0)

	// Debug logging
	if be.config.VerboseOutput {
		fmt.Printf("🔍 Results Analysis: %d final results, %d KEI scores, %d rankings generated\n",
			len(be.finalResults), len(overallKEIScores), len(overallRankings.Rankings))
	}

	// Find best in each category
	var bestCompression, bestSpeed, bestMemory, bestOverall *RankedCombination
	if len(overallRankings.Rankings) > 0 {
		bestOverall = overallRankings.Rankings[0]

		// Find best in each dimension
		for _, ranked := range overallRankings.Rankings {
			if bestCompression == nil || ranked.KEIScore.CompressionScore > bestCompression.KEIScore.CompressionScore {
				bestCompression = ranked
			}
			if bestSpeed == nil || ranked.KEIScore.SpeedScore > bestSpeed.KEIScore.SpeedScore {
				bestSpeed = ranked
			}
			if bestMemory == nil || ranked.KEIScore.MemoryScore > bestMemory.KEIScore.MemoryScore {
				bestMemory = ranked
			}
		}
	}

	return &BenchmarkResult{
		Config:               be.config,
		StartTime:            startTime,
		EndTime:              endTime,
		Duration:             endTime.Sub(startTime),
		TotalCombinations:    be.combinationGen.EstimateTotalCombinations(),
		CompletedTests:       completed,
		FailedTests:          failed,
		TimeoutTests:         timeout,
		ResultsByInputType:   resultsByInputType,
		OverallRankings:      overallRankings,
		BestCompressionRatio: bestCompression,
		BestSpeed:            bestSpeed,
		BestMemoryEfficiency: bestMemory,
		BestOverall:          bestOverall,
		Statistics:           be.keiCalculator.GetStatistics(),
	}
}

// printExecutionPlan prints information about what will be executed
func (be *BenchmarkExecutor) printExecutionPlan() {
	stats := be.combinationGen.GetCombinationStats()

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("                 BENCHMARK EXECUTION PLAN")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Selected Algorithms: %v\n", be.config.SelectedAlgorithms)
	fmt.Printf("Combination Size Range: %d - %d algorithms\n", be.config.MinCombinationSize, be.config.MaxCombinationSize)
	fmt.Printf("Total Combinations: %d\n", stats["total_combinations"])

	fmt.Printf("Input Types: %v\n", be.config.InputTypes)
	fmt.Printf("Input Sizes: ")
	for i, size := range be.config.InputSizes {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(size.Name)
	}
	fmt.Println()

	totalInputs := len(be.config.InputTypes)*len(be.config.InputSizes) + len(be.config.CustomInputs)
	fmt.Printf("Total Test Inputs: %d\n", totalInputs)

	fmt.Printf("Total Tests to Execute: %d\n", stats["total_combinations"].(int64)*int64(totalInputs))
	fmt.Printf("Max Concurrent Tests: %d\n", be.config.MaxConcurrentTests)
	fmt.Printf("Timeout per Test: %s\n", be.config.TimeoutPerTest)

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
}

// Pause pauses the benchmark execution
func (be *BenchmarkExecutor) Pause() {
	be.mutex.Lock()
	defer be.mutex.Unlock()

	if be.isRunning && !be.isPaused {
		be.isPaused = true
		be.progressReporter.Pause()
	}
}

// Resume resumes the benchmark execution
func (be *BenchmarkExecutor) Resume() {
	be.mutex.Lock()
	defer be.mutex.Unlock()

	if be.isRunning && be.isPaused {
		be.isPaused = false
		be.progressReporter.Resume()
	}
}

// Cancel cancels the benchmark execution
func (be *BenchmarkExecutor) Cancel() {
	be.mutex.Lock()
	defer be.mutex.Unlock()

	if be.isRunning && be.cancelFunc != nil {
		be.cancelFunc()
	}
}

// GetProgress returns current execution progress
func (be *BenchmarkExecutor) GetProgress() map[string]interface{} {
	be.mutex.RLock()
	defer be.mutex.RUnlock()

	progress := be.progressReporter.GetProgress()
	progress["is_running"] = be.isRunning
	progress["is_paused"] = be.isPaused
	progress["total_results"] = len(be.allTestResults)
	progress["successful_results"] = len(be.allKEIScores)

	return progress
}

// convertPipelineResultToCompressionResult converts a PipelineResult to CompressionResult
func (be *BenchmarkExecutor) convertPipelineResultToCompressionResult(pipelineResult *pipeline.PipelineResult, algorithmName string) *core.CompressionResult {
	// Calculate basic metrics
	compressionRatio := 1.0
	if pipelineResult.CompressedSize > 0 {
		compressionRatio = float64(pipelineResult.OriginalSize) / float64(pipelineResult.CompressedSize)
	}

	// Create precision metrics with available data
	precisionMetrics := core.StatisticalPrecisionMetrics{
		CompressionTimeNs:   int64(pipelineResult.TotalCompressionTime * 1e9), // Convert to nanoseconds
		DecompressionTimeNs: int64(pipelineResult.TotalDecompressionTime * 1e9),
		TotalTimeNs:         int64((pipelineResult.TotalCompressionTime + pipelineResult.TotalDecompressionTime) * 1e9),

		// Calculate throughput
		ThroughputMBPS:           float64(pipelineResult.OriginalSize) / (1024 * 1024) / pipelineResult.TotalCompressionTime,
		ThroughputBytesPerSecond: float64(pipelineResult.OriginalSize) / pipelineResult.TotalCompressionTime,
		TimePerByteNs:            pipelineResult.TotalCompressionTime * 1e9 / float64(pipelineResult.OriginalSize),

		// Default values for missing metrics
		MemoryPeakBytes:            pipelineResult.CompressedSize * 2, // Estimate
		MemoryDeltaBytes:           pipelineResult.CompressedSize,
		MemoryEfficiencyRatio:      float64(pipelineResult.CompressedSize) / float64(pipelineResult.OriginalSize),
		CPUPercentAvg:              50.0, // Default estimate
		CPUPercentPeak:             75.0, // Default estimate
		DeterminismScore:           0.95, // Default high determinism
		EnergyEfficiencyBytesPerNs: float64(pipelineResult.OriginalSize) / (pipelineResult.TotalCompressionTime * 1e9),

		// Set formatted times
		CompressionTimeFormatted:   fmt.Sprintf("%.3fms", pipelineResult.TotalCompressionTime*1000),
		DecompressionTimeFormatted: fmt.Sprintf("%.3fms", pipelineResult.TotalDecompressionTime*1000),
		TotalTimeFormatted:         fmt.Sprintf("%.3fms", (pipelineResult.TotalCompressionTime+pipelineResult.TotalDecompressionTime)*1000),
	}

	return &core.CompressionResult{
		CompressedData:   nil, // Don't store compressed data to prevent memory leaks
		OriginalSize:     pipelineResult.OriginalSize,
		CompressedSize:   pipelineResult.CompressedSize,
		CompressionRatio: compressionRatio,
		CompressionTime:  pipelineResult.TotalCompressionTime,
		AlgorithmName:    algorithmName,
		Metadata:         pipelineResult.Metadata,
		PrecisionMetrics: precisionMetrics,
	}
}
