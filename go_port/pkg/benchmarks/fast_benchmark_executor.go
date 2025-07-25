package benchmarks

import (
	"context"
	"fmt"
	"sync"
	"time"

	"hybrid-compression-study/pkg/core"
	"hybrid-compression-study/pkg/pipeline"
)

// FastBenchmarkExecutor provides high-performance benchmarking without aerospace monitoring overhead
type FastBenchmarkExecutor struct {
	config           *BenchmarkConfig
	inputGenerator   *InputGenerator
	combinationGen   *FastCombinationGenerator
	keiCalculator    *KEICalculator
	progressReporter *ProgressReporter

	// Execution state
	mutex      sync.RWMutex
	isRunning  bool
	isPaused   bool
	cancelCtx  context.Context
	cancelFunc context.CancelFunc

	// Lightweight results storage
	finalResults   []*TestResult
	finalKEIScores []*KEIScore

	// Algorithm instance cache (reuse across tests)
	algorithmCache map[string]core.CompressionAlgorithm
	algorithmMutex sync.Mutex
}

// NewFastBenchmarkExecutor creates a performance-optimized benchmark executor
func NewFastBenchmarkExecutor(config *BenchmarkConfig) (*FastBenchmarkExecutor, error) {
	if len(config.SelectedAlgorithms) == 0 {
		return nil, fmt.Errorf("no algorithms selected")
	}

	// Create fast components
	inputGenerator := NewInputGenerator()
	combinationGen := NewFastCombinationGenerator()
	keiCalculator := NewKEICalculator(config.KEIWeights)

	// Configure combination generator with fast algorithms only
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
		200*time.Millisecond, // Less frequent updates for better performance
		config.ShowETA,
		config.ShowProgressBar,
		config.ShowRealTimeResults,
	)

	return &FastBenchmarkExecutor{
		config:           config,
		inputGenerator:   inputGenerator,
		combinationGen:   combinationGen,
		keiCalculator:    keiCalculator,
		progressReporter: progressReporter,
		finalResults:     make([]*TestResult, 0),
		finalKEIScores:   make([]*KEIScore, 0),
		algorithmCache:   make(map[string]core.CompressionAlgorithm),
	}, nil
}

// Execute runs the complete benchmark suite with optimized performance
func (fbe *FastBenchmarkExecutor) Execute() (*BenchmarkResult, error) {
	fbe.mutex.Lock()
	if fbe.isRunning {
		fbe.mutex.Unlock()
		return nil, fmt.Errorf("benchmark is already running")
	}
	fbe.isRunning = true
	fbe.cancelCtx, fbe.cancelFunc = context.WithCancel(context.Background())
	fbe.mutex.Unlock()

	defer func() {
		fbe.mutex.Lock()
		fbe.isRunning = false
		fbe.mutex.Unlock()
	}()

	startTime := time.Now()

	// Generate test inputs
	if fbe.config.VerboseOutput {
		fmt.Println("🚀 Generating test inputs...")
	}
	testInputs, err := fbe.generateTestInputs()
	if err != nil {
		return nil, fmt.Errorf("failed to generate test inputs: %w", err)
	}

	// Generate algorithm combinations
	if fbe.config.VerboseOutput {
		fmt.Println("🔧 Generating algorithm combinations...")
	}
	combinations, err := fbe.combinationGen.GenerateAllCombinations()
	if err != nil {
		return nil, fmt.Errorf("failed to generate algorithm combinations: %w", err)
	}

	// Update total tasks with actual numbers
	totalTasks := int64(len(combinations) * len(testInputs))
	fbe.progressReporter.SetTotalTasks(totalTasks)

	if fbe.config.VerboseOutput {
		fmt.Printf("⚡ Running %d tests (%d combinations × %d inputs)...\n",
			totalTasks, len(combinations), len(testInputs))
	}

	// Execute all tests with optimized execution
	err = fbe.executeAllTestsFast(combinations, testInputs)
	if err != nil {
		return nil, fmt.Errorf("failed to execute tests: %w", err)
	}

	// Generate results
	if fbe.config.VerboseOutput {
		fmt.Println("📊 Analyzing results...")
	}
	result := fbe.generateBenchmarkResult(startTime, time.Now())

	// Print final summary
	fbe.progressReporter.PrintFinalSummary()

	return result, nil
}

// executeAllTestsFast runs all tests with performance optimizations
func (fbe *FastBenchmarkExecutor) executeAllTestsFast(combinations []*AlgorithmCombination, inputs []*TestInput) error {
	// Pre-allocate algorithm instances to avoid creation overhead
	fbe.preWarmAlgorithmCache(combinations)

	var wg sync.WaitGroup
	var mutex sync.Mutex

	// Semaphore for concurrency control
	semaphore := make(chan struct{}, fbe.config.MaxConcurrentTests)

	for _, combination := range combinations {
		for _, input := range inputs {
			select {
			case <-fbe.cancelCtx.Done():
				return fmt.Errorf("benchmark execution cancelled")
			default:
			}

			wg.Add(1)
			go func(combo *AlgorithmCombination, testInput *TestInput) {
				defer wg.Done()

				// Acquire semaphore
				semaphore <- struct{}{}
				defer func() { <-semaphore }()

				// Execute test (no retries for performance)
				result := fbe.executeTestFast(combo, testInput)

				// Store results efficiently
				mutex.Lock()
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
				fbe.finalResults = append(fbe.finalResults, finalResult)
				if result.KEIScore != nil {
					fbe.finalKEIScores = append(fbe.finalKEIScores, result.KEIScore)
				}
				mutex.Unlock()

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

				fbe.progressReporter.CompleteTask(progressResult)

			}(combination, input)
		}
	}

	wg.Wait()
	return nil
}

// preWarmAlgorithmCache pre-creates all algorithm instances to avoid overhead
func (fbe *FastBenchmarkExecutor) preWarmAlgorithmCache(combinations []*AlgorithmCombination) {
	algorithmNames := make(map[string]bool)

	// Collect unique algorithm names
	for _, combo := range combinations {
		for _, algName := range combo.Algorithms {
			algorithmNames[algName] = true
		}
	}

	// Pre-create all algorithms
	for algName := range algorithmNames {
		_, _ = fbe.getOrCreateAlgorithm(algName) // Ignore errors for robustness
	}
}

// executeTestFast runs a single test with performance optimizations
func (fbe *FastBenchmarkExecutor) executeTestFast(combination *AlgorithmCombination, input *TestInput) *TestResult {
	result := &TestResult{
		AlgorithmCombination: combination.Name,
		InputName:            input.Name,
		InputType:            input.Type,
		InputSize:            input.Size,
		Success:              false,
	}

	startTime := time.Now()

	// Create timeout context
	ctx, cancel := context.WithTimeout(fbe.cancelCtx, fbe.config.TimeoutPerTest)
	defer cancel()

	// Execute pipeline with optimizations
	compressionResult, err := fbe.executePipelineFast(ctx, combination, input)
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
	// Don't store CompressionResult to save memory

	// Calculate KEI score (lightweight)
	keiScore := fbe.keiCalculator.CalculateKEI(compressionResult, string(input.Type), combination.Name)
	result.KEIScore = keiScore

	return result
}

// getOrCreateAlgorithm gets a cached algorithm instance or creates one if needed
func (fbe *FastBenchmarkExecutor) getOrCreateAlgorithm(algorithmName string) (core.CompressionAlgorithm, error) {
	fbe.algorithmMutex.Lock()
	defer fbe.algorithmMutex.Unlock()

	// Check cache first
	if algorithm, exists := fbe.algorithmCache[algorithmName]; exists {
		return algorithm, nil
	}

	// Create new instance only if not cached
	algInfo, exists := fbe.combinationGen.availableAlgorithms[algorithmName]
	if !exists {
		return nil, fmt.Errorf("algorithm %s not found", algorithmName)
	}

	algorithm, err := algInfo.Creator()
	if err != nil {
		return nil, fmt.Errorf("failed to create algorithm %s: %w", algorithmName, err)
	}

	// Cache for reuse
	fbe.algorithmCache[algorithmName] = algorithm
	return algorithm, nil
}

// executePipelineFast creates and executes a compression pipeline with optimizations
func (fbe *FastBenchmarkExecutor) executePipelineFast(ctx context.Context, combination *AlgorithmCombination, input *TestInput) (*core.CompressionResult, error) {
	// Create pipeline (lightweight)
	pipelineName := fmt.Sprintf("Fast_%s", combination.Name)
	compressionPipeline, err := pipeline.NewCompressionPipeline(pipelineName)
	if err != nil {
		return nil, fmt.Errorf("failed to create pipeline: %w", err)
	}

	// Add stages using cached algorithms
	for _, algorithmName := range combination.Algorithms {
		algorithm, err := fbe.getOrCreateAlgorithm(algorithmName)
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

	// Convert PipelineResult to CompressionResult (lightweight)
	result := &core.CompressionResult{
		CompressedData:   nil, // Don't store for memory efficiency
		OriginalSize:     pipelineResult.OriginalSize,
		CompressedSize:   pipelineResult.CompressedSize,
		CompressionRatio: pipelineResult.CompressionRatio,
		CompressionTime:  pipelineResult.CompressionTime,
		AlgorithmName:    combination.Name,
		Metadata:         pipelineResult.Metadata,
		PrecisionMetrics: pipelineResult.PrecisionMetrics,
	}

	return result, nil
}

// generateTestInputs creates all test inputs based on configuration
func (fbe *FastBenchmarkExecutor) generateTestInputs() ([]*TestInput, error) {
	var allInputs []*TestInput

	// Generate configured input types and sizes
	for _, inputType := range fbe.config.InputTypes {
		for _, inputSize := range fbe.config.InputSizes {
			input, err := fbe.inputGenerator.GenerateInput(inputType, inputSize.Bytes)
			if err != nil {
				return nil, fmt.Errorf("failed to generate input %s/%s: %w", inputType, inputSize.Name, err)
			}
			allInputs = append(allInputs, input)
		}
	}

	// Add custom inputs
	allInputs = append(allInputs, fbe.config.CustomInputs...)

	return allInputs, nil
}

// generateBenchmarkResult creates the final benchmark result using collected data
func (fbe *FastBenchmarkExecutor) generateBenchmarkResult(startTime, endTime time.Time) *BenchmarkResult {
	// Use the analysis engine to process final results
	analysisEngine := NewAnalysisEngine()

	// Convert final results to the format expected by analysis engine
	resultsByInputType := make(map[string]*InputTypeResults)

	for _, testResult := range fbe.finalResults {
		inputTypeKey := string(testResult.InputType)

		if _, exists := resultsByInputType[inputTypeKey]; !exists {
			resultsByInputType[inputTypeKey] = &InputTypeResults{
				InputType:   testResult.InputType,
				TestResults: make([]*TestResult, 0),
				Statistics:  make(map[string]interface{}),
			}
		}

		resultsByInputType[inputTypeKey].TestResults = append(
			resultsByInputType[inputTypeKey].TestResults, testResult)
	}

	// Generate rankings for each input type
	for _, inputResults := range resultsByInputType {
		rankings := GenerateKEIRanking(fbe.finalKEIScores, 10)
		inputResults.Rankings = rankings
		if len(rankings.Rankings) > 0 {
			inputResults.BestOverall = rankings.Rankings[0]
		}
	}

	// Generate overall rankings
	overallRankings := GenerateKEIRanking(fbe.finalKEIScores, 10)

	// Find best combinations
	var bestCompressionRatio, bestSpeed, bestMemoryEfficiency, bestOverall *RankedCombination
	if len(overallRankings.Rankings) > 0 {
		bestOverall = overallRankings.Rankings[0]
		bestCompressionRatio = overallRankings.Rankings[0] // Simplified
		bestSpeed = overallRankings.Rankings[0]
		bestMemoryEfficiency = overallRankings.Rankings[0]
	}

	completedTests := int64(len(fbe.finalResults))
	failedTests := int64(0)
	timeoutTests := int64(0)

	for _, result := range fbe.finalResults {
		if !result.Success {
			if result.Timeout {
				timeoutTests++
			} else {
				failedTests++
			}
		}
	}

	return &BenchmarkResult{
		Config:               fbe.config,
		StartTime:            startTime,
		EndTime:              endTime,
		Duration:             endTime.Sub(startTime),
		TotalCombinations:    int64(len(fbe.finalResults)),
		CompletedTests:       completedTests,
		FailedTests:          failedTests,
		TimeoutTests:         timeoutTests,
		ResultsByInputType:   resultsByInputType,
		OverallRankings:      overallRankings,
		BestCompressionRatio: bestCompressionRatio,
		BestSpeed:            bestSpeed,
		BestMemoryEfficiency: bestMemoryEfficiency,
		BestOverall:          bestOverall,
		Statistics: map[string]interface{}{
			"total_algorithm_combinations": len(fbe.finalResults),
			"average_test_duration_ms":     0, // Calculate if needed
			"fastest_combination":          "",
			"most_memory_efficient":        "",
		},
	}
}
