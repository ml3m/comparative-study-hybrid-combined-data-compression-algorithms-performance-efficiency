package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"hybrid-compression-study/pkg/benchmarks"
)

const (
	banner = `
╔═══════════════════════════════════════════════════════════════╗
║            HYBRID COMPRESSION ALGORITHM BENCHMARK            ║
║                    BRUTE FORCE ANALYSIS                      ║
╚═══════════════════════════════════════════════════════════════╝
`
	version = "v1.0.0"
)

// CLI represents the command-line interface
type CLI struct {
	scanner *bufio.Scanner
	config  *benchmarks.BenchmarkConfig
}

// NewCLI creates a new CLI instance
func NewCLI() *CLI {
	return &CLI{
		scanner: bufio.NewScanner(os.Stdin),
		config:  benchmarks.DefaultBenchmarkConfig(),
	}
}

func main() {
	cli := NewCLI()

	// Print banner
	fmt.Print(banner)
	fmt.Printf("Version: %s\n", version)
	fmt.Printf("Build Time: %s\n\n", time.Now().Format("2006-01-02 15:04:05"))

	// Setup signal handling for graceful shutdown
	cli.setupSignalHandling()

	// Run main CLI loop
	cli.run()
}

// run executes the main CLI flow
func (cli *CLI) run() {
	fmt.Println("Welcome to the Hybrid Compression Algorithm Benchmark!")
	fmt.Println("This tool will brute force test all combinations of selected algorithms")
	fmt.Println("across different input types and provide comprehensive efficiency analysis.\n")

	for {
		fmt.Println("═══════════════════════════════════════════════════════════════")
		fmt.Println("                        MAIN MENU")
		fmt.Println("═══════════════════════════════════════════════════════════════")
		fmt.Println("1. Configure Benchmark")
		fmt.Println("2. View Current Configuration")
		fmt.Println("3. Run Benchmark")
		fmt.Println("4. Load Configuration from File")
		fmt.Println("5. Save Configuration to File")
		fmt.Println("6. View Algorithm Information")
		fmt.Println("7. Estimate Execution Time")
		fmt.Println("8. Exit")
		fmt.Println("═══════════════════════════════════════════════════════════════")

		choice := cli.prompt("Select option [1-8]: ")

		switch choice {
		case "1":
			cli.configureBenchmark()
		case "2":
			cli.viewConfiguration()
		case "3":
			cli.runBenchmark()
		case "4":
			cli.loadConfiguration()
		case "5":
			cli.saveConfiguration()
		case "6":
			cli.viewAlgorithmInfo()
		case "7":
			cli.estimateExecutionTime()
		case "8":
			fmt.Println("Thank you for using the Hybrid Compression Benchmark!")
			return
		default:
			fmt.Println("❌ Invalid option. Please select 1-8.")
		}

		fmt.Println("\nPress Enter to continue...")
		cli.scanner.Scan()
	}
}

// configureBenchmark handles benchmark configuration
func (cli *CLI) configureBenchmark() {
	fmt.Println("\n🔧 BENCHMARK CONFIGURATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	for {
		fmt.Println("\nConfiguration Options:")
		fmt.Println("1. Select Algorithms")
		fmt.Println("2. Set Combination Size Range")
		fmt.Println("3. Configure Input Types")
		fmt.Println("4. Configure Input Sizes")
		fmt.Println("5. Set Execution Parameters")
		fmt.Println("6. Configure KEI Weights")
		fmt.Println("7. Configure Output Options")
		fmt.Println("8. Back to Main Menu")

		choice := cli.prompt("Select configuration option [1-8]: ")

		switch choice {
		case "1":
			cli.selectAlgorithms()
		case "2":
			cli.setCombinationSizeRange()
		case "3":
			cli.configureInputTypes()
		case "4":
			cli.configureInputSizes()
		case "5":
			cli.setExecutionParameters()
		case "6":
			cli.configureKEIWeights()
		case "7":
			cli.configureOutputOptions()
		case "8":
			return
		default:
			fmt.Println("❌ Invalid option. Please select 1-8.")
		}
	}
}

// selectAlgorithms handles algorithm selection
func (cli *CLI) selectAlgorithms() {
	fmt.Println("\n🧮 ALGORITHM SELECTION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Create combination generator to get available algorithms
	combGen := benchmarks.NewCombinationGenerator()
	availableAlgorithms := combGen.GetAvailableAlgorithms()

	fmt.Println("Available Algorithms:")
	algorithms := make([]string, 0, len(availableAlgorithms))
	for name, info := range availableAlgorithms {
		algorithms = append(algorithms, name)
		fmt.Printf("  %s - %s (%s)\n", name, info.Name, info.Category)
	}

	fmt.Println("\nCurrently selected:", cli.config.SelectedAlgorithms)
	fmt.Println("\nOptions:")
	fmt.Println("1. Select All Algorithms")
	fmt.Println("2. Select by Category")
	fmt.Println("3. Select Individual Algorithms")
	fmt.Println("4. Clear Selection")
	fmt.Println("5. Back")

	choice := cli.prompt("Select option [1-5]: ")

	switch choice {
	case "1":
		cli.config.SelectedAlgorithms = algorithms
		fmt.Printf("✅ Selected all %d algorithms\n", len(algorithms))
	case "2":
		cli.selectByCategory(availableAlgorithms)
	case "3":
		cli.selectIndividualAlgorithms(algorithms)
	case "4":
		cli.config.SelectedAlgorithms = []string{}
		fmt.Println("✅ Cleared algorithm selection")
	case "5":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// selectByCategory handles category-based algorithm selection
func (cli *CLI) selectByCategory(availableAlgorithms map[string]*benchmarks.AlgorithmInfo) {
	categories := make(map[string][]string)

	// Group algorithms by category
	for name, info := range availableAlgorithms {
		category := string(info.Category)
		categories[category] = append(categories[category], name)
	}

	fmt.Println("\nAlgorithm Categories:")
	categoryList := make([]string, 0, len(categories))
	for category, algorithms := range categories {
		categoryList = append(categoryList, category)
		fmt.Printf("  %s: %v\n", category, algorithms)
	}

	selectedCategories := cli.promptMultiSelect("Select categories (comma-separated): ", categoryList)

	var selectedAlgorithms []string
	for _, category := range selectedCategories {
		if algorithms, exists := categories[category]; exists {
			selectedAlgorithms = append(selectedAlgorithms, algorithms...)
		}
	}

	cli.config.SelectedAlgorithms = selectedAlgorithms
	fmt.Printf("✅ Selected %d algorithms from %d categories\n",
		len(selectedAlgorithms), len(selectedCategories))
}

// selectIndividualAlgorithms handles individual algorithm selection
func (cli *CLI) selectIndividualAlgorithms(algorithms []string) {
	fmt.Println("\nSelect algorithms (comma-separated names or 'all'): ")
	fmt.Printf("Available: %v\n", algorithms)

	input := cli.prompt("Algorithms: ")

	if input == "all" {
		cli.config.SelectedAlgorithms = algorithms
	} else {
		selected := cli.parseAlgorithmList(input, algorithms)
		cli.config.SelectedAlgorithms = selected
	}

	fmt.Printf("✅ Selected %d algorithms: %v\n",
		len(cli.config.SelectedAlgorithms), cli.config.SelectedAlgorithms)
}

// setCombinationSizeRange sets the combination size range
func (cli *CLI) setCombinationSizeRange() {
	fmt.Println("\n📊 COMBINATION SIZE RANGE")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	maxAlgorithms := len(cli.config.SelectedAlgorithms)
	if maxAlgorithms == 0 {
		fmt.Println("❌ Please select algorithms first!")
		return
	}

	fmt.Printf("Available algorithms: %d\n", maxAlgorithms)
	fmt.Printf("Current range: %d - %d\n", cli.config.MinCombinationSize, cli.config.MaxCombinationSize)

	// Estimate total combinations for different ranges
	combGen := benchmarks.NewCombinationGenerator()
	combGen.SetSelectedAlgorithms(cli.config.SelectedAlgorithms)

	fmt.Println("\nTotal combinations for different ranges:")
	for minSize := 2; minSize <= maxAlgorithms && minSize <= 5; minSize++ {
		for maxSize := minSize; maxSize <= maxAlgorithms && maxSize <= minSize+2; maxSize++ {
			combGen.SetCombinationSizeRange(minSize, maxSize)
			total := combGen.EstimateTotalCombinations()
			fmt.Printf("  %d-%d algorithms: %d combinations\n", minSize, maxSize, total)
		}
	}

	minStr := cli.prompt(fmt.Sprintf("Minimum combination size [2-%d]: ", maxAlgorithms))
	maxStr := cli.prompt(fmt.Sprintf("Maximum combination size [%s-%d]: ", minStr, maxAlgorithms))

	min, err1 := strconv.Atoi(minStr)
	max, err2 := strconv.Atoi(maxStr)

	if err1 != nil || err2 != nil || min < 2 || max < min || max > maxAlgorithms {
		fmt.Println("❌ Invalid range specified.")
		return
	}

	cli.config.MinCombinationSize = min
	cli.config.MaxCombinationSize = max

	// Update estimate
	combGen.SetCombinationSizeRange(min, max)
	total := combGen.EstimateTotalCombinations()

	fmt.Printf("✅ Set combination size range to %d-%d (%d total combinations)\n", min, max, total)
}

// configureInputTypes configures input types for testing
func (cli *CLI) configureInputTypes() {
	fmt.Println("\n📝 INPUT TYPE CONFIGURATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	allInputTypes := []benchmarks.InputType{
		benchmarks.InputTypeRepetitive,
		benchmarks.InputTypeTextPatterns,
		benchmarks.InputTypeRandom,
		benchmarks.InputTypeSequential,
		benchmarks.InputTypeNaturalText,
		benchmarks.InputTypeStructuredBin,
		benchmarks.InputTypeMixed,
		benchmarks.InputTypeSparse,
		benchmarks.InputTypeAlternating,
		benchmarks.InputTypeLog,
	}

	fmt.Println("Available Input Types:")
	for i, inputType := range allInputTypes {
		fmt.Printf("  %d. %s\n", i+1, inputType)
	}

	fmt.Println("\nCurrently selected:", cli.config.InputTypes)

	fmt.Println("\nOptions:")
	fmt.Println("1. Select All Types")
	fmt.Println("2. Select Standard Set (repetitive, text_patterns, random, sequential)")
	fmt.Println("3. Select Custom Set")
	fmt.Println("4. Back")

	choice := cli.prompt("Select option [1-4]: ")

	switch choice {
	case "1":
		cli.config.InputTypes = allInputTypes
		fmt.Printf("✅ Selected all %d input types\n", len(allInputTypes))
	case "2":
		cli.config.InputTypes = []benchmarks.InputType{
			benchmarks.InputTypeRepetitive,
			benchmarks.InputTypeTextPatterns,
			benchmarks.InputTypeRandom,
			benchmarks.InputTypeSequential,
		}
		fmt.Println("✅ Selected standard input type set")
	case "3":
		cli.selectCustomInputTypes(allInputTypes)
	case "4":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// selectCustomInputTypes handles custom input type selection
func (cli *CLI) selectCustomInputTypes(allInputTypes []benchmarks.InputType) {
	fmt.Println("Enter input type numbers (comma-separated, e.g., 1,3,5): ")
	input := cli.prompt("Input types: ")

	var selectedTypes []benchmarks.InputType
	for _, numStr := range strings.Split(input, ",") {
		num, err := strconv.Atoi(strings.TrimSpace(numStr))
		if err != nil || num < 1 || num > len(allInputTypes) {
			fmt.Printf("❌ Invalid input type number: %s\n", numStr)
			continue
		}
		selectedTypes = append(selectedTypes, allInputTypes[num-1])
	}

	cli.config.InputTypes = selectedTypes
	fmt.Printf("✅ Selected %d input types: %v\n", len(selectedTypes), selectedTypes)
}

// configureInputSizes configures input sizes for testing
func (cli *CLI) configureInputSizes() {
	fmt.Println("\n📏 INPUT SIZE CONFIGURATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	standardSizes := benchmarks.StandardSizes

	fmt.Println("Standard Input Sizes:")
	for i, size := range standardSizes {
		fmt.Printf("  %d. %s (%d bytes)\n", i+1, size.Name, size.Bytes)
	}

	fmt.Println("\nCurrently selected:", cli.config.InputSizes)

	fmt.Println("\nOptions:")
	fmt.Println("1. Select All Standard Sizes")
	fmt.Println("2. Select Small Sizes (1KB, 10KB, 100KB)")
	fmt.Println("3. Select Medium Sizes (100KB, 1MB)")
	fmt.Println("4. Select Large Sizes (1MB, 10MB)")
	fmt.Println("5. Select Custom Sizes")
	fmt.Println("6. Back")

	choice := cli.prompt("Select option [1-6]: ")

	switch choice {
	case "1":
		cli.config.InputSizes = standardSizes
		fmt.Printf("✅ Selected all %d standard sizes\n", len(standardSizes))
	case "2":
		cli.config.InputSizes = standardSizes[:3] // First 3 sizes
		fmt.Println("✅ Selected small sizes")
	case "3":
		cli.config.InputSizes = standardSizes[2:4] // 100KB, 1MB
		fmt.Println("✅ Selected medium sizes")
	case "4":
		cli.config.InputSizes = standardSizes[3:] // 1MB, 10MB
		fmt.Println("✅ Selected large sizes")
	case "5":
		cli.selectCustomInputSizes(standardSizes)
	case "6":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// selectCustomInputSizes handles custom input size selection
func (cli *CLI) selectCustomInputSizes(standardSizes []benchmarks.InputSize) {
	fmt.Println("Enter size numbers (comma-separated, e.g., 1,3,5): ")
	input := cli.prompt("Input sizes: ")

	var selectedSizes []benchmarks.InputSize
	for _, numStr := range strings.Split(input, ",") {
		num, err := strconv.Atoi(strings.TrimSpace(numStr))
		if err != nil || num < 1 || num > len(standardSizes) {
			fmt.Printf("❌ Invalid size number: %s\n", numStr)
			continue
		}
		selectedSizes = append(selectedSizes, standardSizes[num-1])
	}

	cli.config.InputSizes = selectedSizes
	fmt.Printf("✅ Selected %d sizes: %v\n", len(selectedSizes), selectedSizes)
}

// setExecutionParameters sets execution parameters
func (cli *CLI) setExecutionParameters() {
	fmt.Println("\n⚙️  EXECUTION PARAMETERS")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Current parameters:\n")
	fmt.Printf("  Timeout per test: %s\n", cli.config.TimeoutPerTest)
	fmt.Printf("  Max concurrent tests: %d\n", cli.config.MaxConcurrentTests)
	fmt.Printf("  Retry failed tests: %d\n", cli.config.RetryFailedTests)

	fmt.Println("\nWhat would you like to configure?")
	fmt.Println("1. Timeout per test")
	fmt.Println("2. Max concurrent tests")
	fmt.Println("3. Retry failed tests")
	fmt.Println("4. Back")

	choice := cli.prompt("Select option [1-4]: ")

	switch choice {
	case "1":
		timeoutStr := cli.prompt("Timeout per test (e.g., 30s, 2m): ")
		if timeout, err := time.ParseDuration(timeoutStr); err == nil {
			cli.config.TimeoutPerTest = timeout
			fmt.Printf("✅ Set timeout to %s\n", timeout)
		} else {
			fmt.Printf("❌ Invalid duration format: %v\n", err)
		}
	case "2":
		concurrentStr := cli.prompt("Max concurrent tests [1-16]: ")
		if concurrent, err := strconv.Atoi(concurrentStr); err == nil && concurrent >= 1 && concurrent <= 16 {
			cli.config.MaxConcurrentTests = concurrent
			fmt.Printf("✅ Set max concurrent tests to %d\n", concurrent)
		} else {
			fmt.Println("❌ Invalid number. Must be between 1 and 16.")
		}
	case "3":
		retryStr := cli.prompt("Retry failed tests [0-3]: ")
		if retry, err := strconv.Atoi(retryStr); err == nil && retry >= 0 && retry <= 3 {
			cli.config.RetryFailedTests = retry
			fmt.Printf("✅ Set retry count to %d\n", retry)
		} else {
			fmt.Println("❌ Invalid number. Must be between 0 and 3.")
		}
	case "4":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// configureKEIWeights configures KEI calculation weights
func (cli *CLI) configureKEIWeights() {
	fmt.Println("\n⚖️  KEI WEIGHTS CONFIGURATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Current KEI weights:\n")
	fmt.Printf("  Compression Ratio: %.2f\n", cli.config.KEIWeights.CompressionRatio)
	fmt.Printf("  Speed: %.2f\n", cli.config.KEIWeights.Speed)
	fmt.Printf("  Memory: %.2f\n", cli.config.KEIWeights.Memory)
	fmt.Printf("  Stability: %.2f\n", cli.config.KEIWeights.Stability)
	fmt.Printf("  Energy: %.2f\n", cli.config.KEIWeights.Energy)

	fmt.Println("\nPreset options:")
	fmt.Println("1. Default (Balanced)")
	fmt.Println("2. Performance-focused")
	fmt.Println("3. Compression-focused")
	fmt.Println("4. Memory-focused")
	fmt.Println("5. Custom weights")
	fmt.Println("6. Back")

	choice := cli.prompt("Select option [1-6]: ")

	switch choice {
	case "1":
		cli.config.KEIWeights = benchmarks.DefaultKEIWeights
		fmt.Println("✅ Set to default balanced weights")
	case "2":
		cli.config.KEIWeights = benchmarks.PerformanceWeights
		fmt.Println("✅ Set to performance-focused weights")
	case "3":
		cli.config.KEIWeights = benchmarks.CompressionWeights
		fmt.Println("✅ Set to compression-focused weights")
	case "4":
		cli.config.KEIWeights = benchmarks.MemoryWeights
		fmt.Println("✅ Set to memory-focused weights")
	case "5":
		cli.setCustomKEIWeights()
	case "6":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// setCustomKEIWeights sets custom KEI weights
func (cli *CLI) setCustomKEIWeights() {
	fmt.Println("Enter custom weights (must sum to 1.0):")

	compressionStr := cli.prompt("Compression Ratio weight [0.0-1.0]: ")
	speedStr := cli.prompt("Speed weight [0.0-1.0]: ")
	memoryStr := cli.prompt("Memory weight [0.0-1.0]: ")
	stabilityStr := cli.prompt("Stability weight [0.0-1.0]: ")
	energyStr := cli.prompt("Energy weight [0.0-1.0]: ")

	compression, err1 := strconv.ParseFloat(compressionStr, 64)
	speed, err2 := strconv.ParseFloat(speedStr, 64)
	memory, err3 := strconv.ParseFloat(memoryStr, 64)
	stability, err4 := strconv.ParseFloat(stabilityStr, 64)
	energy, err5 := strconv.ParseFloat(energyStr, 64)

	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil {
		fmt.Println("❌ Invalid weight values.")
		return
	}

	total := compression + speed + memory + stability + energy
	if total < 0.99 || total > 1.01 { // Allow small floating point errors
		fmt.Printf("❌ Weights must sum to 1.0 (current sum: %.3f)\n", total)
		return
	}

	cli.config.KEIWeights = benchmarks.KEIWeights{
		CompressionRatio: compression,
		Speed:            speed,
		Memory:           memory,
		Stability:        stability,
		Energy:           energy,
	}

	fmt.Println("✅ Set custom KEI weights")
}

// configureOutputOptions configures output options
func (cli *CLI) configureOutputOptions() {
	fmt.Println("\n📄 OUTPUT OPTIONS")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Current output options:\n")
	fmt.Printf("  Save results: %t\n", cli.config.SaveResults)
	fmt.Printf("  Output directory: %s\n", cli.config.OutputDirectory)
	fmt.Printf("  Generate detailed report: %t\n", cli.config.GenerateDetailedReport)
	fmt.Printf("  Verbose output: %t\n", cli.config.VerboseOutput)

	fmt.Println("\nWhat would you like to configure?")
	fmt.Println("1. Toggle save results")
	fmt.Println("2. Set output directory")
	fmt.Println("3. Toggle detailed report")
	fmt.Println("4. Toggle verbose output")
	fmt.Println("5. Back")

	choice := cli.prompt("Select option [1-5]: ")

	switch choice {
	case "1":
		cli.config.SaveResults = !cli.config.SaveResults
		fmt.Printf("✅ Save results: %t\n", cli.config.SaveResults)
	case "2":
		dir := cli.prompt("Output directory: ")
		if dir != "" {
			cli.config.OutputDirectory = dir
			fmt.Printf("✅ Set output directory to: %s\n", dir)
		}
	case "3":
		cli.config.GenerateDetailedReport = !cli.config.GenerateDetailedReport
		fmt.Printf("✅ Generate detailed report: %t\n", cli.config.GenerateDetailedReport)
	case "4":
		cli.config.VerboseOutput = !cli.config.VerboseOutput
		fmt.Printf("✅ Verbose output: %t\n", cli.config.VerboseOutput)
	case "5":
		return
	default:
		fmt.Println("❌ Invalid option.")
	}
}

// viewConfiguration displays current configuration
func (cli *CLI) viewConfiguration() {
	fmt.Println("\n📋 CURRENT CONFIGURATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Selected Algorithms (%d): %v\n",
		len(cli.config.SelectedAlgorithms), cli.config.SelectedAlgorithms)
	fmt.Printf("Combination Size Range: %d - %d\n",
		cli.config.MinCombinationSize, cli.config.MaxCombinationSize)
	fmt.Printf("Input Types (%d): %v\n",
		len(cli.config.InputTypes), cli.config.InputTypes)
	fmt.Printf("Input Sizes (%d): ", len(cli.config.InputSizes))
	for i, size := range cli.config.InputSizes {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(size.Name)
	}
	fmt.Println()

	fmt.Printf("\nExecution Parameters:\n")
	fmt.Printf("  Timeout per test: %s\n", cli.config.TimeoutPerTest)
	fmt.Printf("  Max concurrent tests: %d\n", cli.config.MaxConcurrentTests)
	fmt.Printf("  Retry failed tests: %d\n", cli.config.RetryFailedTests)

	fmt.Printf("\nKEI Weights:\n")
	fmt.Printf("  Compression: %.2f, Speed: %.2f, Memory: %.2f, Stability: %.2f, Energy: %.2f\n",
		cli.config.KEIWeights.CompressionRatio,
		cli.config.KEIWeights.Speed,
		cli.config.KEIWeights.Memory,
		cli.config.KEIWeights.Stability,
		cli.config.KEIWeights.Energy)

	fmt.Printf("\nOutput Options:\n")
	fmt.Printf("  Save results: %t\n", cli.config.SaveResults)
	fmt.Printf("  Output directory: %s\n", cli.config.OutputDirectory)
	fmt.Printf("  Generate detailed report: %t\n", cli.config.GenerateDetailedReport)
	fmt.Printf("  Verbose output: %t\n", cli.config.VerboseOutput)

	// Calculate estimated tests
	if len(cli.config.SelectedAlgorithms) > 0 {
		combGen := benchmarks.NewCombinationGenerator()
		combGen.SetSelectedAlgorithms(cli.config.SelectedAlgorithms)
		combGen.SetCombinationSizeRange(cli.config.MinCombinationSize, cli.config.MaxCombinationSize)
		totalCombinations := combGen.EstimateTotalCombinations()
		totalInputs := len(cli.config.InputTypes) * len(cli.config.InputSizes)
		totalTests := totalCombinations * int64(totalInputs)

		fmt.Printf("\nEstimated Tests:\n")
		fmt.Printf("  Total combinations: %d\n", totalCombinations)
		fmt.Printf("  Total inputs: %d\n", totalInputs)
		fmt.Printf("  Total tests: %d\n", totalTests)
	}
}

// runBenchmark executes the benchmark
func (cli *CLI) runBenchmark() {
	fmt.Println("\n🚀 BENCHMARK EXECUTION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Validate configuration
	if len(cli.config.SelectedAlgorithms) == 0 {
		fmt.Println("❌ No algorithms selected! Please configure algorithms first.")
		return
	}

	if len(cli.config.InputTypes) == 0 {
		fmt.Println("❌ No input types selected! Please configure input types first.")
		return
	}

	if len(cli.config.InputSizes) == 0 {
		fmt.Println("❌ No input sizes selected! Please configure input sizes first.")
		return
	}

	// Show execution summary
	combGen := benchmarks.NewCombinationGenerator()
	combGen.SetSelectedAlgorithms(cli.config.SelectedAlgorithms)
	combGen.SetCombinationSizeRange(cli.config.MinCombinationSize, cli.config.MaxCombinationSize)
	totalCombinations := combGen.EstimateTotalCombinations()
	totalInputs := len(cli.config.InputTypes) * len(cli.config.InputSizes)
	totalTests := totalCombinations * int64(totalInputs)

	fmt.Printf("About to execute:\n")
	fmt.Printf("  %d algorithm combinations\n", totalCombinations)
	fmt.Printf("  %d input configurations\n", totalInputs)
	fmt.Printf("  %d total tests\n", totalTests)
	fmt.Printf("  Estimated duration: %s\n", cli.estimateDuration(totalTests))

	fmt.Println("\n⚠️  This may take a significant amount of time!")
	confirm := cli.prompt("Do you want to proceed? [y/N]: ")

	if strings.ToLower(confirm) != "y" && strings.ToLower(confirm) != "yes" {
		fmt.Println("Benchmark cancelled.")
		return
	}

	// Create and execute benchmark
	executor, err := benchmarks.NewBenchmarkExecutor(cli.config)
	if err != nil {
		fmt.Printf("❌ Failed to create benchmark executor: %v\n", err)
		return
	}

	fmt.Println("\n🎬 Starting benchmark execution...")
	fmt.Println("Press Ctrl+C to gracefully stop the benchmark")

	// Execute benchmark
	result, err := executor.Execute()
	if err != nil {
		fmt.Printf("❌ Benchmark execution failed: %v\n", err)
		return
	}

	// Display results summary
	cli.displayResults(result)

	// Save results if configured
	if cli.config.SaveResults {
		cli.saveResults(result)
	}
}

// displayResults displays benchmark results
func (cli *CLI) displayResults(result *benchmarks.BenchmarkResult) {
	fmt.Println("\n📊 BENCHMARK RESULTS SUMMARY")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	fmt.Printf("Execution completed in: %s\n", result.Duration)
	fmt.Printf("Total tests: %d (Success: %d, Failed: %d, Timeout: %d)\n",
		result.CompletedTests+result.FailedTests+result.TimeoutTests,
		result.CompletedTests, result.FailedTests, result.TimeoutTests)

	if result.BestOverall != nil {
		fmt.Printf("\n🏆 BEST OVERALL COMBINATION:\n")
		fmt.Printf("  %s\n", result.BestOverall.AlgorithmCombination)
		fmt.Printf("  KEI Score: %.2f\n", result.BestOverall.KEIScore.OverallScore)
		fmt.Printf("  Category: %s\n", result.BestOverall.PerformanceCategory)
	}

	if result.BestCompressionRatio != nil {
		fmt.Printf("\n📦 BEST COMPRESSION:\n")
		fmt.Printf("  %s\n", result.BestCompressionRatio.AlgorithmCombination)
		fmt.Printf("  Compression Ratio: %.2f\n", result.BestCompressionRatio.KEIScore.CompressionRatio)
	}

	if result.BestSpeed != nil {
		fmt.Printf("\n⚡ FASTEST COMBINATION:\n")
		fmt.Printf("  %s\n", result.BestSpeed.AlgorithmCombination)
		fmt.Printf("  Speed Score: %.2f\n", result.BestSpeed.KEIScore.SpeedScore)
	}

	if result.BestMemoryEfficiency != nil {
		fmt.Printf("\n🧠 MOST MEMORY EFFICIENT:\n")
		fmt.Printf("  %s\n", result.BestMemoryEfficiency.AlgorithmCombination)
		fmt.Printf("  Memory Score: %.2f\n", result.BestMemoryEfficiency.KEIScore.MemoryScore)
	}

	fmt.Println("\n📈 TOP 5 COMBINATIONS:")
	if result.OverallRankings != nil && len(result.OverallRankings.Rankings) > 0 {
		count := 5
		if len(result.OverallRankings.Rankings) < count {
			count = len(result.OverallRankings.Rankings)
		}

		for i := 0; i < count; i++ {
			ranking := result.OverallRankings.Rankings[i]
			fmt.Printf("  %d. %s (KEI: %.2f)\n",
				i+1, ranking.AlgorithmCombination, ranking.KEIScore.OverallScore)
		}
	} else {
		// Debug: Show why no rankings are displayed
		if result.OverallRankings == nil {
			fmt.Println("  ⚠️  No overall rankings generated")
		} else if len(result.OverallRankings.Rankings) == 0 {
			fmt.Println("  ⚠️  Overall rankings is empty")
		}
		fmt.Printf("  Debug: Completed tests: %d, Failed: %d\n", result.CompletedTests, result.FailedTests)
	}

	// Generate comprehensive analysis if configured
	if cli.config.GenerateDetailedReport {
		fmt.Println("\n📋 Generating comprehensive analysis...")
		analysisEngine := benchmarks.NewAnalysisEngine(result)
		analysis := analysisEngine.GenerateComprehensiveAnalysis()

		fmt.Printf("\n📊 ANALYSIS HIGHLIGHTS:\n")
		fmt.Printf("  Success Rate: %.1f%%\n", analysis.ExecutionSummary.SuccessRate)
		fmt.Printf("  Average KEI Score: %.2f\n", analysis.ExecutionSummary.AverageKEIScore)
		fmt.Printf("  Peak Compression Ratio: %.2f\n", analysis.ExecutionSummary.PeakCompressionRatio)

		// NEW: Display detailed scientific breakdown
		cli.displayScientificBreakdown(result, analysis)

		// Save detailed analysis
		if cli.config.SaveResults {
			cli.saveAnalysis(analysis)
		}
	}
}

// displayTimeoutAnalysis provides detailed analysis of timeout cases
func (cli *CLI) displayTimeoutAnalysis(result *benchmarks.BenchmarkResult) {
	// Collect all timeout cases
	var timeoutCases []*benchmarks.TestResult
	timeoutByCombo := make(map[string]int)
	timeoutByInputType := make(map[string]int)
	timeoutBySize := make(map[string]int)

	for _, inputResults := range result.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			if testResult.Timeout {
				timeoutCases = append(timeoutCases, testResult)
				timeoutByCombo[testResult.AlgorithmCombination]++
				timeoutByInputType[string(testResult.InputType)]++
				timeoutBySize[cli.getSizeName(testResult.InputSize)]++
			}
		}
	}

	if len(timeoutCases) == 0 {
		fmt.Println("\n✅ TIMEOUT ANALYSIS: No timeouts detected - excellent performance!")
		return
	}

	fmt.Println("\n⏱️  TIMEOUT ANALYSIS (Performance Bottlenecks)")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("Total Timeout Cases: %d (%.1f%% of all tests)\n",
		len(timeoutCases),
		float64(len(timeoutCases))/float64(result.CompletedTests+result.FailedTests+result.TimeoutTests)*100)

	// Most problematic combinations
	fmt.Println("\n🚨 WORST PERFORMING COMBINATIONS:")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-30s | %-12s | %-15s\n", "Algorithm Combination", "Timeouts", "Failure Rate")
	fmt.Println(strings.Repeat("-", 50))

	type comboTimeout struct {
		combination string
		timeouts    int
		totalTests  int
	}

	var comboTimeouts []comboTimeout
	comboTotalTests := make(map[string]int)

	// Count total tests per combination
	for _, inputResults := range result.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			comboTotalTests[testResult.AlgorithmCombination]++
		}
	}

	// Create sorted list of problematic combinations
	for combo, timeoutCount := range timeoutByCombo {
		comboTimeouts = append(comboTimeouts, comboTimeout{
			combination: combo,
			timeouts:    timeoutCount,
			totalTests:  comboTotalTests[combo],
		})
	}

	// Sort by timeout count (descending)
	sort.Slice(comboTimeouts, func(i, j int) bool {
		return comboTimeouts[i].timeouts > comboTimeouts[j].timeouts
	})

	// Show top 5 worst combinations
	limit := 5
	if len(comboTimeouts) < limit {
		limit = len(comboTimeouts)
	}

	for i := 0; i < limit; i++ {
		ct := comboTimeouts[i]
		failureRate := float64(ct.timeouts) / float64(ct.totalTests) * 100

		combo := ct.combination
		if len(combo) > 28 {
			combo = combo[:25] + "..."
		}

		fmt.Printf("%-30s | %12d | %14.1f%%\n", combo, ct.timeouts, failureRate)
	}

	// Timeout patterns by input type
	fmt.Println("\n📊 TIMEOUT PATTERNS BY INPUT TYPE:")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-15s | %-12s | %-15s\n", "Input Type", "Timeouts", "Vulnerability")
	fmt.Println(strings.Repeat("-", 50))

	inputOrder := []string{"repetitive", "text_patterns", "random", "sequential", "natural_text", "structured_bin", "mixed", "sparse", "alternating", "log"}

	for _, inputType := range inputOrder {
		if timeouts, exists := timeoutByInputType[inputType]; exists {
			// Calculate total tests for this input type
			totalForInput := 0
			if inputResults, exists := result.ResultsByInputType[inputType]; exists {
				totalForInput = len(inputResults.TestResults)
			}

			vulnerability := float64(timeouts) / float64(totalForInput) * 100
			fmt.Printf("%-15s | %12d | %14.1f%%\n", inputType, timeouts, vulnerability)
		}
	}

	// Timeout patterns by input size
	fmt.Println("\n📏 TIMEOUT PATTERNS BY INPUT SIZE:")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-10s | %-12s | %-15s\n", "Size", "Timeouts", "Scaling Issue")
	fmt.Println(strings.Repeat("-", 50))

	sizes := []string{"512B", "2KB", "8KB"}
	for _, size := range sizes {
		if timeouts, exists := timeoutBySize[size]; exists {
			// Estimate total tests for this size (rough calculation)
			totalTestsPerSize := len(result.Config.SelectedAlgorithms) * len(result.Config.InputTypes)
			if result.Config.MaxCombinationSize > result.Config.MinCombinationSize {
				totalTestsPerSize *= (result.Config.MaxCombinationSize - result.Config.MinCombinationSize + 1)
			}

			scalingIssue := float64(timeouts) / float64(totalTestsPerSize) * 100
			fmt.Printf("%-10s | %12d | %14.1f%%\n", size, timeouts, scalingIssue)
		}
	}

	// Root cause analysis
	fmt.Println("\n🔬 ROOT CAUSE ANALYSIS:")
	fmt.Println(strings.Repeat("-", 80))

	// Analyze timeout patterns
	cli.analyzeTimeoutRootCauses(timeoutCases, timeoutByCombo, timeoutByInputType)

	// Performance recommendations
	fmt.Println("\n💡 TIMEOUT MITIGATION RECOMMENDATIONS:")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Println("1. ALGORITHM-SPECIFIC ISSUES:")

	for i, ct := range comboTimeouts {
		if i >= 3 {
			break
		} // Show only top 3
		fmt.Printf("   • %s: Consider algorithm reordering or parameter tuning\n", ct.combination)
	}

	fmt.Println("\n2. INPUT-SPECIFIC ISSUES:")
	for inputType, timeouts := range timeoutByInputType {
		if timeouts > 1 {
			fmt.Printf("   • %s data: Requires specialized preprocessing or lighter algorithms\n", inputType)
		}
	}

	fmt.Println("\n3. SCALING ISSUES:")
	for size, timeouts := range timeoutBySize {
		if timeouts > 2 {
			fmt.Printf("   • %s inputs: May need chunk-based processing or streaming algorithms\n", size)
		}
	}

	fmt.Println("\n4. GENERAL RECOMMENDATIONS:")
	fmt.Println("   • Increase timeout limits for complex algorithm chains")
	fmt.Printf("   • Current timeout: %s - consider %s for thorough analysis\n",
		result.Config.TimeoutPerTest,
		result.Config.TimeoutPerTest*2)
	fmt.Println("   • Use pipeline parallelization for independent stages")
	fmt.Println("   • Consider lightweight algorithm alternatives for problematic combinations")
}

// analyzeTimeoutRootCauses performs deeper analysis of why timeouts occur
func (cli *CLI) analyzeTimeoutRootCauses(timeoutCases []*benchmarks.TestResult, timeoutByCombo, timeoutByInputType map[string]int) {
	// Pattern analysis
	complexAlgorithms := []string{"BWT", "Arithmetic", "MTF"}
	heavyPipelines := []string{"Pipeline_[BWT", "Pipeline_[Arithmetic"}
	problematicInputs := []string{"natural_text", "structured_bin", "mixed"}

	fmt.Println("Identified Performance Bottlenecks:")

	// Check for complex algorithms
	complexAlgTimeouts := 0
	for combo, timeouts := range timeoutByCombo {
		for _, complexAlg := range complexAlgorithms {
			if strings.Contains(combo, complexAlg) {
				complexAlgTimeouts += timeouts
				break
			}
		}
	}

	if complexAlgTimeouts > 0 {
		fmt.Printf("• Complex Algorithms: %d timeouts from computationally intensive algorithms\n", complexAlgTimeouts)
		fmt.Println("  - BWT has O(n²) memory complexity for rotation generation")
		fmt.Println("  - Arithmetic coding requires precise probability calculations")
		fmt.Println("  - MTF involves repeated list reorganization")
	}

	// Check for pipeline complexity
	heavyPipelineTimeouts := 0
	for combo, timeouts := range timeoutByCombo {
		for _, heavyPipeline := range heavyPipelines {
			if strings.Contains(combo, heavyPipeline) {
				heavyPipelineTimeouts += timeouts
				break
			}
		}
	}

	if heavyPipelineTimeouts > 0 {
		fmt.Printf("• Pipeline Complexity: %d timeouts from multi-stage processing overhead\n", heavyPipelineTimeouts)
		fmt.Println("  - Each pipeline stage multiplies processing time")
		fmt.Println("  - Intermediate data expansion can trigger memory pressure")
	}

	// Check for problematic inputs
	problematicInputTimeouts := 0
	for inputType, timeouts := range timeoutByInputType {
		for _, problematicInput := range problematicInputs {
			if inputType == problematicInput {
				problematicInputTimeouts += timeouts
				break
			}
		}
	}

	if problematicInputTimeouts > 0 {
		fmt.Printf("• Input Complexity: %d timeouts from challenging data patterns\n", problematicInputTimeouts)
		fmt.Println("  - Natural text requires extensive dictionary building")
		fmt.Println("  - Structured binary has mixed entropy characteristics")
		fmt.Println("  - Mixed data defeats algorithm optimizations")
	}

	// Memory pressure analysis
	fmt.Println("• Memory Pressure: Large algorithm chains may exceed available RAM")
	fmt.Println("  - Consider reducing max_concurrent_tests to 1")
	fmt.Println("  - Monitor system memory during benchmark execution")
}

// displayScientificBreakdown displays detailed results breakdown for research
func (cli *CLI) displayScientificBreakdown(result *benchmarks.BenchmarkResult, analysis *benchmarks.ComprehensiveAnalysisReport) {
	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("                    SCIENTIFIC ANALYSIS REPORT")
	fmt.Println("              (Detailed Breakdown for Research Paper)")
	fmt.Println(strings.Repeat("═", 80))

	// 1. Best Performance by Input Type
	fmt.Println("\n🔬 BEST COMBINATIONS BY INPUT TYPE:")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-15s | %-25s | %-8s | %-8s | %-10s\n", "Input Type", "Best Combination", "KEI", "Ratio", "Time (ms)")
	fmt.Println(strings.Repeat("-", 80))

	inputOrder := []string{"repetitive", "text_patterns", "random", "sequential", "natural_text", "structured_bin", "mixed", "sparse", "alternating", "log"}

	for _, inputTypeStr := range inputOrder {
		if inputResults, exists := result.ResultsByInputType[inputTypeStr]; exists && inputResults.BestOverall != nil {
			best := inputResults.BestOverall
			avgTime := cli.calculateAverageTime(inputResults.TestResults, best.AlgorithmCombination)
			fmt.Printf("%-15s | %-25s | %8.2f | %8.2f | %10.2f\n",
				inputTypeStr,
				best.AlgorithmCombination,
				best.KEIScore.OverallScore,
				best.KEIScore.CompressionRatio,
				avgTime)
		}
	}

	// 2. Best Performance by Input Size
	fmt.Println("\n📏 BEST COMBINATIONS BY INPUT SIZE:")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-10s | %-25s | %-8s | %-8s | %-10s | %-6s\n", "Size", "Best Combination", "KEI", "Ratio", "Time (ms)", "Tests")
	fmt.Println(strings.Repeat("-", 80))

	sizeResults := cli.groupResultsBySize(result)
	sizes := []string{"512B", "2KB", "8KB"}

	for _, size := range sizes {
		if sizeData, exists := sizeResults[size]; exists && len(sizeData.Rankings) > 0 {
			best := sizeData.Rankings[0]
			fmt.Printf("%-10s | %-25s | %8.2f | %8.2f | %10.2f | %6d\n",
				size,
				best.AlgorithmCombination,
				best.KEIScore.OverallScore,
				best.KEIScore.CompressionRatio,
				float64(sizeData.AvgTime)/1e6, // Convert to ms
				sizeData.TestCount)
		}
	}

	// 3. Performance Patterns Analysis
	fmt.Println("\n🔍 PERFORMANCE PATTERNS:")
	fmt.Println(strings.Repeat("-", 80))
	for inputType, inputAnalysis := range analysis.InputTypeAnalysis {
		fmt.Printf("\n%s:\n", strings.ToUpper(inputType))
		fmt.Printf("  • Average KEI: %.2f (±%.2f)\n", inputAnalysis.AverageKEI, inputAnalysis.KEIStandardDeviation)
		fmt.Printf("  • Pattern: %s\n", inputAnalysis.PerformancePattern)
		if len(inputAnalysis.OptimalStrategies) > 0 {
			fmt.Printf("  • Optimal Strategy: %s\n", inputAnalysis.OptimalStrategies[0])
		}
		if inputAnalysis.BestCombination != nil {
			fmt.Printf("  • Champion: %s (KEI: %.2f)\n",
				inputAnalysis.BestCombination.AlgorithmCombination,
				inputAnalysis.BestCombination.KEIScore.OverallScore)
		}
	}

	// 4. Algorithm Effectiveness Matrix
	fmt.Println("\n🧮 ALGORITHM EFFECTIVENESS MATRIX:")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("%-10s | %-12s | %-15s | %-15s | %-8s\n", "Algorithm", "Appearances", "Success Rate", "Avg KEI Contrib", "Rating")
	fmt.Println(strings.Repeat("-", 80))

	for algorithm, algAnalysis := range analysis.AlgorithmEffectiveness {
		successRate := float64(algAnalysis.SuccessfulAppearances) / float64(algAnalysis.AppearanceCount) * 100
		fmt.Printf("%-10s | %12d | %14.1f%% | %15.2f | %-8s\n",
			algorithm,
			algAnalysis.AppearanceCount,
			successRate,
			algAnalysis.AverageKEIContribution,
			algAnalysis.EffectivenessRating)
	}

	// 5. Statistical Summary for Research
	fmt.Println("\n📊 STATISTICAL SUMMARY FOR RESEARCH:")
	fmt.Println(strings.Repeat("-", 80))
	summary := analysis.StatisticalSummary
	fmt.Printf("KEI Score Distribution:\n")
	for range_, count := range summary.KEIScoreDistribution {
		fmt.Printf("  %s: %d tests\n", range_, count)
	}

	fmt.Printf("\nCompression Ratio Distribution:\n")
	for range_, count := range summary.CompressionRatioDistribution {
		fmt.Printf("  %s: %d tests\n", range_, count)
	}

	if len(summary.SignificantFindings) > 0 {
		fmt.Printf("\nSignificant Findings:\n")
		for _, finding := range summary.SignificantFindings {
			fmt.Printf("  • %s\n", finding)
		}
	}

	// 6. Research Recommendations
	fmt.Println("\n🎯 RESEARCH RECOMMENDATIONS:")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Println("Based on this comprehensive analysis:")

	if result.BestOverall != nil {
		fmt.Printf("• Overall Champion: %s demonstrates superior performance\n", result.BestOverall.AlgorithmCombination)
	}

	fmt.Println("• Input-specific optimization is crucial for maximum efficiency")
	fmt.Println("• Algorithm order in pipelines significantly impacts performance")
	fmt.Println("• Hybrid approaches outperform single-algorithm solutions")

	if len(analysis.OptimalCombinations.BestForCompression.AlgorithmCombination) > 0 {
		fmt.Printf("• For compression-critical applications: %s\n", analysis.OptimalCombinations.BestForCompression.AlgorithmCombination)
	}
	if len(analysis.OptimalCombinations.BestForSpeed.AlgorithmCombination) > 0 {
		fmt.Printf("• For speed-critical applications: %s\n", analysis.OptimalCombinations.BestForSpeed.AlgorithmCombination)
	}

	// NEW: Add comprehensive timeout analysis
	cli.displayTimeoutAnalysis(result)

	fmt.Println("\n" + strings.Repeat("═", 80))
}

// calculateAverageTime calculates average processing time for a combination across input type
func (cli *CLI) calculateAverageTime(results []*benchmarks.TestResult, combination string) float64 {
	var totalTime time.Duration
	var count int

	for _, result := range results {
		if result.AlgorithmCombination == combination && result.Success {
			totalTime += result.Duration
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return float64(totalTime.Nanoseconds()) / float64(count) / 1e6 // Convert to milliseconds
}

// SizeResults holds grouped results by size
type SizeResults struct {
	Rankings []struct {
		AlgorithmCombination string
		KEIScore             *benchmarks.KEIScore
	}
	TestCount int
	AvgTime   time.Duration
}

// groupResultsBySize groups results by input size for analysis
func (cli *CLI) groupResultsBySize(result *benchmarks.BenchmarkResult) map[string]*SizeResults {
	sizeGroups := make(map[string]*SizeResults)

	// Collect all results and group by size
	sizeData := make(map[string][]*benchmarks.TestResult)

	for _, inputResults := range result.ResultsByInputType {
		for _, testResult := range inputResults.TestResults {
			sizeName := cli.getSizeName(testResult.InputSize)
			sizeData[sizeName] = append(sizeData[sizeName], testResult)
		}
	}

	// Process each size group
	for sizeName, results := range sizeData {
		sizeResult := &SizeResults{
			TestCount: len(results),
		}

		// Calculate combinations performance for this size
		combPerf := make(map[string]*benchmarks.KEIScore)
		combCount := make(map[string]int)
		var totalTime time.Duration

		for _, result := range results {
			if result.Success && result.KEIScore != nil {
				if existing, exists := combPerf[result.AlgorithmCombination]; exists {
					// Average the KEI scores
					existing.OverallScore = (existing.OverallScore + result.KEIScore.OverallScore) / 2
					existing.CompressionRatio = (existing.CompressionRatio + result.KEIScore.CompressionRatio) / 2
				} else {
					combPerf[result.AlgorithmCombination] = result.KEIScore
				}
				combCount[result.AlgorithmCombination]++
				totalTime += result.Duration
			}
		}

		// Create rankings
		for combination, keiScore := range combPerf {
			sizeResult.Rankings = append(sizeResult.Rankings, struct {
				AlgorithmCombination string
				KEIScore             *benchmarks.KEIScore
			}{
				AlgorithmCombination: combination,
				KEIScore:             keiScore,
			})
		}

		// Sort by KEI score
		sort.Slice(sizeResult.Rankings, func(i, j int) bool {
			return sizeResult.Rankings[i].KEIScore.OverallScore > sizeResult.Rankings[j].KEIScore.OverallScore
		})

		if len(results) > 0 {
			sizeResult.AvgTime = totalTime / time.Duration(len(results))
		}

		sizeGroups[sizeName] = sizeResult
	}

	return sizeGroups
}

// getSizeName converts byte size to readable name
func (cli *CLI) getSizeName(bytes int64) string {
	switch bytes {
	case 512:
		return "512B"
	case 2048:
		return "2KB"
	case 8192:
		return "8KB"
	default:
		if bytes < 1024 {
			return fmt.Sprintf("%dB", bytes)
		} else if bytes < 1024*1024 {
			return fmt.Sprintf("%dKB", bytes/1024)
		} else {
			return fmt.Sprintf("%dMB", bytes/(1024*1024))
		}
	}
}

// Helper methods

func (cli *CLI) prompt(message string) string {
	fmt.Print(message)
	cli.scanner.Scan()
	return strings.TrimSpace(cli.scanner.Text())
}

func (cli *CLI) promptMultiSelect(message string, options []string) []string {
	input := cli.prompt(message)
	var selected []string

	for _, item := range strings.Split(input, ",") {
		item = strings.TrimSpace(item)
		for _, option := range options {
			if strings.EqualFold(item, option) {
				selected = append(selected, option)
				break
			}
		}
	}

	return selected
}

func (cli *CLI) parseAlgorithmList(input string, available []string) []string {
	var selected []string

	for _, item := range strings.Split(input, ",") {
		item = strings.TrimSpace(item)
		for _, algorithm := range available {
			if strings.EqualFold(item, algorithm) {
				selected = append(selected, algorithm)
				break
			}
		}
	}

	return selected
}

func (cli *CLI) estimateDuration(totalTests int64) string {
	// Rough estimate: 1-5 seconds per test depending on complexity
	avgTestTime := 3 * time.Second
	estimatedTime := time.Duration(totalTests) * avgTestTime

	if estimatedTime < time.Minute {
		return fmt.Sprintf("%.0fs", estimatedTime.Seconds())
	} else if estimatedTime < time.Hour {
		return fmt.Sprintf("%.0fm", estimatedTime.Minutes())
	} else {
		return fmt.Sprintf("%.1fh", estimatedTime.Hours())
	}
}

func (cli *CLI) saveResults(result *benchmarks.BenchmarkResult) {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(cli.config.OutputDirectory, 0755); err != nil {
		fmt.Printf("❌ Failed to create output directory: %v\n", err)
		return
	}

	// Save JSON results
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("benchmark_results_%s.json", timestamp)
	filepath := filepath.Join(cli.config.OutputDirectory, filename)

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("❌ Failed to marshal results: %v\n", err)
		return
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		fmt.Printf("❌ Failed to save results: %v\n", err)
		return
	}

	fmt.Printf("✅ Results saved to: %s\n", filepath)
}

func (cli *CLI) saveAnalysis(analysis *benchmarks.ComprehensiveAnalysisReport) {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("benchmark_analysis_%s.json", timestamp)
	filepath := filepath.Join(cli.config.OutputDirectory, filename)

	data, err := json.MarshalIndent(analysis, "", "  ")
	if err != nil {
		fmt.Printf("❌ Failed to marshal analysis: %v\n", err)
		return
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		fmt.Printf("❌ Failed to save analysis: %v\n", err)
		return
	}

	fmt.Printf("✅ Analysis saved to: %s\n", filepath)
}

func (cli *CLI) loadConfiguration() {
	filename := cli.prompt("Configuration file path: ")

	data, err := os.ReadFile(filename)
	if err != nil {
		fmt.Printf("❌ Failed to read configuration: %v\n", err)
		return
	}

	var config benchmarks.BenchmarkConfig
	if err := json.Unmarshal(data, &config); err != nil {
		fmt.Printf("❌ Failed to parse configuration: %v\n", err)
		return
	}

	cli.config = &config
	fmt.Printf("✅ Configuration loaded from: %s\n", filename)
}

func (cli *CLI) saveConfiguration() {
	filename := cli.prompt("Save configuration as: ")

	data, err := json.MarshalIndent(cli.config, "", "  ")
	if err != nil {
		fmt.Printf("❌ Failed to marshal configuration: %v\n", err)
		return
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		fmt.Printf("❌ Failed to save configuration: %v\n", err)
		return
	}

	fmt.Printf("✅ Configuration saved to: %s\n", filename)
}

func (cli *CLI) viewAlgorithmInfo() {
	fmt.Println("\n🧮 ALGORITHM INFORMATION")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	combGen := benchmarks.NewCombinationGenerator()
	algorithms := combGen.GetAvailableAlgorithms()

	for name, info := range algorithms {
		fmt.Printf("\n%s:\n", name)
		fmt.Printf("  Category: %s\n", info.Category)
		fmt.Printf("  Available: %t\n", info.Available)
	}
}

func (cli *CLI) estimateExecutionTime() {
	if len(cli.config.SelectedAlgorithms) == 0 {
		fmt.Println("❌ Please select algorithms first!")
		return
	}

	combGen := benchmarks.NewCombinationGenerator()
	combGen.SetSelectedAlgorithms(cli.config.SelectedAlgorithms)
	combGen.SetCombinationSizeRange(cli.config.MinCombinationSize, cli.config.MaxCombinationSize)

	totalCombinations := combGen.EstimateTotalCombinations()
	totalInputs := int64(len(cli.config.InputTypes) * len(cli.config.InputSizes))
	totalTests := totalCombinations * totalInputs

	fmt.Println("\n⏱️  EXECUTION TIME ESTIMATE")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("Total combinations: %d\n", totalCombinations)
	fmt.Printf("Total inputs: %d\n", totalInputs)
	fmt.Printf("Total tests: %d\n", totalTests)
	fmt.Printf("Estimated duration: %s\n", cli.estimateDuration(totalTests))
	fmt.Printf("With %d concurrent tests: %s\n",
		cli.config.MaxConcurrentTests,
		cli.estimateDuration(totalTests/int64(cli.config.MaxConcurrentTests)))
}

func (cli *CLI) setupSignalHandling() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		fmt.Println("\n\n🛑 Received interrupt signal. Gracefully shutting down...")
		fmt.Println("Thank you for using the Hybrid Compression Benchmark!")
		os.Exit(0)
	}()
}
