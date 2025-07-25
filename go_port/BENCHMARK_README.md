# Hybrid Compression Algorithm Brute Force Benchmark

A comprehensive benchmarking system that tests all combinations of compression algorithms to find the most efficient solutions for different types of data and use cases.

## Overview

This benchmark implements a brute force approach to test **all meaningful combinations** of compression algorithms across **10 different engineered input types** with detailed efficiency analysis. The system provides:

- **Combinatorial Algorithm Testing**: Tests all permutations of selected algorithms (P(n,r) combinations)
- **10 Engineered Input Types**: Specifically crafted inputs to favor different algorithms
- **Key Efficiency Index (KEI)**: Multi-dimensional scoring system for comprehensive evaluation
- **Real-time Progress Reporting**: Verbose progress with ETA, progress bars, and live results
- **Comprehensive Analysis**: Deep insights into algorithm effectiveness across different domains
- **Interactive CLI**: User-friendly interface for configuration and execution

## Architecture

### Core Components

1. **Input Generator** (`input_generator.go`): Creates 10 types of engineered test data
2. **Combination Generator** (`combination_generator.go`): Generates all algorithm permutations
3. **KEI Calculator** (`kei_metrics.go`): Calculates Key Efficiency Index scores
4. **Benchmark Executor** (`benchmark_executor.go`): Orchestrates the entire execution
5. **Progress Reporter** (`progress_reporter.go`): Provides verbose progress tracking
6. **Analysis Engine** (`analysis_engine.go`): Performs comprehensive result analysis
7. **CLI Interface** (`cmd/benchmark_brute_force/main.go`): Interactive user interface

### Algorithm Categories

The system includes 10 compression algorithms across 6 categories:

- **Run Length**: RLE
- **Transform**: BWT, MTF
- **Predictive**: Delta
- **Dictionary**: LZ77, LZ78, LZW
- **Entropy Coding**: Huffman
- **Hybrid**: Deflate, Arithmetic

## Engineered Input Types

The system generates 10 specifically engineered input types designed to favor different algorithms:

1. **Repetitive**: Highly repetitive data with long runs (favors RLE)
2. **Text Patterns**: Common text patterns and dictionary words (favors LZ* algorithms)
3. **Random**: Cryptographically random data (favors Huffman/Arithmetic)
4. **Sequential**: Sequential numeric data with small deltas (favors Delta)
5. **Natural Text**: Natural language patterns (favors BWT+MTF combinations)
6. **Structured Binary**: Binary data with headers and structures (favors LZ77/LZW)
7. **Mixed**: Combined characteristics from multiple types (favors hybrid approaches)
8. **Sparse**: Data with many zeros and sparse non-zero values (favors RLE/Huffman)
9. **Alternating**: Regular alternating patterns (favors LZW/RLE)
10. **Log**: Log-like structured text data (favors LZ77/LZW/BWT)

## Key Efficiency Index (KEI)

The KEI system evaluates algorithms across 5 dimensions:

- **Compression Ratio** (default weight: 30%): How well the algorithm compresses data
- **Speed** (25%): Processing speed and throughput
- **Memory** (20%): Memory efficiency and usage patterns
- **Stability** (15%): Performance consistency and predictability
- **Energy** (10%): Energy efficiency for battery-powered devices

### KEI Weight Presets

- **Default**: Balanced across all dimensions
- **Performance**: Speed-focused (40% speed, 20% compression)
- **Compression**: Compression-focused (50% compression, 15% speed)
- **Memory**: Memory-focused (40% memory, 25% compression)

## Quick Start

### 1. Build the System

```bash
cd go_port
go build -o benchmark_brute_force ./cmd/benchmark_brute_force
```

### 2. Run Interactive CLI

```bash
./benchmark_brute_force
```

### 3. Basic Configuration

1. Select algorithms (minimum 2)
2. Set combination size range (e.g., 2-3 algorithms per combination)
3. Choose input types (start with standard set)
4. Configure input sizes (1KB, 10KB, 100KB recommended for testing)
5. Run benchmark

### 4. Quick Example

```go
package main

import (
    "fmt"
    "hybrid-compression-study/pkg/benchmarks"
)

func main() {
    // Create benchmark suite
    suite, err := benchmarks.NewBenchmarkSuite()
    if err != nil {
        panic(err)
    }
    
    // Run quick benchmark (3 algorithms, 2-3 combinations, 3 input types)
    results, err := suite.RunQuickBenchmark()
    if err != nil {
        panic(err)
    }
    
    // Display results
    fmt.Printf("Best combination: %s (KEI: %.2f)\n", 
        results.BestOverall.AlgorithmCombination,
        results.BestOverall.KEIScore.OverallScore)
}
```

## Configuration Options

### Algorithm Selection

Choose which algorithms to include in combinations:

```go
config.SelectedAlgorithms = []string{"RLE", "BWT", "LZ77", "Huffman"}
```

### Combination Size Range

Control the size of algorithm combinations:

```go
config.MinCombinationSize = 2  // Minimum 2 algorithms
config.MaxCombinationSize = 3  // Maximum 3 algorithms
```

**Warning**: With 10 algorithms, combinations grow exponentially:
- 2-3 algorithms: 180 combinations
- 2-4 algorithms: 5,040 combinations
- 2-5 algorithms: 33,600 combinations

### Input Configuration

Select input types and sizes:

```go
config.InputTypes = []benchmarks.InputType{
    benchmarks.InputTypeRepetitive,
    benchmarks.InputTypeTextPatterns,
    benchmarks.InputTypeRandom,
}

config.InputSizes = []benchmarks.InputSize{
    {"1KB", 1024},
    {"10KB", 10 * 1024},
    {"100KB", 100 * 1024},
}
```

### Execution Parameters

Control benchmark execution:

```go
config.TimeoutPerTest = 30 * time.Second  // Max time per test
config.MaxConcurrentTests = 4             // Parallel execution
config.RetryFailedTests = 1               // Retry on failure
```

## Understanding Results

### KEI Scores

KEI scores range from 0-100 with performance categories:

- **90-100**: Exceptional
- **80-89**: Excellent  
- **70-79**: Very Good
- **60-69**: Good
- **50-59**: Average
- **40-49**: Below Average
- **30-39**: Poor
- **0-29**: Very Poor

### Result Analysis

The system provides comprehensive analysis including:

1. **Overall Rankings**: Best combinations across all scenarios
2. **Domain Effectiveness**: Best for compression, speed, memory, etc.
3. **Input Type Analysis**: Which algorithms work best for specific data types
4. **Algorithm Effectiveness**: Individual algorithm performance patterns
5. **Recommendations**: Actionable advice for different use cases

### Example Output

```
🏆 BEST OVERALL COMBINATION:
  Pipeline_[BWT LZ77 Huffman]
  KEI Score: 87.3
  Category: Excellent

📦 BEST COMPRESSION:
  Pipeline_[BWT MTF LZ77 Huffman]  
  Compression Ratio: 15.7

⚡ FASTEST COMBINATION:
  Pipeline_[RLE Huffman]
  Speed Score: 94.2

📈 TOP 5 COMBINATIONS:
  1. Pipeline_[BWT LZ77 Huffman] (KEI: 87.3)
  2. Pipeline_[RLE LZ77 Huffman] (KEI: 84.1)
  3. Pipeline_[Delta LZ77 Huffman] (KEI: 82.5)
  4. Pipeline_[BWT MTF Huffman] (KEI: 81.2)
  5. Pipeline_[LZ77 Huffman] (KEI: 79.8)
```

## Advanced Usage

### Custom Input Generation

```go
suite, _ := benchmarks.NewBenchmarkSuite()

// Generate custom 50KB repetitive input
customInput, err := suite.GenerateCustomInput(
    benchmarks.InputTypeRepetitive, 
    50*1024,
)

// Test specific combination
result, err := suite.BenchmarkSingleCombination(
    []string{"RLE", "Huffman"}, 
    customInput,
)

fmt.Printf("KEI Score: %.2f\n", result.KEIScore.OverallScore)
```

### Custom KEI Weights

```go
config.KEIWeights = benchmarks.KEIWeights{
    CompressionRatio: 0.4,  // 40% weight on compression
    Speed:           0.3,   // 30% weight on speed
    Memory:          0.2,   // 20% weight on memory
    Stability:       0.05,  // 5% weight on stability
    Energy:          0.05,  // 5% weight on energy
}
```

### Full Analysis

```go
// Run comprehensive benchmark
results, err := suite.RunFullBenchmark(allAlgorithms)

// Generate detailed analysis
analysis := suite.AnalyzeResults(results)

// Access specific insights
fmt.Printf("Compression insights: %v\n", 
    analysis.EfficiencyInsights.CompressionEfficiencyInsights)
fmt.Printf("Speed insights: %v\n", 
    analysis.EfficiencyInsights.SpeedEfficiencyInsights)
```

## Performance Considerations

### Execution Time Estimates

Rough estimates for different configurations:

| Algorithms | Combinations | Inputs | Tests | Estimated Time |
|------------|-------------|--------|-------|----------------|
| 3          | 6           | 9      | 54    | ~3 minutes     |
| 4          | 24          | 15     | 360   | ~18 minutes    |
| 5          | 120         | 15     | 1,800 | ~1.5 hours     |
| 6          | 720         | 25     | 18,000| ~15 hours      |

### Memory Usage

- **Light**: 2-3 algorithms, small inputs (< 100MB RAM)
- **Medium**: 4-5 algorithms, medium inputs (< 500MB RAM)  
- **Heavy**: 6+ algorithms, large inputs (> 1GB RAM)

### Optimization Tips

1. **Start Small**: Test with 2-3 algorithms first
2. **Limit Input Sizes**: Use 1KB-100KB for initial testing
3. **Use Concurrency**: Set `MaxConcurrentTests` to your CPU core count
4. **Save Results**: Enable result saving for long-running benchmarks
5. **Filter Algorithms**: Focus on relevant algorithms for your use case

## CLI Features

### Interactive Configuration

The CLI provides step-by-step configuration:

1. **Algorithm Selection**: Choose individual algorithms or categories
2. **Combination Range**: Set min/max combination sizes with estimates
3. **Input Configuration**: Select standard or custom input sets
4. **Execution Tuning**: Configure timeouts, concurrency, retries
5. **KEI Weights**: Choose presets or set custom weights
6. **Output Options**: Configure saving and reporting

### Progress Reporting

Real-time progress includes:

- **Progress Bar**: Visual completion indicator
- **ETA**: Estimated time remaining
- **Live Results**: Recent test results with KEI scores
- **Statistics**: Running averages and success rates
- **Pause/Resume**: Interrupt and continue capability

### Result Export

Results can be exported as:

- **JSON**: Complete results with all metrics
- **Analysis Report**: Comprehensive analysis and insights
- **Configuration**: Save/load benchmark configurations

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce input sizes or algorithm count
2. **Timeouts**: Increase `TimeoutPerTest` for complex combinations
3. **No Results**: Check algorithm selection and input configuration
4. **Slow Performance**: Reduce concurrency if system is overloaded

### Debug Options

Enable verbose output for detailed logging:

```go
config.VerboseOutput = true
config.ShowProgressBar = true
config.ShowRealTimeResults = true
```

## Use Cases

### Research Applications

- **Algorithm Comparison**: Systematic evaluation of compression approaches
- **Performance Analysis**: Understanding algorithm behavior across data types
- **Optimization Studies**: Finding optimal combinations for specific scenarios

### Production Planning

- **System Design**: Choose algorithms for storage/transmission systems
- **Performance Prediction**: Estimate compression performance for real data
- **Resource Planning**: Understand memory/CPU requirements

### Educational Purposes

- **Algorithm Learning**: Understand how different algorithms behave
- **Performance Benchmarking**: Learn about systematic performance evaluation
- **Data Analysis**: Explore relationships between data characteristics and compression

## Contributing

The benchmark system is designed to be extensible:

1. **New Algorithms**: Add to `combination_generator.go`
2. **Input Types**: Extend `input_generator.go`
3. **Metrics**: Add dimensions to KEI calculation
4. **Analysis**: Enhance the analysis engine

## License

This benchmark system is part of the hybrid compression study research project.

---

For more information, run the interactive CLI or consult the source code documentation. 