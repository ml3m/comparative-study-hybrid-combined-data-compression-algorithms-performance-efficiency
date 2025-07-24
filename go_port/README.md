# Hybrid Compression Study - Go Port

**Aerospace-Grade Compression Algorithm Benchmarking and Analysis Tool**

This is a comprehensive Go port of the hybrid compression study framework, designed for mission-critical applications where every byte and nanosecond matters. The tool provides nanosecond-precision performance metrics suitable for aerospace and other high-reliability systems.

## 🚀 Features

### Compression Algorithms
- **Huffman Coding**: Optimal entropy coding with aerospace-grade metrics
- **Run-Length Encoding (RLE)**: Efficient for data with repetitive patterns
- **Lempel-Ziv-Welch (LZW)**: Dictionary-based compression with adaptive coding

### Aerospace-Grade Metrics
- **Nanosecond-precision timing**: Ultra-precise performance measurement
- **Memory usage tracking**: Byte-level memory consumption analysis
- **CPU utilization monitoring**: Real-time CPU usage and efficiency metrics
- **I/O operation tracking**: Detailed disk I/O performance analysis
- **Mission-readiness assessment**: Suitability for real-time and memory-constrained environments

### Scientific Output
- **Comprehensive analysis reports**: Detailed performance and efficiency metrics
- **JSON output support**: Machine-readable results for further analysis
- **Mission-critical reporting**: Aerospace-grade status assessments
- **Resource utilization analysis**: Complete system resource profiling

## 🛠️ Installation

### Prerequisites
- Go 1.21 or later
- Unix-like system (Linux, macOS, etc.)

### Build from Source
```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd hybrid-compression-study/go_port

# Install dependencies
go mod tidy

# Build the CLI tool
go build -o compress cmd/compress/main.go

# Make it executable and optionally install globally
chmod +x compress
# sudo mv compress /usr/local/bin/  # Optional: install globally
```

## 📖 Usage

### Basic Compression
```bash
# Compress a file with Huffman coding (default)
./compress input.txt

# Compress with specific algorithm
./compress -a rle data.bin
./compress -a lzw document.txt

# Specify output file
./compress -i input.txt -o compressed.huff
```

### Decompression
```bash
# Decompress a file
./compress -d compressed.huff

# Decompress with specific algorithm
./compress -d -a rle compressed.rle
```

### Advanced Options
```bash
# Verbose output with detailed metrics
./compress --verbose input.txt

# JSON output for machine processing
./compress --json -a lzw data.txt

# Multiple tries for statistical analysis (aerospace-grade precision)
./compress --tries 10 input.txt
./compress -t 5 --verbose -a rle data.bin

# Algorithm-specific parameters
./compress -a rle --escape-byte 255 --min-run-length 4 data.bin
./compress -a lzw --max-code-bits 14 text.txt
```

### Algorithm-Specific Parameters

#### Statistical Analysis Options
- `-t, --tries N`: Number of compression runs to average for statistical analysis (default: 1)

#### Run-Length Encoding (RLE)
- `--escape-byte N`: Escape byte value (0-255, default: 0)
- `--min-run-length N`: Minimum run length to encode (default: 3)

#### Lempel-Ziv-Welch (LZW)
- `--max-code-bits N`: Maximum code bits (9-16, default: 12)

## 📊 Output Formats

### Scientific Report (Default)
The tool provides comprehensive aerospace-grade analysis:

#### Single Run Analysis
```
🚀 AEROSPACE-GRADE COMPRESSION ANALYSIS
═══════════════════════════════════════════════════════════════

📊 COMPRESSION SUMMARY
Algorithm: Huffman
Input File: example.txt
Output File: example.txt.huffman

📈 DATA INTEGRITY METRICS
Original Size:      10.24KB (10,485 bytes)
Compressed Size:    6.85KB (7,012 bytes)
Compression Ratio:  1.495x
Space Savings:      33.1% (3.39KB)
Effectiveness:      POSITIVE

⚡ PERFORMANCE PROFILE
Compression Time:   1.234ms (1,234,567 ns)
Throughput:         8.50 MB/s
Time per Byte:      117.67 ns/byte
CPU Efficiency:     8,492.31 bytes/cpu-sec

🔧 RESOURCE UTILIZATION
Peak Memory:        15.67KB
Memory Overhead:    1.493x data size
CPU Utilization:    Peak: 45.2%, Avg: 23.1%
I/O Operations:     Read: 1, Write: 1
Determinism Score:  1.000000 (1.0 = perfect)

🎯 MISSION READINESS ASSESSMENT
Worst-case Latency: 1,234,567 ns
Energy Efficiency:  8.50e+06 bytes/ns
Entropy Efficiency: 0.1869
Real-time Suitable: true
Memory Constrained: true
```

#### Statistical Analysis (Multiple Tries)
When using `--tries N` with N > 1, the tool provides comprehensive statistical analysis:

```
🚀 AEROSPACE-GRADE COMPRESSION ANALYSIS (AVERAGED)
═══════════════════════════════════════════════════════════════

📊 COMPRESSION SUMMARY
Algorithm: Huffman
Input File: example.txt  
Output File: example.txt.huffman
Runs: 10 successful / 10 total (100.0% success rate)

📈 DATA INTEGRITY METRICS
Original Size:      10.24KB (10,485 bytes)
Compressed Size:    6.85KB (7,012 bytes)
Avg Compression Ratio: 1.495x ± 0.003
Space Savings:      33.1% (3.39KB)
Effectiveness:      POSITIVE

⚡ PERFORMANCE PROFILE (AVERAGED)
Avg Compression Time:   1.234ms
Time Range:            1.201ms - 1.267ms (σ = 0.018ms)
Avg Throughput:        8.50 MB/s
Throughput Range:      8.31 - 8.69 MB/s
Time per Byte:         117.67 ns/byte
CPU Efficiency:        8,492.31 bytes/cpu-sec

📊 STATISTICAL ANALYSIS
Variability Score:      0.986 (1.0 = perfect consistency)
Consistency Rating:     EXCELLENT
Time Coefficient of Variation: 0.014
Result Reliability:     AEROSPACE_GRADE
```

### JSON Output
```bash
./compress --json input.txt
```

Returns structured JSON data suitable for automated analysis and integration with other systems.

## 🏗️ Architecture

### Core Components
- **`pkg/core`**: Base interfaces and data structures
- **`pkg/algorithms`**: Compression algorithm implementations
- **`pkg/pipeline`**: Multi-algorithm pipeline system
- **`internal/performance`**: Aerospace-grade performance monitoring
- **`cmd/compress`**: Command-line interface

### Performance Monitoring
The system uses advanced performance monitoring with:
- Nanosecond-precision timing using `time.Now().UnixNano()`
- Memory tracking via Go runtime and system calls
- CPU monitoring through `/proc` filesystem (Linux) and system APIs
- Real-time resource sampling during operations

### Algorithm Implementations
Each algorithm follows the `CompressionAlgorithm` interface:
```go
type CompressionAlgorithm interface {
    Compress(ctx context.Context, data []byte) (*CompressionResult, error)
    Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*DecompressionResult, error)
    GetName() string
    GetCategory() AlgorithmCategory
    // ... other methods
}
```

## 🧪 Testing

### Unit Tests
```bash
go test ./pkg/...
go test ./internal/...
```

### Performance Benchmarks
```bash
go test -bench=. ./pkg/algorithms/
```

### Integration Tests
```bash
go test ./tests/integration/
```

## 📈 Performance Characteristics

### Huffman Coding
- **Best for**: Text files, source code, structured data
- **Compression ratio**: 1.2x - 2.5x typical
- **Speed**: Very fast encoding/decoding
- **Memory usage**: Low, proportional to alphabet size

### Run-Length Encoding
- **Best for**: Images, data with repetitive patterns
- **Compression ratio**: Highly variable (1.0x - 50x+)
- **Speed**: Extremely fast
- **Memory usage**: Minimal

### LZW
- **Best for**: General-purpose compression
- **Compression ratio**: 2x - 4x typical
- **Speed**: Moderate
- **Memory usage**: Dictionary size dependent

## 🔧 Development

### Adding New Algorithms
1. Implement the `CompressionAlgorithm` interface
2. Add constructor function
3. Include in CLI algorithm selection
4. Add tests and benchmarks

### Extending Performance Metrics
The performance monitoring system is extensible. Add new metrics by:
1. Extending `AerospacePrecisionMetrics` struct
2. Updating measurement collection in `AerospaceGradeMonitor`
3. Adding calculation logic in metric conversion functions

## 📋 Scientific Applications

This tool is designed for:
- **Aerospace systems**: Mission-critical data compression
- **Real-time systems**: Low-latency compression requirements
- **Embedded systems**: Memory-constrained environments
- **Research**: Algorithm performance comparison and analysis
- **Quality assurance**: Deterministic performance validation

## 🏆 Performance Targets

### Mission-Critical Thresholds
- **Real-time suitability**: < 1 second total processing time
- **Memory efficiency**: < 2x data size peak memory usage
- **CPU efficiency**: < 80% peak CPU utilization
- **Determinism**: Perfect reproducibility (score = 1.0)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all benchmarks pass
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

- Original Python implementation: `../src/hybrid_compression_study/`
- Academic paper: [Link to research paper]
- Benchmark datasets: `../data/test_files/`

## 📞 Support

For issues, questions, or contributions, please open an issue on the project repository.

---

**Built for aerospace-grade reliability and performance** 🚀 