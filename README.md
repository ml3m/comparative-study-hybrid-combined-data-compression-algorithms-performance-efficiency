# Hybrid Compression Study Framework

A comprehensive Python framework for comparative analysis of hybrid and combined data compression algorithms, designed for performance and efficiency benchmarking.

## 🎯 Project Overview

This framework implements and benchmarks various compression algorithms to analyze their performance characteristics, compression ratios, and efficiency across different data types. It's designed for academic research and practical analysis of compression techniques.

## 🏗️ Architecture

The framework follows a modular, object-oriented design with clear separation of concerns:

```
src/hybrid_compression_study/
├── core/               # Core abstractions and base classes
├── algorithms/         # Compression algorithm implementations
├── benchmarks/         # Benchmarking suite
├── pipeline/           # Pipeline system for chaining algorithms
├── utils/             # Utility functions and performance monitoring
├── visualization/     # Data visualization and reporting
└── cli/              # Command-line interface
```

## 📚 Implemented Algorithms

### ✅ Currently Implemented

- **Run-Length Encoding (RLE)** - Simple lossless compression for repetitive data
  - Standard RLE with configurable escape byte and minimum run length
  - Adaptive RLE that optimizes parameters based on data characteristics
  
- **Huffman Coding** - Optimal entropy coding based on symbol frequencies
  - Complete binary tree implementation with bit-level operations
  - Handles all edge cases (empty data, single symbol, etc.)
  - Tree serialization for efficient storage

### 🚧 Planned Implementations

- Burrows-Wheeler Transform (BWT)
- Move-To-Front (MTF)
- Arithmetic Coding
- Lempel-Ziv-Welch (LZW)
- LZ77 and LZ78
- Delta Encoding
- Dictionary-based Compression
- Prediction by Partial Matching (PPM)
- Range Coding
- Deflate (LZ77 + Huffman)
- Modern algorithms (Snappy, Brotli integration)

## 🔧 Core Features

### Performance Monitoring
- Real-time CPU and memory usage tracking
- High-precision timing measurements
- Comprehensive metrics calculation (compression ratio, throughput, etc.)
- System information collection for benchmark context

### Modular Design
- Abstract base classes ensure consistent interfaces
- Easy addition of new algorithms
- Flexible pipeline system for chaining algorithms
- Comprehensive error handling and validation

### Professional Code Quality
- Type hints throughout the codebase
- Comprehensive documentation
- Unit tests and edge case handling
- Modern Python best practices (dataclasses, enums, context managers)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hybrid-compression-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from hybrid_compression_study.algorithms.rle import RunLengthEncoder
from hybrid_compression_study.algorithms.huffman import HuffmanEncoder

# RLE Example
rle = RunLengthEncoder()
data = b"AAAAAABBBBCCCC"
result = rle.compress(data)
print(f"Compression ratio: {result.compression_ratio:.2f}")

# Huffman Example
huffman = HuffmanEncoder()
data = b"hello world"
result = huffman.compress(data)
decompressed = huffman.decompress(result.compressed_data, result.metadata)
```

## 📊 Performance Metrics

The framework tracks comprehensive performance metrics:

- **Compression Ratio**: Original size / Compressed size
- **Compression Percentage**: Space saved as percentage
- **Throughput**: MB/s for compression and decompression
- **Memory Usage**: Peak and average memory consumption
- **CPU Usage**: Processor utilization during operations
- **Timing**: High-precision execution times

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=hybrid_compression_study --cov-report=html

# Type checking
mypy src/

# Code formatting
black src/ tests/
```

## 📈 Algorithm Performance Summary

| Algorithm | Best Use Case | Avg Compression Ratio | Speed | Memory Usage |
|-----------|---------------|----------------------|-------|-------------|
| RLE | Repetitive data | 2.0-8.0x | Very Fast | Low |
| Huffman | Text with varied frequencies | 1.2-2.5x | Fast | Medium |

*Note: Performance varies significantly based on data characteristics*

## 🤝 Contributing

This framework is designed for extensibility. To add a new compression algorithm:

1. Inherit from `CompressionAlgorithm` base class
2. Implement `compress()` and `decompress()` methods
3. Add appropriate error handling and metadata
4. Include comprehensive tests
5. Update documentation

## 📝 Research Applications

This framework is particularly useful for:

- **Academic Research**: Comparative studies of compression algorithms
- **Performance Analysis**: Benchmarking across different data types
- **Algorithm Development**: Testing new compression techniques
- **Educational Purposes**: Understanding compression algorithm internals

## 🏆 Key Features

- ✅ **Professional Quality**: Production-ready code with comprehensive error handling
- ✅ **Extensible**: Easy to add new algorithms and metrics
- ✅ **Well-Documented**: Comprehensive docstrings and examples
- ✅ **Performance-Focused**: Detailed monitoring and optimization
- ✅ **Research-Ready**: Designed for academic and industrial analysis

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.

---

*This framework is designed to support Bachelor's thesis research on "A Comparative Study of Hybrid and Combined Data Compression Algorithms for Performance and Efficiency Benchmarking"* 