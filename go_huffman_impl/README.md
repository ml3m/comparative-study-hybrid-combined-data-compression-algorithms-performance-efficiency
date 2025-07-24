# Huffman Encoding Study

Welcome to the Huffman Encoding Study repository! This project benchmarks various data compression algorithms to evaluate their performance and efficiency. The key algorithms analyzed include:

- **Huffman Encoding**
- **Burrows-Wheeler Transform (BWT)**
- **Run-Length Encoding (RLE)**
- **Combined Approaches**: BWT + RLE + Huffman Encoding

## Features

- **Compression Ratio Calculation**: Measures how well each algorithm compresses data.
- **Performance Metrics**: Tracks compression and decompression times.
- **Space Efficiency**: Analyzes memory usage during compression and decompression.
- **Diverse Testing**: Evaluates algorithms with different input sizes and types.

## Getting Started

1. **Clone the Repository**:
```bash
git clone https://github.com/ml3m/huffman_encoding_study.git
```

Navigate to the Project Directory:

```bash
cd huffman_encoding_study
```
Build the Project:

```bash
go build -o compression_benchmark
```
Run the Benchmark:
```bash
./compression_benchmark [input_file]
```

Results
The benchmarking script compares:

Simple Encoding
Huffman Encoding
BWT + Huffman Encoding
RLE + Huffman Encoding
BWT + RLE + Huffman Encoding
Results include compression sizes, performance times, and compression ratios.

Contributing
Contributions are welcome! To contribute, please fork the repository and submit a pull request. For issues or feature requests, open an issue on GitHub.



### Key Sections:

1. **Overview**: Briefly describes what the project does.
2. **Features**: Lists the main features and metrics of the benchmarks.
3. **Getting Started**: Provides instructions on how to clone, build, and run the project.
4. **Results**: Describes what the benchmarking script evaluates.
6. **Contributing**: Information on how others can contribute to the project.

Feel free to modify any sections or add more information to fit your needs!
