# Test Data Files

This directory contains various test files for compression benchmarking:

## File Types

- **text.txt**: Realistic text with common English words
- **repetitive.bin**: Highly repetitive binary data (good for RLE)
- **structured.json**: JSON data with realistic structure
- **binary.bin**: Mixed binary data with patterns
- **sparse.bin**: Sparse data (mostly zeros)
- **english.txt**: Text with realistic English letter frequencies

## Size Variants

- **small_***: ~5KB files for quick testing
- **medium_***: ~25KB files for moderate testing  
- **large_***: ~100KB files for comprehensive testing

## Usage

These files are automatically used by the benchmarking suite when you specify
the test data directory:

```bash
hcs benchmark --data-dir data/test_files
```

You can also use them individually:

```bash
hcs compress data/test_files/text.txt compressed_output --algorithm huffman
```
