# 📊 Configuration Guide - Hybrid Compression Study

## 🚀 **Your Configuration File: `experiment.conf`**

The main configuration file is **`experiment.conf`** - this is where you define your compression experiments using the simple format you requested.

## 📋 **Step-by-Step Usage**

### **Step 1: Generate Your Configuration File**
```bash
# This creates the main configuration file with examples
./compress --generate-config experiment.conf
```

### **Step 2: View the Generated Configuration**
```bash
cat experiment.conf
```

**Output:**
```conf
# Hybrid Compression Study - Custom Configuration
# Format: key = value
# Pipelines: name = alg1+alg2+alg3
# Algorithm params: alg.param = value

default = text_optimized

# Global Variables
window_size = 4096
compression_level = 6
adaptive_model = true
debug_mode = false

# Pipeline Definitions
high_compression = delta+bwt+mtf+rle+lzw+huffman
fast_compression = rle+lzw
dictionary_test = lz77+lz78+lzw
entropy_test = huffman+arithmetic
deflate_comparison = deflate
everything = delta+bwt+mtf+rle+lz77+lz78+lzw+huffman+arithmetic
text_optimized = bwt+mtf+rle+huffman

# Algorithm Parameters
arithmetic.precision = 32
arithmetic.adaptive_model = true
delta.data_width = 1
delta.signed_data = false
rle.escape_byte = 0
rle.min_run_length = 3
lz77.buffer_size = 256
lz77.window_size = 8192
lz78.max_dict_size = 8192
lz78.reset_threshold = 8192
deflate.enable_huffman = true
deflate.enable_lz77 = true
deflate.compression_level = 9
```

### **Step 3: List All Available Pipelines**
```bash
./compress -c experiment.conf --list-pipelines
```

This shows all your defined pipelines with their algorithms and parameters.

### **Step 4: Run Your Experiments**
```bash
# Run specific experiment
./compress -c experiment.conf -p text_optimized input.txt

# Run with detailed output
./compress -c experiment.conf -p high_compression --verbose input.txt

# Run the default pipeline (if defined)
./compress -c experiment.conf input.txt
```

## ✨ **Your Requested Format Examples**

### **Simple Format You Wanted:**
```conf
experiment1 = rle+huf+lz77
experiment2 = delta+bwt+mtf
experiment3 = lz78+arithmetic
experiment4 = deflate
```

### **With Parameters Using Equals:**
```conf
# Pipeline definitions
experiment1 = rle+huf+lz77
custom_lz77 = lz77

# Algorithm parameters
lz77.window_size = 16384
lz77.buffer_size = 512
rle.min_run_length = 4
huffman.optimize = true

# Global variables
compression_level = 9
debug = true
default = experiment1
```

## 🎯 **Practical Examples**

### **Example 1: Text Compression Experiments**
Create a file `text_experiments.conf`:
```conf
# Text-focused experiments
basic_text = rle+huffman
advanced_text = bwt+mtf+rle+huffman
ultra_text = delta+bwt+mtf+rle+lzw+huffman

# Parameters for better text compression
rle.min_run_length = 2
bwt.block_size = 8192
lzw.max_code_bits = 15

# Default experiment
default = advanced_text
```

**Usage:**
```bash
./compress -c text_experiments.conf -p basic_text document.txt
./compress -c text_experiments.conf -p advanced_text --verbose document.txt
```

### **Example 2: Dictionary-Based Testing**
Create a file `dictionary_tests.conf`:
```conf
# Dictionary algorithm comparisons
lz77_only = lz77
lz78_only = lz78
lzw_only = lzw
combined_dictionary = lz77+lz78+lzw
deflate_vs_manual = deflate

# Custom parameters for each
lz77.window_size = 32768
lz77.buffer_size = 1024
lz78.max_dict_size = 16384
lzw.max_code_bits = 14
deflate.compression_level = 9

default = combined_dictionary
```

### **Example 3: Speed vs Compression Tests**
Create a file `performance_tests.conf`:
```conf
# Speed-focused
fastest = rle
fast = rle+lzw
medium = huffman+lzw

# Compression-focused
good_compression = bwt+mtf+huffman
best_compression = delta+bwt+mtf+rle+lzw+huffman
ultimate = everything = delta+bwt+mtf+rle+lz77+lz78+lzw+huffman+arithmetic

# Balanced
balanced = deflate

default = medium
```

## 🔧 **Algorithm Reference**

### **Available Algorithms:**
- `rle` - Run-Length Encoding (fast, good for repetitive data)
- `huffman` or `huf` - Huffman Coding (entropy coding)
- `lzw` - Lempel-Ziv-Welch (dictionary-based)
- `bwt` - Burrows-Wheeler Transform (preprocessing)
- `mtf` - Move-To-Front (preprocessing)
- `delta` - Delta Encoding (predictive)
- `lz77` - LZ77 sliding window
- `lz78` - LZ78 explicit dictionary
- `deflate` - Deflate (LZ77 + Huffman)
- `arithmetic` - Arithmetic Coding (advanced entropy)

### **Algorithm Parameters:**

#### **RLE Parameters:**
```conf
rle.escape_byte = 0          # Escape byte (0-255)
rle.min_run_length = 3       # Minimum run length
```

#### **LZ77 Parameters:**
```conf
lz77.window_size = 8192      # Sliding window size
lz77.buffer_size = 256       # Look-ahead buffer size
```

#### **LZ78 Parameters:**
```conf
lz78.max_dict_size = 8192    # Maximum dictionary size
lz78.reset_threshold = 8192  # Dictionary reset threshold
```

#### **Deflate Parameters:**
```conf
deflate.compression_level = 9    # Compression level (1-9)
deflate.enable_huffman = true    # Use Huffman coding
deflate.enable_lz77 = true       # Use LZ77 compression
```

#### **Arithmetic Parameters:**
```conf
arithmetic.precision = 32        # Precision bits
arithmetic.adaptive_model = true # Use adaptive model
```

#### **Delta Parameters:**
```conf
delta.data_width = 1        # Data width in bytes
delta.signed_data = false   # Treat data as signed
```

## 📊 **Running and Analyzing Results**

### **Basic Execution:**
```bash
# Run experiment with basic output
./compress -c experiment.conf -p experiment1 test_file.txt
```

### **Detailed Analysis:**
```bash
# Run with verbose pipeline analysis
./compress -c experiment.conf -p experiment1 --verbose test_file.txt
```

### **Statistical Analysis:**
```bash
# Run multiple times for statistical accuracy
./compress -c experiment.conf -p experiment1 --tries 10 test_file.txt
```

### **JSON Output for Processing:**
```bash
# Get machine-readable results
./compress -c experiment.conf -p experiment1 --json test_file.txt
```

## 🎯 **Tips for Best Results**

### **For Text Files:**
```conf
text_optimized = bwt+mtf+rle+huffman
```

### **For Binary Data:**
```conf
binary_optimized = delta+rle+lz77
```

### **For Mixed Data:**
```conf
general_purpose = deflate
```

### **For Maximum Compression:**
```conf
max_compression = delta+bwt+mtf+rle+lz77+lzw+huffman
```

### **For Speed:**
```conf
fast_compression = rle+lzw
```

## 🚀 **Quick Commands Reference**

```bash
# Generate config file
./compress --generate-config experiment.conf

# List all pipelines
./compress -c experiment.conf --list-pipelines

# Run specific pipeline
./compress -c experiment.conf -p experiment1 input.txt

# Run with verbose output
./compress -c experiment.conf -p experiment1 --verbose input.txt

# Run multiple times for statistics
./compress -c experiment.conf -p experiment1 --tries 5 input.txt

# Use default pipeline
./compress -c experiment.conf input.txt
```

## ✅ **Your Configuration is Working!**

The system fully supports your requested format:
- ✅ `experiment1 = rle+huf+lz77`
- ✅ `lz77.window_size = 8192`
- ✅ `compression_level = 6`
- ✅ Any combination of algorithms
- ✅ All parameters configurable
- ✅ Simple, readable format

**Start experimenting with your compression pipelines!** 🎉 