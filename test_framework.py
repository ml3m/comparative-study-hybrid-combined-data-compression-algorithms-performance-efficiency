#!/usr/bin/env python3
"""
Comprehensive test script for the hybrid compression study framework.

This script tests all major components: algorithms, pipelines, and benchmarking.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_compression_study.algorithms.rle import RunLengthEncoder
from hybrid_compression_study.algorithms.huffman import HuffmanEncoder
from hybrid_compression_study.algorithms.lz77 import LZ77Encoder
from hybrid_compression_study.algorithms.lzw import LZWEncoder
from hybrid_compression_study.pipeline.core import PipelineBuilder, PredefinedPipelines
from hybrid_compression_study.benchmarks.suite import (
    CompressionBenchmark, BenchmarkConfig, SyntheticDatasetProvider
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)


def print_step(step: str):
    """Print a test step."""
    print(f"\n📋 {step}")


def print_success(message: str):
    """Print a success message."""
    print(f"   ✅ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"   ❌ {message}")


def test_algorithms():
    """Test individual compression algorithms."""
    print_header("Testing Individual Algorithms")
    
    # Test data
    test_cases = [
        (b"AAAAAABBBBCCCCCCCC", "Repetitive data"),
        (b"hello world this is a test", "Text data"),
        (b"", "Empty data"),
        (b"A", "Single character"),
        (bytes(range(256)), "Binary data")
    ]
    
    algorithms = [
        ("RLE", RunLengthEncoder()),
        ("Huffman", HuffmanEncoder()),
        ("LZ77", LZ77Encoder(window_size=1024, lookahead_size=16)),
        ("LZW", LZWEncoder(max_code_bits=12))
    ]
    
    for algo_name, algorithm in algorithms:
        print_step(f"Testing {algo_name}")
        
        success_count = 0
        for test_data, description in test_cases:
            try:
                # Compress
                start_time = time.time()
                result = algorithm.compress(test_data)
                compression_time = time.time() - start_time
                
                # Decompress
                start_time = time.time()
                decompressed = algorithm.decompress(result.compressed_data, result.metadata)
                decompression_time = time.time() - start_time
                
                # Verify
                if decompressed.decompressed_data == test_data:
                    ratio = result.compression_ratio if result.compressed_size > 0 else float('inf')
                    print_success(f"{description}: {len(test_data)} → {result.compressed_size} bytes "
                                f"(ratio: {ratio:.2f}x, time: {compression_time+decompression_time:.3f}s)")
                    success_count += 1
                else:
                    print_error(f"{description}: Verification failed")
                    
            except Exception as e:
                print_error(f"{description}: {str(e)}")
        
        print(f"   📊 {algo_name}: {success_count}/{len(test_cases)} tests passed")


def test_pipelines():
    """Test compression pipelines."""
    print_header("Testing Compression Pipelines")
    
    test_data = b"The quick brown fox jumps over the lazy dog. " * 20
    
    pipelines = [
        ("Custom Pipeline", PipelineBuilder("Test")
         .add_rle(min_run_length=2)
         .add_huffman()
         .build()),
        ("Deflate-like", PredefinedPipelines.deflate_like()),
        ("Text Optimized", PredefinedPipelines.text_optimized()),
        ("Fast Compression", PredefinedPipelines.fast_compression())
    ]
    
    for pipeline_name, pipeline in pipelines:
        print_step(f"Testing {pipeline_name}")
        
        try:
            # Show pipeline stages
            stages = pipeline.get_stage_info()
            stage_names = [s['name'] for s in stages if s['enabled']]
            print(f"   Stages: {' → '.join(stage_names)}")
            
            # Compress
            start_time = time.time()
            result = pipeline.compress(test_data)
            compression_time = time.time() - start_time
            
            # Decompress
            start_time = time.time()
            decompressed_data = pipeline.decompress(result.compressed_data, result)
            decompression_time = time.time() - start_time
            
            # Verify
            if decompressed_data == test_data:
                print_success(f"Success: {len(test_data)} → {result.compressed_size} bytes "
                            f"(ratio: {result.total_compression_ratio:.2f}x, "
                            f"time: {compression_time+decompression_time:.3f}s)")
                
                # Show stage breakdown
                for stage in result.stage_results[:3]:  # Show first 3 stages
                    stage_ratio = stage['input_size'] / stage['output_size'] if stage['output_size'] > 0 else 0
                    print(f"     {stage['stage_name']}: {stage['input_size']} → {stage['output_size']} "
                          f"(ratio: {stage_ratio:.2f}x)")
            else:
                print_error("Verification failed")
                
        except Exception as e:
            print_error(f"Pipeline test failed: {str(e)}")


def test_benchmarking():
    """Test benchmarking suite."""
    print_header("Testing Benchmarking Suite")
    
    print_step("Setting up benchmark")
    
    try:
        # Create benchmark configuration
        config = BenchmarkConfig(
            name="Framework Test Benchmark",
            description="Testing benchmark functionality",
            test_runs=2,
            warmup_runs=1,
            parallel_execution=False,
            verify_decompression=True
        )
        
        # Create benchmark
        benchmark = CompressionBenchmark(config)
        
        # Add algorithms
        algorithms = [
            RunLengthEncoder(),
            HuffmanEncoder(),
            LZ77Encoder(window_size=512, lookahead_size=8)
        ]
        
        for algo in algorithms:
            benchmark.add_algorithm(algo)
        
        # Add synthetic dataset
        synthetic_dataset = SyntheticDatasetProvider()
        benchmark.add_dataset(synthetic_dataset, "SyntheticTest")
        
        print_success("Benchmark setup completed")
        
        # Run benchmark
        print_step("Running benchmark (this may take a moment)")
        
        summary = benchmark.run_benchmarks()
        
        # Display results
        print_success(f"Benchmark completed: {summary.successful_tests}/{summary.total_tests} tests passed")
        print(f"   Average compression ratio: {summary.avg_compression_ratio:.2f}x")
        print(f"   Average compression time: {summary.avg_compression_time:.3f}s")
        print(f"   Best compression: {summary.best_compression_ratio:.2f}x ({summary.best_compression_algorithm})")
        print(f"   Fastest algorithm: {summary.fastest_algorithm}")
        
        # Save results
        results_file = Path("test_benchmark_results.csv")
        benchmark.export_results_csv(results_file, summary)
        print_success(f"Results saved to: {results_file}")
        
    except Exception as e:
        print_error(f"Benchmark test failed: {str(e)}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print_header("Testing Edge Cases")
    
    algorithms = [HuffmanEncoder(), RunLengthEncoder(), LZ77Encoder(), LZWEncoder()]
    
    edge_cases = [
        (b"", "Empty data"),
        (b"A" * 1000, "Highly repetitive data"),
        (bytes(range(256)) * 4, "Full byte range"),
        (b"A", "Single byte")
    ]
    
    for algo in algorithms:
        algo_name = algo.name
        print_step(f"Testing {algo_name} edge cases")
        
        success_count = 0
        for test_data, description in edge_cases:
            try:
                result = algo.compress(test_data)
                decompressed = algo.decompress(result.compressed_data, result.metadata)
                
                if decompressed.decompressed_data == test_data:
                    print_success(f"{description}: OK")
                    success_count += 1
                else:
                    print_error(f"{description}: Verification failed")
                    
            except Exception as e:
                print_error(f"{description}: {str(e)}")
        
        print(f"   📊 {algo_name}: {success_count}/{len(edge_cases)} edge cases passed")


def main():
    """Run all tests."""
    print("🚀 Starting Hybrid Compression Study Framework Tests")
    print(f"   Python version: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")
    
    start_time = time.time()
    
    try:
        # Run all test suites
        test_algorithms()
        test_pipelines()
        test_edge_cases()
        test_benchmarking()
        
        total_time = time.time() - start_time
        
        print_header("Test Summary")
        print_success(f"All tests completed in {total_time:.2f} seconds")
        print("🎉 Framework is working correctly!")
        
        # Cleanup
        cleanup_files = ["test_benchmark_results.csv"]
        for file in cleanup_files:
            if Path(file).exists():
                Path(file).unlink()
                print(f"   🧹 Cleaned up: {file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 