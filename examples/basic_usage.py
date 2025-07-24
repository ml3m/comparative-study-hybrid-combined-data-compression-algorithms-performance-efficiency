#!/usr/bin/env python3
"""
Basic usage examples for the Hybrid Compression Study Framework.

This script demonstrates how to use individual algorithms, pipelines,
and benchmarking capabilities.
"""

import sys
import os
from pathlib import Path

# Add src to path for examples
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_compression_study.algorithms.rle import RunLengthEncoder
from hybrid_compression_study.algorithms.huffman import HuffmanEncoder
from hybrid_compression_study.algorithms.lz77 import LZ77Encoder
from hybrid_compression_study.algorithms.lzw import LZWEncoder
from hybrid_compression_study.pipeline.core import PipelineBuilder, PredefinedPipelines
from hybrid_compression_study.benchmarks.suite import (
    CompressionBenchmark, BenchmarkConfig, SyntheticDatasetProvider
)


def example_individual_algorithms():
    """Demonstrate individual algorithm usage."""
    print("🔧 Example 1: Individual Algorithms")
    print("=" * 50)
    
    # Test data
    test_data = b"Hello, world! This is a test message for compression. " * 10
    print(f"Original data: {len(test_data)} bytes")
    
    # Test different algorithms
    algorithms = [
        ("RLE", RunLengthEncoder()),
        ("Huffman", HuffmanEncoder()),
        ("LZ77", LZ77Encoder(window_size=1024)),
        ("LZW", LZWEncoder(max_code_bits=12))
    ]
    
    for name, algorithm in algorithms:
        print(f"\n📦 Testing {name}:")
        
        try:
            # Compress
            result = algorithm.compress(test_data)
            print(f"   Compressed: {result.original_size} → {result.compressed_size} bytes")
            print(f"   Ratio: {result.compression_ratio:.2f}x")
            print(f"   Time: {result.compression_time:.4f}s")
            
            # Decompress and verify
            decompressed = algorithm.decompress(result.compressed_data, result.metadata)
            success = decompressed.decompressed_data == test_data
            print(f"   Verification: {'✅' if success else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def example_compression_pipelines():
    """Demonstrate compression pipeline usage."""
    print("\n\n🔗 Example 2: Compression Pipelines")
    print("=" * 50)
    
    test_data = b"The quick brown fox jumps over the lazy dog. " * 50
    print(f"Original data: {len(test_data)} bytes")
    
    # Custom pipeline
    print(f"\n🏗️  Custom Pipeline (RLE → Huffman):")
    custom_pipeline = (PipelineBuilder("Custom")
                      .add_rle(min_run_length=2)
                      .add_huffman()
                      .build())
    
    try:
        result = custom_pipeline.compress(test_data)
        print(f"   Compressed: {result.original_size} → {result.compressed_size} bytes")
        print(f"   Ratio: {result.total_compression_ratio:.2f}x")
        print(f"   Time: {result.total_compression_time:.4f}s")
        
        # Show stage breakdown
        for stage in result.stage_results:
            stage_ratio = stage['input_size'] / stage['output_size'] if stage['output_size'] > 0 else 0
            print(f"     {stage['stage_name']}: {stage['input_size']} → {stage['output_size']} "
                  f"(ratio: {stage_ratio:.2f}x)")
        
        # Verify decompression
        decompressed = custom_pipeline.decompress(result.compressed_data, result)
        success = decompressed == test_data
        print(f"   Verification: {'✅' if success else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Predefined pipelines
    predefined_pipelines = [
        ("Deflate-like", PredefinedPipelines.deflate_like()),
        ("Text Optimized", PredefinedPipelines.text_optimized()),
        ("Fast Compression", PredefinedPipelines.fast_compression())
    ]
    
    for name, pipeline in predefined_pipelines:
        print(f"\n🔗 {name}:")
        
        try:
            result = pipeline.compress(test_data)
            print(f"   Compressed: {result.original_size} → {result.compressed_size} bytes")
            print(f"   Ratio: {result.total_compression_ratio:.2f}x")
            print(f"   Stages: {len(result.stage_results)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def example_benchmarking():
    """Demonstrate benchmarking usage."""
    print("\n\n📊 Example 3: Benchmarking")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        name="Example Benchmark",
        description="Comparing compression algorithms",
        test_runs=2,
        warmup_runs=1,
        verify_decompression=True
    )
    
    # Create benchmark
    benchmark = CompressionBenchmark(config)
    
    # Add algorithms to test
    algorithms = [
        HuffmanEncoder(),
        LZ77Encoder(window_size=512),
        LZWEncoder(max_code_bits=10)
    ]
    
    for algo in algorithms:
        benchmark.add_algorithm(algo)
    
    # Add synthetic dataset
    synthetic_data = SyntheticDatasetProvider()
    benchmark.add_dataset(synthetic_data, "Synthetic")
    
    try:
        print("Running benchmark (this may take a moment)...")
        summary = benchmark.run_benchmarks()
        
        print(f"\n📈 Benchmark Results:")
        print(f"   Total tests: {summary.total_tests}")
        print(f"   Successful: {summary.successful_tests}")
        print(f"   Failed: {summary.failed_tests}")
        print(f"   Average compression ratio: {summary.avg_compression_ratio:.2f}x")
        print(f"   Best compression: {summary.best_compression_ratio:.2f}x ({summary.best_compression_algorithm})")
        print(f"   Fastest algorithm: {summary.fastest_algorithm}")
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        for result in summary.results[:5]:  # Show first 5 results
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.algorithm_name} on {result.file_name}: "
                  f"{result.compression_ratio:.2f}x in {result.total_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def example_advanced_usage():
    """Demonstrate advanced features."""
    print("\n\n🚀 Example 4: Advanced Features")
    print("=" * 50)
    
    # Algorithm parameters
    print("🔧 Algorithm with custom parameters:")
    rle_custom = RunLengthEncoder(escape_byte=0xFF, min_run_length=4)
    test_data = b"AAAABBBBCCCCCCCCDDDDDDDD"
    
    result = rle_custom.compress(test_data)
    print(f"   Custom RLE: {result.compression_ratio:.2f}x")
    print(f"   Parameters: {rle_custom.get_parameters()}")
    
    # Pipeline validation
    print(f"\n🔍 Pipeline validation:")
    pipeline = PipelineBuilder("Test").add_huffman().add_lz77().build()
    issues = pipeline.validate_pipeline()
    
    if issues:
        print(f"   ⚠️  Validation issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"   ✅ Pipeline validation passed")
    
    # Performance monitoring
    print(f"\n⏱️  Performance monitoring:")
    from hybrid_compression_study.utils.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    huffman = HuffmanEncoder()
    
    with monitor.profile_operation("huffman_test", len(test_data)) as profile:
        result = huffman.compress(test_data)
    
    print(f"   Duration: {profile.duration:.4f}s")
    print(f"   Peak memory: {profile.peak_memory_mb:.2f} MB")
    print(f"   Throughput: {profile.throughput_mbps:.2f} MB/s")


def main():
    """Run all examples."""
    print("🎯 Hybrid Compression Study Framework - Basic Examples")
    print("=" * 60)
    
    try:
        example_individual_algorithms()
        example_compression_pipelines()
        example_benchmarking()
        example_advanced_usage()
        
        print("\n\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Try the CLI: `hcs --help`")
        print("- Run benchmarks: `hcs benchmark --synthetic`")
        print("- Compress files: `hcs compress myfile.txt output`")
        print("- Create pipelines: `hcs pipeline myfile.txt --pipeline deflate-like`")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 