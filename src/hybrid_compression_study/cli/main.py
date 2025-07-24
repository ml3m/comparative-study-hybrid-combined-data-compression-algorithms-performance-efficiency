"""
Main CLI interface for the hybrid compression study framework.

This module provides command-line access to compression algorithms,
pipelines, and benchmarking capabilities.
"""

import click
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import sys
import time

from ..algorithms.rle import RunLengthEncoder, AdaptiveRunLengthEncoder
from ..algorithms.huffman import HuffmanEncoder
from ..algorithms.lz77 import LZ77Encoder, OptimizedLZ77Encoder
from ..algorithms.lzw import LZWEncoder, AdaptiveLZWEncoder
from ..pipeline.core import CompressionPipeline, PipelineBuilder, PredefinedPipelines
from ..benchmarks.suite import (
    CompressionBenchmark, BenchmarkConfig, BenchmarkSuite,
    FileDatasetProvider, SyntheticDatasetProvider
)
from ..core.base import CompressionAlgorithm


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC) -> None:
    """Print colored text to terminal."""
    click.echo(f"{color}{text}{Colors.ENDC}")


def get_algorithm_by_name(name: str, **kwargs) -> CompressionAlgorithm:
    """Get algorithm instance by name."""
    algorithms = {
        'rle': RunLengthEncoder,
        'adaptive-rle': AdaptiveRunLengthEncoder,
        'huffman': HuffmanEncoder,
        'lz77': LZ77Encoder,
        'optimized-lz77': OptimizedLZ77Encoder,
        'lzw': LZWEncoder,
        'adaptive-lzw': AdaptiveLZWEncoder
    }
    
    if name.lower() not in algorithms:
        available = ', '.join(algorithms.keys())
        raise click.BadParameter(f"Unknown algorithm '{name}'. Available: {available}")
    
    try:
        return algorithms[name.lower()](**kwargs)
    except Exception as e:
        raise click.BadParameter(f"Failed to create algorithm '{name}': {str(e)}")


@click.group()
@click.version_option(version="1.0.0", prog_name="Hybrid Compression Study")
def cli():
    """
    🗜️  Hybrid Compression Study Framework
    
    A comprehensive toolkit for analyzing and benchmarking
    compression algorithms and hybrid compression pipelines.
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
@click.option('--algorithm', '-a', default='huffman', 
              help='Compression algorithm to use (rle, huffman, lz77, lzw, etc.)')
@click.option('--parameters', '-p', help='Algorithm parameters as JSON string')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--verify', is_flag=True, default=True, help='Verify decompression')
def compress(input_file: str, output_file: str, algorithm: str, 
             parameters: Optional[str], verbose: bool, verify: bool):
    """
    Compress a file using the specified algorithm.
    
    Example: hcs compress input.txt output.hcs --algorithm huffman
    """
    print_colored("🗜️  Compression Task", Colors.HEADER)
    
    # Parse parameters
    params = {}
    if parameters:
        try:
            params = json.loads(parameters)
        except json.JSONDecodeError as e:
            print_colored(f"❌ Invalid parameters JSON: {e}", Colors.FAIL)
            sys.exit(1)
    
    try:
        # Create algorithm
        if verbose:
            print_colored(f"Creating {algorithm} algorithm with params: {params}", Colors.OKBLUE)
        
        algo = get_algorithm_by_name(algorithm, **params)
        
        # Read input file
        if verbose:
            print_colored(f"Reading input file: {input_file}", Colors.OKBLUE)
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        
        # Compress
        if verbose:
            print_colored("Compressing...", Colors.OKBLUE)
        
        start_time = time.time()
        result = algo.compress(data)
        compression_time = time.time() - start_time
        
        # Write compressed data and metadata
        output_path = Path(output_file)
        compressed_file = output_path.with_suffix('.hcs')
        metadata_file = output_path.with_suffix('.meta.json')
        
        with open(compressed_file, 'wb') as f:
            f.write(result.compressed_data)
        
        # Save metadata
        metadata = {
            'algorithm': algorithm,
            'algorithm_params': params,
            'original_size': original_size,
            'compressed_size': result.compressed_size,
            'compression_ratio': result.compression_ratio,
            'compression_time': compression_time,
            'metadata': result.metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Verify if requested
        if verify:
            if verbose:
                print_colored("Verifying decompression...", Colors.OKBLUE)
            
            decompressed = algo.decompress(result.compressed_data, result.metadata)
            if decompressed.decompressed_data != data:
                print_colored("❌ Verification failed: decompressed data doesn't match", Colors.FAIL)
                sys.exit(1)
            else:
                if verbose:
                    print_colored("✅ Verification passed", Colors.OKGREEN)
        
        # Print results
        print_colored("\n📊 Compression Results:", Colors.HEADER)
        print(f"   Original size:     {original_size:,} bytes")
        print(f"   Compressed size:   {result.compressed_size:,} bytes")
        print(f"   Compression ratio: {result.compression_ratio:.2f}x")
        print(f"   Space saved:       {result.compression_percentage:.1f}%")
        print(f"   Compression time:  {compression_time:.3f} seconds")
        print(f"   Output files:")
        print(f"     Compressed data: {compressed_file}")
        print(f"     Metadata:        {metadata_file}")
        
        print_colored("✅ Compression completed successfully!", Colors.OKGREEN)
        
    except Exception as e:
        print_colored(f"❌ Compression failed: {str(e)}", Colors.FAIL)
        sys.exit(1)


@cli.command()
@click.argument('compressed_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
@click.option('--metadata', '-m', type=click.Path(exists=True, readable=True),
              help='Metadata file (auto-detected if not specified)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def decompress(compressed_file: str, output_file: str, 
               metadata: Optional[str], verbose: bool):
    """
    Decompress a file that was compressed with this framework.
    
    Example: hcs decompress output.hcs original.txt
    """
    print_colored("📂 Decompression Task", Colors.HEADER)
    
    try:
        # Find metadata file if not specified
        if metadata is None:
            compressed_path = Path(compressed_file)
            if compressed_path.suffix == '.hcs':
                metadata_path = compressed_path.with_suffix('.meta.json')
            else:
                metadata_path = compressed_path.with_suffix(compressed_path.suffix + '.meta.json')
            
            if not metadata_path.exists():
                print_colored(f"❌ Metadata file not found: {metadata_path}", Colors.FAIL)
                print_colored("   Use --metadata to specify the metadata file", Colors.WARNING)
                sys.exit(1)
            
            metadata = str(metadata_path)
        
        # Load metadata
        if verbose:
            print_colored(f"Loading metadata from: {metadata}", Colors.OKBLUE)
        
        with open(metadata, 'r') as f:
            meta = json.load(f)
        
        # Load compressed data
        if verbose:
            print_colored(f"Loading compressed data from: {compressed_file}", Colors.OKBLUE)
        
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        # Create algorithm
        algorithm_name = meta['algorithm']
        algorithm_params = meta.get('algorithm_params', {})
        
        if verbose:
            print_colored(f"Creating {algorithm_name} algorithm", Colors.OKBLUE)
        
        algo = get_algorithm_by_name(algorithm_name, **algorithm_params)
        
        # Decompress
        if verbose:
            print_colored("Decompressing...", Colors.OKBLUE)
        
        start_time = time.time()
        result = algo.decompress(compressed_data, meta['metadata'])
        decompression_time = time.time() - start_time
        
        # Write decompressed data
        with open(output_file, 'wb') as f:
            f.write(result.decompressed_data)
        
        # Print results
        print_colored("\n📊 Decompression Results:", Colors.HEADER)
        print(f"   Compressed size:   {len(compressed_data):,} bytes")
        print(f"   Decompressed size: {len(result.decompressed_data):,} bytes")
        print(f"   Decompression time: {decompression_time:.3f} seconds")
        print(f"   Output file:       {output_file}")
        
        print_colored("✅ Decompression completed successfully!", Colors.OKGREEN)
        
    except Exception as e:
        print_colored(f"❌ Decompression failed: {str(e)}", Colors.FAIL)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.option('--pipeline', '-p', default='deflate-like',
              help='Pipeline name (deflate-like, text-optimized, binary-optimized, high-compression, fast-compression)')
@click.option('--custom', '-c', help='Custom pipeline specification as JSON')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--output', '-o', help='Output file for compressed data')
def pipeline(input_file: str, pipeline: str, custom: Optional[str], 
             verbose: bool, output: Optional[str]):
    """
    Compress a file using compression pipelines.
    
    Example: hcs pipeline input.txt --pipeline deflate-like
    """
    print_colored("🔗 Pipeline Compression", Colors.HEADER)
    
    try:
        # Create pipeline
        if custom:
            print_colored("❌ Custom pipeline specification not yet implemented", Colors.FAIL)
            sys.exit(1)
        else:
            # Use predefined pipeline
            pipelines = {
                'deflate-like': PredefinedPipelines.deflate_like,
                'text-optimized': PredefinedPipelines.text_optimized,
                'binary-optimized': PredefinedPipelines.binary_optimized,
                'high-compression': PredefinedPipelines.high_compression,
                'fast-compression': PredefinedPipelines.fast_compression
            }
            
            if pipeline not in pipelines:
                available = ', '.join(pipelines.keys())
                print_colored(f"❌ Unknown pipeline '{pipeline}'. Available: {available}", Colors.FAIL)
                sys.exit(1)
            
            comp_pipeline = pipelines[pipeline]()
        
        if verbose:
            print_colored(f"Created pipeline: {comp_pipeline}", Colors.OKBLUE)
            stages = comp_pipeline.get_stage_info()
            for stage in stages:
                print(f"   Stage {stage['index']}: {stage['name']} ({stage['component_type']})")
        
        # Read input file
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # Compress through pipeline
        if verbose:
            print_colored("Running compression pipeline...", Colors.OKBLUE)
        
        result = comp_pipeline.compress(data)
        
        # Decompress for verification
        if verbose:
            print_colored("Verifying with decompression...", Colors.OKBLUE)
        
        decompressed_data = comp_pipeline.decompress(result.compressed_data, result)
        
        if decompressed_data != data:
            print_colored("❌ Pipeline verification failed", Colors.FAIL)
            sys.exit(1)
        
        # Print results
        print_colored("\n📊 Pipeline Results:", Colors.HEADER)
        print(f"   Pipeline:          {result.pipeline_name}")
        print(f"   Stages:            {len(result.stage_results)}")
        print(f"   Original size:     {result.original_size:,} bytes")
        print(f"   Compressed size:   {result.compressed_size:,} bytes")
        print(f"   Compression ratio: {result.total_compression_ratio:.2f}x")
        print(f"   Space saved:       {result.compression_percentage:.1f}%")
        print(f"   Compression time:  {result.total_compression_time:.3f} seconds")
        print(f"   Decompression time: {result.total_decompression_time:.3f} seconds")
        
        if verbose:
            print_colored("\n📋 Stage Details:", Colors.HEADER)
            for stage in result.stage_results:
                ratio = stage['input_size'] / stage['output_size'] if stage['output_size'] > 0 else 0
                print(f"   {stage['stage_name']}: {stage['input_size']:,} → {stage['output_size']:,} bytes (ratio: {ratio:.2f}x)")
        
        # Save output if requested
        if output:
            output_path = Path(output)
            compressed_file = output_path.with_suffix('.hcp')  # Hybrid Compression Pipeline
            metadata_file = output_path.with_suffix('.pipeline.json')
            
            with open(compressed_file, 'wb') as f:
                f.write(result.compressed_data)
            
            pipeline_metadata = {
                'pipeline_name': result.pipeline_name,
                'original_size': result.original_size,
                'compressed_size': result.compressed_size,
                'compression_ratio': result.total_compression_ratio,
                'metadata': result.metadata,
                'stage_results': result.stage_results
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(pipeline_metadata, f, indent=2)
            
            print(f"   Saved to:          {compressed_file}")
            print(f"   Metadata:          {metadata_file}")
        
        print_colored("✅ Pipeline compression completed successfully!", Colors.OKGREEN)
        
    except Exception as e:
        print_colored(f"❌ Pipeline compression failed: {str(e)}", Colors.FAIL)
        sys.exit(1)


@cli.command()
@click.option('--algorithms', '-a', multiple=True, default=['huffman', 'lz77', 'lzw'],
              help='Algorithms to benchmark (can be specified multiple times)')
@click.option('--data-dir', '-d', type=click.Path(exists=True, file_okay=False),
              help='Directory containing test files')
@click.option('--synthetic', '-s', is_flag=True, help='Include synthetic datasets')
@click.option('--output', '-o', type=click.Path(), default='benchmark_results.csv',
              help='Output file for results')
@click.option('--runs', '-r', default=3, help='Number of test runs per algorithm')
@click.option('--parallel', is_flag=True, help='Enable parallel execution')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def benchmark(algorithms: List[str], data_dir: Optional[str], synthetic: bool,
              output: str, runs: int, parallel: bool, verbose: bool):
    """
    Run comprehensive benchmarks on compression algorithms.
    
    Example: hcs benchmark -a huffman -a lz77 --data-dir ./test_files
    """
    print_colored("📊 Compression Benchmark Suite", Colors.HEADER)
    
    try:
        # Create benchmark configuration
        config = BenchmarkConfig(
            name="CLI Benchmark",
            description=f"Benchmarking {len(algorithms)} algorithms",
            test_runs=runs,
            parallel_execution=parallel,
            verify_decompression=True
        )
        
        # Create benchmark
        benchmark_obj = CompressionBenchmark(config)
        
        # Add algorithms
        for algo_name in algorithms:
            if verbose:
                print_colored(f"Adding algorithm: {algo_name}", Colors.OKBLUE)
            
            try:
                algo = get_algorithm_by_name(algo_name)
                benchmark_obj.add_algorithm(algo)
            except Exception as e:
                print_colored(f"⚠️  Skipping {algo_name}: {str(e)}", Colors.WARNING)
        
        # Add datasets
        if data_dir:
            if verbose:
                print_colored(f"Adding file dataset: {data_dir}", Colors.OKBLUE)
            
            file_dataset = FileDatasetProvider(Path(data_dir))
            benchmark_obj.add_dataset(file_dataset, "FileDataset")
        
        if synthetic:
            if verbose:
                print_colored("Adding synthetic datasets", Colors.OKBLUE)
            
            synthetic_dataset = SyntheticDatasetProvider()
            benchmark_obj.add_dataset(synthetic_dataset, "SyntheticDataset")
        
        if not data_dir and not synthetic:
            print_colored("⚠️  No datasets specified, adding synthetic datasets", Colors.WARNING)
            synthetic_dataset = SyntheticDatasetProvider()
            benchmark_obj.add_dataset(synthetic_dataset, "SyntheticDataset")
        
        # Run benchmark
        if verbose:
            print_colored("Starting benchmark execution...", Colors.OKBLUE)
        
        summary = benchmark_obj.run_benchmarks()
        
        # Export results
        benchmark_obj.export_results_csv(Path(output), summary)
        
        # Print summary
        print_colored("\n📈 Benchmark Summary:", Colors.HEADER)
        print(f"   Total tests:       {summary.total_tests}")
        print(f"   Successful:        {summary.successful_tests}")
        print(f"   Failed:            {summary.failed_tests}")
        print(f"   Avg compression:   {summary.avg_compression_ratio:.2f}x")
        print(f"   Avg comp. time:    {summary.avg_compression_time:.3f}s")
        print(f"   Avg decomp. time:  {summary.avg_decompression_time:.3f}s")
        print(f"   Best compression:  {summary.best_compression_ratio:.2f}x ({summary.best_compression_algorithm})")
        print(f"   Fastest algorithm: {summary.fastest_algorithm}")
        
        print_colored("✅ Benchmark completed successfully!", Colors.OKGREEN)
        
    except Exception as e:
        print_colored(f"❌ Benchmark failed: {str(e)}", Colors.FAIL)
        sys.exit(1)


@cli.command()
def list_algorithms():
    """List all available compression algorithms."""
    print_colored("🔧 Available Compression Algorithms", Colors.HEADER)
    
    algorithms = [
        ("rle", "Run-Length Encoding", "Basic RLE with configurable parameters"),
        ("adaptive-rle", "Adaptive RLE", "RLE with automatic parameter optimization"),
        ("huffman", "Huffman Coding", "Optimal entropy coding based on frequencies"),
        ("lz77", "LZ77", "Sliding window dictionary compression"),
        ("optimized-lz77", "Optimized LZ77", "LZ77 with hash table optimization"),
        ("lzw", "LZW", "Lempel-Ziv-Welch dictionary compression"),
        ("adaptive-lzw", "Adaptive LZW", "LZW with dictionary reset capability")
    ]
    
    for name, title, description in algorithms:
        print_colored(f"\n📦 {name}", Colors.OKGREEN)
        print(f"   {title}")
        print(f"   {description}")


@cli.command()
def list_pipelines():
    """List all available compression pipelines."""
    print_colored("🔗 Available Compression Pipelines", Colors.HEADER)
    
    pipelines = [
        ("deflate-like", "Deflate-like Pipeline", "LZ77 + Huffman (similar to gzip)"),
        ("text-optimized", "Text Optimized", "RLE + LZW + Huffman for text data"),
        ("binary-optimized", "Binary Optimized", "LZ77 + Huffman for binary data"),
        ("high-compression", "High Compression", "LZ77 + LZW + Huffman for maximum compression"),
        ("fast-compression", "Fast Compression", "RLE + Huffman for speed")
    ]
    
    for name, title, description in pipelines:
        print_colored(f"\n🔗 {name}", Colors.OKGREEN)
        print(f"   {title}")
        print(f"   {description}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Operation cancelled by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ Unexpected error: {str(e)}", Colors.FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main() 