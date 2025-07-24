"""
Comprehensive benchmarking suite for compression algorithms and pipelines.

This module provides tools for systematically testing and comparing
compression algorithms across different data types and conditions.
"""

import time
import gc
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd

from ..core.base import CompressionAlgorithm, DatasetProvider
from ..pipeline.core import CompressionPipeline, PipelineResult
from ..utils.performance import PerformanceMonitor, get_system_info, calculate_compression_metrics


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    description: str = ""
    warmup_runs: int = 1
    test_runs: int = 3
    timeout_seconds: int = 300
    memory_limit_mb: int = 1024
    enable_gc: bool = True
    parallel_execution: bool = False
    max_workers: int = 4
    save_compressed_data: bool = False
    verify_decompression: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    algorithm_name: str
    dataset_name: str
    file_name: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    total_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_mbps: float
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def compression_percentage(self) -> float:
        """Calculate compression percentage (space saved)."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100
    
    @property
    def bits_per_byte(self) -> float:
        """Calculate bits per byte in compressed data."""
        if self.original_size == 0:
            return 0.0
        return (self.compressed_size * 8) / self.original_size


@dataclass
class BenchmarkSummary:
    """Summary statistics for a set of benchmark results."""
    config_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_compression_ratio: float
    avg_compression_time: float
    avg_decompression_time: float
    avg_memory_usage_mb: float
    avg_throughput_mbps: float
    best_compression_ratio: float
    best_compression_algorithm: str
    fastest_algorithm: str
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)


class FileDatasetProvider(DatasetProvider):
    """Dataset provider that loads files from a directory."""
    
    def __init__(self, directory: Path, file_patterns: List[str] = None):
        self.directory = Path(directory)
        self.file_patterns = file_patterns or ["*"]
        
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
    
    def get_files(self) -> List[Path]:
        """Get list of files matching the patterns."""
        files = []
        for pattern in self.file_patterns:
            files.extend(self.directory.glob(pattern))
        
        # Filter to only include files (not directories)
        return [f for f in files if f.is_file()]
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about a specific file."""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'extension': file_path.suffix,
            'path': str(file_path),
            'modified': stat.st_mtime
        }
    
    def load_file(self, file_path: Path) -> bytes:
        """Load file content as bytes."""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to load file {file_path}: {str(e)}")


class SyntheticDatasetProvider(DatasetProvider):
    """Dataset provider that generates synthetic test data."""
    
    def __init__(self):
        self.datasets = {
            'random': self._generate_random_data,
            'repetitive': self._generate_repetitive_data,
            'text_like': self._generate_text_like_data,
            'structured': self._generate_structured_data,
            'sparse': self._generate_sparse_data
        }
    
    def get_files(self) -> List[Path]:
        """Return synthetic dataset names as Path objects."""
        return [Path(f"synthetic_{name}.bin") for name in self.datasets.keys()]
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about synthetic data."""
        name = file_path.stem.replace('synthetic_', '')
        size = 10240  # Default 10KB
        return {
            'name': file_path.name,
            'size': size,
            'extension': '.bin',
            'path': str(file_path),
            'type': 'synthetic',
            'data_type': name
        }
    
    def load_file(self, file_path: Path) -> bytes:
        """Generate and return synthetic data."""
        name = file_path.stem.replace('synthetic_', '')
        if name in self.datasets:
            return self.datasets[name](10240)  # 10KB default
        raise ValueError(f"Unknown synthetic dataset: {name}")
    
    def _generate_random_data(self, size: int) -> bytes:
        """Generate random data."""
        import random
        return bytes(random.randint(0, 255) for _ in range(size))
    
    def _generate_repetitive_data(self, size: int) -> bytes:
        """Generate data with many repetitions."""
        pattern = b"ABCDEFGH" * 10
        return (pattern * (size // len(pattern) + 1))[:size]
    
    def _generate_text_like_data(self, size: int) -> bytes:
        """Generate text-like data with realistic letter frequencies."""
        import random
        # English letter frequencies (approximate)
        letters = "etaoinshrdlcumwfgypbvkjxqz"
        weights = [12.0, 9.1, 8.1, 7.5, 7.0, 6.9, 6.3, 6.1, 5.9, 4.3, 4.0, 3.4, 2.8, 2.4, 2.4, 2.1, 1.9, 1.5, 0.95, 0.15, 0.074, 0.019, 0.015, 0.0074]
        
        text = []
        for _ in range(size):
            if random.random() < 0.15:  # Space probability
                text.append(' ')
            else:
                char = random.choices(letters, weights=weights)[0]
                text.append(char.upper() if random.random() < 0.1 else char)
        
        return ''.join(text).encode('utf-8')[:size]
    
    def _generate_structured_data(self, size: int) -> bytes:
        """Generate structured data (JSON-like)."""
        import json
        import random
        
        data = []
        while len(json.dumps(data).encode()) < size:
            entry = {
                'id': random.randint(1, 10000),
                'name': f"Item{random.randint(1, 1000)}",
                'value': round(random.uniform(0, 100), 2),
                'active': random.choice([True, False])
            }
            data.append(entry)
        
        return json.dumps(data).encode('utf-8')[:size]
    
    def _generate_sparse_data(self, size: int) -> bytes:
        """Generate sparse data (mostly zeros)."""
        import random
        data = bytearray(size)
        # Add some non-zero values randomly
        for _ in range(size // 50):  # 2% non-zero
            pos = random.randint(0, size - 1)
            data[pos] = random.randint(1, 255)
        return bytes(data)


class CompressionBenchmark:
    """
    Main benchmarking class for compression algorithms and pipelines.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        self.results: List[BenchmarkResult] = []
        
    def add_algorithm(self, algorithm: CompressionAlgorithm) -> 'CompressionBenchmark':
        """Add an algorithm to be benchmarked."""
        if not hasattr(self, '_algorithms'):
            self._algorithms = []
        self._algorithms.append(algorithm)
        return self
    
    def add_pipeline(self, pipeline: CompressionPipeline) -> 'CompressionBenchmark':
        """Add a pipeline to be benchmarked."""
        if not hasattr(self, '_pipelines'):
            self._pipelines = []
        self._pipelines.append(pipeline)
        return self
    
    def add_dataset(self, dataset: DatasetProvider, name: str = None) -> 'CompressionBenchmark':
        """Add a dataset to be tested."""
        if not hasattr(self, '_datasets'):
            self._datasets = []
        
        dataset_name = name or f"Dataset{len(self._datasets)}"
        self._datasets.append((dataset_name, dataset))
        return self
    
    def run_benchmarks(self) -> BenchmarkSummary:
        """
        Run all configured benchmarks.
        
        Returns:
            BenchmarkSummary with results and statistics
        """
        print(f"🚀 Starting benchmark: {self.config.name}")
        print(f"   Description: {self.config.description}")
        
        if self.config.enable_gc:
            gc.collect()
        
        # Collect all test cases
        test_cases = self._generate_test_cases()
        
        print(f"   Total test cases: {len(test_cases)}")
        
        # Run tests
        if self.config.parallel_execution and len(test_cases) > 1:
            results = self._run_parallel_tests(test_cases)
        else:
            results = self._run_sequential_tests(test_cases)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        print(f"✅ Benchmark completed: {summary.successful_tests}/{summary.total_tests} tests passed")
        
        return summary
    
    def _generate_test_cases(self) -> List[tuple]:
        """Generate all test cases to run."""
        test_cases = []
        
        # Add algorithm test cases
        if hasattr(self, '_algorithms'):
            for algorithm in self._algorithms:
                for dataset_name, dataset in getattr(self, '_datasets', []):
                    for file_path in dataset.get_files():
                        test_cases.append(('algorithm', algorithm, dataset_name, dataset, file_path))
        
        # Add pipeline test cases
        if hasattr(self, '_pipelines'):
            for pipeline in self._pipelines:
                for dataset_name, dataset in getattr(self, '_datasets', []):
                    for file_path in dataset.get_files():
                        test_cases.append(('pipeline', pipeline, dataset_name, dataset, file_path))
        
        return test_cases
    
    def _run_sequential_tests(self, test_cases: List[tuple]) -> List[BenchmarkResult]:
        """Run tests sequentially."""
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"   [{i}/{len(test_cases)}] Running {test_case[0]}: {test_case[1].name} on {test_case[4].name}")
            
            try:
                result = self._run_single_test(*test_case)
                results.append(result)
                
                if result.success:
                    print(f"      ✅ Ratio: {result.compression_ratio:.2f}, Time: {result.total_time:.3f}s")
                else:
                    print(f"      ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                # Create failed result
                results.append(BenchmarkResult(
                    algorithm_name=test_case[1].name,
                    dataset_name=test_case[2],
                    file_name=test_case[4].name,
                    original_size=0,
                    compressed_size=0,
                    compression_ratio=0.0,
                    compression_time=0.0,
                    decompression_time=0.0,
                    total_time=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    throughput_mbps=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _run_parallel_tests(self, test_cases: List[tuple]) -> List[BenchmarkResult]:
        """Run tests in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_test = {
                executor.submit(self._run_single_test, *test_case): test_case 
                for test_case in test_cases
            }
            
            # Collect results
            for future in future_to_test:
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    test_case = future_to_test[future]
                    results.append(BenchmarkResult(
                        algorithm_name=test_case[1].name,
                        dataset_name=test_case[2],
                        file_name=test_case[4].name,
                        original_size=0,
                        compressed_size=0,
                        compression_ratio=0.0,
                        compression_time=0.0,
                        decompression_time=0.0,
                        total_time=0.0,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        throughput_mbps=0.0,
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _run_single_test(self, test_type: str, component: Union[CompressionAlgorithm, CompressionPipeline],
                        dataset_name: str, dataset: DatasetProvider, file_path: Path) -> BenchmarkResult:
        """Run a single benchmark test."""
        
        # Load data
        try:
            data = dataset.load_file(file_path)
            if len(data) == 0:
                raise ValueError("Empty file")
        except Exception as e:
            return BenchmarkResult(
                algorithm_name=component.name,
                dataset_name=dataset_name,
                file_name=file_path.name,
                original_size=0,
                compressed_size=0,
                compression_ratio=0.0,
                compression_time=0.0,
                decompression_time=0.0,
                total_time=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                throughput_mbps=0.0,
                success=False,
                error_message=f"Failed to load data: {str(e)}"
            )
        
        # Run warmup if configured
        for _ in range(self.config.warmup_runs):
            try:
                if test_type == 'algorithm':
                    component.compress(data[:1000])  # Small warmup
                else:  # pipeline
                    component.compress(data[:1000])
            except:
                pass  # Ignore warmup failures
        
        # Force garbage collection
        if self.config.enable_gc:
            gc.collect()
        
        # Run actual test multiple times and average
        compression_times = []
        decompression_times = []
        memory_usages = []
        compressed_sizes = []
        success_count = 0
        last_error = ""
        
        for run in range(self.config.test_runs):
            try:
                # Run compression test
                with self.performance_monitor.profile_operation(f"{component.name}_compress", len(data)) as profile:
                    if test_type == 'algorithm':
                        result = component.compress(data)
                        compressed_data = result.compressed_data
                        metadata = result.metadata
                    else:  # pipeline
                        result = component.compress(data)
                        compressed_data = result.compressed_data
                        metadata = result.metadata
                
                # Verify decompression if enabled
                decompression_time = 0.0
                if self.config.verify_decompression:
                    decomp_start = time.perf_counter()
                    
                    if test_type == 'algorithm':
                        decompressed_result = component.decompress(compressed_data, metadata)
                        decompressed_data = decompressed_result.decompressed_data
                    else:  # pipeline
                        decompressed_data = component.decompress(compressed_data, result)
                    
                    decompression_time = time.perf_counter() - decomp_start
                    
                    # Verify correctness
                    if decompressed_data != data:
                        raise ValueError("Decompression verification failed")
                
                # Record metrics
                compression_times.append(profile.duration)
                decompression_times.append(decompression_time)
                memory_usages.append(profile.peak_memory_mb)
                compressed_sizes.append(len(compressed_data))
                success_count += 1
                
            except Exception as e:
                last_error = str(e)
                continue
        
        # Calculate averages
        if success_count > 0:
            avg_compression_time = sum(compression_times) / len(compression_times)
            avg_decompression_time = sum(decompression_times) / len(decompression_times) 
            avg_memory_usage = sum(memory_usages) / len(memory_usages)
            avg_compressed_size = sum(compressed_sizes) / len(compressed_sizes)
            
            compression_ratio = len(data) / avg_compressed_size if avg_compressed_size > 0 else 0.0
            total_time = avg_compression_time + avg_decompression_time
            throughput = (len(data) / 1024 / 1024) / total_time if total_time > 0 else 0.0
            
            return BenchmarkResult(
                algorithm_name=component.name,
                dataset_name=dataset_name,
                file_name=file_path.name,
                original_size=len(data),
                compressed_size=int(avg_compressed_size),
                compression_ratio=compression_ratio,
                compression_time=avg_compression_time,
                decompression_time=avg_decompression_time,
                total_time=total_time,
                memory_usage_mb=avg_memory_usage,
                cpu_usage_percent=0.0,  # Would need more complex monitoring
                throughput_mbps=throughput,
                success=True,
                metadata={'test_runs': success_count, 'total_runs': self.config.test_runs}
            )
        else:
            return BenchmarkResult(
                algorithm_name=component.name,
                dataset_name=dataset_name,
                file_name=file_path.name,
                original_size=len(data),
                compressed_size=0,
                compression_ratio=0.0,
                compression_time=0.0,
                decompression_time=0.0,
                total_time=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                throughput_mbps=0.0,
                success=False,
                error_message=last_error or "All test runs failed"
            )
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics from results."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return BenchmarkSummary(
                config_name=self.config.name,
                total_tests=len(results),
                successful_tests=0,
                failed_tests=len(results),
                avg_compression_ratio=0.0,
                avg_compression_time=0.0,
                avg_decompression_time=0.0,
                avg_memory_usage_mb=0.0,
                avg_throughput_mbps=0.0,
                best_compression_ratio=0.0,
                best_compression_algorithm="None",
                fastest_algorithm="None",
                results=results,
                system_info=get_system_info()
            )
        
        # Calculate averages
        avg_compression_ratio = sum(r.compression_ratio for r in successful_results) / len(successful_results)
        avg_compression_time = sum(r.compression_time for r in successful_results) / len(successful_results)
        avg_decompression_time = sum(r.decompression_time for r in successful_results) / len(successful_results)
        avg_memory_usage = sum(r.memory_usage_mb for r in successful_results) / len(successful_results)
        avg_throughput = sum(r.throughput_mbps for r in successful_results) / len(successful_results)
        
        # Find best performers
        best_compression = max(successful_results, key=lambda r: r.compression_ratio)
        fastest = min(successful_results, key=lambda r: r.total_time)
        
        return BenchmarkSummary(
            config_name=self.config.name,
            total_tests=len(results),
            successful_tests=len(successful_results),
            failed_tests=len(results) - len(successful_results),
            avg_compression_ratio=avg_compression_ratio,
            avg_compression_time=avg_compression_time,
            avg_decompression_time=avg_decompression_time,
            avg_memory_usage_mb=avg_memory_usage,
            avg_throughput_mbps=avg_throughput,
            best_compression_ratio=best_compression.compression_ratio,
            best_compression_algorithm=best_compression.algorithm_name,
            fastest_algorithm=fastest.algorithm_name,
            results=results,
            system_info=get_system_info()
        )
    
    def export_results_csv(self, filepath: Path, summary: BenchmarkSummary) -> None:
        """Export results to CSV file."""
        # Convert results to DataFrame
        data = []
        for result in summary.results:
            data.append({
                'Algorithm': result.algorithm_name,
                'Dataset': result.dataset_name,
                'File': result.file_name,
                'Original_Size_KB': result.original_size / 1024,
                'Compressed_Size_KB': result.compressed_size / 1024,
                'Compression_Ratio': result.compression_ratio,
                'Compression_Percentage': result.compression_percentage,
                'Compression_Time_s': result.compression_time,
                'Decompression_Time_s': result.decompression_time,
                'Total_Time_s': result.total_time,
                'Memory_Usage_MB': result.memory_usage_mb,
                'Throughput_MBps': result.throughput_mbps,
                'Bits_Per_Byte': result.bits_per_byte,
                'Success': result.success,
                'Error': result.error_message
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"📊 Results exported to: {filepath}")


class BenchmarkSuite:
    """
    High-level benchmark suite that manages multiple benchmark configurations.
    """
    
    def __init__(self, name: str = "CompressionBenchmarkSuite"):
        self.name = name
        self.benchmarks: List[CompressionBenchmark] = []
        
    def add_benchmark(self, benchmark: CompressionBenchmark) -> 'BenchmarkSuite':
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
        return self
    
    def run_all(self) -> List[BenchmarkSummary]:
        """Run all benchmarks in the suite."""
        print(f"🏁 Starting benchmark suite: {self.name}")
        print(f"   Total benchmarks: {len(self.benchmarks)}")
        
        summaries = []
        for i, benchmark in enumerate(self.benchmarks, 1):
            print(f"\n📋 Running benchmark {i}/{len(self.benchmarks)}")
            summary = benchmark.run_benchmarks()
            summaries.append(summary)
        
        print(f"\n🎉 Benchmark suite completed!")
        return summaries 