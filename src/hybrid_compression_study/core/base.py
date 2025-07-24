"""
Enhanced core base classes and interfaces for aerospace-grade compression framework.

This module defines the foundational components with mission-critical precision
suitable for space applications where every byte and nanosecond matters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import time


class CompressionType(Enum):
    """Types of compression algorithms."""
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class AlgorithmCategory(Enum):
    """Categories of compression algorithms for analysis."""
    ENTROPY_CODING = "entropy_coding"      # Huffman, Arithmetic
    DICTIONARY = "dictionary"              # LZ77, LZ78, LZW
    STATISTICAL = "statistical"           # PPM, Context modeling
    TRANSFORM = "transform"                # BWT, DCT, Wavelet
    PREDICTIVE = "predictive"             # Delta, Linear prediction
    RUN_LENGTH = "run_length"             # RLE variants
    HYBRID = "hybrid"                     # Combined approaches


@dataclass
class AerospacePrecisionMetrics:
    """Aerospace-grade precision metrics for mission-critical applications."""
    
    # Nanosecond-precision timing
    compression_time_ns: int = 0
    decompression_time_ns: int = 0
    total_time_ns: int = 0
    
    # Formatted time strings for readability
    compression_time_formatted: str = "0ns"
    decompression_time_formatted: str = "0ns"
    total_time_formatted: str = "0ns"
    
    # Detailed memory metrics (byte precision)
    memory_peak_bytes: int = 0
    memory_delta_bytes: int = 0
    memory_before_bytes: int = 0
    memory_after_bytes: int = 0
    tracemalloc_peak_mb: float = 0.0
    tracemalloc_diff_mb: float = 0.0
    
    # CPU utilization metrics
    cpu_percent_avg: float = 0.0
    cpu_percent_peak: float = 0.0
    cpu_time_user_s: float = 0.0
    cpu_time_system_s: float = 0.0
    cpu_freq_avg_mhz: float = 0.0
    
    # System resource metrics
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_ops: int = 0
    io_write_ops: int = 0
    page_faults: int = 0
    context_switches: int = 0
    gc_collections: List[int] = field(default_factory=list)
    threads_count: int = 0
    file_descriptors: int = 0
    
    # Performance efficiency metrics
    throughput_mbps: float = 0.0
    throughput_bytes_per_second: float = 0.0
    time_per_byte_ns: float = 0.0
    bytes_per_cpu_cycle: float = 0.0
    memory_efficiency_ratio: float = 0.0
    
    # Mission-critical compression metrics
    bits_per_byte: float = 0.0
    entropy_efficiency: float = 0.0  # Ratio to theoretical maximum
    energy_efficiency_bytes_per_ns: float = 0.0
    worst_case_latency_ns: int = 0
    determinism_score: float = 1.0  # 1.0 = perfectly deterministic
    
    # Resource overhead metrics
    memory_overhead_ratio: float = 0.0  # peak_memory / data_size
    cpu_efficiency_bytes_per_cpu_second: float = 0.0
    io_efficiency_ratio: float = 0.0
    
    def update_formatted_times(self):
        """Update formatted time strings based on nanosecond values."""
        from ..utils.performance import format_time_precision
        self.compression_time_formatted = format_time_precision(self.compression_time_ns)
        self.decompression_time_formatted = format_time_precision(self.decompression_time_ns)
        self.total_time_formatted = format_time_precision(self.total_time_ns)


@dataclass
class CompressionResult:
    """Enhanced result from compression operation with aerospace-grade metrics."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float  # Legacy compatibility (seconds)
    algorithm_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced aerospace metrics
    precision_metrics: AerospacePrecisionMetrics = field(default_factory=AerospacePrecisionMetrics)
    
    @property
    def compression_percentage(self) -> float:
        """Calculate compression percentage (space saved)."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100
    
    @property
    def space_savings_bytes(self) -> int:
        """Calculate absolute space savings in bytes."""
        return max(0, self.original_size - self.compressed_size)
    
    @property  
    def is_effective(self) -> bool:
        """Check if compression was effective (ratio > 1.0)."""
        return self.compression_ratio > 1.0
    
    def to_mission_report(self) -> Dict[str, Any]:
        """Generate a comprehensive mission-critical report."""
        return {
            'algorithm': self.algorithm_name,
            'data_integrity': {
                'original_size_bytes': self.original_size,
                'compressed_size_bytes': self.compressed_size,
                'compression_ratio': f"{self.compression_ratio:.6f}x",
                'space_savings_bytes': self.space_savings_bytes,
                'space_savings_percent': f"{self.compression_percentage:.3f}%",
                'effectiveness': 'POSITIVE' if self.is_effective else 'NEGATIVE'
            },
            'performance_profile': {
                'compression_time': self.precision_metrics.compression_time_formatted,
                'compression_time_ns': self.precision_metrics.compression_time_ns,
                'throughput_mbps': f"{self.precision_metrics.throughput_mbps:.6f}",
                'time_per_byte': f"{self.precision_metrics.time_per_byte_ns:.2f}ns",
                'cpu_efficiency': f"{self.precision_metrics.cpu_efficiency_bytes_per_cpu_second:.2f} bytes/cpu-sec"
            },
            'resource_utilization': {
                'peak_memory': f"{self.precision_metrics.memory_peak_bytes:,} bytes",
                'memory_overhead_ratio': f"{self.precision_metrics.memory_overhead_ratio:.4f}",
                'cpu_peak_percent': f"{self.precision_metrics.cpu_percent_peak:.2f}%",
                'io_operations': f"R:{self.precision_metrics.io_read_ops} W:{self.precision_metrics.io_write_ops}",
                'determinism_score': f"{self.precision_metrics.determinism_score:.6f}"
            },
            'mission_readiness': {
                'worst_case_latency': self.precision_metrics.worst_case_latency_ns,
                'energy_efficiency': f"{self.precision_metrics.energy_efficiency_bytes_per_ns:.2e} bytes/ns",
                'entropy_efficiency': f"{self.precision_metrics.entropy_efficiency:.4f}",
                'suitable_for_realtime': self.precision_metrics.worst_case_latency_ns < 1_000_000_000,  # < 1 second
                'memory_constrained_safe': self.precision_metrics.memory_overhead_ratio < 2.0
            }
        }


@dataclass
class DecompressionResult:
    """Enhanced result from decompression operation with aerospace-grade metrics."""
    decompressed_data: bytes
    original_compressed_size: int
    decompressed_size: int
    decompression_time: float  # Legacy compatibility (seconds)
    algorithm_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced aerospace metrics
    precision_metrics: AerospacePrecisionMetrics = field(default_factory=AerospacePrecisionMetrics)
    
    @property
    def expansion_ratio(self) -> float:
        """Calculate expansion ratio during decompression."""
        if self.original_compressed_size == 0:
            return float('inf')
        return self.decompressed_size / self.original_compressed_size
    
    def verify_integrity(self, original_data: bytes) -> bool:
        """Verify data integrity after decompression."""
        return self.decompressed_data == original_data
    
    def to_mission_report(self) -> Dict[str, Any]:
        """Generate a comprehensive decompression mission report."""
        return {
            'algorithm': self.algorithm_name,
            'decompression_profile': {
                'compressed_size_bytes': self.original_compressed_size,
                'decompressed_size_bytes': self.decompressed_size,
                'expansion_ratio': f"{self.expansion_ratio:.6f}x",
                'decompression_time': self.precision_metrics.decompression_time_formatted,
                'throughput_mbps': f"{self.precision_metrics.throughput_mbps:.6f}"
            },
            'resource_utilization': {
                'peak_memory': f"{self.precision_metrics.memory_peak_bytes:,} bytes", 
                'cpu_utilization': f"{self.precision_metrics.cpu_percent_avg:.2f}%",
                'determinism_score': f"{self.precision_metrics.determinism_score:.6f}"
            }
        }


@dataclass 
class PerformanceMetrics:
    """Comprehensive performance metrics for aerospace applications."""
    compression_time_ns: int
    decompression_time_ns: int
    peak_memory_bytes: int
    avg_memory_bytes: int
    cpu_utilization_percent: float
    throughput_mbps: float
    energy_efficiency: float
    determinism_score: float
    
    # Mission-critical thresholds
    REALTIME_THRESHOLD_NS = 1_000_000_000  # 1 second
    MEMORY_EFFICIENCY_THRESHOLD = 2.0      # 2x data size max
    CPU_EFFICIENCY_THRESHOLD = 80.0        # 80% max CPU
    
    @property
    def is_realtime_suitable(self) -> bool:
        """Check if performance is suitable for real-time applications."""
        total_time = self.compression_time_ns + self.decompression_time_ns
        return total_time < self.REALTIME_THRESHOLD_NS
    
    @property
    def is_memory_efficient(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        return self.peak_memory_bytes < (self.avg_memory_bytes * self.MEMORY_EFFICIENCY_THRESHOLD)
    
    @property
    def mission_readiness_score(self) -> float:
        """Calculate overall mission readiness score (0.0 - 1.0)."""
        scores = []
        
        # Time efficiency (lower is better)
        time_score = max(0, 1.0 - (self.compression_time_ns + self.decompression_time_ns) / self.REALTIME_THRESHOLD_NS)
        scores.append(time_score)
        
        # Memory efficiency
        memory_score = 1.0 if self.is_memory_efficient else 0.5
        scores.append(memory_score)
        
        # CPU efficiency
        cpu_score = max(0, 1.0 - self.cpu_utilization_percent / 100.0)
        scores.append(cpu_score)
        
        # Determinism (higher is better)
        scores.append(self.determinism_score)
        
        return sum(scores) / len(scores)


class CompressionAlgorithm(ABC):
    """Enhanced abstract base class for compression algorithms with aerospace-grade monitoring."""
    
    def __init__(self, name: str, category: AlgorithmCategory, compression_type: CompressionType = CompressionType.LOSSLESS):
        self.name = name
        self.category = category
        self.compression_type = compression_type
        self._parameters: Dict[str, Any] = {}
        
        # Initialize aerospace-grade monitor
        from ..utils.performance import AerospaceGradeMonitor
        self._monitor = AerospaceGradeMonitor(sampling_interval_ms=0.1)  # 100μs precision
    
    @abstractmethod
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress data with aerospace-grade performance monitoring.
        
        Args:
            data: Raw bytes to compress
            
        Returns:
            CompressionResult with detailed precision metrics
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> DecompressionResult:
        """
        Decompress data with aerospace-grade performance monitoring.
        
        Args:
            compressed_data: Compressed bytes
            metadata: Compression metadata
            
        Returns:
            DecompressionResult with detailed precision metrics
        """
        pass
    
    def _create_compression_result(self, compressed_data: bytes, original_size: int, 
                                 profile, metadata: Dict[str, Any] = None) -> CompressionResult:
        """Create enhanced compression result with aerospace metrics."""
        from ..utils.performance import calculate_aerospace_metrics, format_time_precision
        
        if metadata is None:
            metadata = {}
        
        # Create precision metrics
        precision_metrics = AerospacePrecisionMetrics()
        precision_metrics.compression_time_ns = profile.duration_ns
        precision_metrics.total_time_ns = profile.duration_ns
        precision_metrics.memory_peak_bytes = profile.memory_peak_bytes
        precision_metrics.memory_delta_bytes = profile.memory_delta_bytes
        precision_metrics.memory_before_bytes = profile.memory_before_bytes
        precision_metrics.memory_after_bytes = profile.memory_after_bytes
        precision_metrics.tracemalloc_peak_mb = profile.tracemalloc_peak_mb
        precision_metrics.tracemalloc_diff_mb = profile.tracemalloc_diff_mb
        precision_metrics.cpu_percent_avg = profile.cpu_percent_avg
        precision_metrics.cpu_percent_peak = profile.cpu_percent_peak
        precision_metrics.cpu_time_user_s = profile.cpu_time_user_s
        precision_metrics.cpu_time_system_s = profile.cpu_time_system_s
        precision_metrics.cpu_freq_avg_mhz = profile.cpu_freq_avg_mhz
        precision_metrics.io_read_bytes = profile.io_read_bytes
        precision_metrics.io_write_bytes = profile.io_write_bytes
        precision_metrics.io_read_ops = profile.io_read_ops
        precision_metrics.io_write_ops = profile.io_write_ops
        precision_metrics.page_faults = profile.page_faults
        precision_metrics.context_switches = profile.context_switches
        precision_metrics.gc_collections = profile.gc_collections
        precision_metrics.throughput_mbps = profile.throughput_mbps
        precision_metrics.throughput_bytes_per_second = profile.throughput_mbps * 1024 * 1024
        precision_metrics.time_per_byte_ns = profile.time_per_byte_ns
        precision_metrics.bytes_per_cpu_cycle = profile.bytes_per_cpu_cycle
        precision_metrics.memory_efficiency_ratio = profile.memory_efficiency_ratio
        
        # Calculate aerospace-specific metrics
        aerospace_metrics = calculate_aerospace_metrics(original_size, len(compressed_data), profile)
        precision_metrics.bits_per_byte = aerospace_metrics['bits_per_byte']
        precision_metrics.entropy_efficiency = aerospace_metrics['entropy_efficiency']
        precision_metrics.energy_efficiency_bytes_per_ns = aerospace_metrics['energy_efficiency_bytes_per_ns']
        precision_metrics.worst_case_latency_ns = aerospace_metrics['worst_case_latency_ns']
        precision_metrics.determinism_score = aerospace_metrics['determinism_score']
        precision_metrics.memory_overhead_ratio = aerospace_metrics['memory_overhead_ratio']
        precision_metrics.cpu_efficiency_bytes_per_cpu_second = aerospace_metrics['cpu_efficiency_bytes_per_cpu_second']
        precision_metrics.io_efficiency_ratio = aerospace_metrics['io_efficiency_ratio']
        
        # Update formatted times
        precision_metrics.update_formatted_times()
        
        return CompressionResult(
            compressed_data=compressed_data,
            original_size=original_size,
            compressed_size=len(compressed_data),
            compression_ratio=original_size / len(compressed_data) if compressed_data else float('inf'),
            compression_time=profile.duration_s,  # Legacy compatibility
            algorithm_name=self.name,
            metadata=metadata,
            precision_metrics=precision_metrics
        )
    
    def _create_decompression_result(self, decompressed_data: bytes, compressed_size: int,
                                   profile, metadata: Dict[str, Any] = None) -> DecompressionResult:
        """Create enhanced decompression result with aerospace metrics."""
        if metadata is None:
            metadata = {}
        
        # Create precision metrics for decompression
        precision_metrics = AerospacePrecisionMetrics()
        precision_metrics.decompression_time_ns = profile.duration_ns
        precision_metrics.total_time_ns = profile.duration_ns
        precision_metrics.memory_peak_bytes = profile.memory_peak_bytes
        precision_metrics.memory_delta_bytes = profile.memory_delta_bytes
        precision_metrics.cpu_percent_avg = profile.cpu_percent_avg
        precision_metrics.cpu_percent_peak = profile.cpu_percent_peak
        precision_metrics.throughput_mbps = profile.throughput_mbps
        precision_metrics.determinism_score = 1.0 - (profile.memory_peak_delta_bytes / max(1, profile.memory_delta_bytes))
        precision_metrics.determinism_score = max(0.0, min(1.0, precision_metrics.determinism_score))
        
        # Update formatted times
        precision_metrics.update_formatted_times()
        
        return DecompressionResult(
            decompressed_data=decompressed_data,
            original_compressed_size=compressed_size,
            decompressed_size=len(decompressed_data),
            decompression_time=profile.duration_s,  # Legacy compatibility
            algorithm_name=self.name,
            metadata=metadata,
            precision_metrics=precision_metrics
        )
    
    def set_parameters(self, **kwargs) -> None:
        """Set algorithm parameters."""
        self._parameters.update(kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return self._parameters.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information."""
        return {
            'name': self.name,
            'category': self.category.value,
            'compression_type': self.compression_type.value,
            'parameters': self.get_parameters(),
            'supports_streaming': False,  # Override in subclasses if supported
            'thread_safe': False,         # Override in subclasses if thread-safe
            'deterministic': True,        # Override if non-deterministic
            'memory_bounded': True        # Override if memory usage can be unbounded
        }


class PipelineComponent(ABC):
    """Abstract base class for pipeline components."""
    
    @abstractmethod
    def process(self, data: bytes, metadata: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
        """
        Process data through this pipeline component.
        
        Args:
            data: Input data
            metadata: Input metadata
            
        Returns:
            Tuple of (processed_data, output_metadata)
        """
        pass
    
    @abstractmethod
    def reverse_process(self, data: bytes, metadata: Dict[str, Any]) -> bytes:
        """
        Reverse the processing operation.
        
        Args:
            data: Processed data
            metadata: Processing metadata
            
        Returns:
            Original data
        """
        pass


class BenchmarkMetric(ABC):
    """Abstract base class for benchmark metrics."""
    
    @abstractmethod
    def calculate(self, result: Union[CompressionResult, DecompressionResult]) -> float:
        """Calculate metric value from result."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass
    
    @property
    @abstractmethod
    def unit(self) -> str:
        """Get metric unit."""
        pass


class DatasetProvider(ABC):
    """Abstract base class for dataset providers."""
    
    @abstractmethod
    def get_files(self):
        """Get list of available files."""
        pass
    
    @abstractmethod
    def load_file(self, file_path) -> bytes:
        """Load file content."""
        pass


class ResultsExporter(ABC):
    """Abstract base class for results exporters."""
    
    @abstractmethod
    def export(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Export results to specified format."""
        pass


# Custom exceptions with enhanced error context
class CompressionError(Exception):
    """Enhanced exception for compression operations."""
    
    def __init__(self, message: str, algorithm: str = None, data_size: int = None, 
                 error_context: Dict[str, Any] = None):
        super().__init__(message)
        self.algorithm = algorithm
        self.data_size = data_size
        self.error_context = error_context or {}
        self.timestamp_ns = time.time_ns()
    
    def to_mission_report(self) -> Dict[str, Any]:
        """Generate mission-critical error report."""
        return {
            'error_type': 'COMPRESSION_FAILURE',
            'message': str(self),
            'algorithm': self.algorithm,
            'data_size_bytes': self.data_size,
            'timestamp_ns': self.timestamp_ns,
            'context': self.error_context,
            'criticality': 'HIGH' if self.data_size and self.data_size > 1024*1024 else 'MEDIUM'
        }


class DecompressionError(Exception):
    """Enhanced exception for decompression operations."""
    
    def __init__(self, message: str, algorithm: str = None, compressed_size: int = None,
                 error_context: Dict[str, Any] = None):
        super().__init__(message)
        self.algorithm = algorithm
        self.compressed_size = compressed_size
        self.error_context = error_context or {}
        self.timestamp_ns = time.time_ns()
    
    def to_mission_report(self) -> Dict[str, Any]:
        """Generate mission-critical error report."""
        return {
            'error_type': 'DECOMPRESSION_FAILURE',
            'message': str(self),
            'algorithm': self.algorithm,
            'compressed_size_bytes': self.compressed_size,
            'timestamp_ns': self.timestamp_ns,
            'context': self.error_context,
            'criticality': 'CRITICAL'  # Decompression failures are always critical
        }


class PipelineError(Exception):
    """Enhanced exception for pipeline operations."""
    
    def __init__(self, message: str, stage_name: str = None, stage_index: int = None,
                 pipeline_name: str = None, error_context: Dict[str, Any] = None):
        super().__init__(message)
        self.stage_name = stage_name
        self.stage_index = stage_index
        self.pipeline_name = pipeline_name
        self.error_context = error_context or {}
        self.timestamp_ns = time.time_ns()
    
    def to_mission_report(self) -> Dict[str, Any]:
        """Generate mission-critical pipeline error report."""
        return {
            'error_type': 'PIPELINE_FAILURE',
            'message': str(self),
            'pipeline_name': self.pipeline_name,
            'failed_stage': self.stage_name,
            'stage_index': self.stage_index,
            'timestamp_ns': self.timestamp_ns,
            'context': self.error_context,
            'criticality': 'HIGH'
        } 