"""
Enhanced Run-Length Encoding implementation with aerospace-grade precision monitoring.

This implementation provides nanosecond-precision metrics suitable for
mission-critical applications where every byte and microsecond matters.
"""

import time
from typing import Dict, Any, List, Tuple
from collections import Counter

from ..core.base import (
    CompressionAlgorithm,
    CompressionResult,
    DecompressionResult,
    AlgorithmCategory,
    CompressionError,
    DecompressionError
)


class RunLengthEncoder(CompressionAlgorithm):
    """
    Enhanced Run-Length Encoding implementation with aerospace-grade monitoring.
    
    Provides nanosecond-precision performance metrics suitable for
    mission-critical space applications.
    """
    
    def __init__(self, escape_byte: int = 0x00, min_run_length: int = 3):
        """
        Initialize RLE encoder with aerospace-grade monitoring.
        
        Args:
            escape_byte: Byte value used as escape sequence marker (0-255)
            min_run_length: Minimum run length to encode (saves space)
        """
        super().__init__("RLE", AlgorithmCategory.RUN_LENGTH)
        
        if not 0 <= escape_byte <= 255:
            raise ValueError("Escape byte must be between 0 and 255")
        if min_run_length < 1:
            raise ValueError("Minimum run length must be at least 1")
        
        self.escape_byte = escape_byte
        self.min_run_length = min_run_length
        
        self.set_parameters(
            escape_byte=escape_byte,
            min_run_length=min_run_length
        )
    
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress data using RLE with aerospace-grade monitoring.
        
        Args:
            data: Raw bytes to compress
            
        Returns:
            CompressionResult with nanosecond-precision metrics
        """
        if not data:
            return self._create_compression_result(
                compressed_data=b'',
                original_size=0,
                profile=self._create_empty_profile(),
                metadata={'runs_encoded': 0, 'escape_conflicts': 0}
            )
        
        # Use aerospace-grade monitoring
        with self._monitor.profile_operation("rle_compress", len(data)) as profile:
            try:
                compressed_data, metadata = self._encode_rle(data)
                
                return self._create_compression_result(
                    compressed_data=compressed_data,
                    original_size=len(data),
                    profile=profile,
                    metadata=metadata
                )
                
            except Exception as e:
                raise CompressionError(
                    f"RLE compression failed: {str(e)}",
                    algorithm=self.name,
                    data_size=len(data),
                    error_context={
                        'escape_byte': self.escape_byte,
                        'min_run_length': self.min_run_length
                    }
                )
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> DecompressionResult:
        """
        Decompress RLE-compressed data with aerospace-grade monitoring.
        
        Args:
            compressed_data: Compressed bytes
            metadata: Compression metadata
            
        Returns:
            DecompressionResult with nanosecond-precision metrics
        """
        if not compressed_data:
            return self._create_decompression_result(
                decompressed_data=b'',
                compressed_size=0,
                profile=self._create_empty_profile(),
                metadata={}
            )
        
        with self._monitor.profile_operation("rle_decompress", len(compressed_data)) as profile:
            try:
                decompressed_data = self._decode_rle(compressed_data, metadata)
                
                return self._create_decompression_result(
                    decompressed_data=decompressed_data,
                    compressed_size=len(compressed_data),
                    profile=profile,
                    metadata=metadata
                )
                
            except Exception as e:
                raise DecompressionError(
                    f"RLE decompression failed: {str(e)}",
                    algorithm=self.name,
                    compressed_size=len(compressed_data),
                    error_context=metadata
                )
    
    def _encode_rle(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Core RLE encoding with detailed metrics tracking.
        
        Returns:
            Tuple of (compressed_data, metadata)
        """
        if not data:
            return b'', {'runs_encoded': 0, 'escape_conflicts': 0}
        
        result = bytearray()
        
        # Statistics for mission-critical analysis
        runs_encoded = 0
        escape_conflicts = 0
        total_runs_found = 0
        bytes_saved = 0
        literal_sequences = 0
        
        i = 0
        while i < len(data):
            current_byte = data[i]
            run_length = 1
            
            # Count consecutive identical bytes
            while (i + run_length < len(data) and 
                   data[i + run_length] == current_byte and 
                   run_length < 255):  # Max run length for single byte
                run_length += 1
            
            total_runs_found += 1
            
            if run_length >= self.min_run_length:
                # Encode as run: [escape_byte][actual_byte][run_length]
                if current_byte == self.escape_byte:
                    escape_conflicts += 1
                
                result.extend([self.escape_byte, current_byte, run_length])
                runs_encoded += 1
                bytes_saved += max(0, run_length - 3)  # 3 bytes used for encoding
                i += run_length
            else:
                # Handle escape byte conflicts in literal data
                if current_byte == self.escape_byte:
                    # Encode escaped escape byte: [escape_byte][escape_byte][1]
                    result.extend([self.escape_byte, self.escape_byte, 1])
                    escape_conflicts += 1
                else:
                    # Literal byte
                    result.append(current_byte)
                    literal_sequences += 1
                i += 1
        
        # Calculate compression efficiency metrics
        compression_effectiveness = bytes_saved / len(data) if len(data) > 0 else 0.0
        run_density = runs_encoded / total_runs_found if total_runs_found > 0 else 0.0
        
        metadata = {
            'runs_encoded': runs_encoded,
            'total_runs_found': total_runs_found,
            'escape_conflicts': escape_conflicts,
            'literal_sequences': literal_sequences,
            'bytes_saved': bytes_saved,
            'compression_effectiveness': compression_effectiveness,
            'run_density': run_density,
            'min_run_length': self.min_run_length,
            'escape_byte': self.escape_byte,
            'original_entropy': self._calculate_entropy(data)
        }
        
        return bytes(result), metadata
    
    def _decode_rle(self, compressed_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """
        Core RLE decoding with safety checks.
        
        Args:
            compressed_data: Compressed bytes
            metadata: Metadata from compression
            
        Returns:
            Decompressed bytes
        """
        if not compressed_data:
            return b''
        
        result = bytearray()
        i = 0
        
        # Safety limits for mission-critical applications
        max_iterations = len(compressed_data) * 1000  # Prevent infinite loops
        iterations = 0
        
        while i < len(compressed_data) and iterations < max_iterations:
            iterations += 1
            
            if compressed_data[i] == self.escape_byte:
                # Handle escape sequence
                if i + 2 >= len(compressed_data):
                    raise DecompressionError(
                        f"Incomplete escape sequence at position {i}",
                        algorithm=self.name,
                        compressed_size=len(compressed_data)
                    )
                
                actual_byte = compressed_data[i + 1]
                run_length = compressed_data[i + 2]
                
                if run_length == 0:
                    raise DecompressionError(
                        f"Invalid run length 0 at position {i}",
                        algorithm=self.name,
                        compressed_size=len(compressed_data)
                    )
                
                # Decode run
                result.extend([actual_byte] * run_length)
                i += 3
            else:
                # Literal byte
                result.append(compressed_data[i])
                i += 1
        
        if iterations >= max_iterations:
            raise DecompressionError(
                "Maximum iteration limit reached (possible infinite loop)",
                algorithm=self.name,
                compressed_size=len(compressed_data)
            )
        
        return bytes(result)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy for analysis."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        frequencies = Counter(data)
        total_bytes = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in frequencies.values():
            if count > 0:
                probability = count / total_bytes
                entropy -= probability * (probability.bit_length() - 1)  # log2 approximation
        
        return entropy
    
    def _create_empty_profile(self):
        """Create an empty performance profile for edge cases."""
        from ..utils.performance import PrecisionPerformanceProfile
        
        return PrecisionPerformanceProfile(
            operation_name="empty_operation",
            data_size_bytes=0,
            start_time_ns=time.time_ns(),
            end_time_ns=time.time_ns(),
            duration_ns=0,
            duration_us=0.0,
            duration_ms=0.0,
            duration_s=0.0,
            memory_before_bytes=0,
            memory_after_bytes=0,
            memory_peak_bytes=0,
            memory_delta_bytes=0,
            memory_peak_delta_bytes=0,
            tracemalloc_current_mb=0.0,
            tracemalloc_peak_mb=0.0,
            tracemalloc_diff_mb=0.0,
            cpu_percent_avg=0.0,
            cpu_percent_peak=0.0,
            cpu_time_user_s=0.0,
            cpu_time_system_s=0.0,
            cpu_freq_avg_mhz=0.0,
            io_read_bytes=0,
            io_write_bytes=0,
            io_read_ops=0,
            io_write_ops=0,
            io_read_time_ms=0.0,
            io_write_time_ms=0.0,
            page_faults=0,
            context_switches=0,
            threads_created=0,
            gc_collections=[],
            throughput_mbps=0.0,
            throughput_ops_per_sec=0.0,
            bytes_per_cpu_cycle=0.0,
            memory_efficiency_ratio=0.0,
            compression_efficiency=0.0,
            time_per_byte_ns=0.0
        )


class AdaptiveRunLengthEncoder(RunLengthEncoder):
    """
    Adaptive RLE with automatic parameter optimization and aerospace-grade monitoring.
    
    Analyzes data patterns to optimize compression parameters automatically.
    """
    
    def __init__(self, escape_byte: int = 0x00):
        # Start with default parameters
        super().__init__(escape_byte, min_run_length=3)
        self.name = "Adaptive-RLE"
    
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress with automatic parameter optimization and aerospace monitoring.
        """
        if not data:
            return self._create_compression_result(
                compressed_data=b'',
                original_size=0,
                profile=self._create_empty_profile(),
                metadata={'adaptive_optimization': 'skipped_empty_data'}
            )
        
        with self._monitor.profile_operation("adaptive_rle_compress", len(data)) as profile:
            try:
                # Analyze data for optimal parameters
                optimal_params = self._analyze_data_for_optimal_params(data)
                
                # Update parameters
                original_min_run = self.min_run_length
                self.min_run_length = optimal_params['optimal_min_run_length']
                
                # Compress with optimized parameters
                compressed_data, base_metadata = self._encode_rle(data)
                
                # Add adaptive optimization metadata
                metadata = {
                    **base_metadata,
                    'adaptive_optimization': {
                        'original_min_run_length': original_min_run,
                        'optimized_min_run_length': self.min_run_length,
                        'optimization_analysis': optimal_params,
                        'adaptation_effective': optimal_params['estimated_improvement'] > 0.05
                    }
                }
                
                return self._create_compression_result(
                    compressed_data=compressed_data,
                    original_size=len(data),
                    profile=profile,
                    metadata=metadata
                )
                
            except Exception as e:
                raise CompressionError(
                    f"Adaptive RLE compression failed: {str(e)}",
                    algorithm=self.name,
                    data_size=len(data),
                    error_context={'adaptive_analysis': True}
                )
    
    def _analyze_data_for_optimal_params(self, data: bytes) -> Dict[str, Any]:
        """
        Analyze data patterns to determine optimal compression parameters.
        
        Returns:
            Dictionary with analysis results and optimal parameters
        """
        if len(data) < 10:
            return {
                'optimal_min_run_length': 3,
                'run_length_distribution': {},
                'estimated_improvement': 0.0,
                'analysis_method': 'insufficient_data'
            }
        
        # Analyze run length distribution
        run_analysis = self._analyze_run_lengths(data)
        
        # Find optimal minimum run length
        optimal_min_run = self._find_optimal_min_run_length(run_analysis, len(data))
        
        # Estimate improvement
        current_efficiency = self._estimate_compression_efficiency(data, self.min_run_length)
        optimal_efficiency = self._estimate_compression_efficiency(data, optimal_min_run)
        improvement = optimal_efficiency - current_efficiency
        
        return {
            'optimal_min_run_length': optimal_min_run,
            'run_length_distribution': run_analysis['distribution'],
            'average_run_length': run_analysis['average_run_length'],
            'max_run_length': run_analysis['max_run_length'],
            'total_runs': run_analysis['total_runs'],
            'estimated_improvement': improvement,
            'current_efficiency': current_efficiency,
            'optimal_efficiency': optimal_efficiency,
            'analysis_method': 'run_length_distribution'
        }
    
    def _analyze_run_lengths(self, data: bytes) -> Dict[str, Any]:
        """Analyze run length patterns in the data."""
        run_lengths = []
        run_distribution = {}
        
        i = 0
        while i < len(data):
            current_byte = data[i]
            run_length = 1
            
            # Count run length
            while (i + run_length < len(data) and 
                   data[i + run_length] == current_byte):
                run_length += 1
            
            run_lengths.append(run_length)
            run_distribution[run_length] = run_distribution.get(run_length, 0) + 1
            i += run_length
        
        return {
            'distribution': run_distribution,
            'total_runs': len(run_lengths),
            'average_run_length': sum(run_lengths) / len(run_lengths) if run_lengths else 0,
            'max_run_length': max(run_lengths) if run_lengths else 0,
            'run_lengths': run_lengths[:100]  # Sample for analysis
        }
    
    def _find_optimal_min_run_length(self, run_analysis: Dict[str, Any], data_size: int) -> int:
        """Find the optimal minimum run length based on analysis."""
        distribution = run_analysis['distribution']
        
        if not distribution:
            return 3  # Default fallback
        
        # Calculate savings for different minimum run lengths
        best_min_run = 3
        best_savings = -float('inf')
        
        for candidate_min_run in range(2, min(8, max(distribution.keys()) + 1)):
            total_savings = 0
            
            for run_length, count in distribution.items():
                if run_length >= candidate_min_run:
                    # Savings: run_length bytes replaced by 3 bytes (escape + byte + count)
                    savings_per_run = run_length - 3
                    total_savings += savings_per_run * count
                else:
                    # Cost: might need escape sequences for conflicts
                    total_savings -= count * 0.1  # Small penalty
            
            if total_savings > best_savings:
                best_savings = total_savings
                best_min_run = candidate_min_run
        
        return max(2, min(7, best_min_run))  # Reasonable bounds
    
    def _estimate_compression_efficiency(self, data: bytes, min_run_length: int) -> float:
        """Estimate compression efficiency for given parameters."""
        if not data:
            return 0.0
        
        # Quick simulation without actual compression
        savings = 0
        i = 0
        
        while i < len(data):
            current_byte = data[i]
            run_length = 1
            
            while (i + run_length < len(data) and 
                   data[i + run_length] == current_byte and 
                   run_length < 255):
                run_length += 1
            
            if run_length >= min_run_length:
                # Would be encoded as run (3 bytes)
                savings += run_length - 3
            
            i += run_length
        
        return savings / len(data) if len(data) > 0 else 0.0 