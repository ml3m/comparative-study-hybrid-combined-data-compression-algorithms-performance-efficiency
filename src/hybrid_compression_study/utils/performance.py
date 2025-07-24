"""
Enhanced performance monitoring utilities for aerospace-grade precision.

This module provides comprehensive performance monitoring capabilities
suitable for mission-critical applications where every byte and nanosecond matters.
"""

import time
import threading
import gc
import sys
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import tracemalloc
from collections import deque


@dataclass
class DetailedResourceSnapshot:
    """Comprehensive system resource snapshot with aerospace-grade precision."""
    timestamp_ns: int  # Nanosecond precision timestamp
    cpu_percent: float  # CPU usage percentage
    cpu_freq_mhz: float  # CPU frequency in MHz
    memory_rss_bytes: int  # Resident Set Size in bytes
    memory_vms_bytes: int  # Virtual Memory Size in bytes
    memory_percent: float  # Memory usage percentage
    memory_available_bytes: int  # Available memory in bytes
    memory_cached_bytes: int  # Cached memory in bytes
    memory_shared_bytes: int  # Shared memory in bytes
    page_faults: int  # Number of page faults
    context_switches: int  # Number of context switches
    io_read_bytes: int  # Bytes read from disk
    io_write_bytes: int  # Bytes written to disk
    io_read_count: int  # Number of read operations
    io_write_count: int  # Number of write operations
    threads_count: int  # Number of threads
    file_descriptors: int  # Number of open file descriptors
    gc_collections: List[int]  # Garbage collection counts per generation


@dataclass
class PrecisionPerformanceProfile:
    """Aerospace-grade performance profile with nanosecond precision."""
    operation_name: str
    data_size_bytes: int
    
    # Timing metrics (nanosecond precision)
    start_time_ns: int
    end_time_ns: int
    duration_ns: int
    duration_us: float  # Microseconds
    duration_ms: float  # Milliseconds
    duration_s: float   # Seconds
    
    # Memory metrics (byte precision)
    memory_before_bytes: int
    memory_after_bytes: int
    memory_peak_bytes: int
    memory_delta_bytes: int
    memory_peak_delta_bytes: int
    
    # Detailed memory breakdown
    tracemalloc_current_mb: float
    tracemalloc_peak_mb: float
    tracemalloc_diff_mb: float
    
    # CPU metrics
    cpu_percent_avg: float
    cpu_percent_peak: float
    cpu_time_user_s: float
    cpu_time_system_s: float
    cpu_freq_avg_mhz: float
    
    # I/O metrics
    io_read_bytes: int
    io_write_bytes: int
    io_read_ops: int
    io_write_ops: int
    io_read_time_ms: float
    io_write_time_ms: float
    
    # System metrics
    page_faults: int
    context_switches: int
    threads_created: int
    gc_collections: List[int]
    
    # Performance efficiency metrics
    throughput_mbps: float
    throughput_ops_per_sec: float
    bytes_per_cpu_cycle: float
    memory_efficiency_ratio: float  # data_size / peak_memory
    
    # Compression-specific metrics
    compression_efficiency: float  # bytes_saved / processing_time
    time_per_byte_ns: float  # processing_time / data_size
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.duration_s > 0:
            self.throughput_mbps = (self.data_size_bytes / (1024 * 1024)) / self.duration_s
            self.throughput_ops_per_sec = 1.0 / self.duration_s if self.duration_s > 0 else 0
            self.time_per_byte_ns = self.duration_ns / max(1, self.data_size_bytes)
        
        if self.memory_peak_bytes > 0:
            self.memory_efficiency_ratio = self.data_size_bytes / self.memory_peak_bytes
        
        # Estimate CPU cycles (rough approximation)
        if self.cpu_freq_avg_mhz > 0 and self.duration_s > 0:
            estimated_cycles = self.cpu_freq_avg_mhz * 1_000_000 * self.duration_s
            self.bytes_per_cpu_cycle = self.data_size_bytes / max(1, estimated_cycles)


class AerospaceGradeMonitor:
    """
    Aerospace-grade performance monitor with nanosecond precision.
    
    Suitable for mission-critical applications where every resource matters.
    """
    
    def __init__(self, sampling_interval_ms: float = 1.0):
        """
        Initialize high-precision monitor.
        
        Args:
            sampling_interval_ms: Sampling interval in milliseconds
        """
        self.sampling_interval_s = sampling_interval_ms / 1000.0
        self._monitoring = False
        self._samples: deque = deque(maxlen=10000)  # Circular buffer
        self._monitor_thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        
        # Enable tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def _get_detailed_snapshot(self) -> DetailedResourceSnapshot:
        """Get comprehensive system resource snapshot."""
        try:
            # Get process info
            proc_info = self._process.as_dict([
                'cpu_percent', 'memory_info', 'memory_percent', 'num_threads',
                'num_fds', 'io_counters'
            ])
            
            # Get system info
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            # Get I/O counters
            io_counters = proc_info.get('io_counters')
            if io_counters is None:
                io_counters = psutil._common.pio(read_count=0, write_count=0, 
                                                read_bytes=0, write_bytes=0)
            
            # Get memory info
            memory_info = proc_info.get('memory_info', psutil._common.pmem(rss=0, vms=0))
            
            return DetailedResourceSnapshot(
                timestamp_ns=time.time_ns(),
                cpu_percent=proc_info.get('cpu_percent', 0.0),
                cpu_freq_mhz=cpu_freq.current if cpu_freq else 0.0,
                memory_rss_bytes=memory_info.rss,
                memory_vms_bytes=memory_info.vms,
                memory_percent=proc_info.get('memory_percent', 0.0),
                memory_available_bytes=memory.available,
                memory_cached_bytes=getattr(memory, 'cached', 0),
                memory_shared_bytes=getattr(memory, 'shared', 0),
                page_faults=getattr(memory_info, 'pfaults', 0),
                context_switches=0,  # Would need platform-specific implementation
                io_read_bytes=io_counters.read_bytes,
                io_write_bytes=io_counters.write_bytes,
                io_read_count=io_counters.read_count,
                io_write_count=io_counters.write_count,
                threads_count=proc_info.get('num_threads', 0),
                file_descriptors=proc_info.get('num_fds', 0),
                gc_collections=list(gc.get_counts())
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Return zero snapshot if process monitoring fails
            return DetailedResourceSnapshot(
                timestamp_ns=time.time_ns(),
                cpu_percent=0.0, cpu_freq_mhz=0.0,
                memory_rss_bytes=0, memory_vms_bytes=0, memory_percent=0.0,
                memory_available_bytes=0, memory_cached_bytes=0, memory_shared_bytes=0,
                page_faults=0, context_switches=0,
                io_read_bytes=0, io_write_bytes=0, io_read_count=0, io_write_count=0,
                threads_count=0, file_descriptors=0, gc_collections=[0, 0, 0]
            )
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._samples.clear()
        
        def monitor_loop():
            while self._monitoring:
                snapshot = self._get_detailed_snapshot()
                self._samples.append(snapshot)
                time.sleep(self.sampling_interval_s)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    @contextmanager
    def profile_operation(self, operation_name: str, data_size_bytes: int):
        """
        Profile an operation with aerospace-grade precision.
        
        Args:
            operation_name: Name of the operation
            data_size_bytes: Size of data being processed
            
        Yields:
            PrecisionPerformanceProfile: Detailed performance profile
        """
        # Pre-operation setup
        gc.collect()  # Clean slate for memory measurements
        
        # Get initial state
        start_snapshot = self._get_detailed_snapshot()
        
        # Memory tracking
        tracemalloc_start = tracemalloc.get_traced_memory()
        
        # CPU time tracking
        cpu_times_start = self._process.cpu_times()
        
        # High-precision timing
        start_time_ns = time.time_ns()
        
        # Start monitoring if not already running
        was_monitoring = self._monitoring
        if not was_monitoring:
            self.start_monitoring()
        
        try:
            # Create profile object that will be populated
            profile = PrecisionPerformanceProfile(
                operation_name=operation_name,
                data_size_bytes=data_size_bytes,
                start_time_ns=start_time_ns,
                end_time_ns=0,
                duration_ns=0,
                duration_us=0.0,
                duration_ms=0.0,
                duration_s=0.0,
                memory_before_bytes=start_snapshot.memory_rss_bytes,
                memory_after_bytes=0,
                memory_peak_bytes=0,
                memory_delta_bytes=0,
                memory_peak_delta_bytes=0,
                tracemalloc_current_mb=tracemalloc_start[0] / 1024 / 1024,
                tracemalloc_peak_mb=0.0,
                tracemalloc_diff_mb=0.0,
                cpu_percent_avg=0.0,
                cpu_percent_peak=0.0,
                cpu_time_user_s=cpu_times_start.user,
                cpu_time_system_s=cpu_times_start.system,
                cpu_freq_avg_mhz=start_snapshot.cpu_freq_mhz,
                io_read_bytes=start_snapshot.io_read_bytes,
                io_write_bytes=start_snapshot.io_write_bytes,
                io_read_ops=start_snapshot.io_read_count,
                io_write_ops=start_snapshot.io_write_count,
                io_read_time_ms=0.0,
                io_write_time_ms=0.0,
                page_faults=start_snapshot.page_faults,
                context_switches=start_snapshot.context_switches,
                threads_created=0,
                gc_collections=start_snapshot.gc_collections.copy(),
                throughput_mbps=0.0,
                throughput_ops_per_sec=0.0,
                bytes_per_cpu_cycle=0.0,
                memory_efficiency_ratio=0.0,
                compression_efficiency=0.0,
                time_per_byte_ns=0.0
            )
            
            yield profile
            
        finally:
            # Post-operation measurements
            end_time_ns = time.time_ns()
            end_snapshot = self._get_detailed_snapshot()
            
            # CPU time tracking
            cpu_times_end = self._process.cpu_times()
            
            # Memory tracking
            tracemalloc_end = tracemalloc.get_traced_memory()
            
            # Calculate timing metrics
            profile.end_time_ns = end_time_ns
            profile.duration_ns = end_time_ns - start_time_ns
            profile.duration_us = profile.duration_ns / 1_000
            profile.duration_ms = profile.duration_ns / 1_000_000
            profile.duration_s = profile.duration_ns / 1_000_000_000
            
            # Calculate memory metrics
            profile.memory_after_bytes = end_snapshot.memory_rss_bytes
            profile.memory_delta_bytes = end_snapshot.memory_rss_bytes - start_snapshot.memory_rss_bytes
            
            # Tracemalloc metrics
            profile.tracemalloc_peak_mb = tracemalloc_end[1] / 1024 / 1024
            profile.tracemalloc_diff_mb = (tracemalloc_end[0] - tracemalloc_start[0]) / 1024 / 1024
            
            # CPU metrics from monitoring samples
            if len(self._samples) > 1:
                recent_samples = [s for s in self._samples 
                                if s.timestamp_ns >= start_time_ns]
                if recent_samples:
                    cpu_values = [s.cpu_percent for s in recent_samples]
                    memory_values = [s.memory_rss_bytes for s in recent_samples]
                    
                    profile.cpu_percent_avg = sum(cpu_values) / len(cpu_values)
                    profile.cpu_percent_peak = max(cpu_values)
                    profile.memory_peak_bytes = max(memory_values)
                    profile.memory_peak_delta_bytes = profile.memory_peak_bytes - start_snapshot.memory_rss_bytes
                    
                    # Average CPU frequency during operation
                    freq_values = [s.cpu_freq_mhz for s in recent_samples if s.cpu_freq_mhz > 0]
                    if freq_values:
                        profile.cpu_freq_avg_mhz = sum(freq_values) / len(freq_values)
            
            # CPU time deltas
            profile.cpu_time_user_s = cpu_times_end.user - cpu_times_start.user
            profile.cpu_time_system_s = cpu_times_end.system - cpu_times_start.system
            
            # I/O metrics
            profile.io_read_bytes = end_snapshot.io_read_bytes - start_snapshot.io_read_bytes
            profile.io_write_bytes = end_snapshot.io_write_bytes - start_snapshot.io_write_bytes
            profile.io_read_ops = end_snapshot.io_read_count - start_snapshot.io_read_count
            profile.io_write_ops = end_snapshot.io_write_count - start_snapshot.io_write_count
            
            # GC metrics
            end_gc = end_snapshot.gc_collections
            start_gc = start_snapshot.gc_collections
            profile.gc_collections = [end_gc[i] - start_gc[i] for i in range(min(len(end_gc), len(start_gc)))]
            
            # Calculate derived metrics
            profile.__post_init__()
            
            # Stop monitoring if we started it
            if not was_monitoring:
                self.stop_monitoring()


def format_time_precision(duration_ns: int) -> str:
    """Format time with appropriate precision for aerospace applications."""
    if duration_ns < 1_000:  # Less than 1 microsecond
        return f"{duration_ns}ns"
    elif duration_ns < 1_000_000:  # Less than 1 millisecond
        return f"{duration_ns / 1_000:.2f}μs"
    elif duration_ns < 1_000_000_000:  # Less than 1 second
        return f"{duration_ns / 1_000_000:.3f}ms"
    else:
        return f"{duration_ns / 1_000_000_000:.6f}s"


def format_memory_precision(bytes_value: int) -> str:
    """Format memory with appropriate precision for space applications."""
    if bytes_value == 0:
        return "0B"
    elif bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.2f}KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.3f}MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.6f}GB"


def calculate_aerospace_metrics(data_size: int, compressed_size: int, 
                              profile: PrecisionPerformanceProfile) -> Dict[str, Any]:
    """Calculate aerospace-specific compression metrics."""
    compression_ratio = data_size / max(1, compressed_size)
    space_savings_bytes = data_size - compressed_size
    space_savings_percent = (space_savings_bytes / data_size) * 100 if data_size > 0 else 0
    
    # Mission-critical metrics
    bits_per_byte = (compressed_size * 8) / max(1, data_size)
    entropy_efficiency = compression_ratio / 8.0  # Theoretical max is 8:1 for 8-bit data
    
    # Resource efficiency
    memory_overhead_ratio = profile.memory_peak_bytes / max(1, data_size)
    energy_efficiency = data_size / max(1, profile.duration_ns)  # bytes per nanosecond (proxy for energy)
    
    # Real-time performance
    worst_case_latency_ns = profile.duration_ns  # Worst case is the actual time taken
    determinism_score = 1.0 - (profile.memory_peak_delta_bytes / max(1, profile.memory_delta_bytes))
    
    return {
        'compression_ratio': compression_ratio,
        'space_savings_bytes': space_savings_bytes,
        'space_savings_percent': space_savings_percent,
        'bits_per_byte': bits_per_byte,
        'entropy_efficiency': entropy_efficiency,
        'memory_overhead_ratio': memory_overhead_ratio,
        'energy_efficiency_bytes_per_ns': energy_efficiency,
        'worst_case_latency_ns': worst_case_latency_ns,
        'determinism_score': max(0.0, min(1.0, determinism_score)),
        'throughput_bytes_per_second': profile.throughput_mbps * 1024 * 1024,
        'cpu_efficiency_bytes_per_cpu_second': data_size / max(0.001, profile.cpu_time_user_s + profile.cpu_time_system_s),
        'io_efficiency_ratio': data_size / max(1, profile.io_read_bytes + profile.io_write_bytes),
        'memory_efficiency_ratio': profile.memory_efficiency_ratio
    }


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for aerospace applications."""
    cpu_info = {}
    memory_info = {}
    
    try:
        # CPU information
        cpu_freq = psutil.cpu_freq()
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'current_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'min_freq_mhz': cpu_freq.min if cpu_freq else 0,
            'max_freq_mhz': cpu_freq.max if cpu_freq else 0,
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown',
            'cache_info': 'N/A'  # Would need platform-specific implementation
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'total_bytes': memory.total,
            'available_bytes': memory.available,
            'used_bytes': memory.used,
            'free_bytes': memory.free,
            'cached_bytes': getattr(memory, 'cached', 0),
            'shared_bytes': getattr(memory, 'shared', 0),
            'page_size_bytes': os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096
        }
        
    except Exception:
        pass
    
    return {
        'platform': sys.platform,
        'python_version': sys.version,
        'cpu': cpu_info,
        'memory': memory_info,
        'pid': os.getpid(),
        'timestamp_ns': time.time_ns()
    } 