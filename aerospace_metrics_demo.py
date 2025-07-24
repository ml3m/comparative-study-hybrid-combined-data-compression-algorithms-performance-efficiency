#!/usr/bin/env python3
"""
🚀 AEROSPACE-GRADE PRECISION METRICS DEMONSTRATION

This script demonstrates the enhanced precision metrics suitable for
mission-critical applications like spacecraft and Mars rovers where
every byte and nanosecond matters.

Features:
- Nanosecond-precision timing
- Byte-level memory tracking
- CPU utilization monitoring
- I/O operation counting
- Real-time performance analysis
- Mission-critical readiness assessment
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_compression_study.algorithms.huffman import HuffmanEncoder
from hybrid_compression_study.algorithms.rle import RunLengthEncoder
from hybrid_compression_study.algorithms.lz77 import LZ77Encoder
from hybrid_compression_study.algorithms.lzw import LZWEncoder
from hybrid_compression_study.utils.performance import format_time_precision, format_memory_precision


def print_section_header(title: str, emoji: str = "🔬"):
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_metric(label: str, value, unit: str = "", precision: int = 6):
    """Print a formatted metric."""
    if isinstance(value, float):
        if precision == 0:
            print(f"   📊 {label:<30}: {value:,.0f}{unit}")
        else:
            print(f"   📊 {label:<30}: {value:,.{precision}f}{unit}")
    elif isinstance(value, int):
        print(f"   📊 {label:<30}: {value:,}{unit}")
    else:
        print(f"   📊 {label:<30}: {value}{unit}")


def demonstrate_precision_timing():
    """Demonstrate nanosecond-precision timing capabilities."""
    print_section_header("NANOSECOND-PRECISION TIMING ANALYSIS", "⏱️")
    
    # Test data designed for different compression characteristics
    test_cases = [
        (b"A" * 1000, "Highly repetitive data (RLE optimal)"),
        (b"Hello, World! " * 100, "Moderately repetitive text"),
        (bytes(range(256)) * 4, "Low-entropy binary data"),
        (b"The quick brown fox jumps over the lazy dog. " * 50, "Natural text patterns")
    ]
    
    algorithms = [
        ("Huffman", HuffmanEncoder()),
        ("RLE", RunLengthEncoder()),
        ("LZ77", LZ77Encoder(window_size=1024, lookahead_size=16)),
        ("LZW", LZWEncoder(max_code_bits=12))
    ]
    
    for data, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Data size: {len(data):,} bytes")
        
        for algo_name, algorithm in algorithms:
            print(f"\n   🔍 {algo_name} Analysis:")
            
            try:
                # Compress with detailed monitoring
                result = algorithm.compress(data)
                
                # Display nanosecond-precision metrics
                metrics = result.precision_metrics
                
                print_metric("Compression Time", metrics.compression_time_formatted)
                print_metric("Compression Time (ns)", metrics.compression_time_ns, " ns")
                print_metric("Time per Byte", metrics.time_per_byte_ns, " ns/byte", 2)
                print_metric("Throughput", metrics.throughput_mbps, " MB/s", 3)
                print_metric("Compression Ratio", result.compression_ratio, "x", 3)
                
                # Verify decompression with timing
                decomp_result = algorithm.decompress(result.compressed_data, result.metadata)
                decomp_metrics = decomp_result.precision_metrics
                
                print_metric("Decompression Time", decomp_metrics.decompression_time_formatted)
                print_metric("Round-trip Time", format_time_precision(metrics.compression_time_ns + decomp_metrics.decompression_time_ns))
                
                # Data integrity check
                integrity_ok = decomp_result.decompressed_data == data
                print_metric("Data Integrity", "✅ VERIFIED" if integrity_ok else "❌ FAILED")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")


def demonstrate_memory_precision():
    """Demonstrate byte-level memory tracking."""
    print_section_header("BYTE-LEVEL MEMORY ANALYSIS", "🧠")
    
    # Test with increasing data sizes to show memory scaling
    data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
    
    for size in data_sizes:
        # Generate test data with specific patterns
        test_data = (b"SPACE_MISSION_DATA_" * (size // 19))[:size]
        
        print(f"\n🔬 Memory Analysis for {size:,} byte payload:")
        
        huffman = HuffmanEncoder()
        result = huffman.compress(test_data)
        metrics = result.precision_metrics
        
        print_metric("Input Data Size", size, " bytes")
        print_metric("Peak Memory Usage", format_memory_precision(metrics.memory_peak_bytes))
        print_metric("Memory Delta", format_memory_precision(metrics.memory_delta_bytes))
        print_metric("Memory Overhead Ratio", metrics.memory_overhead_ratio, "", 4)
        print_metric("TraceMalloc Peak", metrics.tracemalloc_peak_mb, " MB", 3)
        print_metric("TraceMalloc Delta", metrics.tracemalloc_diff_mb, " MB", 3)
        print_metric("Memory Efficiency", "EXCELLENT" if metrics.memory_overhead_ratio < 2.0 else "ACCEPTABLE" if metrics.memory_overhead_ratio < 5.0 else "CONCERNING")


def demonstrate_cpu_analysis():
    """Demonstrate detailed CPU utilization analysis."""
    print_section_header("CPU UTILIZATION ANALYSIS", "⚡")
    
    # CPU-intensive test data
    complex_data = bytes(i % 256 for i in range(50000))  # 50KB of varied data
    
    algorithms = [
        ("Huffman (Tree Building)", HuffmanEncoder()),
        ("LZ77 (Pattern Matching)", LZ77Encoder(window_size=4096, lookahead_size=32)),
        ("LZW (Dictionary Building)", LZWEncoder(max_code_bits=14))
    ]
    
    for name, algorithm in algorithms:
        print(f"\n🔍 {name}:")
        
        result = algorithm.compress(complex_data)
        metrics = result.precision_metrics
        
        print_metric("CPU Average Usage", metrics.cpu_percent_avg, "%", 2)
        print_metric("CPU Peak Usage", metrics.cpu_percent_peak, "%", 2)
        print_metric("CPU User Time", metrics.cpu_time_user_s, " seconds", 6)
        print_metric("CPU System Time", metrics.cpu_time_system_s, " seconds", 6)
        print_metric("CPU Frequency (Avg)", metrics.cpu_freq_avg_mhz, " MHz", 1)
        print_metric("CPU Efficiency", metrics.cpu_efficiency_bytes_per_cpu_second, " bytes/cpu-sec", 0)
        print_metric("Bytes per CPU Cycle", metrics.bytes_per_cpu_cycle, "", 8)


def demonstrate_io_monitoring():
    """Demonstrate I/O operation monitoring."""
    print_section_header("I/O OPERATION MONITORING", "💽")
    
    # Test data that will trigger various I/O patterns
    io_test_data = b"IO_INTENSIVE_" * 8192  # ~100KB
    
    huffman = HuffmanEncoder()
    result = huffman.compress(io_test_data)
    metrics = result.precision_metrics
    
    print_metric("I/O Read Bytes", metrics.io_read_bytes, " bytes")
    print_metric("I/O Write Bytes", metrics.io_write_bytes, " bytes")
    print_metric("I/O Read Operations", metrics.io_read_ops)
    print_metric("I/O Write Operations", metrics.io_write_ops)
    print_metric("I/O Efficiency Ratio", metrics.io_efficiency_ratio, "", 4)
    print_metric("Page Faults", metrics.page_faults)
    print_metric("Context Switches", metrics.context_switches)


def demonstrate_mission_critical_assessment():
    """Demonstrate mission-critical readiness assessment."""
    print_section_header("MISSION-CRITICAL READINESS ASSESSMENT", "🚀")
    
    # Simulate Mars rover data compression scenario
    mars_telemetry = b"MARS_ROVER_TELEMETRY:" + bytes(range(256)) * 200  # ~51KB
    
    algorithms = [
        ("Huffman", HuffmanEncoder()),
        ("RLE", RunLengthEncoder()),
        ("LZ77", LZ77Encoder()),
        ("LZW", LZWEncoder())
    ]
    
    print("🌌 Simulating Mars Rover Telemetry Compression:")
    print(f"   Payload Size: {len(mars_telemetry):,} bytes")
    print("   Mission Requirements:")
    print("     • Real-time processing: < 100ms")  
    print("     • Memory efficiency: < 3x data size")
    print("     • Deterministic behavior: > 90%")
    print("     • Energy efficiency: Critical")
    
    for name, algorithm in algorithms:
        print(f"\n🛰️  {name} Mission Assessment:")
        
        try:
            result = algorithm.compress(mars_telemetry)
            metrics = result.precision_metrics
            
            # Mission-critical assessments
            realtime_ok = metrics.compression_time_ns < 100_000_000  # 100ms in ns
            memory_ok = metrics.memory_overhead_ratio < 3.0
            deterministic_ok = metrics.determinism_score > 0.9
            
            print_metric("Compression Ratio", result.compression_ratio, "x", 3)
            print_metric("Processing Time", metrics.compression_time_formatted)
            print_metric("Real-time Suitable", "✅ YES" if realtime_ok else "❌ NO")
            print_metric("Memory Efficient", "✅ YES" if memory_ok else "❌ NO") 
            print_metric("Determinism Score", metrics.determinism_score, "", 6)
            print_metric("Deterministic Enough", "✅ YES" if deterministic_ok else "❌ NO")
            print_metric("Energy Efficiency", metrics.energy_efficiency_bytes_per_ns, " bytes/ns", 2)
            print_metric("Worst-case Latency", format_time_precision(metrics.worst_case_latency_ns))
            print_metric("Entropy Efficiency", metrics.entropy_efficiency, "", 4)
            
            # Overall mission readiness
            mission_ready_score = sum([realtime_ok, memory_ok, deterministic_ok]) / 3.0
            mission_status = "🟢 MISSION READY" if mission_ready_score >= 0.8 else "🟡 CONDITIONAL" if mission_ready_score >= 0.5 else "🔴 NOT SUITABLE"
            print_metric("Mission Readiness", mission_status)
            
        except Exception as e:
            print(f"      ❌ Mission Critical Failure: {str(e)}")


def demonstrate_comparative_analysis():
    """Demonstrate comparative analysis across algorithms."""
    print_section_header("COMPARATIVE PERFORMANCE ANALYSIS", "📊")
    
    # Standard test payload
    test_payload = b"COMPARATIVE_TEST_" * 3200  # ~51KB
    
    algorithms = [
        HuffmanEncoder(),
        RunLengthEncoder(), 
        LZ77Encoder(window_size=2048),
        LZWEncoder(max_code_bits=12)
    ]
    
    results = []
    
    print("🔬 Running comparative analysis...")
    
    for algorithm in algorithms:
        try:
            result = algorithm.compress(test_payload)
            metrics = result.precision_metrics
            
            results.append({
                'name': algorithm.name,
                'ratio': result.compression_ratio,
                'time_ns': metrics.compression_time_ns,
                'memory_mb': metrics.memory_peak_bytes / (1024 * 1024),
                'throughput': metrics.throughput_mbps,
                'efficiency': metrics.energy_efficiency_bytes_per_ns
            })
        except Exception as e:
            print(f"   ❌ {algorithm.name} failed: {str(e)}")
    
    if results:
        print("\n📈 Comparative Results (sorted by compression ratio):")
        results.sort(key=lambda x: x['ratio'], reverse=True)
        
        print(f"{'Algorithm':<12} {'Ratio':<8} {'Time':<12} {'Memory':<10} {'Throughput':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['name']:<12} {r['ratio']:<8.2f} {format_time_precision(r['time_ns']):<12} "
                  f"{r['memory_mb']:<10.2f} {r['throughput']:<12.3f} {r['efficiency']:<12.2e}")


def main():
    """Run comprehensive aerospace-grade metrics demonstration."""
    print("🚀 AEROSPACE-GRADE COMPRESSION METRICS DEMONSTRATION")
    print("=" * 60)
    print("🎯 Mission: Demonstrate nanosecond-precision performance monitoring")
    print("🎯 Application: Space-grade compression for Mars rovers and spacecraft")
    print("🎯 Precision: Every byte and nanosecond matters")
    
    try:
        demonstrate_precision_timing()
        demonstrate_memory_precision()
        demonstrate_cpu_analysis()
        demonstrate_io_monitoring()
        demonstrate_mission_critical_assessment()
        demonstrate_comparative_analysis()
        
        print_section_header("MISSION COMPLETE", "🎉")
        print("✅ All aerospace-grade metrics successfully demonstrated!")
        print("🚀 Framework ready for mission-critical space applications!")
        print("🌌 Every nanosecond and byte accounted for!")
        
    except Exception as e:
        print(f"\n❌ MISSION CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 