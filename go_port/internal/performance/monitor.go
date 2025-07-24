// Package performance provides aerospace-grade performance monitoring utilities.
//
// This package provides comprehensive performance monitoring capabilities
// suitable for mission-critical applications where every byte and nanosecond matters.
package performance

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

// DetailedResourceSnapshot represents comprehensive system resource snapshot with aerospace-grade precision
type DetailedResourceSnapshot struct {
	TimestampNs         int64   `json:"timestamp_ns"`          // Nanosecond precision timestamp
	CPUPercent          float64 `json:"cpu_percent"`           // CPU usage percentage
	CPUFreqMHz          float64 `json:"cpu_freq_mhz"`          // CPU frequency in MHz
	MemoryRSSBytes      int64   `json:"memory_rss_bytes"`      // Resident Set Size in bytes
	MemoryVMSBytes      int64   `json:"memory_vms_bytes"`      // Virtual Memory Size in bytes
	MemoryPercent       float64 `json:"memory_percent"`        // Memory usage percentage
	MemoryAvailableBytes int64   `json:"memory_available_bytes"` // Available memory in bytes
	MemoryCachedBytes   int64   `json:"memory_cached_bytes"`   // Cached memory in bytes
	MemorySharedBytes   int64   `json:"memory_shared_bytes"`   // Shared memory in bytes
	PageFaults          int64   `json:"page_faults"`           // Number of page faults
	ContextSwitches     int64   `json:"context_switches"`      // Number of context switches
	IOReadBytes         int64   `json:"io_read_bytes"`         // Bytes read from disk
	IOWriteBytes        int64   `json:"io_write_bytes"`        // Bytes written to disk
	IOReadCount         int64   `json:"io_read_count"`         // Number of read operations
	IOWriteCount        int64   `json:"io_write_count"`        // Number of write operations
	ThreadsCount        int32   `json:"threads_count"`         // Number of threads
	FileDescriptors     int32   `json:"file_descriptors"`      // Number of open file descriptors
	GCCollections       []int64 `json:"gc_collections"`        // Garbage collection counts
}

// PrecisionPerformanceProfile contains aerospace-grade performance profile with nanosecond precision
type PrecisionPerformanceProfile struct {
	OperationName string `json:"operation_name"`
	DataSizeBytes int64  `json:"data_size_bytes"`
	
	// Timing metrics (nanosecond precision)
	StartTimeNs int64   `json:"start_time_ns"`
	EndTimeNs   int64   `json:"end_time_ns"`
	DurationNs  int64   `json:"duration_ns"`
	DurationUs  float64 `json:"duration_us"` // Microseconds
	DurationMs  float64 `json:"duration_ms"` // Milliseconds
	DurationS   float64 `json:"duration_s"`  // Seconds
	
	// Memory metrics (byte precision)
	MemoryBeforeBytes     int64   `json:"memory_before_bytes"`
	MemoryAfterBytes      int64   `json:"memory_after_bytes"`
	MemoryPeakBytes       int64   `json:"memory_peak_bytes"`
	MemoryDeltaBytes      int64   `json:"memory_delta_bytes"`
	MemoryPeakDeltaBytes  int64   `json:"memory_peak_delta_bytes"`
	
	// Detailed memory breakdown
	RuntimeAllocMB    float64 `json:"runtime_alloc_mb"`
	RuntimeSysMB      float64 `json:"runtime_sys_mb"`
	RuntimeHeapMB     float64 `json:"runtime_heap_mb"`
	RuntimeStackMB    float64 `json:"runtime_stack_mb"`
	
	// CPU metrics
	CPUPercentAvg  float64 `json:"cpu_percent_avg"`
	CPUPercentPeak float64 `json:"cpu_percent_peak"`
	CPUTimeUserS   float64 `json:"cpu_time_user_s"`
	CPUTimeSystemS float64 `json:"cpu_time_system_s"`
	CPUFreqAvgMHz  float64 `json:"cpu_freq_avg_mhz"`
	
	// I/O metrics
	IOReadBytes    int64   `json:"io_read_bytes"`
	IOWriteBytes   int64   `json:"io_write_bytes"`
	IOReadOps      int64   `json:"io_read_ops"`
	IOWriteOps     int64   `json:"io_write_ops"`
	IOReadTimeMs   float64 `json:"io_read_time_ms"`
	IOWriteTimeMs  float64 `json:"io_write_time_ms"`
	
	// System metrics
	PageFaults      int64   `json:"page_faults"`
	ContextSwitches int64   `json:"context_switches"`
	ThreadsCreated  int32   `json:"threads_created"`
	GCCollections   []int64 `json:"gc_collections"`
	
	// Performance efficiency metrics
	ThroughputMBPS        float64 `json:"throughput_mbps"`
	ThroughputOpsPerSec   float64 `json:"throughput_ops_per_sec"`
	BytesPerCPUCycle      float64 `json:"bytes_per_cpu_cycle"`
	MemoryEfficiencyRatio float64 `json:"memory_efficiency_ratio"` // data_size / peak_memory
	
	// Compression-specific metrics
	CompressionEfficiency float64 `json:"compression_efficiency"` // bytes_saved / processing_time
	TimePerByteNs         float64 `json:"time_per_byte_ns"`       // processing_time / data_size
	
	// Samples collected during monitoring
	Samples []DetailedResourceSnapshot `json:"samples"`
}

// CalculateDerivedMetrics calculates derived performance metrics
func (p *PrecisionPerformanceProfile) CalculateDerivedMetrics() {
	if p.DurationS > 0 {
		p.ThroughputMBPS = (float64(p.DataSizeBytes) / (1024 * 1024)) / p.DurationS
		p.ThroughputOpsPerSec = 1.0 / p.DurationS
		p.TimePerByteNs = float64(p.DurationNs) / max(1, float64(p.DataSizeBytes))
	}
	
	if p.MemoryPeakBytes > 0 {
		p.MemoryEfficiencyRatio = float64(p.DataSizeBytes) / float64(p.MemoryPeakBytes)
	}
	
	// Estimate CPU cycles (rough approximation)
	if p.CPUFreqAvgMHz > 0 && p.DurationS > 0 {
		estimatedCycles := p.CPUFreqAvgMHz * 1_000_000 * p.DurationS
		p.BytesPerCPUCycle = float64(p.DataSizeBytes) / max(1, estimatedCycles)
	}
	
	// Calculate timing conversions
	p.DurationUs = float64(p.DurationNs) / 1_000
	p.DurationMs = float64(p.DurationNs) / 1_000_000
	p.DurationS = float64(p.DurationNs) / 1_000_000_000
}

// AerospaceGradeMonitor provides aerospace-grade performance monitoring with nanosecond precision
type AerospaceGradeMonitor struct {
	samplingIntervalMs float64
	monitoring         bool
	samples            []DetailedResourceSnapshot
	monitorMutex       sync.RWMutex
	process            *process.Process
}

// NewAerospaceGradeMonitor creates a new aerospace-grade monitor
func NewAerospaceGradeMonitor(samplingIntervalMs float64) (*AerospaceGradeMonitor, error) {
	proc, err := process.NewProcess(int32(runtime.GOMAXPROCS(0)))
	if err != nil {
		// Fallback to current process
		proc = nil
	}
	
	return &AerospaceGradeMonitor{
		samplingIntervalMs: samplingIntervalMs,
		monitoring:         false,
		samples:            make([]DetailedResourceSnapshot, 0, 10000), // Circular buffer
		process:            proc,
	}, nil
}

// GetDetailedSnapshot gets comprehensive system resource snapshot
func (m *AerospaceGradeMonitor) GetDetailedSnapshot() DetailedResourceSnapshot {
	snapshot := DetailedResourceSnapshot{
		TimestampNs: time.Now().UnixNano(),
	}
	
	// Get CPU information
	if cpuPercent, err := cpu.Percent(0, false); err == nil && len(cpuPercent) > 0 {
		snapshot.CPUPercent = cpuPercent[0]
	}
	
	if cpuInfo, err := cpu.Info(); err == nil && len(cpuInfo) > 0 {
		snapshot.CPUFreqMHz = cpuInfo[0].Mhz
	}
	
	// Get memory information
	if memInfo, err := mem.VirtualMemory(); err == nil {
		snapshot.MemoryPercent = memInfo.UsedPercent
		snapshot.MemoryAvailableBytes = int64(memInfo.Available)
		snapshot.MemoryCachedBytes = int64(memInfo.Cached)
		snapshot.MemorySharedBytes = int64(memInfo.Shared)
	}
	
	// Get process-specific information if available
	if m.process != nil {
		if memInfo, err := m.process.MemoryInfo(); err == nil {
			snapshot.MemoryRSSBytes = int64(memInfo.RSS)
			snapshot.MemoryVMSBytes = int64(memInfo.VMS)
		}
		
		if ioCounters, err := m.process.IOCounters(); err == nil {
			snapshot.IOReadBytes = int64(ioCounters.ReadBytes)
			snapshot.IOWriteBytes = int64(ioCounters.WriteBytes)
			snapshot.IOReadCount = int64(ioCounters.ReadCount)
			snapshot.IOWriteCount = int64(ioCounters.WriteCount)
		}
		
		if numThreads, err := m.process.NumThreads(); err == nil {
			snapshot.ThreadsCount = numThreads
		}
		
		if numFDs, err := m.process.NumFDs(); err == nil {
			snapshot.FileDescriptors = numFDs
		}
	}
	
	// Get Go runtime GC stats
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	snapshot.GCCollections = []int64{int64(memStats.NumGC)}
	
	return snapshot
}

// StartMonitoring starts continuous resource monitoring
func (m *AerospaceGradeMonitor) StartMonitoring(ctx context.Context) {
	m.monitorMutex.Lock()
	if m.monitoring {
		m.monitorMutex.Unlock()
		return
	}
	m.monitoring = true
	m.samples = m.samples[:0] // Clear existing samples
	m.monitorMutex.Unlock()
	
	go func() {
		interval := time.Duration(m.samplingIntervalMs * float64(time.Millisecond))
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				m.monitorMutex.Lock()
				m.monitoring = false
				m.monitorMutex.Unlock()
				return
			case <-ticker.C:
				m.monitorMutex.RLock()
				if !m.monitoring {
					m.monitorMutex.RUnlock()
					return
				}
				m.monitorMutex.RUnlock()
				
				snapshot := m.GetDetailedSnapshot()
				
				m.monitorMutex.Lock()
				// Keep only the last 10000 samples (circular buffer)
				if len(m.samples) >= 10000 {
					copy(m.samples, m.samples[1:])
					m.samples = m.samples[:9999]
				}
				m.samples = append(m.samples, snapshot)
				m.monitorMutex.Unlock()
			}
		}
	}()
}

// StopMonitoring stops continuous resource monitoring
func (m *AerospaceGradeMonitor) StopMonitoring() {
	m.monitorMutex.Lock()
	m.monitoring = false
	m.monitorMutex.Unlock()
}

// ProfileOperation profiles an operation with aerospace-grade precision
func (m *AerospaceGradeMonitor) ProfileOperation(ctx context.Context, operationName string, dataSizeBytes int64, operation func() error) (*PrecisionPerformanceProfile, error) {
	// Pre-operation setup
	runtime.GC() // Clean slate for memory measurements
	
	// Get initial state
	startSnapshot := m.GetDetailedSnapshot()
	
	// Runtime memory tracking
	var startMemStats, endMemStats runtime.MemStats
	runtime.ReadMemStats(&startMemStats)
	
	// High-precision timing
	startTimeNs := time.Now().UnixNano()
	
	// Start monitoring if not already running
	wasMonitoring := m.monitoring
	if !wasMonitoring {
		m.StartMonitoring(ctx)
	}
	
	// Execute the operation
	operationErr := operation()
	
	// Post-operation measurements
	endTimeNs := time.Now().UnixNano()
	endSnapshot := m.GetDetailedSnapshot()
	runtime.ReadMemStats(&endMemStats)
	
	// Calculate timing metrics
	durationNs := endTimeNs - startTimeNs
	
	// Calculate memory metrics
	memoryDelta := endSnapshot.MemoryRSSBytes - startSnapshot.MemoryRSSBytes
	
	// Get monitoring samples from the operation period
	m.monitorMutex.RLock()
	operationSamples := make([]DetailedResourceSnapshot, 0)
	for _, sample := range m.samples {
		if sample.TimestampNs >= startTimeNs && sample.TimestampNs <= endTimeNs {
			operationSamples = append(operationSamples, sample)
		}
	}
	m.monitorMutex.RUnlock()
	
	// Calculate aggregated metrics from samples
	var cpuValues []float64
	var memoryValues []int64
	var freqValues []float64
	
	for _, sample := range operationSamples {
		cpuValues = append(cpuValues, sample.CPUPercent)
		memoryValues = append(memoryValues, sample.MemoryRSSBytes)
		if sample.CPUFreqMHz > 0 {
			freqValues = append(freqValues, sample.CPUFreqMHz)
		}
	}
	
	// Calculate averages and peaks
	var cpuPercentAvg, cpuPercentPeak, cpuFreqAvgMHz float64
	var memoryPeakBytes int64
	
	if len(cpuValues) > 0 {
		cpuSum := 0.0
		cpuPercentPeak = cpuValues[0]
		for _, cpu := range cpuValues {
			cpuSum += cpu
			if cpu > cpuPercentPeak {
				cpuPercentPeak = cpu
			}
		}
		cpuPercentAvg = cpuSum / float64(len(cpuValues))
	}
	
	if len(memoryValues) > 0 {
		memoryPeakBytes = memoryValues[0]
		for _, mem := range memoryValues {
			if mem > memoryPeakBytes {
				memoryPeakBytes = mem
			}
		}
	}
	
	if len(freqValues) > 0 {
		freqSum := 0.0
		for _, freq := range freqValues {
			freqSum += freq
		}
		cpuFreqAvgMHz = freqSum / float64(len(freqValues))
	}
	
	// I/O metrics
	ioReadBytes := endSnapshot.IOReadBytes - startSnapshot.IOReadBytes
	ioWriteBytes := endSnapshot.IOWriteBytes - startSnapshot.IOWriteBytes
	ioReadOps := endSnapshot.IOReadCount - startSnapshot.IOReadCount
	ioWriteOps := endSnapshot.IOWriteCount - startSnapshot.IOWriteCount
	
	// GC metrics
	gcCollections := []int64{}
	if len(endSnapshot.GCCollections) > 0 && len(startSnapshot.GCCollections) > 0 {
		for i := 0; i < len(endSnapshot.GCCollections) && i < len(startSnapshot.GCCollections); i++ {
			gcCollections = append(gcCollections, endSnapshot.GCCollections[i]-startSnapshot.GCCollections[i])
		}
	}
	
	// Create performance profile
	profile := &PrecisionPerformanceProfile{
		OperationName: operationName,
		DataSizeBytes: dataSizeBytes,
		StartTimeNs:   startTimeNs,
		EndTimeNs:     endTimeNs,
		DurationNs:    durationNs,
		
		MemoryBeforeBytes:    startSnapshot.MemoryRSSBytes,
		MemoryAfterBytes:     endSnapshot.MemoryRSSBytes,
		MemoryPeakBytes:      memoryPeakBytes,
		MemoryDeltaBytes:     memoryDelta,
		MemoryPeakDeltaBytes: memoryPeakBytes - startSnapshot.MemoryRSSBytes,
		
		RuntimeAllocMB: float64(endMemStats.Alloc) / 1024 / 1024,
		RuntimeSysMB:   float64(endMemStats.Sys) / 1024 / 1024,
		RuntimeHeapMB:  float64(endMemStats.HeapAlloc) / 1024 / 1024,
		RuntimeStackMB: float64(endMemStats.StackInuse) / 1024 / 1024,
		
		CPUPercentAvg:  cpuPercentAvg,
		CPUPercentPeak: cpuPercentPeak,
		CPUFreqAvgMHz:  cpuFreqAvgMHz,
		
		IOReadBytes:  ioReadBytes,
		IOWriteBytes: ioWriteBytes,
		IOReadOps:    ioReadOps,
		IOWriteOps:   ioWriteOps,
		
		PageFaults:      endSnapshot.PageFaults - startSnapshot.PageFaults,
		ContextSwitches: endSnapshot.ContextSwitches - startSnapshot.ContextSwitches,
		GCCollections:   gcCollections,
		
		Samples: operationSamples,
	}
	
	// Calculate derived metrics
	profile.CalculateDerivedMetrics()
	
	// Stop monitoring if we started it
	if !wasMonitoring {
		m.StopMonitoring()
	}
	
	return profile, operationErr
}

// Helper function for max
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
} 