// Package core provides the foundational components for advanced statistical compression framework.
//
// This package defines the core interfaces and structures with precision measurement
// capabilities suitable for comprehensive statistical analysis where every byte and nanosecond matters.
package core

import (
	"context"
	"fmt"
)

// CompressionType represents the type of compression algorithm
type CompressionType string

const (
	CompressionTypeLossless CompressionType = "lossless"
	CompressionTypeLossy    CompressionType = "lossy"
)

// AlgorithmCategory represents categories of compression algorithms for analysis
type AlgorithmCategory string

const (
	AlgorithmCategoryEntropyCoding AlgorithmCategory = "entropy_coding" // Huffman, Arithmetic
	AlgorithmCategoryDictionary    AlgorithmCategory = "dictionary"     // LZ77, LZ78, LZW
	AlgorithmCategoryStatistical   AlgorithmCategory = "statistical"    // PPM, Context modeling
	AlgorithmCategoryTransform     AlgorithmCategory = "transform"      // BWT, DCT, Wavelet
	AlgorithmCategoryPredictive    AlgorithmCategory = "predictive"     // Delta, Linear prediction
	AlgorithmCategoryRunLength     AlgorithmCategory = "run_length"     // RLE variants
	AlgorithmCategoryHybrid        AlgorithmCategory = "hybrid"         // Combined approaches
)

// StatisticalPrecisionMetrics contains advanced precision metrics for comprehensive statistical analysis
type StatisticalPrecisionMetrics struct {
	// Nanosecond-precision timing
	CompressionTimeNs   int64 `json:"compression_time_ns"`
	DecompressionTimeNs int64 `json:"decompression_time_ns"`
	TotalTimeNs         int64 `json:"total_time_ns"`

	// Formatted time strings for readability
	CompressionTimeFormatted   string `json:"compression_time_formatted"`
	DecompressionTimeFormatted string `json:"decompression_time_formatted"`
	TotalTimeFormatted         string `json:"total_time_formatted"`

	// Detailed memory metrics (byte precision)
	MemoryPeakBytes   int64   `json:"memory_peak_bytes"`
	MemoryDeltaBytes  int64   `json:"memory_delta_bytes"`
	MemoryBeforeBytes int64   `json:"memory_before_bytes"`
	MemoryAfterBytes  int64   `json:"memory_after_bytes"`
	MemoryAllocMB     float64 `json:"memory_alloc_mb"`
	MemorySystemMB    float64 `json:"memory_system_mb"`

	// CPU utilization metrics
	CPUPercentAvg  float64 `json:"cpu_percent_avg"`
	CPUPercentPeak float64 `json:"cpu_percent_peak"`
	CPUTimeUserS   float64 `json:"cpu_time_user_s"`
	CPUTimeSystemS float64 `json:"cpu_time_system_s"`
	CPUFreqAvgMHz  float64 `json:"cpu_freq_avg_mhz"`

	// System resource metrics
	IOReadBytes     int64   `json:"io_read_bytes"`
	IOWriteBytes    int64   `json:"io_write_bytes"`
	IOReadOps       int64   `json:"io_read_ops"`
	IOWriteOps      int64   `json:"io_write_ops"`
	PageFaults      int64   `json:"page_faults"`
	ContextSwitches int64   `json:"context_switches"`
	GCCollections   []int64 `json:"gc_collections"`
	ThreadsCount    int     `json:"threads_count"`
	FileDescriptors int     `json:"file_descriptors"`

	// Performance efficiency metrics
	ThroughputMBPS           float64 `json:"throughput_mbps"`
	ThroughputBytesPerSecond float64 `json:"throughput_bytes_per_second"`
	TimePerByteNs            float64 `json:"time_per_byte_ns"`
	BytesPerCPUCycle         float64 `json:"bytes_per_cpu_cycle"`
	MemoryEfficiencyRatio    float64 `json:"memory_efficiency_ratio"`

	// Advanced compression metrics for statistical analysis
	BitsPerByte                float64 `json:"bits_per_byte"`
	EntropyEfficiency          float64 `json:"entropy_efficiency"` // Ratio to theoretical maximum
	EnergyEfficiencyBytesPerNs float64 `json:"energy_efficiency_bytes_per_ns"`
	WorstCaseLatencyNs         int64   `json:"worst_case_latency_ns"`
	DeterminismScore           float64 `json:"determinism_score"` // 1.0 = perfectly deterministic

	// Resource overhead metrics
	MemoryOverheadRatio            float64 `json:"memory_overhead_ratio"` // peak_memory / data_size
	CPUEfficiencyBytesPerCPUSecond float64 `json:"cpu_efficiency_bytes_per_cpu_second"`
	IOEfficiencyRatio              float64 `json:"io_efficiency_ratio"`
}

// UpdateFormattedTimes updates formatted time strings based on nanosecond values
func (m *StatisticalPrecisionMetrics) UpdateFormattedTimes() {
	m.CompressionTimeFormatted = FormatTimePrecision(m.CompressionTimeNs)
	m.DecompressionTimeFormatted = FormatTimePrecision(m.DecompressionTimeNs)
	m.TotalTimeFormatted = FormatTimePrecision(m.TotalTimeNs)
}

// CompressionResult represents the result from a compression operation with advanced statistical metrics
type CompressionResult struct {
	CompressedData   []byte                 `json:"compressed_data"`
	OriginalSize     int64                  `json:"original_size"`
	CompressedSize   int64                  `json:"compressed_size"`
	CompressionRatio float64                `json:"compression_ratio"`
	CompressionTime  float64                `json:"compression_time"` // Legacy compatibility (seconds)
	AlgorithmName    string                 `json:"algorithm_name"`
	Metadata         map[string]interface{} `json:"metadata"`

	// Enhanced statistical metrics
	PrecisionMetrics StatisticalPrecisionMetrics `json:"precision_metrics"`
}

// CompressionPercentage calculates compression percentage (space saved)
func (r *CompressionResult) CompressionPercentage() float64 {
	if r.OriginalSize == 0 {
		return 0.0
	}
	return (1.0 - float64(r.CompressedSize)/float64(r.OriginalSize)) * 100.0
}

// SpaceSavingsBytes calculates absolute space savings in bytes
func (r *CompressionResult) SpaceSavingsBytes() int64 {
	if r.CompressedSize > r.OriginalSize {
		return 0
	}
	return r.OriginalSize - r.CompressedSize
}

// IsEffective checks if compression was effective (ratio > 1.0)
func (r *CompressionResult) IsEffective() bool {
	return r.CompressionRatio > 1.0
}

// ToMissionReport generates a comprehensive mission-critical report
func (r *CompressionResult) ToMissionReport() map[string]interface{} {
	effectiveness := "NEGATIVE"
	if r.IsEffective() {
		effectiveness = "POSITIVE"
	}

	suitableForRealtime := r.PrecisionMetrics.WorstCaseLatencyNs < 1_000_000_000 // < 1 second
	memoryConstrainedSafe := r.PrecisionMetrics.MemoryOverheadRatio < 2.0

	return map[string]interface{}{
		"algorithm": r.AlgorithmName,
		"data_integrity": map[string]interface{}{
			"original_size_bytes":   r.OriginalSize,
			"compressed_size_bytes": r.CompressedSize,
			"compression_ratio":     fmt.Sprintf("%.6fx", r.CompressionRatio),
			"space_savings_bytes":   r.SpaceSavingsBytes(),
			"space_savings_percent": fmt.Sprintf("%.3f%%", r.CompressionPercentage()),
			"effectiveness":         effectiveness,
		},
		"performance_profile": map[string]interface{}{
			"compression_time":    r.PrecisionMetrics.CompressionTimeFormatted,
			"compression_time_ns": r.PrecisionMetrics.CompressionTimeNs,
			"throughput_mbps":     fmt.Sprintf("%.6f", r.PrecisionMetrics.ThroughputMBPS),
			"time_per_byte":       fmt.Sprintf("%.2fns", r.PrecisionMetrics.TimePerByteNs),
			"cpu_efficiency":      fmt.Sprintf("%.2f bytes/cpu-sec", r.PrecisionMetrics.CPUEfficiencyBytesPerCPUSecond),
		},
		"resource_utilization": map[string]interface{}{
			"peak_memory":           fmt.Sprintf("%d bytes", r.PrecisionMetrics.MemoryPeakBytes),
			"memory_overhead_ratio": fmt.Sprintf("%.4f", r.PrecisionMetrics.MemoryOverheadRatio),
			"cpu_peak_percent":      fmt.Sprintf("%.2f%%", r.PrecisionMetrics.CPUPercentPeak),
			"io_operations":         fmt.Sprintf("R:%d W:%d", r.PrecisionMetrics.IOReadOps, r.PrecisionMetrics.IOWriteOps),
			"determinism_score":     fmt.Sprintf("%.6f", r.PrecisionMetrics.DeterminismScore),
		},
		"mission_readiness": map[string]interface{}{
			"worst_case_latency":      r.PrecisionMetrics.WorstCaseLatencyNs,
			"energy_efficiency":       fmt.Sprintf("%.2e bytes/ns", r.PrecisionMetrics.EnergyEfficiencyBytesPerNs),
			"entropy_efficiency":      fmt.Sprintf("%.4f", r.PrecisionMetrics.EntropyEfficiency),
			"suitable_for_realtime":   suitableForRealtime,
			"memory_constrained_safe": memoryConstrainedSafe,
		},
	}
}

// DecompressionResult represents the result from a decompression operation with advanced statistical metrics
type DecompressionResult struct {
	DecompressedData       []byte                 `json:"decompressed_data"`
	OriginalCompressedSize int64                  `json:"original_compressed_size"`
	DecompressedSize       int64                  `json:"decompressed_size"`
	DecompressionTime      float64                `json:"decompression_time"` // Legacy compatibility
	AlgorithmName          string                 `json:"algorithm_name"`
	Metadata               map[string]interface{} `json:"metadata"`

	// Enhanced statistical metrics
	PrecisionMetrics StatisticalPrecisionMetrics `json:"precision_metrics"`
}

// ExpansionRatio calculates expansion ratio during decompression
func (r *DecompressionResult) ExpansionRatio() float64 {
	if r.OriginalCompressedSize == 0 {
		return 0.0
	}
	return float64(r.DecompressedSize) / float64(r.OriginalCompressedSize)
}

// VerifyIntegrity verifies data integrity after decompression
func (r *DecompressionResult) VerifyIntegrity(originalData []byte) bool {
	if len(r.DecompressedData) != len(originalData) {
		return false
	}

	for i, b := range r.DecompressedData {
		if b != originalData[i] {
			return false
		}
	}
	return true
}

// ToMissionReport generates a comprehensive decompression mission report
func (r *DecompressionResult) ToMissionReport() map[string]interface{} {
	return map[string]interface{}{
		"algorithm": r.AlgorithmName,
		"decompression_profile": map[string]interface{}{
			"compressed_size_bytes":   r.OriginalCompressedSize,
			"decompressed_size_bytes": r.DecompressedSize,
			"expansion_ratio":         fmt.Sprintf("%.6fx", r.ExpansionRatio()),
			"decompression_time":      r.PrecisionMetrics.DecompressionTimeFormatted,
			"throughput_mbps":         fmt.Sprintf("%.6f", r.PrecisionMetrics.ThroughputMBPS),
		},
		"resource_utilization": map[string]interface{}{
			"peak_memory":       fmt.Sprintf("%d bytes", r.PrecisionMetrics.MemoryPeakBytes),
			"cpu_utilization":   fmt.Sprintf("%.2f%%", r.PrecisionMetrics.CPUPercentAvg),
			"determinism_score": fmt.Sprintf("%.6f", r.PrecisionMetrics.DeterminismScore),
		},
	}
}

// PerformanceMetrics contains comprehensive performance metrics for advanced statistical analysis
type PerformanceMetrics struct {
	CompressionTimeNs     int64   `json:"compression_time_ns"`
	DecompressionTimeNs   int64   `json:"decompression_time_ns"`
	PeakMemoryBytes       int64   `json:"peak_memory_bytes"`
	AvgMemoryBytes        int64   `json:"avg_memory_bytes"`
	CPUUtilizationPercent float64 `json:"cpu_utilization_percent"`
	ThroughputMBPS        float64 `json:"throughput_mbps"`
	EnergyEfficiency      float64 `json:"energy_efficiency"`
	DeterminismScore      float64 `json:"determinism_score"`
}

// Statistical analysis thresholds
const (
	RealtimeThresholdNs       = 1_000_000_000 // 1 second
	MemoryEfficiencyThreshold = 2.0           // 2x data size max
	CPUEfficiencyThreshold    = 80.0          // 80% max CPU
)

// IsRealtimeSuitable checks if performance is suitable for real-time applications
func (p *PerformanceMetrics) IsRealtimeSuitable() bool {
	totalTime := p.CompressionTimeNs + p.DecompressionTimeNs
	return totalTime < RealtimeThresholdNs
}

// IsMemoryEfficient checks if memory usage is within acceptable limits
func (p *PerformanceMetrics) IsMemoryEfficient() bool {
	return float64(p.PeakMemoryBytes) < float64(p.AvgMemoryBytes)*MemoryEfficiencyThreshold
}

// MissionReadinessScore calculates overall mission readiness score (0.0 - 1.0)
func (p *PerformanceMetrics) MissionReadinessScore() float64 {
	var scores []float64

	// Time efficiency (lower is better)
	timeScore := max(0.0, 1.0-float64(p.CompressionTimeNs+p.DecompressionTimeNs)/float64(RealtimeThresholdNs))
	scores = append(scores, timeScore)

	// Memory efficiency
	memoryScore := 1.0
	if !p.IsMemoryEfficient() {
		memoryScore = 0.5
	}
	scores = append(scores, memoryScore)

	// CPU efficiency
	cpuScore := max(0.0, 1.0-p.CPUUtilizationPercent/100.0)
	scores = append(scores, cpuScore)

	// Determinism (higher is better)
	scores = append(scores, p.DeterminismScore)

	sum := 0.0
	for _, score := range scores {
		sum += score
	}
	return sum / float64(len(scores))
}

// CompressionAlgorithm defines the interface for compression algorithms with aerospace-grade monitoring
type CompressionAlgorithm interface {
	// Compress data with aerospace-grade performance monitoring
	Compress(ctx context.Context, data []byte) (*CompressionResult, error)

	// Decompress data with aerospace-grade performance monitoring
	Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*DecompressionResult, error)

	// GetName returns the algorithm name
	GetName() string

	// GetCategory returns the algorithm category
	GetCategory() AlgorithmCategory

	// GetCompressionType returns the compression type
	GetCompressionType() CompressionType

	// SetParameters sets algorithm parameters
	SetParameters(params map[string]interface{}) error

	// GetParameters gets current algorithm parameters
	GetParameters() map[string]interface{}

	// GetInfo gets comprehensive algorithm information
	GetInfo() map[string]interface{}
}

// PipelineComponent defines the interface for pipeline components
type PipelineComponent interface {
	// Process data through this pipeline component
	Process(ctx context.Context, data []byte, metadata map[string]interface{}) ([]byte, map[string]interface{}, error)

	// ReverseProcess reverses the processing operation
	ReverseProcess(ctx context.Context, data []byte, metadata map[string]interface{}) ([]byte, error)
}

// BenchmarkMetric defines the interface for benchmark metrics
type BenchmarkMetric interface {
	// Calculate metric value from result
	Calculate(result interface{}) (float64, error)

	// GetName returns metric name
	GetName() string

	// GetUnit returns metric unit
	GetUnit() string
}

// DatasetProvider defines the interface for dataset providers
type DatasetProvider interface {
	// GetFiles returns list of available files
	GetFiles() ([]string, error)

	// LoadFile loads file content
	LoadFile(filePath string) ([]byte, error)
}

// ResultsExporter defines the interface for results exporters
type ResultsExporter interface {
	// Export results to specified format
	Export(results []map[string]interface{}, outputPath string) error
}

// CompressionError represents compression operation errors
type CompressionError struct {
	Message      string                 `json:"message"`
	Algorithm    string                 `json:"algorithm,omitempty"`
	DataSize     int64                  `json:"data_size,omitempty"`
	ErrorContext map[string]interface{} `json:"error_context,omitempty"`
	TimestampNs  int64                  `json:"timestamp_ns"`
}

func (e *CompressionError) Error() string {
	return e.Message
}

// ToMissionReport generates mission-critical error report
func (e *CompressionError) ToMissionReport() map[string]interface{} {
	criticality := "MEDIUM"
	if e.DataSize > 1024*1024 { // > 1MB
		criticality = "HIGH"
	}

	return map[string]interface{}{
		"error_type":      "COMPRESSION_FAILURE",
		"message":         e.Message,
		"algorithm":       e.Algorithm,
		"data_size_bytes": e.DataSize,
		"timestamp_ns":    e.TimestampNs,
		"context":         e.ErrorContext,
		"criticality":     criticality,
	}
}

// DecompressionError represents decompression operation errors
type DecompressionError struct {
	Message        string                 `json:"message"`
	Algorithm      string                 `json:"algorithm,omitempty"`
	CompressedSize int64                  `json:"compressed_size,omitempty"`
	ErrorContext   map[string]interface{} `json:"error_context,omitempty"`
	TimestampNs    int64                  `json:"timestamp_ns"`
}

func (e *DecompressionError) Error() string {
	return e.Message
}

// ToMissionReport generates mission-critical error report
func (e *DecompressionError) ToMissionReport() map[string]interface{} {
	return map[string]interface{}{
		"error_type":            "DECOMPRESSION_FAILURE",
		"message":               e.Message,
		"algorithm":             e.Algorithm,
		"compressed_size_bytes": e.CompressedSize,
		"timestamp_ns":          e.TimestampNs,
		"context":               e.ErrorContext,
		"criticality":           "CRITICAL", // Decompression failures are always critical
	}
}

// PipelineError represents pipeline operation errors
type PipelineError struct {
	Message      string                 `json:"message"`
	StageName    string                 `json:"stage_name,omitempty"`
	StageIndex   int                    `json:"stage_index,omitempty"`
	PipelineName string                 `json:"pipeline_name,omitempty"`
	ErrorContext map[string]interface{} `json:"error_context,omitempty"`
	TimestampNs  int64                  `json:"timestamp_ns"`
}

func (e *PipelineError) Error() string {
	return e.Message
}

// ToMissionReport generates mission-critical pipeline error report
func (e *PipelineError) ToMissionReport() map[string]interface{} {
	return map[string]interface{}{
		"error_type":    "PIPELINE_FAILURE",
		"message":       e.Message,
		"pipeline_name": e.PipelineName,
		"failed_stage":  e.StageName,
		"stage_index":   e.StageIndex,
		"timestamp_ns":  e.TimestampNs,
		"context":       e.ErrorContext,
		"criticality":   "HIGH",
	}
}

// FormatTimePrecision formats time with appropriate precision for aerospace applications
func FormatTimePrecision(durationNs int64) string {
	if durationNs < 1_000 { // Less than 1 microsecond
		return fmt.Sprintf("%dns", durationNs)
	} else if durationNs < 1_000_000 { // Less than 1 millisecond
		return fmt.Sprintf("%.2fμs", float64(durationNs)/1_000)
	} else if durationNs < 1_000_000_000 { // Less than 1 second
		return fmt.Sprintf("%.3fms", float64(durationNs)/1_000_000)
	} else {
		return fmt.Sprintf("%.6fs", float64(durationNs)/1_000_000_000)
	}
}

// FormatMemoryPrecision formats memory with appropriate precision for space applications
func FormatMemoryPrecision(bytesValue int64) string {
	if bytesValue == 0 {
		return "0B"
	} else if bytesValue < 1024 {
		return fmt.Sprintf("%dB", bytesValue)
	} else if bytesValue < 1024*1024 {
		return fmt.Sprintf("%.2fKB", float64(bytesValue)/1024)
	} else if bytesValue < 1024*1024*1024 {
		return fmt.Sprintf("%.3fMB", float64(bytesValue)/(1024*1024))
	} else {
		return fmt.Sprintf("%.6fGB", float64(bytesValue)/(1024*1024*1024))
	}
}

// Helper function for max
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
