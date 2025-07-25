// Package algorithms provides BWT (Burrows-Wheeler Transform) implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// BWTEncoder implements Burrows-Wheeler Transform with aerospace-grade precision monitoring
type BWTEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// BWT-specific parameters
	endMarker string
}

// NewBWTEncoder creates a new BWT encoder
func NewBWTEncoder() (*BWTEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	return &BWTEncoder{
		name:            "BWT",
		category:        core.AlgorithmCategoryTransform,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		endMarker:       "$", // Default end marker
	}, nil
}

// GetName returns the algorithm name
func (b *BWTEncoder) GetName() string {
	return b.name
}

// GetCategory returns the algorithm category
func (b *BWTEncoder) GetCategory() core.AlgorithmCategory {
	return b.category
}

// GetCompressionType returns the compression type
func (b *BWTEncoder) GetCompressionType() core.CompressionType {
	return b.compressionType
}

// SetParameters sets algorithm parameters
func (b *BWTEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		b.parameters[k] = v

		switch k {
		case "end_marker":
			if val, ok := v.(string); ok && len(val) == 1 {
				b.endMarker = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (b *BWTEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range b.parameters {
		result[k] = v
	}
	result["end_marker"] = b.endMarker
	return result
}

// GetInfo gets comprehensive algorithm information
func (b *BWTEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 b.name,
		"category":             string(b.category),
		"compression_type":     string(b.compressionType),
		"parameters":           b.GetParameters(),
		"supports_streaming":   false,
		"thread_safe":          false,
		"deterministic":        true,
		"memory_bounded":       false, // BWT can use O(n²) memory in worst case
		"end_marker":           b.endMarker,
		"algorithm_complexity": "O(n²) time, O(n²) space worst case",
		"best_use_case":        "Text data preprocessing for better compression",
	}
}

// performBWT performs the actual Burrows-Wheeler Transform
func (b *BWTEncoder) performBWT(input []byte) ([]byte, int) {
	if len(input) == 0 {
		return input, 0
	}

	// Convert to string and add end marker
	str := string(input) + b.endMarker
	n := len(str)

	// Generate all rotations
	rotations := make([]string, n)
	for i := 0; i < n; i++ {
		rotations[i] = str[i:] + str[:i]
	}

	// Sort rotations lexicographically
	sort.Strings(rotations)

	// Extract the last column and find the original string index
	lastColumn := make([]byte, n)
	originalIndex := 0

	for i, rotation := range rotations {
		lastColumn[i] = rotation[n-1]
		if rotation == str {
			originalIndex = i
		}
	}

	return lastColumn, originalIndex
}

// performInverseBWT performs the inverse Burrows-Wheeler Transform
func (b *BWTEncoder) performInverseBWT(lastColumn []byte, originalIndex int) []byte {
	if len(lastColumn) == 0 {
		return lastColumn
	}

	n := len(lastColumn)

	// Create first column by sorting last column
	firstColumn := make([]byte, n)
	copy(firstColumn, lastColumn)
	sort.Slice(firstColumn, func(i, j int) bool {
		return firstColumn[i] < firstColumn[j]
	})

	// Create transformation array
	transform := make([]int, n)
	count := make(map[byte]int)

	// Build rank arrays
	firstRank := make([]int, n)
	lastRank := make([]int, n)

	for i := 0; i < n; i++ {
		firstRank[i] = count[firstColumn[i]]
		count[firstColumn[i]]++
	}

	count = make(map[byte]int)
	for i := 0; i < n; i++ {
		lastRank[i] = count[lastColumn[i]]
		count[lastColumn[i]]++
	}

	// Build transformation array
	for i := 0; i < n; i++ {
		char := lastColumn[i]
		rank := lastRank[i]

		// Find corresponding position in first column
		for j := 0; j < n; j++ {
			if firstColumn[j] == char && firstRank[j] == rank {
				transform[i] = j
				break
			}
		}
	}

	// Reconstruct original string
	result := make([]byte, 0, n-1)
	pos := originalIndex

	for i := 0; i < n-1; i++ {
		pos = transform[pos]
		if firstColumn[pos] != byte(b.endMarker[0]) {
			result = append(result, firstColumn[pos])
		}
	}

	return result
}

// Compress compresses data using BWT with aerospace-grade performance monitoring
func (b *BWTEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    b.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var transformedData []byte
	var originalIndex int
	var rotationsGenerated int

	profile, err := b.monitor.ProfileOperation(ctx, "bwt_compress", int64(len(data)), func() error {
		transformedData, originalIndex = b.performBWT(data)
		rotationsGenerated = len(data) + 1 // +1 for end marker
		return nil
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("BWT compression failed: %v", err),
			Algorithm:   b.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}

	// Encode the result: [original_index(4 bytes)][transformed_data]
	compressedData := make([]byte, 4+len(transformedData))

	// Store original index (big-endian)
	compressedData[0] = byte(originalIndex >> 24)
	compressedData[1] = byte(originalIndex >> 16)
	compressedData[2] = byte(originalIndex >> 8)
	compressedData[3] = byte(originalIndex)

	// Store transformed data
	copy(compressedData[4:], transformedData)

	// BWT usually increases size but improves compressibility for subsequent algorithms
	compressionRatio := float64(len(data)) / float64(len(compressedData))

	// Convert profile to aerospace precision metrics
	precisionMetrics := b.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))

	metadata := map[string]interface{}{
		"original_index":      originalIndex,
		"rotations_generated": rotationsGenerated,
		"end_marker":          b.endMarker,
		"transformed_size":    len(transformedData),
		"overhead_bytes":      4,                           // 4 bytes for original index
		"memory_usage_factor": float64(rotationsGenerated), // Approximation
		"preprocessing_stage": true,
	}

	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    b.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses BWT-transformed data with aerospace-grade performance monitoring
func (b *BWTEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          b.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.StatisticalPrecisionMetrics{},
		}, nil
	}

	if len(compressedData) < 4 {
		return nil, &core.DecompressionError{
			Message:        "compressed data too short for BWT",
			Algorithm:      b.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	var decompressedData []byte

	profile, err := b.monitor.ProfileOperation(ctx, "bwt_decompress", int64(len(compressedData)), func() error {
		// Extract original index (big-endian)
		originalIndex := int(compressedData[0])<<24 |
			int(compressedData[1])<<16 |
			int(compressedData[2])<<8 |
			int(compressedData[3])

		// Extract transformed data
		transformedData := compressedData[4:]

		// Perform inverse BWT
		decompressedData = b.performInverseBWT(transformedData, originalIndex)
		return nil
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("BWT decompression failed: %v", err),
			Algorithm:      b.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	// Convert profile to aerospace precision metrics
	precisionMetrics := b.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))

	decompMetadata := map[string]interface{}{
		"original_compressed_size": len(compressedData),
		"decompressed_size":        len(decompressedData),
		"algorithm":                b.name,
		"preprocessing_reverse":    true,
	}

	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          b.name,
		Metadata:               decompMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (b *BWTEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
	metrics := core.StatisticalPrecisionMetrics{
		CompressionTimeNs:          profile.DurationNs,
		DecompressionTimeNs:        0, // Not available in compression profile
		TotalTimeNs:                profile.DurationNs,
		CompressionTimeFormatted:   core.FormatTimePrecision(profile.DurationNs),
		DecompressionTimeFormatted: "N/A",
		TotalTimeFormatted:         core.FormatTimePrecision(profile.DurationNs),

		// Memory metrics
		MemoryPeakBytes:   profile.MemoryPeakBytes,
		MemoryDeltaBytes:  profile.MemoryDeltaBytes,
		MemoryBeforeBytes: profile.MemoryBeforeBytes,
		MemoryAfterBytes:  profile.MemoryAfterBytes,
		MemoryAllocMB:     profile.RuntimeAllocMB,
		MemorySystemMB:    profile.RuntimeSysMB,

		// CPU metrics
		CPUPercentAvg:  profile.CPUPercentAvg,
		CPUPercentPeak: profile.CPUPercentPeak,
		CPUTimeUserS:   profile.CPUTimeUserS,
		CPUTimeSystemS: profile.CPUTimeSystemS,
		CPUFreqAvgMHz:  profile.CPUFreqAvgMHz,

		// I/O metrics
		IOReadBytes:  profile.IOReadBytes,
		IOWriteBytes: profile.IOWriteBytes,
		IOReadOps:    profile.IOReadOps,
		IOWriteOps:   profile.IOWriteOps,

		// System metrics
		PageFaults:      profile.PageFaults,
		ContextSwitches: profile.ContextSwitches,
		GCCollections:   profile.GCCollections,

		// Performance metrics
		ThroughputMBPS:           profile.ThroughputMBPS,
		ThroughputBytesPerSecond: profile.ThroughputMBPS * 1024 * 1024,
		TimePerByteNs:            profile.TimePerByteNs,
		BytesPerCPUCycle:         profile.BytesPerCPUCycle,
		MemoryEfficiencyRatio:    profile.MemoryEfficiencyRatio,

		// BWT-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // BWT is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}
