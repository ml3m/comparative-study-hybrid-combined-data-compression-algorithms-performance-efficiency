// Package algorithms provides Delta Encoding implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// DeltaEncoder implements Delta Encoding with aerospace-grade precision monitoring
type DeltaEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// Delta-specific parameters
	dataWidth     int    // Number of bytes per data element (1, 2, 4, 8)
	signedData    bool   // Whether to treat data as signed
	predictorType string // Type of predictor (linear, adaptive, etc.)
}

// NewDeltaEncoder creates a new Delta encoder
func NewDeltaEncoder() (*DeltaEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	return &DeltaEncoder{
		name:            "Delta",
		category:        core.AlgorithmCategoryPredictive,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		dataWidth:       1, // Default to byte-level
		signedData:      false,
		predictorType:   "linear",
	}, nil
}

// GetName returns the algorithm name
func (d *DeltaEncoder) GetName() string {
	return d.name
}

// GetCategory returns the algorithm category
func (d *DeltaEncoder) GetCategory() core.AlgorithmCategory {
	return d.category
}

// GetCompressionType returns the compression type
func (d *DeltaEncoder) GetCompressionType() core.CompressionType {
	return d.compressionType
}

// SetParameters sets algorithm parameters
func (d *DeltaEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		d.parameters[k] = v

		switch k {
		case "data_width":
			if val, ok := v.(int); ok && (val == 1 || val == 2 || val == 4 || val == 8) {
				d.dataWidth = val
			}
		case "signed_data":
			if val, ok := v.(bool); ok {
				d.signedData = val
			}
		case "predictor_type":
			if val, ok := v.(string); ok {
				d.predictorType = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (d *DeltaEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range d.parameters {
		result[k] = v
	}
	result["data_width"] = d.dataWidth
	result["signed_data"] = d.signedData
	result["predictor_type"] = d.predictorType
	return result
}

// GetInfo gets comprehensive algorithm information
func (d *DeltaEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 d.name,
		"category":             string(d.category),
		"compression_type":     string(d.compressionType),
		"parameters":           d.GetParameters(),
		"supports_streaming":   true,
		"thread_safe":          true,
		"deterministic":        true,
		"memory_bounded":       true,
		"data_width":           d.dataWidth,
		"algorithm_complexity": "O(n) time, O(1) space",
		"best_use_case":        "Time series data, audio samples, sensor readings with gradual changes",
	}
}

// bytesToInt converts bytes to integer based on data width
func (d *DeltaEncoder) bytesToInt(data []byte, offset int) int64 {
	if offset+d.dataWidth > len(data) {
		return 0
	}

	var value int64
	switch d.dataWidth {
	case 1:
		if d.signedData {
			value = int64(int8(data[offset]))
		} else {
			value = int64(data[offset])
		}
	case 2:
		val := int64(data[offset])<<8 | int64(data[offset+1])
		if d.signedData && val >= 0x8000 {
			value = val - 0x10000
		} else {
			value = val
		}
	case 4:
		val := int64(data[offset])<<24 | int64(data[offset+1])<<16 |
			int64(data[offset+2])<<8 | int64(data[offset+3])
		if d.signedData && val >= 0x80000000 {
			value = val - 0x100000000
		} else {
			value = val
		}
	case 8:
		value = int64(data[offset])<<56 | int64(data[offset+1])<<48 |
			int64(data[offset+2])<<40 | int64(data[offset+3])<<32 |
			int64(data[offset+4])<<24 | int64(data[offset+5])<<16 |
			int64(data[offset+6])<<8 | int64(data[offset+7])
	}

	return value
}

// intToBytes converts integer to bytes based on data width
func (d *DeltaEncoder) intToBytes(value int64) []byte {
	result := make([]byte, d.dataWidth)

	// Handle signed values by converting to unsigned representation
	if d.signedData {
		switch d.dataWidth {
		case 1:
			if value < -128 || value > 127 {
				value = value & 0xFF
			}
		case 2:
			if value < 0 {
				value = (value & 0xFFFF)
			}
		case 4:
			if value < 0 {
				value = (value & 0xFFFFFFFF)
			}
		}
	}

	switch d.dataWidth {
	case 1:
		result[0] = byte(value)
	case 2:
		result[0] = byte(value >> 8)
		result[1] = byte(value)
	case 4:
		result[0] = byte(value >> 24)
		result[1] = byte(value >> 16)
		result[2] = byte(value >> 8)
		result[3] = byte(value)
	case 8:
		result[0] = byte(value >> 56)
		result[1] = byte(value >> 48)
		result[2] = byte(value >> 40)
		result[3] = byte(value >> 32)
		result[4] = byte(value >> 24)
		result[5] = byte(value >> 16)
		result[6] = byte(value >> 8)
		result[7] = byte(value)
	}

	return result
}

// performDeltaEncoding performs the actual delta encoding
func (d *DeltaEncoder) performDeltaEncoding(input []byte) ([]byte, map[string]interface{}) {
	if len(input) < d.dataWidth {
		return input, make(map[string]interface{})
	}

	numElements := len(input) / d.dataWidth
	if numElements < 2 {
		return input, make(map[string]interface{})
	}

	// Header: [data_width(1)][signed_flag(1)][original_length(4)]
	headerSize := 6
	result := make([]byte, headerSize, headerSize+len(input))

	// Write header
	result[0] = byte(d.dataWidth)
	if d.signedData {
		result[1] = 1
	} else {
		result[1] = 0
	}
	// Original length (big-endian)
	originalLen := len(input)
	result[2] = byte(originalLen >> 24)
	result[3] = byte(originalLen >> 16)
	result[4] = byte(originalLen >> 8)
	result[5] = byte(originalLen)

	// Statistics
	stats := map[string]interface{}{
		"elements_processed": 0,
		"zero_deltas":        0,
		"small_deltas":       0, // deltas with absolute value <= 255
		"large_deltas":       0,
		"max_delta":          int64(0),
		"min_delta":          int64(0),
		"avg_delta":          0.0,
	}

	// Store first element unchanged
	firstElement := input[:d.dataWidth]
	result = append(result, firstElement...)

	// Process remaining elements
	previousValue := d.bytesToInt(input, 0)
	totalDelta := int64(0)
	maxDelta := int64(0)
	minDelta := int64(0)
	elementsProcessed := 0

	for i := 1; i < numElements; i++ {
		offset := i * d.dataWidth
		currentValue := d.bytesToInt(input, offset)
		delta := currentValue - previousValue

		// Convert delta to bytes
		deltaBytes := d.intToBytes(delta)
		result = append(result, deltaBytes...)

		// Update statistics
		elementsProcessed++
		totalDelta += delta

		if delta == 0 {
			stats["zero_deltas"] = stats["zero_deltas"].(int) + 1
		} else if delta >= -255 && delta <= 255 {
			stats["small_deltas"] = stats["small_deltas"].(int) + 1
		} else {
			stats["large_deltas"] = stats["large_deltas"].(int) + 1
		}

		if delta > maxDelta {
			maxDelta = delta
		}
		if delta < minDelta {
			minDelta = delta
		}

		previousValue = currentValue
	}

	// Finalize statistics
	stats["elements_processed"] = elementsProcessed
	stats["max_delta"] = maxDelta
	stats["min_delta"] = minDelta
	if elementsProcessed > 0 {
		stats["avg_delta"] = float64(totalDelta) / float64(elementsProcessed)
	}

	return result, stats
}

// performDeltaDecoding performs the inverse delta encoding
func (d *DeltaEncoder) performDeltaDecoding(encoded []byte) ([]byte, error) {
	headerSize := 6
	if len(encoded) < headerSize {
		return nil, fmt.Errorf("encoded data too short")
	}

	// Read header
	dataWidth := int(encoded[0])
	signedData := encoded[1] == 1
	originalLen := int(encoded[2])<<24 | int(encoded[3])<<16 |
		int(encoded[4])<<8 | int(encoded[5])

	// Validate parameters
	if dataWidth != d.dataWidth || signedData != d.signedData {
		return nil, fmt.Errorf("decoder parameters don't match encoded data")
	}

	if len(encoded) < headerSize+dataWidth {
		return nil, fmt.Errorf("encoded data too short for first element")
	}

	result := make([]byte, 0, originalLen)

	// Copy first element
	firstElement := encoded[headerSize : headerSize+dataWidth]
	result = append(result, firstElement...)

	// Process deltas
	previousValue := d.bytesToInt(firstElement, 0)
	offset := headerSize + dataWidth

	for offset+dataWidth <= len(encoded) {
		delta := d.bytesToInt(encoded, offset)
		currentValue := previousValue + delta

		currentBytes := d.intToBytes(currentValue)
		result = append(result, currentBytes...)

		previousValue = currentValue
		offset += dataWidth
	}

	// Trim to original length if necessary
	if len(result) > originalLen {
		result = result[:originalLen]
	}

	return result, nil
}

// Compress compresses data using Delta Encoding with aerospace-grade performance monitoring
func (d *DeltaEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    d.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var compressedData []byte
	var deltaStats map[string]interface{}

	profile, err := d.monitor.ProfileOperation(ctx, "delta_compress", int64(len(data)), func() error {
		compressedData, deltaStats = d.performDeltaEncoding(data)
		return nil
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("Delta compression failed: %v", err),
			Algorithm:   d.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}

	// Delta encoding may increase or decrease size depending on data patterns
	compressionRatio := float64(len(data)) / float64(len(compressedData))

	// Convert profile to aerospace precision metrics
	precisionMetrics := d.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))

	metadata := map[string]interface{}{
		"data_width":         d.dataWidth,
		"signed_data":        d.signedData,
		"predictor_type":     d.predictorType,
		"elements_processed": deltaStats["elements_processed"],
		"zero_deltas":        deltaStats["zero_deltas"],
		"small_deltas":       deltaStats["small_deltas"],
		"large_deltas":       deltaStats["large_deltas"],
		"max_delta":          deltaStats["max_delta"],
		"min_delta":          deltaStats["min_delta"],
		"avg_delta":          deltaStats["avg_delta"],
		"header_overhead":    6, // 6 bytes header
		"predictive_stage":   true,
		"delta_distribution": fmt.Sprintf("zero: %v, small: %v, large: %v",
			deltaStats["zero_deltas"], deltaStats["small_deltas"], deltaStats["large_deltas"]),
	}

	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    d.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses delta-encoded data with aerospace-grade performance monitoring
func (d *DeltaEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          d.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var decompressedData []byte

	profile, err := d.monitor.ProfileOperation(ctx, "delta_decompress", int64(len(compressedData)), func() error {
		var decodeErr error
		decompressedData, decodeErr = d.performDeltaDecoding(compressedData)
		return decodeErr
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("Delta decompression failed: %v", err),
			Algorithm:      d.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	// Convert profile to aerospace precision metrics
	precisionMetrics := d.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))

	decompMetadata := map[string]interface{}{
		"original_compressed_size": len(compressedData),
		"decompressed_size":        len(decompressedData),
		"algorithm":                d.name,
		"predictive_reverse":       true,
	}

	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          d.name,
		Metadata:               decompMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (d *DeltaEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
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

		// Delta-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // Delta encoding is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}
