// Package algorithms provides Run-Length Encoding implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"time"

	"hybrid-compression-study/pkg/core"
	"hybrid-compression-study/internal/performance"
)

// RLEEncoder implements Run-Length Encoding with aerospace-grade precision monitoring
type RLEEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.AerospaceGradeMonitor
	
	// RLE-specific parameters
	escapeByte    byte
	minRunLength  int
}

// NewRLEEncoder creates a new RLE encoder
func NewRLEEncoder() (*RLEEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}
	
	return &RLEEncoder{
		name:            "RLE",
		category:        core.AlgorithmCategoryRunLength,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		escapeByte:      0x00, // Default escape byte
		minRunLength:    3,    // Minimum run length to encode
	}, nil
}

// GetName returns the algorithm name
func (r *RLEEncoder) GetName() string {
	return r.name
}

// GetCategory returns the algorithm category
func (r *RLEEncoder) GetCategory() core.AlgorithmCategory {
	return r.category
}

// GetCompressionType returns the compression type
func (r *RLEEncoder) GetCompressionType() core.CompressionType {
	return r.compressionType
}

// SetParameters sets algorithm parameters
func (r *RLEEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		r.parameters[k] = v
		
		switch k {
		case "escape_byte":
			if val, ok := v.(int); ok && val >= 0 && val <= 255 {
				r.escapeByte = byte(val)
			}
		case "min_run_length":
			if val, ok := v.(int); ok && val > 0 {
				r.minRunLength = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (r *RLEEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range r.parameters {
		result[k] = v
	}
	result["escape_byte"] = int(r.escapeByte)
	result["min_run_length"] = r.minRunLength
	return result
}

// GetInfo gets comprehensive algorithm information
func (r *RLEEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               r.name,
		"category":           string(r.category),
		"compression_type":   string(r.compressionType),
		"parameters":         r.GetParameters(),
		"supports_streaming": true,
		"thread_safe":        false,
		"deterministic":      true,
		"memory_bounded":     true,
		"escape_byte":        int(r.escapeByte),
		"min_run_length":     r.minRunLength,
	}
}

// Compress compresses data using Run-Length Encoding with aerospace-grade performance monitoring
func (r *RLEEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    r.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.AerospacePrecisionMetrics{},
		}, nil
	}
	
	var compressedData []byte
	var runCount, maxRunLength, totalRuns int
	var compressionSavings int64
	
	profile, err := r.monitor.ProfileOperation(ctx, "rle_compress", int64(len(data)), func() error {
		result := make([]byte, 0, len(data))
		
		i := 0
		for i < len(data) {
			currentByte := data[i]
			runLength := 1
			
			// Count consecutive identical bytes
			for i+runLength < len(data) && data[i+runLength] == currentByte && runLength < 255 {
				runLength++
			}
			
			if runLength >= r.minRunLength {
				// Encode as run: [escape_byte][run_length][value]
				result = append(result, r.escapeByte)
				result = append(result, byte(runLength))
				result = append(result, currentByte)
				totalRuns++
				if runLength > maxRunLength {
					maxRunLength = runLength
				}
				compressionSavings += int64(runLength - 3) // Saved bytes
			} else {
				// Copy literal bytes, but escape any escape bytes
				for j := 0; j < runLength; j++ {
					if data[i+j] == r.escapeByte {
						// Escape the escape byte: [escape_byte][0][escape_byte]
						result = append(result, r.escapeByte, 0, r.escapeByte)
					} else {
						result = append(result, data[i+j])
					}
				}
			}
			
			i += runLength
		}
		
		compressedData = result
		runCount = totalRuns
		return nil
	})
	
	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("RLE compression failed: %v", err),
			Algorithm:   r.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}
	
	// Calculate metrics
	compressionRatio := float64(len(data)) / float64(len(compressedData))
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := r.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))
	
	metadata := map[string]interface{}{
		"total_runs":           runCount,
		"max_run_length":       maxRunLength,
		"compression_savings":  compressionSavings,
		"escape_byte":          int(r.escapeByte),
		"min_run_length":       r.minRunLength,
		"compression_efficiency": compressionRatio,
		"runs_per_kb":          float64(runCount) / (float64(len(data)) / 1024.0),
	}
	
	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    r.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses RLE-encoded data with aerospace-grade performance monitoring
func (r *RLEEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          r.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.AerospacePrecisionMetrics{},
		}, nil
	}
	
	var decompressedData []byte
	var decodedRuns int
	
	profile, err := r.monitor.ProfileOperation(ctx, "rle_decompress", int64(len(compressedData)), func() error {
		result := make([]byte, 0, len(compressedData)*2) // Estimate expansion
		
		i := 0
		for i < len(compressedData) {
			if compressedData[i] == r.escapeByte {
				if i+2 >= len(compressedData) {
					return fmt.Errorf("incomplete RLE sequence at position %d", i)
				}
				
				runLength := int(compressedData[i+1])
				value := compressedData[i+2]
				
				if runLength == 0 {
					// Escaped escape byte
					result = append(result, r.escapeByte)
					i += 3
				} else {
					// Run sequence
					for j := 0; j < runLength; j++ {
						result = append(result, value)
					}
					decodedRuns++
					i += 3
				}
			} else {
				// Literal byte
				result = append(result, compressedData[i])
				i++
			}
		}
		
		decompressedData = result
		return nil
	})
	
	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("RLE decompression failed: %v", err),
			Algorithm:      r.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := r.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))
	
	// Add decompression-specific metadata
	decompressionMetadata := make(map[string]interface{})
	for k, v := range metadata {
		decompressionMetadata[k] = v
	}
	decompressionMetadata["decoded_runs"] = decodedRuns
	
	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          r.name,
		Metadata:               decompressionMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (r *RLEEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, dataSize, outputSize int64) core.AerospacePrecisionMetrics {
	metrics := core.AerospacePrecisionMetrics{
		CompressionTimeNs:             profile.DurationNs,
		TotalTimeNs:                   profile.DurationNs,
		MemoryPeakBytes:               profile.MemoryPeakBytes,
		MemoryDeltaBytes:              profile.MemoryDeltaBytes,
		MemoryBeforeBytes:             profile.MemoryBeforeBytes,
		MemoryAfterBytes:              profile.MemoryAfterBytes,
		MemoryAllocMB:                 profile.RuntimeAllocMB,
		MemorySystemMB:                profile.RuntimeSysMB,
		CPUPercentAvg:                 profile.CPUPercentAvg,
		CPUPercentPeak:                profile.CPUPercentPeak,
		CPUFreqAvgMHz:                 profile.CPUFreqAvgMHz,
		IOReadBytes:                   profile.IOReadBytes,
		IOWriteBytes:                  profile.IOWriteBytes,
		IOReadOps:                     profile.IOReadOps,
		IOWriteOps:                    profile.IOWriteOps,
		PageFaults:                    profile.PageFaults,
		ContextSwitches:               profile.ContextSwitches,
		GCCollections:                 profile.GCCollections,
		ThroughputMBPS:                profile.ThroughputMBPS,
		ThroughputBytesPerSecond:      profile.ThroughputMBPS * 1024 * 1024,
		TimePerByteNs:                 profile.TimePerByteNs,
		BytesPerCPUCycle:              profile.BytesPerCPUCycle,
		MemoryEfficiencyRatio:         profile.MemoryEfficiencyRatio,
		BitsPerByte:                   (float64(outputSize) * 8) / float64(dataSize),
		EntropyEfficiency:             float64(dataSize) / float64(outputSize) / 8.0,
		EnergyEfficiencyBytesPerNs:    float64(dataSize) / float64(profile.DurationNs),
		WorstCaseLatencyNs:            profile.DurationNs,
		DeterminismScore:              1.0, // RLE is deterministic
		MemoryOverheadRatio:           float64(profile.MemoryPeakBytes) / float64(dataSize),
		CPUEfficiencyBytesPerCPUSecond: float64(dataSize) / max(0.001, profile.CPUTimeUserS+profile.CPUTimeSystemS),
		IOEfficiencyRatio:             float64(dataSize) / max(1.0, float64(profile.IOReadBytes+profile.IOWriteBytes)),
	}
	
	metrics.UpdateFormattedTimes()
	return metrics
} 