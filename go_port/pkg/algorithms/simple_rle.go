package algorithms

import (
	"context"
	"time"

	"hybrid-compression-study/pkg/core"
)

// SimpleRLEEncoder implements fast RLE without aerospace monitoring
type SimpleRLEEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
}

// NewSimpleRLEEncoder creates a new fast RLE encoder
func NewSimpleRLEEncoder() (*SimpleRLEEncoder, error) {
	return &SimpleRLEEncoder{
		name:            "SimpleRLE",
		category:        core.AlgorithmCategoryRunLength,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
	}, nil
}

// GetName returns the algorithm name
func (s *SimpleRLEEncoder) GetName() string {
	return s.name
}

// GetCategory returns the algorithm category
func (s *SimpleRLEEncoder) GetCategory() core.AlgorithmCategory {
	return s.category
}

// GetCompressionType returns the compression type
func (s *SimpleRLEEncoder) GetCompressionType() core.CompressionType {
	return s.compressionType
}

// SetParameters sets algorithm parameters
func (s *SimpleRLEEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		s.parameters[k] = v
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (s *SimpleRLEEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range s.parameters {
		result[k] = v
	}
	return result
}

// GetInfo gets comprehensive algorithm information
func (s *SimpleRLEEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               s.name,
		"category":           string(s.category),
		"compression_type":   string(s.compressionType),
		"parameters":         s.GetParameters(),
		"supports_streaming": false,
		"thread_safe":        true,
		"deterministic":      true,
		"memory_bounded":     true,
		"time_complexity":    "O(n)",
		"space_complexity":   "O(n)",
		"best_use_case":      "Data with long runs of identical bytes",
	}
}

// Compress compresses data using simple RLE
func (s *SimpleRLEEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	startTime := time.Now()

	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   nil,
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    s.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.StatisticalPrecisionMetrics{},
		}, nil
	}

	compressed := make([]byte, 0, len(data))

	i := 0
	for i < len(data) {
		current := data[i]
		count := 1

		// Count consecutive identical bytes (max 255)
		for i+count < len(data) && data[i+count] == current && count < 255 {
			count++
		}

		if count >= 3 {
			// Use RLE: [count][value]
			compressed = append(compressed, byte(count), current)
		} else {
			// Just copy the bytes
			for j := 0; j < count; j++ {
				compressed = append(compressed, current)
			}
		}

		i += count
	}

	duration := time.Since(startTime)
	compressionRatio := 1.0
	if len(compressed) > 0 {
		compressionRatio = float64(len(data)) / float64(len(compressed))
	}

	return &core.CompressionResult{
		CompressedData:   nil, // Don't store for memory efficiency
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressed)),
		CompressionRatio: compressionRatio,
		CompressionTime:  duration.Seconds(),
		AlgorithmName:    s.name,
		Metadata: map[string]interface{}{
			"runs_encoded": "variable",
		},
		PrecisionMetrics: core.StatisticalPrecisionMetrics{
			CompressionTimeNs: duration.Nanoseconds(),
			ThroughputMBPS:    (float64(len(data)) / (1024 * 1024)) / duration.Seconds(),
			MemoryPeakBytes:   int64(len(data) + len(compressed)),
		},
	}, nil
}

// Decompress decompresses RLE data
func (s *SimpleRLEEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	startTime := time.Now()

	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          s.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.StatisticalPrecisionMetrics{},
		}, nil
	}

	decompressed := make([]byte, 0, len(compressedData)*2) // Estimate

	i := 0
	for i < len(compressedData) {
		if i+1 >= len(compressedData) {
			// Single byte, just copy
			decompressed = append(decompressed, compressedData[i])
			i++
			continue
		}

		count := int(compressedData[i])
		value := compressedData[i+1]

		if count >= 3 {
			// RLE encoded: repeat value count times
			for j := 0; j < count; j++ {
				decompressed = append(decompressed, value)
			}
			i += 2
		} else {
			// Not RLE encoded, just copy
			decompressed = append(decompressed, compressedData[i])
			i++
		}
	}

	duration := time.Since(startTime)

	return &core.DecompressionResult{
		DecompressedData:       decompressed,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressed)),
		DecompressionTime:      duration.Seconds(),
		AlgorithmName:          s.name,
		Metadata: map[string]interface{}{
			"decompressed_size": len(decompressed),
		},
		PrecisionMetrics: core.StatisticalPrecisionMetrics{
			CompressionTimeNs: duration.Nanoseconds(),
			ThroughputMBPS:    (float64(len(decompressed)) / (1024 * 1024)) / duration.Seconds(),
			MemoryPeakBytes:   int64(len(compressedData) + len(decompressed)),
		},
	}, nil
}
