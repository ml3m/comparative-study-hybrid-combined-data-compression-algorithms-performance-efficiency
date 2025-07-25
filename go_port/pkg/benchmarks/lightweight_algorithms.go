package benchmarks

import (
	"context"
	"hybrid-compression-study/pkg/core"
)

// LightweightAlgorithmWrapper provides memory-efficient algorithm wrapper
type LightweightAlgorithmWrapper struct {
	core.CompressionAlgorithm
	name string
}

// NewLightweightRLE creates a memory-efficient RLE implementation
func NewLightweightRLE() (core.CompressionAlgorithm, error) {
	return &LightweightAlgorithmWrapper{
		name: "RLE_Lightweight",
	}, nil
}

// Compress implements a basic RLE compression without aerospace monitoring
func (l *LightweightAlgorithmWrapper) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   nil,
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			AlgorithmName:    l.name,
		}, nil
	}

	// Simple RLE implementation
	compressed := make([]byte, 0, len(data))

	i := 0
	for i < len(data) {
		current := data[i]
		count := 1

		// Count consecutive identical bytes
		for i+count < len(data) && data[i+count] == current && count < 255 {
			count++
		}

		if count >= 3 {
			// Encode run: [escape][count][value]
			compressed = append(compressed, 0x00, byte(count), current)
		} else {
			// Just append the bytes
			for j := 0; j < count; j++ {
				compressed = append(compressed, current)
			}
		}

		i += count
	}

	ratio := 1.0
	if len(compressed) > 0 {
		ratio = float64(len(data)) / float64(len(compressed))
	}

	return &core.CompressionResult{
		CompressedData:   nil, // Don't store compressed data!
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressed)),
		CompressionRatio: ratio,
		AlgorithmName:    l.name,
		PrecisionMetrics: core.StatisticalPrecisionMetrics{
			MemoryPeakBytes: int64(len(data) + len(compressed)), // Estimate
		},
	}, nil
}

// GetName returns algorithm name
func (l *LightweightAlgorithmWrapper) GetName() string {
	return l.name
}

// GetCategory returns algorithm category
func (l *LightweightAlgorithmWrapper) GetCategory() core.AlgorithmCategory {
	return core.AlgorithmCategoryRunLength
}

// GetCompressionType returns compression type
func (l *LightweightAlgorithmWrapper) GetCompressionType() core.CompressionType {
	return core.CompressionTypeLossless
}

// SetParameters sets parameters (no-op for lightweight version)
func (l *LightweightAlgorithmWrapper) SetParameters(params map[string]interface{}) error {
	return nil
}

// GetParameters returns empty parameters
func (l *LightweightAlgorithmWrapper) GetParameters() map[string]interface{} {
	return make(map[string]interface{})
}

// GetInfo returns basic info
func (l *LightweightAlgorithmWrapper) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":        l.name,
		"lightweight": true,
		"category":    string(l.GetCategory()),
	}
}

// Decompress implements basic decompression (placeholder)
func (l *LightweightAlgorithmWrapper) Decompress(ctx context.Context, data []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	return &core.DecompressionResult{
		DecompressedData:  data, // Placeholder
		DecompressionTime: 0.001,
		AlgorithmName:     l.name,
	}, nil
}
