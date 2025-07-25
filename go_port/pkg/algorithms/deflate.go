// Package algorithms provides Deflate implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// DeflateEncoder implements Deflate compression (LZ77 + Huffman) with aerospace-grade precision monitoring
type DeflateEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// Deflate uses internal LZ77 and Huffman encoders
	lz77Encoder    *LZ77Encoder
	huffmanEncoder *HuffmanEncoder

	// Deflate-specific parameters
	compressionLevel int  // 1-9, affects LZ77 parameters
	enableHuffman    bool // Enable/disable Huffman stage
	enableLZ77       bool // Enable/disable LZ77 stage
}

// NewDeflateEncoder creates a new Deflate encoder
func NewDeflateEncoder() (*DeflateEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	// Create internal LZ77 encoder
	lz77, err := NewLZ77Encoder()
	if err != nil {
		return nil, fmt.Errorf("failed to create LZ77 encoder: %w", err)
	}

	// Create internal Huffman encoder
	huffman, err := NewHuffmanEncoder()
	if err != nil {
		return nil, fmt.Errorf("failed to create Huffman encoder: %w", err)
	}

	deflate := &DeflateEncoder{
		name:             "Deflate",
		category:         core.AlgorithmCategoryHybrid,
		compressionType:  core.CompressionTypeLossless,
		parameters:       make(map[string]interface{}),
		monitor:          monitor,
		lz77Encoder:      lz77,
		huffmanEncoder:   huffman,
		compressionLevel: 6,    // Default compression level
		enableHuffman:    true, // Enable both stages
		enableLZ77:       true,
	}

	// Set default LZ77 parameters for deflate
	deflate.updateLZ77Parameters()

	return deflate, nil
}

// updateLZ77Parameters updates LZ77 parameters based on compression level
func (d *DeflateEncoder) updateLZ77Parameters() {
	var windowSize, bufferSize int

	switch d.compressionLevel {
	case 1: // Fast
		windowSize = 1024
		bufferSize = 64
	case 2, 3: // Fast-Medium
		windowSize = 2048
		bufferSize = 128
	case 4, 5, 6: // Medium (default)
		windowSize = 4096
		bufferSize = 256
	case 7, 8: // Medium-High
		windowSize = 8192
		bufferSize = 256
	case 9: // Maximum
		windowSize = 32768
		bufferSize = 258
	default:
		windowSize = 4096
		bufferSize = 256
	}

	lz77Params := map[string]interface{}{
		"window_size":   windowSize,
		"buffer_size":   bufferSize,
		"min_match_len": 3,
		"max_match_len": 258,
	}

	d.lz77Encoder.SetParameters(lz77Params)
}

// GetName returns the algorithm name
func (d *DeflateEncoder) GetName() string {
	return d.name
}

// GetCategory returns the algorithm category
func (d *DeflateEncoder) GetCategory() core.AlgorithmCategory {
	return d.category
}

// GetCompressionType returns the compression type
func (d *DeflateEncoder) GetCompressionType() core.CompressionType {
	return d.compressionType
}

// SetParameters sets algorithm parameters
func (d *DeflateEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		d.parameters[k] = v

		switch k {
		case "compression_level":
			if val, ok := v.(int); ok && val >= 1 && val <= 9 {
				d.compressionLevel = val
				d.updateLZ77Parameters()
			}
		case "enable_huffman":
			if val, ok := v.(bool); ok {
				d.enableHuffman = val
			}
		case "enable_lz77":
			if val, ok := v.(bool); ok {
				d.enableLZ77 = val
			}
		case "window_size":
			// Pass through to LZ77
			if d.lz77Encoder != nil {
				lz77Params := map[string]interface{}{"window_size": v}
				d.lz77Encoder.SetParameters(lz77Params)
			}
		case "buffer_size":
			// Pass through to LZ77
			if d.lz77Encoder != nil {
				lz77Params := map[string]interface{}{"buffer_size": v}
				d.lz77Encoder.SetParameters(lz77Params)
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (d *DeflateEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range d.parameters {
		result[k] = v
	}
	result["compression_level"] = d.compressionLevel
	result["enable_huffman"] = d.enableHuffman
	result["enable_lz77"] = d.enableLZ77

	// Include LZ77 parameters
	if d.lz77Encoder != nil {
		lz77Params := d.lz77Encoder.GetParameters()
		for k, v := range lz77Params {
			result["lz77_"+k] = v
		}
	}

	return result
}

// GetInfo gets comprehensive algorithm information
func (d *DeflateEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 d.name,
		"category":             string(d.category),
		"compression_type":     string(d.compressionType),
		"parameters":           d.GetParameters(),
		"supports_streaming":   true,
		"thread_safe":          false,
		"deterministic":        true,
		"memory_bounded":       true,
		"compression_level":    d.compressionLevel,
		"hybrid_algorithm":     true,
		"stage1":               "LZ77 (dictionary compression)",
		"stage2":               "Huffman (entropy encoding)",
		"algorithm_complexity": "O(n*w + n*log(n)) time where w is window size",
		"best_use_case":        "General purpose, web content, archives (ZIP, gzip, PNG)",
		"standard_compliance":  "RFC 1951 (Deflate Compressed Data Format)",
	}
}

// performDeflateCompression performs the actual Deflate compression
func (d *DeflateEncoder) performDeflateCompression(ctx context.Context, input []byte) ([]byte, map[string]interface{}, error) {
	if len(input) == 0 {
		return input, make(map[string]interface{}), nil
	}

	stats := map[string]interface{}{
		"original_size":             len(input),
		"lz77_enabled":              d.enableLZ77,
		"huffman_enabled":           d.enableHuffman,
		"compression_level":         d.compressionLevel,
		"lz77_compression_ratio":    1.0,
		"huffman_compression_ratio": 1.0,
		"total_compression_ratio":   1.0,
		"lz77_time_ns":              int64(0),
		"huffman_time_ns":           int64(0),
		"total_time_ns":             int64(0),
	}

	var currentData []byte = input

	// Stage 1: LZ77 compression (if enabled)
	if d.enableLZ77 {
		lz77Start := time.Now()
		lz77Result, err := d.lz77Encoder.Compress(ctx, currentData)
		lz77Duration := time.Since(lz77Start)

		if err != nil {
			return nil, stats, fmt.Errorf("LZ77 stage failed: %w", err)
		}

		currentData = lz77Result.CompressedData
		stats["lz77_compression_ratio"] = lz77Result.CompressionRatio
		stats["lz77_time_ns"] = lz77Duration.Nanoseconds()
		stats["lz77_compressed_size"] = len(currentData)

		// Merge LZ77 metadata
		for k, v := range lz77Result.Metadata {
			stats["lz77_"+k] = v
		}
	}

	// Stage 2: Huffman compression (if enabled)
	if d.enableHuffman {
		huffmanStart := time.Now()
		huffmanResult, err := d.huffmanEncoder.Compress(ctx, currentData)
		huffmanDuration := time.Since(huffmanStart)

		if err != nil {
			return nil, stats, fmt.Errorf("Huffman stage failed: %w", err)
		}

		currentData = huffmanResult.CompressedData
		stats["huffman_compression_ratio"] = huffmanResult.CompressionRatio
		stats["huffman_time_ns"] = huffmanDuration.Nanoseconds()
		stats["huffman_compressed_size"] = len(currentData)

		// Merge Huffman metadata
		for k, v := range huffmanResult.Metadata {
			stats["huffman_"+k] = v
		}
	}

	// Calculate total statistics
	stats["final_compressed_size"] = len(currentData)
	stats["total_compression_ratio"] = float64(len(input)) / float64(len(currentData))
	stats["total_time_ns"] = stats["lz77_time_ns"].(int64) + stats["huffman_time_ns"].(int64)

	return currentData, stats, nil
}

// performDeflateDecompression performs Deflate decompression
func (d *DeflateEncoder) performDeflateDecompression(ctx context.Context, compressed []byte, metadata map[string]interface{}) ([]byte, error) {
	if len(compressed) == 0 {
		return compressed, nil
	}

	var currentData []byte = compressed

	// Stage 1: Huffman decompression (reverse order - if enabled)
	if d.enableHuffman {
		huffmanResult, err := d.huffmanEncoder.Decompress(ctx, currentData, metadata)
		if err != nil {
			return nil, fmt.Errorf("Huffman decompression stage failed: %w", err)
		}
		currentData = huffmanResult.DecompressedData
	}

	// Stage 2: LZ77 decompression (if enabled)
	if d.enableLZ77 {
		lz77Result, err := d.lz77Encoder.Decompress(ctx, currentData, metadata)
		if err != nil {
			return nil, fmt.Errorf("LZ77 decompression stage failed: %w", err)
		}
		currentData = lz77Result.DecompressedData
	}

	return currentData, nil
}

// Compress compresses data using Deflate with aerospace-grade performance monitoring
func (d *DeflateEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
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
	var deflateStats map[string]interface{}

	profile, err := d.monitor.ProfileOperation(ctx, "deflate_compress", int64(len(data)), func() error {
		var compErr error
		compressedData, deflateStats, compErr = d.performDeflateCompression(ctx, data)
		return compErr
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("Deflate compression failed: %v", err),
			Algorithm:   d.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}

	// Calculate compression ratio
	compressionRatio := float64(len(data)) / float64(len(compressedData))

	// Convert profile to aerospace precision metrics
	precisionMetrics := d.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))

	// Merge deflate statistics with metadata
	metadata := make(map[string]interface{})
	for k, v := range deflateStats {
		metadata[k] = v
	}
	metadata["hybrid_compression"] = true
	metadata["deflate_compliant"] = true
	metadata["stages_completed"] = 2
	metadata["algorithm_family"] = "LZ77+Huffman"

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

// Decompress decompresses Deflate-compressed data with aerospace-grade performance monitoring
func (d *DeflateEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
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

	profile, err := d.monitor.ProfileOperation(ctx, "deflate_decompress", int64(len(compressedData)), func() error {
		var decompErr error
		decompressedData, decompErr = d.performDeflateDecompression(ctx, compressedData, metadata)
		return decompErr
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("Deflate decompression failed: %v", err),
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
		"hybrid_decompression":     true,
		"stages_processed":         2,
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
func (d *DeflateEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
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

		// Deflate-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // Deflate is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}
