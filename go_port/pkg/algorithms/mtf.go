// Package algorithms provides MTF (Move-To-Front) implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// MTFEncoder implements Move-To-Front transform with aerospace-grade precision monitoring
type MTFEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// MTF-specific parameters
	alphabet string
}

// NewMTFEncoder creates a new MTF encoder
func NewMTFEncoder() (*MTFEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	// Default alphabet includes common ASCII characters
	defaultAlphabet := ""
	for i := 0; i < 256; i++ {
		defaultAlphabet += string(byte(i))
	}

	return &MTFEncoder{
		name:            "MTF",
		category:        core.AlgorithmCategoryTransform,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		alphabet:        defaultAlphabet,
	}, nil
}

// GetName returns the algorithm name
func (m *MTFEncoder) GetName() string {
	return m.name
}

// GetCategory returns the algorithm category
func (m *MTFEncoder) GetCategory() core.AlgorithmCategory {
	return m.category
}

// GetCompressionType returns the compression type
func (m *MTFEncoder) GetCompressionType() core.CompressionType {
	return m.compressionType
}

// SetParameters sets algorithm parameters
func (m *MTFEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		m.parameters[k] = v

		switch k {
		case "alphabet":
			if val, ok := v.(string); ok && len(val) > 0 {
				m.alphabet = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (m *MTFEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range m.parameters {
		result[k] = v
	}
	result["alphabet"] = m.alphabet
	result["alphabet_size"] = len(m.alphabet)
	return result
}

// GetInfo gets comprehensive algorithm information
func (m *MTFEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 m.name,
		"category":             string(m.category),
		"compression_type":     string(m.compressionType),
		"parameters":           m.GetParameters(),
		"supports_streaming":   true,
		"thread_safe":          false,
		"deterministic":        true,
		"memory_bounded":       true,
		"alphabet_size":        len(m.alphabet),
		"algorithm_complexity": "O(n*k) time, O(k) space where k is alphabet size",
		"best_use_case":        "Text preprocessing after BWT for improved compression",
	}
}

// performMTF performs the actual Move-To-Front transform
func (m *MTFEncoder) performMTF(input []byte) ([]byte, map[string]interface{}) {
	if len(input) == 0 {
		return input, make(map[string]interface{})
	}

	// Create working alphabet list
	alphabet := make([]byte, len(m.alphabet))
	for i, char := range m.alphabet {
		alphabet[i] = byte(char)
	}

	result := make([]byte, 0, len(input))
	stats := map[string]interface{}{
		"transformations": 0,
		"max_rank":        0,
		"avg_rank":        0.0,
	}

	totalRank := 0
	maxRank := 0
	transformations := 0

	for _, inputByte := range input {
		// Find the position of the byte in the current alphabet
		rank := -1
		for i, alphabetByte := range alphabet {
			if alphabetByte == inputByte {
				rank = i
				break
			}
		}

		if rank == -1 {
			// Character not in alphabet, add it at the beginning
			newAlphabet := make([]byte, len(alphabet)+1)
			newAlphabet[0] = inputByte
			copy(newAlphabet[1:], alphabet)
			alphabet = newAlphabet
			rank = 0
		}

		// Output the rank
		result = append(result, byte(rank))

		// Move the character to front if not already there
		if rank > 0 {
			// Move character to front
			char := alphabet[rank]
			copy(alphabet[1:rank+1], alphabet[0:rank])
			alphabet[0] = char
			transformations++
		}

		// Update statistics
		totalRank += rank
		if rank > maxRank {
			maxRank = rank
		}
	}

	// Calculate statistics
	stats["transformations"] = transformations
	stats["max_rank"] = maxRank
	if len(input) > 0 {
		stats["avg_rank"] = float64(totalRank) / float64(len(input))
	}
	stats["final_alphabet_size"] = len(alphabet)
	stats["alphabet_expansion"] = len(alphabet) - len(m.alphabet)

	return result, stats
}

// performInverseMTF performs the inverse Move-To-Front transform
func (m *MTFEncoder) performInverseMTF(ranks []byte, alphabetSize int) []byte {
	if len(ranks) == 0 {
		return ranks
	}

	// Initialize alphabet - use the size from metadata if available
	alphabet := make([]byte, 0, alphabetSize)
	for i := 0; i < min(len(m.alphabet), alphabetSize); i++ {
		alphabet = append(alphabet, byte(m.alphabet[i]))
	}

	result := make([]byte, 0, len(ranks))

	for _, rank := range ranks {
		rankInt := int(rank)

		// Handle case where rank refers to a position beyond current alphabet
		for len(alphabet) <= rankInt {
			// Extend alphabet with new characters
			alphabet = append(alphabet, byte(len(alphabet)))
		}

		// Get character at rank position
		char := alphabet[rankInt]
		result = append(result, char)

		// Move character to front if not already there
		if rankInt > 0 {
			copy(alphabet[1:rankInt+1], alphabet[0:rankInt])
			alphabet[0] = char
		}
	}

	return result
}

// Compress compresses data using MTF with aerospace-grade performance monitoring
func (m *MTFEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    m.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var transformedData []byte
	var mtfStats map[string]interface{}

	profile, err := m.monitor.ProfileOperation(ctx, "mtf_compress", int64(len(data)), func() error {
		transformedData, mtfStats = m.performMTF(data)
		return nil
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("MTF compression failed: %v", err),
			Algorithm:   m.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}

	// Encode the result: [alphabet_size(4 bytes)][transformed_data]
	compressedData := make([]byte, 4+len(transformedData))

	// Store alphabet size (big-endian)
	alphabetSize := len(m.alphabet)
	compressedData[0] = byte(alphabetSize >> 24)
	compressedData[1] = byte(alphabetSize >> 16)
	compressedData[2] = byte(alphabetSize >> 8)
	compressedData[3] = byte(alphabetSize)

	// Store transformed data
	copy(compressedData[4:], transformedData)

	// MTF is a transform that may increase or decrease size depending on data
	compressionRatio := float64(len(data)) / float64(len(compressedData))

	// Convert profile to aerospace precision metrics
	precisionMetrics := m.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))

	metadata := map[string]interface{}{
		"alphabet_size":       alphabetSize,
		"transformations":     mtfStats["transformations"],
		"max_rank":            mtfStats["max_rank"],
		"avg_rank":            mtfStats["avg_rank"],
		"final_alphabet_size": mtfStats["final_alphabet_size"],
		"alphabet_expansion":  mtfStats["alphabet_expansion"],
		"overhead_bytes":      4, // 4 bytes for alphabet size
		"preprocessing_stage": true,
		"rank_distribution":   fmt.Sprintf("max: %v, avg: %.2f", mtfStats["max_rank"], mtfStats["avg_rank"]),
	}

	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    m.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses MTF-transformed data with aerospace-grade performance monitoring
func (m *MTFEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          m.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.StatisticalPrecisionMetrics{},
		}, nil
	}

	if len(compressedData) < 4 {
		return nil, &core.DecompressionError{
			Message:        "compressed data too short for MTF",
			Algorithm:      m.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	var decompressedData []byte

	profile, err := m.monitor.ProfileOperation(ctx, "mtf_decompress", int64(len(compressedData)), func() error {
		// Extract alphabet size (big-endian)
		alphabetSize := int(compressedData[0])<<24 |
			int(compressedData[1])<<16 |
			int(compressedData[2])<<8 |
			int(compressedData[3])

		// Extract transformed data
		transformedData := compressedData[4:]

		// Perform inverse MTF
		decompressedData = m.performInverseMTF(transformedData, alphabetSize)
		return nil
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("MTF decompression failed: %v", err),
			Algorithm:      m.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	// Convert profile to aerospace precision metrics
	precisionMetrics := m.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))

	decompMetadata := map[string]interface{}{
		"original_compressed_size": len(compressedData),
		"decompressed_size":        len(decompressedData),
		"algorithm":                m.name,
		"preprocessing_reverse":    true,
	}

	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          m.name,
		Metadata:               decompMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (m *MTFEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
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

		// MTF-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // MTF is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
