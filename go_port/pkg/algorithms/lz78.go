// Package algorithms provides LZ78 implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// LZ78Encoder implements LZ78 compression with aerospace-grade precision monitoring
type LZ78Encoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// LZ78-specific parameters
	maxDictSize    int // Maximum dictionary size
	resetThreshold int // Reset dictionary when this size is reached
}

// LZ78Token represents an LZ78 output token (dictionary_index, character)
type LZ78Token struct {
	Index     int  // Dictionary index (0 = no match)
	Character byte // Next character
}

// LZ78Dictionary manages the LZ78 dictionary
type LZ78Dictionary struct {
	entries   map[string]int // string -> index mapping
	strings   []string       // index -> string mapping
	nextIndex int            // next available index
	maxSize   int            // maximum dictionary size
}

// NewLZ78Dictionary creates a new LZ78 dictionary
func NewLZ78Dictionary(maxSize int) *LZ78Dictionary {
	return &LZ78Dictionary{
		entries:   make(map[string]int),
		strings:   make([]string, 0, maxSize),
		nextIndex: 1, // Start from 1 (0 means no match)
		maxSize:   maxSize,
	}
}

// Add adds a string to the dictionary and returns its index
func (d *LZ78Dictionary) Add(s string) int {
	if len(d.strings) >= d.maxSize {
		return 0 // Dictionary full
	}

	index := d.nextIndex
	d.entries[s] = index
	d.strings = append(d.strings, s)
	d.nextIndex++
	return index
}

// Find finds a string in the dictionary and returns its index (0 if not found)
func (d *LZ78Dictionary) Find(s string) int {
	if index, exists := d.entries[s]; exists {
		return index
	}
	return 0
}

// Get retrieves a string by its index
func (d *LZ78Dictionary) Get(index int) string {
	if index <= 0 || index > len(d.strings) {
		return ""
	}
	return d.strings[index-1] // Convert to 0-based indexing
}

// Size returns the current dictionary size
func (d *LZ78Dictionary) Size() int {
	return len(d.strings)
}

// Reset clears the dictionary
func (d *LZ78Dictionary) Reset() {
	d.entries = make(map[string]int)
	d.strings = d.strings[:0]
	d.nextIndex = 1
}

// NewLZ78Encoder creates a new LZ78 encoder
func NewLZ78Encoder() (*LZ78Encoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	return &LZ78Encoder{
		name:            "LZ78",
		category:        core.AlgorithmCategoryDictionary,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		maxDictSize:     4096, // 4K dictionary entries
		resetThreshold:  4096, // Reset when full
	}, nil
}

// GetName returns the algorithm name
func (l *LZ78Encoder) GetName() string {
	return l.name
}

// GetCategory returns the algorithm category
func (l *LZ78Encoder) GetCategory() core.AlgorithmCategory {
	return l.category
}

// GetCompressionType returns the compression type
func (l *LZ78Encoder) GetCompressionType() core.CompressionType {
	return l.compressionType
}

// SetParameters sets algorithm parameters
func (l *LZ78Encoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		l.parameters[k] = v

		switch k {
		case "max_dict_size":
			if val, ok := v.(int); ok && val > 0 && val <= 65536 {
				l.maxDictSize = val
			}
		case "reset_threshold":
			if val, ok := v.(int); ok && val > 0 && val <= 65536 {
				l.resetThreshold = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (l *LZ78Encoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range l.parameters {
		result[k] = v
	}
	result["max_dict_size"] = l.maxDictSize
	result["reset_threshold"] = l.resetThreshold
	return result
}

// GetInfo gets comprehensive algorithm information
func (l *LZ78Encoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 l.name,
		"category":             string(l.category),
		"compression_type":     string(l.compressionType),
		"parameters":           l.GetParameters(),
		"supports_streaming":   true,
		"thread_safe":          false,
		"deterministic":        true,
		"memory_bounded":       true,
		"max_dict_size":        l.maxDictSize,
		"algorithm_complexity": "O(n*log(d)) time, O(d) space where d is dictionary size",
		"best_use_case":        "General purpose, good for structured data with recurring patterns",
	}
}

// performLZ78Compression performs the actual LZ78 compression
func (l *LZ78Encoder) performLZ78Compression(input []byte) ([]byte, map[string]interface{}) {
	if len(input) == 0 {
		return input, make(map[string]interface{})
	}

	dict := NewLZ78Dictionary(l.maxDictSize)
	var tokens []LZ78Token
	var result []byte

	stats := map[string]interface{}{
		"tokens_generated":       0,
		"dictionary_entries":     0,
		"dictionary_resets":      0,
		"avg_match_length":       0.0,
		"max_match_length":       0,
		"compression_efficiency": 0.0,
	}

	i := 0
	tokensGenerated := 0
	totalMatchLength := 0
	maxMatchLength := 0
	dictResets := 0

	// Write header: [max_dict_size(2)][reset_threshold(2)]
	header := make([]byte, 4)
	header[0] = byte(l.maxDictSize >> 8)
	header[1] = byte(l.maxDictSize)
	header[2] = byte(l.resetThreshold >> 8)
	header[3] = byte(l.resetThreshold)
	result = append(result, header...)

	for i < len(input) {
		// Find the longest match in the dictionary
		currentString := ""
		matchIndex := 0
		matchLength := 0

		// Try to extend the match as long as possible
		for j := i; j < len(input) && j-i < 255; j++ { // Limit match length to 255
			testString := string(input[i : j+1])
			if dictIndex := dict.Find(testString); dictIndex > 0 {
				currentString = testString
				matchIndex = dictIndex
				matchLength = len(testString)
			} else {
				break
			}
		}

		// Determine the next character
		var nextChar byte
		nextCharPos := i + matchLength
		if nextCharPos < len(input) {
			nextChar = input[nextCharPos]
		}

		// Create token
		token := LZ78Token{
			Index:     matchIndex,
			Character: nextChar,
		}
		tokens = append(tokens, token)
		tokensGenerated++

		// Encode token: [index(2)][character(1)]
		tokenBytes := make([]byte, 3)
		tokenBytes[0] = byte(token.Index >> 8)
		tokenBytes[1] = byte(token.Index)
		tokenBytes[2] = token.Character
		result = append(result, tokenBytes...)

		// Add new string to dictionary (current match + next character)
		if nextCharPos < len(input) {
			newString := currentString + string(nextChar)
			dict.Add(newString)

			// Track statistics
			if matchLength > maxMatchLength {
				maxMatchLength = matchLength
			}
			totalMatchLength += matchLength
		}

		// Check if dictionary should be reset
		if dict.Size() >= l.resetThreshold {
			dict.Reset()
			dictResets++
		}

		// Move to next position
		if nextCharPos < len(input) {
			i = nextCharPos + 1
		} else {
			i = nextCharPos
		}
	}

	// Calculate statistics
	stats["tokens_generated"] = tokensGenerated
	stats["dictionary_entries"] = dict.Size()
	stats["dictionary_resets"] = dictResets
	stats["max_match_length"] = maxMatchLength

	if tokensGenerated > 0 {
		stats["avg_match_length"] = float64(totalMatchLength) / float64(tokensGenerated)
	}

	if len(input) > 0 {
		stats["compression_efficiency"] = float64(totalMatchLength) / float64(len(input))
	}

	return result, stats
}

// performLZ78Decompression performs LZ78 decompression
func (l *LZ78Encoder) performLZ78Decompression(compressed []byte) ([]byte, error) {
	if len(compressed) < 4 {
		return nil, fmt.Errorf("compressed data too short for LZ78 header")
	}

	// Read header
	maxDictSize := int(compressed[0])<<8 | int(compressed[1])
	resetThreshold := int(compressed[2])<<8 | int(compressed[3])

	// Validate parameters
	if maxDictSize != l.maxDictSize || resetThreshold != l.resetThreshold {
		return nil, fmt.Errorf("decoder parameters don't match encoded data")
	}

	dict := NewLZ78Dictionary(maxDictSize)
	var result []byte
	pos := 4 // Skip header

	for pos+2 < len(compressed) {
		// Read token: [index(2)][character(1)]
		index := int(compressed[pos])<<8 | int(compressed[pos+1])
		character := compressed[pos+2]
		pos += 3

		// Reconstruct string
		var tokenString string
		if index > 0 {
			dictString := dict.Get(index)
			if dictString == "" {
				return nil, fmt.Errorf("invalid dictionary index: %d", index)
			}
			tokenString = dictString + string(character)
		} else {
			tokenString = string(character)
		}

		// Append to result
		result = append(result, []byte(tokenString)...)

		// Add to dictionary
		dict.Add(tokenString)

		// Check for dictionary reset
		if dict.Size() >= resetThreshold {
			dict.Reset()
		}
	}

	return result, nil
}

// Compress compresses data using LZ78 with aerospace-grade performance monitoring
func (l *LZ78Encoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    l.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var compressedData []byte
	var lz78Stats map[string]interface{}

	profile, err := l.monitor.ProfileOperation(ctx, "lz78_compress", int64(len(data)), func() error {
		compressedData, lz78Stats = l.performLZ78Compression(data)
		return nil
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("LZ78 compression failed: %v", err),
			Algorithm:   l.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}

	// Calculate compression ratio
	compressionRatio := float64(len(data)) / float64(len(compressedData))

	// Convert profile to aerospace precision metrics
	precisionMetrics := l.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))

	metadata := map[string]interface{}{
		"max_dict_size":          l.maxDictSize,
		"reset_threshold":        l.resetThreshold,
		"tokens_generated":       lz78Stats["tokens_generated"],
		"dictionary_entries":     lz78Stats["dictionary_entries"],
		"dictionary_resets":      lz78Stats["dictionary_resets"],
		"avg_match_length":       lz78Stats["avg_match_length"],
		"max_match_length":       lz78Stats["max_match_length"],
		"compression_efficiency": lz78Stats["compression_efficiency"],
		"header_overhead":        4, // 4 bytes header
		"dictionary_adaptive":    true,
		"explicit_dictionary":    true,
	}

	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    l.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses LZ78-compressed data with aerospace-grade performance monitoring
func (l *LZ78Encoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          l.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.StatisticalPrecisionMetrics{},
		}, nil
	}

	var decompressedData []byte

	profile, err := l.monitor.ProfileOperation(ctx, "lz78_decompress", int64(len(compressedData)), func() error {
		var decompErr error
		decompressedData, decompErr = l.performLZ78Decompression(compressedData)
		return decompErr
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("LZ78 decompression failed: %v", err),
			Algorithm:      l.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	// Convert profile to aerospace precision metrics
	precisionMetrics := l.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))

	decompMetadata := map[string]interface{}{
		"original_compressed_size":  len(compressedData),
		"decompressed_size":         len(decompressedData),
		"algorithm":                 l.name,
		"dictionary_reconstruction": true,
	}

	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          l.name,
		Metadata:               decompMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (l *LZ78Encoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
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

		// LZ78-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // LZ78 is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}
