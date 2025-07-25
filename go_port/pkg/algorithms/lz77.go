// Package algorithms provides LZ77 implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"math"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// LZ77Encoder implements LZ77 compression with aerospace-grade precision monitoring
type LZ77Encoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor

	// LZ77-specific parameters
	windowSize  int // Size of the sliding window (search buffer)
	bufferSize  int // Size of the lookahead buffer
	minMatchLen int // Minimum match length
	maxMatchLen int // Maximum match length
}

// LZ77Match represents a match found in the sliding window
type LZ77Match struct {
	Distance int  // Distance back to the match
	Length   int  // Length of the match
	NextByte byte // Next byte after the match
}

// NewLZ77Encoder creates a new LZ77 encoder
func NewLZ77Encoder() (*LZ77Encoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	return &LZ77Encoder{
		name:            "LZ77",
		category:        core.AlgorithmCategoryDictionary,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		windowSize:      4096, // 4KB default window
		bufferSize:      256,  // 256 byte lookahead buffer
		minMatchLen:     3,    // Minimum 3 bytes to be worth encoding
		maxMatchLen:     258,  // Standard maximum match length
	}, nil
}

// GetName returns the algorithm name
func (l *LZ77Encoder) GetName() string {
	return l.name
}

// GetCategory returns the algorithm category
func (l *LZ77Encoder) GetCategory() core.AlgorithmCategory {
	return l.category
}

// GetCompressionType returns the compression type
func (l *LZ77Encoder) GetCompressionType() core.CompressionType {
	return l.compressionType
}

// SetParameters sets algorithm parameters
func (l *LZ77Encoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		l.parameters[k] = v

		switch k {
		case "window_size":
			if val, ok := v.(int); ok && val > 0 && val <= 32768 {
				l.windowSize = val
			}
		case "buffer_size":
			if val, ok := v.(int); ok && val > 0 && val <= 256 {
				l.bufferSize = val
			}
		case "min_match_len":
			if val, ok := v.(int); ok && val >= 2 && val <= 10 {
				l.minMatchLen = val
			}
		case "max_match_len":
			if val, ok := v.(int); ok && val >= l.minMatchLen && val <= 258 {
				l.maxMatchLen = val
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (l *LZ77Encoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range l.parameters {
		result[k] = v
	}
	result["window_size"] = l.windowSize
	result["buffer_size"] = l.bufferSize
	result["min_match_len"] = l.minMatchLen
	result["max_match_len"] = l.maxMatchLen
	return result
}

// GetInfo gets comprehensive algorithm information
func (l *LZ77Encoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 l.name,
		"category":             string(l.category),
		"compression_type":     string(l.compressionType),
		"parameters":           l.GetParameters(),
		"supports_streaming":   true,
		"thread_safe":          false,
		"deterministic":        true,
		"memory_bounded":       true,
		"window_size":          l.windowSize,
		"algorithm_complexity": "O(n*w) time, O(w) space where w is window size",
		"best_use_case":        "General purpose, text files, binary data with repeated patterns",
	}
}

// findLongestMatch finds the longest match in the sliding window
func (l *LZ77Encoder) findLongestMatch(data []byte, pos int) LZ77Match {
	bestMatch := LZ77Match{Distance: 0, Length: 0}

	// Determine search window boundaries
	windowStart := pos - l.windowSize
	if windowStart < 0 {
		windowStart = 0
	}

	// Determine lookahead buffer size
	remainingBytes := len(data) - pos
	lookAheadSize := l.bufferSize
	if remainingBytes < lookAheadSize {
		lookAheadSize = remainingBytes
	}

	if lookAheadSize < l.minMatchLen {
		// Not enough data for minimum match
		if lookAheadSize > 0 {
			bestMatch.NextByte = data[pos]
		}
		return bestMatch
	}

	// Search for matches in the sliding window
	for searchPos := windowStart; searchPos < pos; searchPos++ {
		matchLen := 0

		// Count matching bytes
		for matchLen < lookAheadSize &&
			matchLen < l.maxMatchLen &&
			searchPos+matchLen < pos &&
			data[searchPos+matchLen] == data[pos+matchLen] {
			matchLen++
		}

		// Check if this is a better match
		if matchLen >= l.minMatchLen && matchLen > bestMatch.Length {
			bestMatch.Distance = pos - searchPos
			bestMatch.Length = matchLen
		}
	}

	// Set next byte after the match
	nextBytePos := pos + bestMatch.Length
	if nextBytePos < len(data) {
		bestMatch.NextByte = data[nextBytePos]
	}

	return bestMatch
}

// encodeLZ77Token encodes an LZ77 token (distance, length, literal)
func (l *LZ77Encoder) encodeLZ77Token(match LZ77Match, hasNextByte bool) []byte {
	// Simple encoding format:
	// [flag(1)] [distance(2)] [length(2)] [literal(1)]
	// flag: 0x80 = has match, 0x40 = has literal

	var token []byte

	if match.Length > 0 {
		// Match found
		flag := byte(0x80) // Has match
		if hasNextByte {
			flag |= 0x40 // Has literal
		}

		token = append(token, flag)
		token = append(token, byte(match.Distance>>8), byte(match.Distance))
		token = append(token, byte(match.Length>>8), byte(match.Length))

		if hasNextByte {
			token = append(token, match.NextByte)
		}
	} else {
		// No match, just literal
		flag := byte(0x40) // Has literal only
		token = append(token, flag)
		token = append(token, match.NextByte)
	}

	return token
}

// decodeLZ77Token decodes an LZ77 token
func (l *LZ77Encoder) decodeLZ77Token(tokenData []byte, pos *int) (LZ77Match, bool, error) {
	if *pos >= len(tokenData) {
		return LZ77Match{}, false, fmt.Errorf("unexpected end of token data")
	}

	flag := tokenData[*pos]
	*pos++

	match := LZ77Match{}
	hasLiteral := (flag & 0x40) != 0
	hasMatch := (flag & 0x80) != 0

	if hasMatch {
		if *pos+4 > len(tokenData) {
			return LZ77Match{}, false, fmt.Errorf("insufficient data for match")
		}

		match.Distance = int(tokenData[*pos])<<8 | int(tokenData[*pos+1])
		match.Length = int(tokenData[*pos+2])<<8 | int(tokenData[*pos+3])
		*pos += 4
	}

	if hasLiteral {
		if *pos >= len(tokenData) {
			return LZ77Match{}, false, fmt.Errorf("insufficient data for literal")
		}
		match.NextByte = tokenData[*pos]
		*pos++
	}

	return match, hasLiteral, nil
}

// performLZ77Compression performs the actual LZ77 compression
func (l *LZ77Encoder) performLZ77Compression(input []byte) ([]byte, map[string]interface{}) {
	if len(input) == 0 {
		return input, make(map[string]interface{})
	}

	var result []byte
	stats := map[string]interface{}{
		"matches_found":          0,
		"literals":               0,
		"total_match_len":        0,
		"avg_match_len":          0.0,
		"max_match_len":          0,
		"avg_distance":           0.0,
		"compression_efficiency": 0.0,
	}

	pos := 0
	matchesFound := 0
	totalMatchLen := 0
	totalDistance := 0
	literals := 0
	maxMatchLen := 0

	// Write header: [window_size(2)][buffer_size(2)][min_match(1)][max_match(2)]
	header := make([]byte, 7)
	header[0] = byte(l.windowSize >> 8)
	header[1] = byte(l.windowSize)
	header[2] = byte(l.bufferSize >> 8)
	header[3] = byte(l.bufferSize)
	header[4] = byte(l.minMatchLen)
	header[5] = byte(l.maxMatchLen >> 8)
	header[6] = byte(l.maxMatchLen)
	result = append(result, header...)

	for pos < len(input) {
		match := l.findLongestMatch(input, pos)

		hasNextByte := (pos + match.Length) < len(input)
		token := l.encodeLZ77Token(match, hasNextByte)
		result = append(result, token...)

		if match.Length > 0 {
			matchesFound++
			totalMatchLen += match.Length
			totalDistance += match.Distance
			if match.Length > maxMatchLen {
				maxMatchLen = match.Length
			}
			pos += match.Length
		}

		if hasNextByte {
			literals++
			pos++
		} else {
			break
		}
	}

	// Calculate statistics
	stats["matches_found"] = matchesFound
	stats["literals"] = literals
	stats["total_match_len"] = totalMatchLen
	stats["max_match_len"] = maxMatchLen

	if matchesFound > 0 {
		stats["avg_match_len"] = float64(totalMatchLen) / float64(matchesFound)
		stats["avg_distance"] = float64(totalDistance) / float64(matchesFound)
	}

	if len(input) > 0 {
		stats["compression_efficiency"] = float64(totalMatchLen) / float64(len(input))
	}

	return result, stats
}

// performLZ77Decompression performs LZ77 decompression
func (l *LZ77Encoder) performLZ77Decompression(compressed []byte) ([]byte, error) {
	if len(compressed) < 7 {
		return nil, fmt.Errorf("compressed data too short for LZ77 header")
	}

	// Read header
	windowSize := int(compressed[0])<<8 | int(compressed[1])
	bufferSize := int(compressed[2])<<8 | int(compressed[3])
	minMatchLen := int(compressed[4])
	maxMatchLen := int(compressed[5])<<8 | int(compressed[6])

	// Validate parameters
	if windowSize != l.windowSize || bufferSize != l.bufferSize ||
		minMatchLen != l.minMatchLen || maxMatchLen != l.maxMatchLen {
		return nil, fmt.Errorf("decoder parameters don't match encoded data")
	}

	var result []byte
	pos := 7 // Skip header

	for pos < len(compressed) {
		match, hasLiteral, err := l.decodeLZ77Token(compressed, &pos)
		if err != nil {
			return nil, fmt.Errorf("failed to decode token: %w", err)
		}

		// Copy match if it exists
		if match.Length > 0 {
			if match.Distance > len(result) {
				return nil, fmt.Errorf("invalid match distance: %d > %d", match.Distance, len(result))
			}

			// Copy bytes from the match
			startPos := len(result) - match.Distance
			for i := 0; i < match.Length; i++ {
				if startPos+i >= len(result) {
					return nil, fmt.Errorf("match extends beyond available data")
				}
				result = append(result, result[startPos+i])
			}
		}

		// Add literal if it exists
		if hasLiteral {
			result = append(result, match.NextByte)
		}
	}

	return result, nil
}

// Compress compresses data using LZ77 with aerospace-grade performance monitoring
func (l *LZ77Encoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
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
	var lz77Stats map[string]interface{}

	profile, err := l.monitor.ProfileOperation(ctx, "lz77_compress", int64(len(data)), func() error {
		compressedData, lz77Stats = l.performLZ77Compression(data)
		return nil
	})

	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("LZ77 compression failed: %v", err),
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
		"window_size":            l.windowSize,
		"buffer_size":            l.bufferSize,
		"min_match_len":          l.minMatchLen,
		"max_match_len":          l.maxMatchLen,
		"matches_found":          lz77Stats["matches_found"],
		"literals":               lz77Stats["literals"],
		"total_match_len":        lz77Stats["total_match_len"],
		"avg_match_len":          lz77Stats["avg_match_len"],
		"max_match_len_found":    lz77Stats["max_match_len"],
		"avg_distance":           lz77Stats["avg_distance"],
		"compression_efficiency": lz77Stats["compression_efficiency"],
		"header_overhead":        7, // 7 bytes header
		"dictionary_stage":       true,
		"sliding_window":         true,
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

// Decompress decompresses LZ77-compressed data with aerospace-grade performance monitoring
func (l *LZ77Encoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
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

	profile, err := l.monitor.ProfileOperation(ctx, "lz77_decompress", int64(len(compressedData)), func() error {
		var decompErr error
		decompressedData, decompErr = l.performLZ77Decompression(compressedData)
		return decompErr
	})

	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("LZ77 decompression failed: %v", err),
			Algorithm:      l.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}

	// Convert profile to aerospace precision metrics
	precisionMetrics := l.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))

	decompMetadata := map[string]interface{}{
		"original_compressed_size": len(compressedData),
		"decompressed_size":        len(decompressedData),
		"algorithm":                l.name,
		"dictionary_reverse":       true,
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
func (l *LZ77Encoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, inputSize, outputSize int64) core.StatisticalPrecisionMetrics {
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

		// LZ77-specific metrics
		MemoryOverheadRatio:            float64(profile.MemoryPeakBytes) / float64(inputSize),
		CPUEfficiencyBytesPerCPUSecond: float64(inputSize) / math.Max(profile.DurationS, 0.000001),
		DeterminismScore:               1.0, // LZ77 is deterministic
		WorstCaseLatencyNs:             profile.DurationNs,
		EnergyEfficiencyBytesPerNs:     float64(inputSize) / math.Max(float64(profile.DurationNs), 1.0),
		BitsPerByte:                    float64(outputSize*8) / float64(inputSize),
		EntropyEfficiency:              float64(inputSize) / float64(outputSize*8),
	}

	// Update formatted times
	metrics.UpdateFormattedTimes()

	return metrics
}
