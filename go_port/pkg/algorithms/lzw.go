// Package algorithms provides LZW (Lempel-Ziv-Welch) implementation with aerospace-grade precision.
package algorithms

import (
	"context"
	"fmt"
	"time"

	"hybrid-compression-study/pkg/core"
	"hybrid-compression-study/internal/performance"
)

// LZWEncoder implements LZW compression with aerospace-grade precision monitoring
type LZWEncoder struct {
	name            string
	category        core.AlgorithmCategory
	compressionType core.CompressionType
	parameters      map[string]interface{}
	monitor         *performance.StatisticalMonitor
	
	// LZW-specific parameters
	maxCodeBits int
	maxCodes    int
}

// NewLZWEncoder creates a new LZW encoder
func NewLZWEncoder() (*LZWEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}
	
	maxCodeBits := 12 // Default 12-bit codes
	maxCodes := (1 << maxCodeBits) - 1
	
	return &LZWEncoder{
		name:            "LZW",
		category:        core.AlgorithmCategoryDictionary,
		compressionType: core.CompressionTypeLossless,
		parameters:      make(map[string]interface{}),
		monitor:         monitor,
		maxCodeBits:     maxCodeBits,
		maxCodes:        maxCodes,
	}, nil
}

// GetName returns the algorithm name
func (l *LZWEncoder) GetName() string {
	return l.name
}

// GetCategory returns the algorithm category
func (l *LZWEncoder) GetCategory() core.AlgorithmCategory {
	return l.category
}

// GetCompressionType returns the compression type
func (l *LZWEncoder) GetCompressionType() core.CompressionType {
	return l.compressionType
}

// SetParameters sets algorithm parameters
func (l *LZWEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		l.parameters[k] = v
		
		switch k {
		case "max_code_bits":
			if val, ok := v.(int); ok && val >= 9 && val <= 16 {
				l.maxCodeBits = val
				l.maxCodes = (1 << val) - 1
			}
		}
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (l *LZWEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range l.parameters {
		result[k] = v
	}
	result["max_code_bits"] = l.maxCodeBits
	result["max_codes"] = l.maxCodes
	return result
}

// GetInfo gets comprehensive algorithm information
func (l *LZWEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               l.name,
		"category":           string(l.category),
		"compression_type":   string(l.compressionType),
		"parameters":         l.GetParameters(),
		"supports_streaming": false,
		"thread_safe":        false,
		"deterministic":      true,
		"memory_bounded":     true,
		"max_code_bits":      l.maxCodeBits,
		"max_codes":          l.maxCodes,
	}
}

// writeCode writes a variable-length code to the output buffer
func (l *LZWEncoder) writeCode(output *[]byte, code int, codeBits int) {
	// Simple implementation: write as big-endian bytes
	// This is not optimal but works for demonstration
	if codeBits <= 8 {
		*output = append(*output, byte(code))
	} else if codeBits <= 16 {
		*output = append(*output, byte(code>>8), byte(code))
	} else {
		*output = append(*output, byte(code>>16), byte(code>>8), byte(code))
	}
}

// readCode reads a variable-length code from the input buffer
func (l *LZWEncoder) readCode(input []byte, pos *int, codeBits int) (int, bool) {
	if codeBits <= 8 {
		if *pos >= len(input) {
			return 0, false
		}
		code := int(input[*pos])
		*pos++
		return code, true
	} else if codeBits <= 16 {
		if *pos+1 >= len(input) {
			return 0, false
		}
		code := int(input[*pos])<<8 | int(input[*pos+1])
		*pos += 2
		return code, true
	} else {
		if *pos+2 >= len(input) {
			return 0, false
		}
		code := int(input[*pos])<<16 | int(input[*pos+1])<<8 | int(input[*pos+2])
		*pos += 3
		return code, true
	}
}

// Compress compresses data using LZW with aerospace-grade performance monitoring
func (l *LZWEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
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
	var dictionarySize, codesGenerated int
	var currentCodeBits int
	
	profile, err := l.monitor.ProfileOperation(ctx, "lzw_compress", int64(len(data)), func() error {
		// Initialize dictionary with all single-byte strings (0-255)
		dictionary := make(map[string]int)
		for i := 0; i < 256; i++ {
			dictionary[string([]byte{byte(i)})] = i
		}
		
		nextCode := 256
		currentCodeBits = 9 // Start with 9 bits
		
		var output []byte
		
		// Write initial parameters
		output = append(output, byte(l.maxCodeBits)) // Store max code bits
		
		w := ""
		
		for _, c := range data {
			wc := w + string(c)
			
			if _, exists := dictionary[wc]; exists {
				w = wc
			} else {
				// Output code for w
				if code, exists := dictionary[w]; exists {
					l.writeCode(&output, code, currentCodeBits)
					codesGenerated++
				}
				
				// Add wc to dictionary if there's space
				if nextCode <= l.maxCodes {
					dictionary[wc] = nextCode
					nextCode++
					
					// Increase code bits if needed
					if nextCode > (1 << currentCodeBits) && currentCodeBits < l.maxCodeBits {
						currentCodeBits++
					}
				}
				
				w = string(c)
			}
		}
		
		// Output code for remaining w
		if w != "" {
			if code, exists := dictionary[w]; exists {
				l.writeCode(&output, code, currentCodeBits)
				codesGenerated++
			}
		}
		
		compressedData = output
		dictionarySize = len(dictionary)
		return nil
	})
	
	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("LZW compression failed: %v", err),
			Algorithm:   l.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}
	
	// Calculate metrics
	compressionRatio := float64(len(data)) / float64(len(compressedData))
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := l.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))
	
	metadata := map[string]interface{}{
		"dictionary_size":      dictionarySize,
		"codes_generated":      codesGenerated,
		"final_code_bits":      currentCodeBits,
		"max_code_bits":        l.maxCodeBits,
		"compression_efficiency": compressionRatio,
		"dictionary_utilization": float64(dictionarySize) / float64(l.maxCodes),
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

// Decompress decompresses LZW-encoded data with aerospace-grade performance monitoring
func (l *LZWEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
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
	
	if len(compressedData) < 1 {
		return nil, &core.DecompressionError{
			Message:        "compressed data too short",
			Algorithm:      l.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}
	
	var decompressedData []byte
	var dictionarySize, codesDecoded int
	
	profile, err := l.monitor.ProfileOperation(ctx, "lzw_decompress", int64(len(compressedData)), func() error {
		// Read max code bits
		maxCodeBits := int(compressedData[0])
		if maxCodeBits < 9 || maxCodeBits > 16 {
			return fmt.Errorf("invalid max code bits: %d", maxCodeBits)
		}
		
		// Initialize reverse dictionary
		dictionary := make(map[int]string)
		for i := 0; i < 256; i++ {
			dictionary[i] = string([]byte{byte(i)})
		}
		
		nextCode := 256
		currentCodeBits := 9
		pos := 1 // Skip the header byte
		
		var result []byte
		
		// Read first code
		oldCode, ok := l.readCode(compressedData, &pos, currentCodeBits)
		if !ok {
			return fmt.Errorf("failed to read first code")
		}
		
		if entry, exists := dictionary[oldCode]; exists {
			result = append(result, []byte(entry)...)
			codesDecoded++
		} else {
			return fmt.Errorf("invalid first code: %d", oldCode)
		}
		
		for pos < len(compressedData) {
			newCode, ok := l.readCode(compressedData, &pos, currentCodeBits)
			if !ok {
				break // End of data
			}
			
			var entry string
			if dictEntry, exists := dictionary[newCode]; exists {
				entry = dictEntry
			} else if newCode == nextCode {
				// Special case: code not in dictionary yet
				if oldEntry, exists := dictionary[oldCode]; exists {
					entry = oldEntry + string(oldEntry[0])
				} else {
					return fmt.Errorf("invalid code sequence at code %d", newCode)
				}
			} else {
				return fmt.Errorf("invalid code: %d", newCode)
			}
			
			result = append(result, []byte(entry)...)
			codesDecoded++
			
			// Add new entry to dictionary
			if nextCode <= (1 << maxCodeBits) - 1 {
				if oldEntry, exists := dictionary[oldCode]; exists {
					dictionary[nextCode] = oldEntry + string(entry[0])
					nextCode++
					
					// Increase code bits if needed
					if nextCode > (1 << currentCodeBits) && currentCodeBits < maxCodeBits {
						currentCodeBits++
					}
				}
			}
			
			oldCode = newCode
		}
		
		decompressedData = result
		dictionarySize = len(dictionary)
		return nil
	})
	
	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("LZW decompression failed: %v", err),
			Algorithm:      l.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := l.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))
	
	// Add decompression-specific metadata
	decompressionMetadata := make(map[string]interface{})
	for k, v := range metadata {
		decompressionMetadata[k] = v
	}
	decompressionMetadata["dictionary_size"] = dictionarySize
	decompressionMetadata["codes_decoded"] = codesDecoded
	
	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          l.name,
		Metadata:               decompressionMetadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (l *LZWEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, dataSize, outputSize int64) core.StatisticalPrecisionMetrics {
	metrics := core.StatisticalPrecisionMetrics{
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
		DeterminismScore:              1.0, // LZW is deterministic
		MemoryOverheadRatio:           float64(profile.MemoryPeakBytes) / float64(dataSize),
		CPUEfficiencyBytesPerCPUSecond: float64(dataSize) / max(0.001, profile.CPUTimeUserS+profile.CPUTimeSystemS),
		IOEfficiencyRatio:             float64(dataSize) / max(1.0, float64(profile.IOReadBytes+profile.IOWriteBytes)),
	}
	
	metrics.UpdateFormattedTimes()
	return metrics
} 