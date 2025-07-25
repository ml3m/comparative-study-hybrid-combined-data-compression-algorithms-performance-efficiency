// Package benchmarks provides comprehensive benchmarking infrastructure for compression algorithms.
//
// This package contains sophisticated test data generators and benchmarking frameworks
// designed to thoroughly evaluate compression algorithm combinations across multiple dimensions.
package benchmarks

import (
	"crypto/rand"
	"fmt"
	"strings"
	"time"
)

// InputType represents different categories of test input data
type InputType string

const (
	InputTypeRepetitive    InputType = "repetitive"     // Highly repetitive data (favors RLE)
	InputTypeTextPatterns  InputType = "text_patterns"  // Text with common patterns (favors LZ*)
	InputTypeRandom        InputType = "random"         // Random data (favors Huffman/Arithmetic)
	InputTypeSequential    InputType = "sequential"     // Sequential numeric data (favors Delta)
	InputTypeNaturalText   InputType = "natural_text"   // Natural language (favors BWT+MTF)
	InputTypeStructuredBin InputType = "structured_bin" // Structured binary data
	InputTypeMixed         InputType = "mixed"          // Mixed characteristics
	InputTypeSparse        InputType = "sparse"         // Sparse data with many zeros
	InputTypeAlternating   InputType = "alternating"    // Alternating patterns
	InputTypeLog           InputType = "log"            // Log-like structured text
)

// InputSize represents different test data sizes
type InputSize struct {
	Name  string
	Bytes int64
}

var StandardSizes = []InputSize{
	{"1KB", 1024},
	{"10KB", 10 * 1024},
	{"100KB", 100 * 1024},
	{"1MB", 1024 * 1024},
	{"10MB", 10 * 1024 * 1024},
}

// TestInput represents a generated test input with metadata
type TestInput struct {
	Data            []byte                 `json:"data"`
	Type            InputType              `json:"type"`
	Size            int64                  `json:"size"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	ExpectedFavors  []string               `json:"expected_favors"` // Algorithms expected to perform well
	Characteristics map[string]interface{} `json:"characteristics"` // Input characteristics for analysis
}

// InputGenerator creates engineered test inputs for comprehensive algorithm evaluation
type InputGenerator struct {
	seed int64
}

// NewInputGenerator creates a new input generator
func NewInputGenerator() *InputGenerator {
	return &InputGenerator{
		seed: time.Now().UnixNano(),
	}
}

// GenerateAllInputs creates a comprehensive set of test inputs for all scenarios
func (ig *InputGenerator) GenerateAllInputs() ([]*TestInput, error) {
	var inputs []*TestInput

	inputTypes := []InputType{
		InputTypeRepetitive,
		InputTypeTextPatterns,
		InputTypeRandom,
		InputTypeSequential,
		InputTypeNaturalText,
		InputTypeStructuredBin,
		InputTypeMixed,
		InputTypeSparse,
		InputTypeAlternating,
		InputTypeLog,
	}

	for _, inputType := range inputTypes {
		for _, size := range StandardSizes {
			input, err := ig.GenerateInput(inputType, size.Bytes)
			if err != nil {
				return nil, fmt.Errorf("failed to generate %s input of size %s: %w", inputType, size.Name, err)
			}
			input.Name = fmt.Sprintf("%s_%s", inputType, size.Name)
			inputs = append(inputs, input)
		}
	}

	return inputs, nil
}

// GenerateInput creates a specific type of test input
func (ig *InputGenerator) GenerateInput(inputType InputType, size int64) (*TestInput, error) {
	switch inputType {
	case InputTypeRepetitive:
		return ig.generateRepetitiveData(size)
	case InputTypeTextPatterns:
		return ig.generateTextPatterns(size)
	case InputTypeRandom:
		return ig.generateRandomData(size)
	case InputTypeSequential:
		return ig.generateSequentialData(size)
	case InputTypeNaturalText:
		return ig.generateNaturalText(size)
	case InputTypeStructuredBin:
		return ig.generateStructuredBinary(size)
	case InputTypeMixed:
		return ig.generateMixedData(size)
	case InputTypeSparse:
		return ig.generateSparseData(size)
	case InputTypeAlternating:
		return ig.generateAlternatingData(size)
	case InputTypeLog:
		return ig.generateLogData(size)
	default:
		return nil, fmt.Errorf("unknown input type: %s", inputType)
	}
}

// generateRepetitiveData creates highly repetitive data that should favor RLE
func (ig *InputGenerator) generateRepetitiveData(size int64) (*TestInput, error) {
	data := make([]byte, size)

	// Create blocks of repeated bytes with varying run lengths
	patterns := []struct {
		byte_val   byte
		run_length int
	}{
		{0xAA, 50},  // Long runs of 0xAA
		{0x00, 100}, // Very long runs of zeros
		{0xFF, 25},  // Medium runs of 0xFF
		{0x55, 75},  // Long runs of 0x55
	}

	pos := int64(0)
	patternIndex := 0

	for pos < size {
		pattern := patterns[patternIndex%len(patterns)]
		runLength := int64(pattern.run_length)

		if pos+runLength > size {
			runLength = size - pos
		}

		for i := int64(0); i < runLength; i++ {
			data[pos+i] = pattern.byte_val
		}

		pos += runLength
		patternIndex++
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeRepetitive,
		Size:           size,
		Description:    "Highly repetitive data with long runs of identical bytes",
		ExpectedFavors: []string{"RLE"},
		Characteristics: map[string]interface{}{
			"repetition_factor": 0.95,
			"unique_bytes":      4,
			"avg_run_length":    62.5,
		},
	}, nil
}

// generateTextPatterns creates text with common patterns that should favor dictionary algorithms
func (ig *InputGenerator) generateTextPatterns(size int64) (*TestInput, error) {
	commonWords := []string{
		"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
	}

	commonPhrases := []string{
		"in the", "to the", "of the", "and the", "for the", "with the", "on the", "at the",
		"this is", "that is", "there is", "here is", "what is", "when is", "where is",
		"compression algorithm", "data structure", "performance metrics", "statistical analysis",
	}

	data := make([]byte, 0, size)

	for int64(len(data)) < size {
		// Mix words and phrases
		if len(data)%3 == 0 && int64(len(data)) < size-50 {
			phrase := commonPhrases[len(data)%len(commonPhrases)]
			data = append(data, []byte(phrase)...)
			data = append(data, ' ')
		} else {
			word := commonWords[len(data)%len(commonWords)]
			data = append(data, []byte(word)...)
			data = append(data, ' ')
		}

		// Add some punctuation
		if len(data)%50 == 0 {
			data = append(data, '.', ' ')
		}
	}

	// Trim to exact size
	if int64(len(data)) > size {
		data = data[:size]
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeTextPatterns,
		Size:           size,
		Description:    "Text with repeated words and phrases, common dictionary patterns",
		ExpectedFavors: []string{"LZ77", "LZ78", "LZW"},
		Characteristics: map[string]interface{}{
			"vocabulary_size":   len(commonWords) + len(commonPhrases),
			"repetition_factor": 0.8,
			"pattern_diversity": "medium",
		},
	}, nil
}

// generateRandomData creates random data that should favor entropy-based algorithms
func (ig *InputGenerator) generateRandomData(size int64) (*TestInput, error) {
	data := make([]byte, size)
	_, err := rand.Read(data)
	if err != nil {
		return nil, fmt.Errorf("failed to generate random data: %w", err)
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeRandom,
		Size:           size,
		Description:    "Cryptographically random data with high entropy",
		ExpectedFavors: []string{"Huffman", "Arithmetic"},
		Characteristics: map[string]interface{}{
			"entropy":           "high",
			"predictability":    0.0,
			"compression_limit": "theoretical_entropy",
		},
	}, nil
}

// generateSequentialData creates sequential numeric data that should favor delta encoding
func (ig *InputGenerator) generateSequentialData(size int64) (*TestInput, error) {
	data := make([]byte, size)

	// Create sequences of increasing numbers with small variations
	pos := int64(0)
	value := uint32(1000) // Starting value

	for pos < size-4 {
		// Add small random variations to create delta-compressible sequences
		variation := int32(-5 + (pos % 11)) // Small variations
		value = uint32(int32(value) + variation + 1)

		// Store as 4-byte little-endian
		data[pos] = byte(value)
		data[pos+1] = byte(value >> 8)
		data[pos+2] = byte(value >> 16)
		data[pos+3] = byte(value >> 24)

		pos += 4
	}

	// Fill remaining bytes
	for pos < size {
		data[pos] = byte(value)
		pos++
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeSequential,
		Size:           size,
		Description:    "Sequential numeric data with small deltas, ideal for delta compression",
		ExpectedFavors: []string{"Delta"},
		Characteristics: map[string]interface{}{
			"sequence_type":  "numeric",
			"delta_variance": "low",
			"predictability": 0.9,
		},
	}, nil
}

// generateNaturalText creates natural language text that should favor BWT+MTF combinations
func (ig *InputGenerator) generateNaturalText(size int64) (*TestInput, error) {
	// Sample of natural English text patterns
	sentences := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Compression algorithms analyze patterns in data to reduce storage requirements.",
		"Statistical analysis reveals optimal performance characteristics under various conditions.",
		"Advanced aerospace applications demand precise measurement and monitoring capabilities.",
		"Data transformation techniques enable efficient storage and transmission systems.",
		"Performance optimization requires careful consideration of time and space complexity.",
		"Hybrid approaches combine multiple strategies to achieve superior compression ratios.",
		"Entropy coding methods exploit statistical properties of input data distributions.",
		"Dictionary-based algorithms identify and eliminate redundant sequence patterns.",
		"Transform-based preprocessing can significantly improve subsequent compression stages.",
	}

	var text strings.Builder
	sentenceIndex := 0

	for int64(text.Len()) < size {
		sentence := sentences[sentenceIndex%len(sentences)]
		text.WriteString(sentence)
		text.WriteString(" ")
		sentenceIndex++

		// Add paragraph breaks
		if sentenceIndex%5 == 0 {
			text.WriteString("\n\n")
		}
	}

	data := []byte(text.String())
	if int64(len(data)) > size {
		data = data[:size]
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeNaturalText,
		Size:           size,
		Description:    "Natural English text with typical language patterns and structure",
		ExpectedFavors: []string{"BWT", "MTF", "Huffman"},
		Characteristics: map[string]interface{}{
			"language":               "english",
			"structure":              "sentences",
			"character_distribution": "natural",
		},
	}, nil
}

// generateStructuredBinary creates structured binary data
func (ig *InputGenerator) generateStructuredBinary(size int64) (*TestInput, error) {
	data := make([]byte, size)
	pos := int64(0)

	// Create structured binary with headers, repeated structures
	for pos < size {
		// Header pattern
		if pos+16 < size {
			copy(data[pos:], []byte{0xDE, 0xAD, 0xBE, 0xEF}) // Magic header
			pos += 4

			// Length field
			length := uint32(64 + (pos % 128))
			data[pos] = byte(length)
			data[pos+1] = byte(length >> 8)
			data[pos+2] = byte(length >> 16)
			data[pos+3] = byte(length >> 24)
			pos += 4

			// Structured payload
			for i := int64(0); i < int64(length) && pos < size; i++ {
				data[pos] = byte(i % 256)
				pos++
			}
		} else {
			// Fill remaining
			for pos < size {
				data[pos] = 0x00
				pos++
			}
		}
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeStructuredBin,
		Size:           size,
		Description:    "Structured binary data with headers and repeated patterns",
		ExpectedFavors: []string{"LZ77", "LZW", "RLE"},
		Characteristics: map[string]interface{}{
			"structure":    "binary_records",
			"header_ratio": 0.1,
			"pattern_size": "variable",
		},
	}, nil
}

// generateMixedData creates data with mixed characteristics
func (ig *InputGenerator) generateMixedData(size int64) (*TestInput, error) {
	data := make([]byte, size)
	pos := int64(0)

	// Mix different data types
	sections := []func([]byte, int64, int64) int64{
		ig.addRandomSection,
		ig.addRepetitiveSection,
		ig.addTextSection,
		ig.addSequentialSection,
	}

	sectionIndex := 0
	for pos < size {
		sectionSize := size / 4
		if pos+sectionSize > size {
			sectionSize = size - pos
		}

		pos = sections[sectionIndex%len(sections)](data, pos, sectionSize)
		sectionIndex++
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeMixed,
		Size:           size,
		Description:    "Mixed data combining random, repetitive, text, and sequential patterns",
		ExpectedFavors: []string{"Deflate", "Arithmetic"},
		Characteristics: map[string]interface{}{
			"sections":   4,
			"diversity":  "high",
			"complexity": "mixed",
		},
	}, nil
}

// Helper functions for mixed data generation
func (ig *InputGenerator) addRandomSection(data []byte, start, length int64) int64 {
	for i := start; i < start+length && i < int64(len(data)); i++ {
		data[i] = byte(i * 17 % 256) // Pseudo-random
	}
	return start + length
}

func (ig *InputGenerator) addRepetitiveSection(data []byte, start, length int64) int64 {
	val := byte(0xAA)
	for i := start; i < start+length && i < int64(len(data)); i++ {
		data[i] = val
	}
	return start + length
}

func (ig *InputGenerator) addTextSection(data []byte, start, length int64) int64 {
	text := "the quick brown fox jumps over the lazy dog "
	textBytes := []byte(text)

	for i := start; i < start+length && i < int64(len(data)); i++ {
		data[i] = textBytes[(i-start)%int64(len(textBytes))]
	}
	return start + length
}

func (ig *InputGenerator) addSequentialSection(data []byte, start, length int64) int64 {
	for i := start; i < start+length && i < int64(len(data)); i++ {
		data[i] = byte((i - start) % 256)
	}
	return start + length
}

// generateSparseData creates sparse data with many zeros
func (ig *InputGenerator) generateSparseData(size int64) (*TestInput, error) {
	data := make([]byte, size)

	// Fill with zeros and add sparse non-zero values
	for i := int64(0); i < size; i += 17 {
		if i < size {
			data[i] = byte(i%127 + 1) // Non-zero values
		}
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeSparse,
		Size:           size,
		Description:    "Sparse data with mostly zeros and occasional non-zero values",
		ExpectedFavors: []string{"RLE", "Huffman"},
		Characteristics: map[string]interface{}{
			"sparsity":     0.94,
			"zero_ratio":   0.94,
			"distribution": "sparse",
		},
	}, nil
}

// generateAlternatingData creates alternating patterns
func (ig *InputGenerator) generateAlternatingData(size int64) (*TestInput, error) {
	data := make([]byte, size)

	patterns := [][]byte{
		{0xAA, 0x55},
		{0xFF, 0x00},
		{0x01, 0x02, 0x04, 0x08},
		{0xF0, 0x0F},
	}

	patternIndex := 0
	pos := int64(0)

	for pos < size {
		pattern := patterns[patternIndex%len(patterns)]
		for i := 0; i < len(pattern) && pos < size; i++ {
			data[pos] = pattern[i]
			pos++
		}
		patternIndex++
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeAlternating,
		Size:           size,
		Description:    "Alternating byte patterns with regular periodicity",
		ExpectedFavors: []string{"LZW", "RLE"},
		Characteristics: map[string]interface{}{
			"periodicity":   "regular",
			"pattern_count": len(patterns),
			"predictable":   true,
		},
	}, nil
}

// generateLogData creates log-like structured text data
func (ig *InputGenerator) generateLogData(size int64) (*TestInput, error) {
	logFormats := []string{
		"2024-01-15 10:30:15 [INFO] Processing request ID: %d",
		"2024-01-15 10:30:16 [ERROR] Connection timeout after %d seconds",
		"2024-01-15 10:30:17 [DEBUG] Memory usage: %d MB",
		"2024-01-15 10:30:18 [WARN] High CPU utilization: %d%%",
		"2024-01-15 10:30:19 [INFO] Transaction completed successfully",
	}

	var logText strings.Builder
	counter := 1000

	for int64(logText.Len()) < size {
		format := logFormats[counter%len(logFormats)]
		var line string

		if strings.Contains(format, "%d") {
			line = fmt.Sprintf(format, counter)
		} else {
			line = format
		}

		logText.WriteString(line)
		logText.WriteString("\n")
		counter++
	}

	data := []byte(logText.String())
	if int64(len(data)) > size {
		data = data[:size]
	}

	return &TestInput{
		Data:           data,
		Type:           InputTypeLog,
		Size:           size,
		Description:    "Structured log data with timestamps and repeated format patterns",
		ExpectedFavors: []string{"LZ77", "LZW", "BWT"},
		Characteristics: map[string]interface{}{
			"structure":       "log_entries",
			"timestamp_ratio": 0.3,
			"format_patterns": len(logFormats),
		},
	}, nil
}
