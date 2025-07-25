package benchmarks

import (
	"hybrid-compression-study/pkg/algorithms"
	"hybrid-compression-study/pkg/core"
)

// FastCombinationGenerator creates algorithm combinations using fast implementations
type FastCombinationGenerator struct {
	*CombinationGenerator
}

// NewFastCombinationGenerator creates a combination generator with fast algorithms
func NewFastCombinationGenerator() *FastCombinationGenerator {
	baseGen := NewCombinationGenerator()

	// Override algorithm creators with fast implementations
	fastGen := &FastCombinationGenerator{
		CombinationGenerator: baseGen,
	}

	// Replace with fast algorithm implementations
	fastGen.availableAlgorithms = map[string]*AlgorithmInfo{
		"SimpleRLE": &AlgorithmInfo{
			Name:      "SimpleRLE",
			Category:  core.AlgorithmCategoryRunLength,
			Creator:   func() (core.CompressionAlgorithm, error) { return algorithms.NewSimpleRLEEncoder() },
			Available: true,
		},
		"LZW": &AlgorithmInfo{
			Name:      "LZW",
			Category:  core.AlgorithmCategoryDictionary,
			Creator:   func() (core.CompressionAlgorithm, error) { return algorithms.NewLZWEncoder() },
			Available: true,
		},
		"Huffman": &AlgorithmInfo{
			Name:      "Huffman",
			Category:  core.AlgorithmCategoryEntropyCoding,
			Creator:   func() (core.CompressionAlgorithm, error) { return algorithms.NewHuffmanEncoder() },
			Available: true,
		},
		// NOTE: Excluded BWT, MTF, Arithmetic due to performance issues
		// NOTE: Excluded heavy monitoring algorithms
	}

	return fastGen
}

// GetFastAlgorithmNames returns the names of fast algorithms available
func (fcg *FastCombinationGenerator) GetFastAlgorithmNames() []string {
	return []string{"SimpleRLE", "LZW", "Huffman"}
}
