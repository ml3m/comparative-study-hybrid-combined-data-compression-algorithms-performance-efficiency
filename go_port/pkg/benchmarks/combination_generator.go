package benchmarks

import (
	"fmt"
	"hybrid-compression-study/pkg/algorithms"
	"hybrid-compression-study/pkg/core"
	"sort"
)

// AlgorithmInfo holds metadata about available algorithms
type AlgorithmInfo struct {
	Name      string                                    `json:"name"`
	Category  core.AlgorithmCategory                    `json:"category"`
	Creator   func() (core.CompressionAlgorithm, error) `json:"-"`
	Available bool                                      `json:"available"`
}

// CombinationGenerator generates algorithm combinations for brute force testing
type CombinationGenerator struct {
	availableAlgorithms map[string]*AlgorithmInfo
	selectedAlgorithms  []string
	minCombinationSize  int
	maxCombinationSize  int
}

// NewCombinationGenerator creates a new combination generator
func NewCombinationGenerator() *CombinationGenerator {
	// Initialize all available algorithms
	algorithms := map[string]*AlgorithmInfo{
		"RLE": {
			Name:     "RLE",
			Category: core.AlgorithmCategoryRunLength,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewRLEEncoder()
			},
			Available: true,
		},
		"Delta": {
			Name:     "Delta",
			Category: core.AlgorithmCategoryPredictive,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewDeltaEncoder()
			},
			Available: true,
		},
		"BWT": {
			Name:     "BWT",
			Category: core.AlgorithmCategoryTransform,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewBWTEncoder()
			},
			Available: true,
		},
		"MTF": {
			Name:     "MTF",
			Category: core.AlgorithmCategoryTransform,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewMTFEncoder()
			},
			Available: true,
		},
		"LZ77": {
			Name:     "LZ77",
			Category: core.AlgorithmCategoryDictionary,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewLZ77Encoder()
			},
			Available: true,
		},
		"LZ78": {
			Name:     "LZ78",
			Category: core.AlgorithmCategoryDictionary,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewLZ78Encoder()
			},
			Available: true,
		},
		"LZW": {
			Name:     "LZW",
			Category: core.AlgorithmCategoryDictionary,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewLZWEncoder()
			},
			Available: true,
		},
		"Huffman": {
			Name:     "Huffman",
			Category: core.AlgorithmCategoryEntropyCoding,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewHuffmanEncoder()
			},
			Available: true,
		},
		"Arithmetic": {
			Name:     "Arithmetic",
			Category: core.AlgorithmCategoryHybrid,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewArithmeticEncoder()
			},
			Available: true,
		},
		"Deflate": {
			Name:     "Deflate",
			Category: core.AlgorithmCategoryHybrid,
			Creator: func() (core.CompressionAlgorithm, error) {
				return algorithms.NewDeflateEncoder()
			},
			Available: true,
		},
	}

	return &CombinationGenerator{
		availableAlgorithms: algorithms,
		selectedAlgorithms:  []string{},
		minCombinationSize:  2, // Default minimum
		maxCombinationSize:  4, // Default maximum to keep it manageable
	}
}

// GetAvailableAlgorithms returns the list of available algorithms
func (cg *CombinationGenerator) GetAvailableAlgorithms() map[string]*AlgorithmInfo {
	return cg.availableAlgorithms
}

// SetSelectedAlgorithms sets which algorithms to use in combinations
func (cg *CombinationGenerator) SetSelectedAlgorithms(algorithms []string) error {
	// Validate that all selected algorithms exist
	for _, alg := range algorithms {
		if _, exists := cg.availableAlgorithms[alg]; !exists {
			return fmt.Errorf("algorithm %s is not available", alg)
		}
	}

	cg.selectedAlgorithms = algorithms
	return nil
}

// SetCombinationSizeRange sets the range of combination sizes to generate
func (cg *CombinationGenerator) SetCombinationSizeRange(min, max int) error {
	if min < 1 {
		return fmt.Errorf("minimum combination size must be at least 1")
	}
	if max < min {
		return fmt.Errorf("maximum combination size must be >= minimum")
	}
	if len(cg.selectedAlgorithms) > 0 && max > len(cg.selectedAlgorithms) {
		return fmt.Errorf("maximum combination size cannot exceed number of selected algorithms")
	}

	cg.minCombinationSize = min
	cg.maxCombinationSize = max
	return nil
}

// AlgorithmCombination represents a specific combination of algorithms in order
type AlgorithmCombination struct {
	Algorithms       []string                 `json:"algorithms"`
	Name             string                   `json:"name"`
	Description      string                   `json:"description"`
	ExpectedStrength []string                 `json:"expected_strength"` // What this combination should be good at
	Categories       []core.AlgorithmCategory `json:"categories"`
	Size             int                      `json:"size"`
}

// GenerateAllCombinations generates all possible combinations of selected algorithms
// This implements the brute force approach from bruteForce.txt
func (cg *CombinationGenerator) GenerateAllCombinations() ([]*AlgorithmCombination, error) {
	if len(cg.selectedAlgorithms) == 0 {
		return nil, fmt.Errorf("no algorithms selected")
	}

	var combinations []*AlgorithmCombination

	// Generate combinations for each size from min to max
	for size := cg.minCombinationSize; size <= cg.maxCombinationSize; size++ {
		if size > len(cg.selectedAlgorithms) {
			break
		}

		sizeCombinations := cg.generatePermutations(cg.selectedAlgorithms, size)
		combinations = append(combinations, sizeCombinations...)
	}

	return combinations, nil
}

// generatePermutations generates all permutations of r items from n algorithms
// This calculates P(n,r) = n!/(n-r)! as described in bruteForce.txt
func (cg *CombinationGenerator) generatePermutations(algorithms []string, r int) []*AlgorithmCombination {
	var result []*AlgorithmCombination

	// Generate all permutations of size r
	cg.permute(algorithms, r, []string{}, &result)

	return result
}

// permute is a recursive function to generate permutations
func (cg *CombinationGenerator) permute(remaining []string, r int, current []string, result *[]*AlgorithmCombination) {
	if len(current) == r {
		// Create a copy of current combination
		combination := make([]string, len(current))
		copy(combination, current)

		// Create the algorithm combination
		algoComb := &AlgorithmCombination{
			Algorithms:       combination,
			Name:             cg.generateCombinationName(combination),
			Description:      cg.generateCombinationDescription(combination),
			ExpectedStrength: cg.predictCombinationStrengths(combination),
			Categories:       cg.getCombinationCategories(combination),
			Size:             len(combination),
		}

		*result = append(*result, algoComb)
		return
	}

	// Try each remaining algorithm
	for i, alg := range remaining {
		// Create new slices for recursion
		newRemaining := make([]string, 0, len(remaining)-1)
		newRemaining = append(newRemaining, remaining[:i]...)
		newRemaining = append(newRemaining, remaining[i+1:]...)

		newCurrent := make([]string, len(current)+1)
		copy(newCurrent, current)
		newCurrent[len(current)] = alg

		cg.permute(newRemaining, r, newCurrent, result)
	}
}

// generateCombinationName creates a descriptive name for the combination
func (cg *CombinationGenerator) generateCombinationName(algorithms []string) string {
	return fmt.Sprintf("Pipeline_%s", fmt.Sprintf("%v", algorithms))
}

// generateCombinationDescription creates a description based on the algorithm sequence
func (cg *CombinationGenerator) generateCombinationDescription(algorithms []string) string {
	if len(algorithms) == 0 {
		return "Empty pipeline"
	}

	description := fmt.Sprintf("%d-stage pipeline: ", len(algorithms))

	for i, alg := range algorithms {
		if i > 0 {
			description += " → "
		}
		description += alg
	}

	return description
}

// predictCombinationStrengths predicts what types of data this combination should handle well
func (cg *CombinationGenerator) predictCombinationStrengths(algorithms []string) []string {
	strengths := make(map[string]bool)

	for _, alg := range algorithms {
		algInfo, exists := cg.availableAlgorithms[alg]
		if !exists {
			continue
		}

		// Add strengths based on algorithm category
		switch algInfo.Category {
		case core.AlgorithmCategoryRunLength:
			strengths["repetitive_data"] = true
			strengths["sparse_data"] = true
		case core.AlgorithmCategoryDictionary:
			strengths["text_patterns"] = true
			strengths["structured_data"] = true
			strengths["redundant_sequences"] = true
		case core.AlgorithmCategoryEntropyCoding:
			strengths["random_data"] = true
			strengths["entropy_optimization"] = true
		case core.AlgorithmCategoryTransform:
			strengths["natural_text"] = true
			strengths["symbol_reordering"] = true
		case core.AlgorithmCategoryPredictive:
			strengths["sequential_data"] = true
			strengths["numerical_sequences"] = true
		case core.AlgorithmCategoryHybrid:
			strengths["mixed_data"] = true
			strengths["general_purpose"] = true
		}
	}

	// Convert map to slice
	var result []string
	for strength := range strengths {
		result = append(result, strength)
	}

	sort.Strings(result)
	return result
}

// getCombinationCategories returns the categories represented in this combination
func (cg *CombinationGenerator) getCombinationCategories(algorithms []string) []core.AlgorithmCategory {
	categories := make(map[core.AlgorithmCategory]bool)

	for _, alg := range algorithms {
		if algInfo, exists := cg.availableAlgorithms[alg]; exists {
			categories[algInfo.Category] = true
		}
	}

	var result []core.AlgorithmCategory
	for category := range categories {
		result = append(result, category)
	}

	return result
}

// EstimateTotalCombinations calculates the total number of combinations that will be generated
// This implements the sum from bruteForce.txt: Σ(r=min to max) P(n,r)
func (cg *CombinationGenerator) EstimateTotalCombinations() int64 {
	if len(cg.selectedAlgorithms) == 0 {
		return 0
	}

	n := len(cg.selectedAlgorithms)
	total := int64(0)

	for r := cg.minCombinationSize; r <= cg.maxCombinationSize && r <= n; r++ {
		// Calculate P(n,r) = n!/(n-r)!
		permutations := int64(1)
		for i := 0; i < r; i++ {
			permutations *= int64(n - i)
		}
		total += permutations
	}

	return total
}

// GetCombinationStats returns statistics about the combination generation
func (cg *CombinationGenerator) GetCombinationStats() map[string]interface{} {
	stats := map[string]interface{}{
		"selected_algorithms":  len(cg.selectedAlgorithms),
		"min_combination_size": cg.minCombinationSize,
		"max_combination_size": cg.maxCombinationSize,
		"total_combinations":   cg.EstimateTotalCombinations(),
		"available_algorithms": len(cg.availableAlgorithms),
	}

	// Add per-size breakdown
	breakdown := make(map[string]int64)
	n := len(cg.selectedAlgorithms)

	for r := cg.minCombinationSize; r <= cg.maxCombinationSize && r <= n; r++ {
		permutations := int64(1)
		for i := 0; i < r; i++ {
			permutations *= int64(n - i)
		}
		breakdown[fmt.Sprintf("size_%d", r)] = permutations
	}

	stats["size_breakdown"] = breakdown
	return stats
}

// ValidateCombination checks if a combination is logically valid
func (cg *CombinationGenerator) ValidateCombination(algorithms []string) error {
	if len(algorithms) == 0 {
		return fmt.Errorf("empty combination")
	}

	// Check for duplicates
	seen := make(map[string]bool)
	for _, alg := range algorithms {
		if seen[alg] {
			return fmt.Errorf("duplicate algorithm in combination: %s", alg)
		}
		seen[alg] = true
	}

	// Check algorithm order constraints
	categories := make([]core.AlgorithmCategory, len(algorithms))
	for i, alg := range algorithms {
		if algInfo, exists := cg.availableAlgorithms[alg]; exists {
			categories[i] = algInfo.Category
		} else {
			return fmt.Errorf("unknown algorithm: %s", alg)
		}
	}

	// Apply logical ordering constraints
	for i := 0; i < len(categories)-1; i++ {
		current := categories[i]
		next := categories[i+1]

		// Entropy coding should typically come last
		if current == core.AlgorithmCategoryEntropyCoding &&
			next != core.AlgorithmCategoryEntropyCoding {
			return fmt.Errorf("entropy coding algorithm %s should typically be the last stage, not followed by %s",
				algorithms[i], algorithms[i+1])
		}

		// Transform algorithms typically come before dictionary algorithms
		if current == core.AlgorithmCategoryDictionary &&
			next == core.AlgorithmCategoryTransform {
			// This might be suboptimal but not invalid
			// Log as a warning rather than error
		}
	}

	return nil
}
