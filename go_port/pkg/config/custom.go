// Package config provides custom configuration format for compression pipelines.
package config

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// CustomConfig represents the custom configuration format
type CustomConfig struct {
	Variables       map[string]interface{}            // Global variables
	Pipelines       map[string]*CustomPipeline        // Pipeline definitions
	AlgParams       map[string]map[string]interface{} // Algorithm-specific parameters
	DefaultPipeline string                            // Default pipeline name
}

// CustomPipeline represents a pipeline in custom format
type CustomPipeline struct {
	Name        string   // Pipeline name
	Algorithms  []string // Algorithm sequence
	Description string   // Optional description
}

// NewCustomConfig creates a new custom configuration
func NewCustomConfig() *CustomConfig {
	return &CustomConfig{
		Variables: make(map[string]interface{}),
		Pipelines: make(map[string]*CustomPipeline),
		AlgParams: make(map[string]map[string]interface{}),
	}
}

// ParseCustomConfig parses a custom configuration file
func ParseCustomConfig(filename string) (*CustomConfig, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	config := NewCustomConfig()
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		if err := parseLine(config, line, lineNum); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Set default pipeline if not specified
	if config.DefaultPipeline == "" && len(config.Pipelines) > 0 {
		for name := range config.Pipelines {
			config.DefaultPipeline = name
			break
		}
	}

	return config, nil
}

// parseLine parses a single configuration line
func parseLine(config *CustomConfig, line string, lineNum int) error {
	// Handle different line types
	if strings.Contains(line, "=") {
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid assignment syntax")
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		// Remove quotes if present
		if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
			(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
			value = value[1 : len(value)-1]
		}

		// Handle special directives
		if key == "default" {
			config.DefaultPipeline = value
			return nil
		}

		// Check if it's a pipeline definition (contains + or is a single algorithm)
		if strings.Contains(value, "+") || isAlgorithmName(value) {
			return parsePipelineDefinition(config, key, value)
		}

		// Check if it's an algorithm-specific parameter (contains .)
		if strings.Contains(key, ".") {
			return parseAlgorithmParameter(config, key, value)
		}

		// Otherwise, it's a global variable
		return parseVariable(config, key, value)
	}

	return fmt.Errorf("unrecognized line format")
}

// parsePipelineDefinition parses a pipeline definition like "experiment1 = rle+huffman+lz77"
func parsePipelineDefinition(config *CustomConfig, name, definition string) error {
	algorithms := strings.Split(definition, "+")

	// Clean up algorithm names and validate
	var cleanAlgorithms []string
	for _, alg := range algorithms {
		alg = strings.TrimSpace(alg)
		if alg == "" {
			continue
		}

		// Normalize algorithm names
		alg = normalizeAlgorithmName(alg)
		cleanAlgorithms = append(cleanAlgorithms, alg)
	}

	if len(cleanAlgorithms) == 0 {
		return fmt.Errorf("pipeline definition contains no algorithms")
	}

	config.Pipelines[name] = &CustomPipeline{
		Name:       name,
		Algorithms: cleanAlgorithms,
	}

	return nil
}

// parseAlgorithmParameter parses algorithm-specific parameters like "lz77.window_size = 4096"
func parseAlgorithmParameter(config *CustomConfig, key, value string) error {
	parts := strings.SplitN(key, ".", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid algorithm parameter format")
	}

	algName := normalizeAlgorithmName(strings.TrimSpace(parts[0]))
	paramName := strings.TrimSpace(parts[1])

	if config.AlgParams[algName] == nil {
		config.AlgParams[algName] = make(map[string]interface{})
	}

	// Parse value based on type
	parsedValue, err := parseValue(value)
	if err != nil {
		return fmt.Errorf("invalid parameter value: %w", err)
	}

	config.AlgParams[algName][paramName] = parsedValue
	return nil
}

// parseVariable parses global variables
func parseVariable(config *CustomConfig, key, value string) error {
	parsedValue, err := parseValue(value)
	if err != nil {
		return fmt.Errorf("invalid variable value: %w", err)
	}

	config.Variables[key] = parsedValue
	return nil
}

// parseValue parses a string value into appropriate Go type
func parseValue(value string) (interface{}, error) {
	// Try boolean
	if strings.ToLower(value) == "true" {
		return true, nil
	}
	if strings.ToLower(value) == "false" {
		return false, nil
	}

	// Try integer
	if intVal, err := strconv.Atoi(value); err == nil {
		return intVal, nil
	}

	// Try float
	if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
		return floatVal, nil
	}

	// Default to string
	return value, nil
}

// isAlgorithmName checks if a string is a valid algorithm name
func isAlgorithmName(name string) bool {
	normalized := normalizeAlgorithmName(name)
	validAlgorithms := []string{
		"huffman", "rle", "lzw", "bwt", "mtf", "delta",
		"lz77", "lz78", "deflate", "arithmetic",
	}

	for _, alg := range validAlgorithms {
		if normalized == alg {
			return true
		}
	}
	return false
}

// normalizeAlgorithmName normalizes algorithm names to standard form
func normalizeAlgorithmName(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))

	// Handle common abbreviations and variants
	switch name {
	case "huf", "huffman":
		return "huffman"
	case "rle", "runlength":
		return "rle"
	case "lzw":
		return "lzw"
	case "bwt", "burrows-wheeler":
		return "bwt"
	case "mtf", "move-to-front":
		return "mtf"
	case "delta":
		return "delta"
	case "lz77":
		return "lz77"
	case "lz78":
		return "lz78"
	case "deflate":
		return "deflate"
	case "arithmetic", "arith":
		return "arithmetic"
	default:
		return name
	}
}

// GetPipeline retrieves a pipeline by name
func (c *CustomConfig) GetPipeline(name string) (*CustomPipeline, error) {
	pipeline, exists := c.Pipelines[name]
	if !exists {
		return nil, fmt.Errorf("pipeline '%s' not found", name)
	}
	return pipeline, nil
}

// GetAlgorithmParameters gets parameters for a specific algorithm
func (c *CustomConfig) GetAlgorithmParameters(algName string) map[string]interface{} {
	params := make(map[string]interface{})

	// Add algorithm-specific parameters
	if algParams, exists := c.AlgParams[algName]; exists {
		for k, v := range algParams {
			params[k] = v
		}
	}

	return params
}

// ListPipelines returns all pipeline names
func (c *CustomConfig) ListPipelines() []string {
	var names []string
	for name := range c.Pipelines {
		names = append(names, name)
	}
	return names
}

// GetDefaultPipeline returns the default pipeline
func (c *CustomConfig) GetDefaultPipeline() (*CustomPipeline, string, error) {
	if c.DefaultPipeline == "" {
		return nil, "", fmt.Errorf("no default pipeline specified")
	}

	pipeline, err := c.GetPipeline(c.DefaultPipeline)
	return pipeline, c.DefaultPipeline, err
}

// SaveCustomConfig saves configuration in custom format
func SaveCustomConfig(config *CustomConfig, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer file.Close()

	fmt.Fprintln(file, "# Hybrid Compression Study - Custom Configuration")
	fmt.Fprintln(file, "# Format: key = value")
	fmt.Fprintln(file, "# Pipelines: name = alg1+alg2+alg3")
	fmt.Fprintln(file, "# Algorithm params: alg.param = value")
	fmt.Fprintln(file)

	// Write default pipeline
	if config.DefaultPipeline != "" {
		fmt.Fprintf(file, "default = %s\n", config.DefaultPipeline)
		fmt.Fprintln(file)
	}

	// Write global variables
	if len(config.Variables) > 0 {
		fmt.Fprintln(file, "# Global Variables")
		for key, value := range config.Variables {
			fmt.Fprintf(file, "%s = %v\n", key, value)
		}
		fmt.Fprintln(file)
	}

	// Write pipelines
	if len(config.Pipelines) > 0 {
		fmt.Fprintln(file, "# Pipeline Definitions")
		for name, pipeline := range config.Pipelines {
			fmt.Fprintf(file, "%s = %s\n", name, strings.Join(pipeline.Algorithms, "+"))
		}
		fmt.Fprintln(file)
	}

	// Write algorithm parameters
	if len(config.AlgParams) > 0 {
		fmt.Fprintln(file, "# Algorithm Parameters")
		for algName, params := range config.AlgParams {
			for param, value := range params {
				fmt.Fprintf(file, "%s.%s = %v\n", algName, param, value)
			}
		}
		fmt.Fprintln(file)
	}

	return nil
}

// CreateExampleCustomConfig creates an example custom configuration
func CreateExampleCustomConfig() *CustomConfig {
	config := NewCustomConfig()

	// Set default
	config.DefaultPipeline = "text_optimized"

	// Global variables
	config.Variables["window_size"] = 4096
	config.Variables["compression_level"] = 6
	config.Variables["adaptive_model"] = true
	config.Variables["debug_mode"] = false

	// Pipeline definitions
	config.Pipelines["text_optimized"] = &CustomPipeline{
		Name:       "text_optimized",
		Algorithms: []string{"bwt", "mtf", "rle", "huffman"},
	}

	config.Pipelines["high_compression"] = &CustomPipeline{
		Name:       "high_compression",
		Algorithms: []string{"delta", "bwt", "mtf", "rle", "lzw", "huffman"},
	}

	config.Pipelines["fast_compression"] = &CustomPipeline{
		Name:       "fast_compression",
		Algorithms: []string{"rle", "lzw"},
	}

	config.Pipelines["dictionary_test"] = &CustomPipeline{
		Name:       "dictionary_test",
		Algorithms: []string{"lz77", "lz78", "lzw"},
	}

	config.Pipelines["entropy_test"] = &CustomPipeline{
		Name:       "entropy_test",
		Algorithms: []string{"huffman", "arithmetic"},
	}

	config.Pipelines["deflate_comparison"] = &CustomPipeline{
		Name:       "deflate_comparison",
		Algorithms: []string{"deflate"},
	}

	config.Pipelines["everything"] = &CustomPipeline{
		Name:       "everything",
		Algorithms: []string{"delta", "bwt", "mtf", "rle", "lz77", "lz78", "lzw", "huffman", "arithmetic"},
	}

	// Algorithm-specific parameters
	config.AlgParams["lz77"] = map[string]interface{}{
		"window_size": 8192,
		"buffer_size": 256,
	}

	config.AlgParams["lz78"] = map[string]interface{}{
		"max_dict_size":   8192,
		"reset_threshold": 8192,
	}

	config.AlgParams["deflate"] = map[string]interface{}{
		"compression_level": 9,
		"enable_huffman":    true,
		"enable_lz77":       true,
	}

	config.AlgParams["arithmetic"] = map[string]interface{}{
		"precision":      32,
		"adaptive_model": true,
	}

	config.AlgParams["delta"] = map[string]interface{}{
		"data_width":  1,
		"signed_data": false,
	}

	config.AlgParams["rle"] = map[string]interface{}{
		"escape_byte":    0,
		"min_run_length": 3,
	}

	return config
}

// GenerateExampleCustomConfigFile generates an example custom config file
func GenerateExampleCustomConfigFile(filename string) error {
	config := CreateExampleCustomConfig()
	return SaveCustomConfig(config, filename)
}
