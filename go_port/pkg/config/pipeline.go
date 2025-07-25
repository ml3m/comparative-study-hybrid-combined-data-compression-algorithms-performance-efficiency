// Package config provides configuration management for compression pipelines.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// AlgorithmConfig represents configuration for a single algorithm in the pipeline
type AlgorithmConfig struct {
	Name       string                 `json:"name"`
	Enabled    bool                   `json:"enabled"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

// PipelineConfig represents configuration for a complete compression pipeline
type PipelineConfig struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Algorithms  []AlgorithmConfig      `json:"algorithms"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// CompressionConfig represents the complete configuration file structure
type CompressionConfig struct {
	Version         string                    `json:"version"`
	DefaultPipeline string                    `json:"default_pipeline,omitempty"`
	Pipelines       map[string]PipelineConfig `json:"pipelines"`
	GlobalSettings  map[string]interface{}    `json:"global_settings,omitempty"`
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(configPath string) (*CompressionConfig, error) {
	// Check if file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file not found: %s", configPath)
	}

	// Read file content
	content, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config CompressionConfig

	// Parse JSON
	err = json.Unmarshal(content, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}

// SaveConfig saves configuration to a JSON file
func SaveConfig(config *CompressionConfig, configPath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Marshal to JSON with pretty formatting
	content, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Write to file
	if err := os.WriteFile(configPath, content, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Validate validates the configuration structure
func (c *CompressionConfig) Validate() error {
	if c.Version == "" {
		c.Version = "1.0"
	}

	if len(c.Pipelines) == 0 {
		return fmt.Errorf("no pipelines defined")
	}

	// Validate each pipeline
	for name, pipeline := range c.Pipelines {
		if err := pipeline.Validate(); err != nil {
			return fmt.Errorf("invalid pipeline '%s': %w", name, err)
		}
	}

	// Validate default pipeline exists
	if c.DefaultPipeline != "" {
		if _, exists := c.Pipelines[c.DefaultPipeline]; !exists {
			return fmt.Errorf("default pipeline '%s' not found", c.DefaultPipeline)
		}
	}

	return nil
}

// Validate validates a pipeline configuration
func (p *PipelineConfig) Validate() error {
	if p.Name == "" {
		return fmt.Errorf("pipeline name is required")
	}

	if len(p.Algorithms) == 0 {
		return fmt.Errorf("pipeline must have at least one algorithm")
	}

	// Validate each algorithm
	for i, alg := range p.Algorithms {
		if err := alg.Validate(); err != nil {
			return fmt.Errorf("invalid algorithm at position %d: %w", i, err)
		}
	}

	return nil
}

// Validate validates an algorithm configuration
func (a *AlgorithmConfig) Validate() error {
	if a.Name == "" {
		return fmt.Errorf("algorithm name is required")
	}

	// Validate algorithm name against known algorithms
	validAlgorithms := []string{
		"huffman", "rle", "lzw", "bwt", "mtf", "delta",
		"lz77", "lz78", "arithmetic", "ppm", "range", "deflate",
		"snappy", "brotli",
	}

	found := false
	for _, valid := range validAlgorithms {
		if strings.ToLower(a.Name) == valid {
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("unknown algorithm: %s", a.Name)
	}

	return nil
}

// GetPipeline returns a specific pipeline configuration
func (c *CompressionConfig) GetPipeline(name string) (*PipelineConfig, error) {
	pipeline, exists := c.Pipelines[name]
	if !exists {
		return nil, fmt.Errorf("pipeline '%s' not found", name)
	}

	return &pipeline, nil
}

// GetDefaultPipeline returns the default pipeline or the first available pipeline
func (c *CompressionConfig) GetDefaultPipeline() (*PipelineConfig, string, error) {
	if c.DefaultPipeline != "" {
		pipeline, err := c.GetPipeline(c.DefaultPipeline)
		return pipeline, c.DefaultPipeline, err
	}

	// Return first pipeline if no default is set
	for name, pipeline := range c.Pipelines {
		return &pipeline, name, nil
	}

	return nil, "", fmt.Errorf("no pipelines available")
}

// ListPipelines returns a list of all available pipeline names
func (c *CompressionConfig) ListPipelines() []string {
	names := make([]string, 0, len(c.Pipelines))
	for name := range c.Pipelines {
		names = append(names, name)
	}
	return names
}

// CreateExampleConfig creates an example configuration file
func CreateExampleConfig() *CompressionConfig {
	return &CompressionConfig{
		Version:         "1.0",
		DefaultPipeline: "text_optimized",
		Pipelines: map[string]PipelineConfig{
			"text_optimized": {
				Name:        "Text Optimized",
				Description: "Optimized pipeline for text data compression",
				Algorithms: []AlgorithmConfig{
					{
						Name:    "bwt",
						Enabled: true,
						Parameters: map[string]interface{}{
							"end_marker": "$",
						},
					},
					{
						Name:    "mtf",
						Enabled: true,
					},
					{
						Name:    "rle",
						Enabled: true,
						Parameters: map[string]interface{}{
							"escape_byte":    0,
							"min_run_length": 3,
						},
					},
					{
						Name:    "huffman",
						Enabled: true,
					},
				},
				Metadata: map[string]interface{}{
					"best_for":          []string{"text", "source_code", "documents"},
					"complexity":        "high",
					"compression_ratio": "excellent",
				},
			},
			"fast_compression": {
				Name:        "Fast Compression",
				Description: "Fast compression pipeline optimized for speed",
				Algorithms: []AlgorithmConfig{
					{
						Name:    "rle",
						Enabled: true,
						Parameters: map[string]interface{}{
							"escape_byte":    0,
							"min_run_length": 2,
						},
					},
					{
						Name:    "lzw",
						Enabled: true,
						Parameters: map[string]interface{}{
							"max_code_bits": 12,
						},
					},
				},
				Metadata: map[string]interface{}{
					"best_for":   []string{"binary", "mixed_data"},
					"complexity": "low",
					"speed":      "very_fast",
				},
			},
			"high_compression": {
				Name:        "High Compression",
				Description: "Maximum compression ratio pipeline",
				Algorithms: []AlgorithmConfig{
					{
						Name:    "delta",
						Enabled: true,
						Parameters: map[string]interface{}{
							"data_width":     1,
							"signed_data":    false,
							"predictor_type": "linear",
						},
					},
					{
						Name:    "bwt",
						Enabled: true,
					},
					{
						Name:    "mtf",
						Enabled: true,
					},
					{
						Name:    "rle",
						Enabled: true,
						Parameters: map[string]interface{}{
							"escape_byte":    0,
							"min_run_length": 3,
						},
					},
					{
						Name:    "lzw",
						Enabled: true,
						Parameters: map[string]interface{}{
							"max_code_bits": 14,
						},
					},
					{
						Name:    "huffman",
						Enabled: true,
					},
				},
				Metadata: map[string]interface{}{
					"best_for":          []string{"text", "repetitive_data"},
					"complexity":        "very_high",
					"compression_ratio": "maximum",
					"speed":             "slow",
				},
			},
			"audio_optimized": {
				Name:        "Audio Optimized",
				Description: "Optimized for audio and time-series data",
				Algorithms: []AlgorithmConfig{
					{
						Name:    "delta",
						Enabled: true,
						Parameters: map[string]interface{}{
							"data_width":     2,
							"signed_data":    true,
							"predictor_type": "adaptive",
						},
					},
					{
						Name:    "rle",
						Enabled: true,
						Parameters: map[string]interface{}{
							"escape_byte":    0,
							"min_run_length": 4,
						},
					},
					{
						Name:    "huffman",
						Enabled: true,
					},
				},
				Metadata: map[string]interface{}{
					"best_for":    []string{"audio", "time_series", "sensor_data"},
					"complexity":  "medium",
					"specialized": true,
				},
			},
		},
		GlobalSettings: map[string]interface{}{
			"default_tries":    1,
			"verbose_output":   false,
			"json_output":      false,
			"benchmark_mode":   false,
			"output_directory": "./compressed/",
		},
	}
}

// GenerateExampleConfigFile creates an example configuration file at the specified path
func GenerateExampleConfigFile(configPath string) error {
	config := CreateExampleConfig()
	return SaveConfig(config, configPath)
}

// ValidateAlgorithmParameters validates parameters for specific algorithms
func ValidateAlgorithmParameters(algorithmName string, parameters map[string]interface{}) error {
	switch strings.ToLower(algorithmName) {
	case "rle":
		if escByte, ok := parameters["escape_byte"]; ok {
			if val, ok := escByte.(int); !ok || val < 0 || val > 255 {
				return fmt.Errorf("escape_byte must be integer between 0-255")
			}
		}
		if minRun, ok := parameters["min_run_length"]; ok {
			if val, ok := minRun.(int); !ok || val < 1 || val > 255 {
				return fmt.Errorf("min_run_length must be integer between 1-255")
			}
		}
	case "lzw":
		if maxBits, ok := parameters["max_code_bits"]; ok {
			if val, ok := maxBits.(int); !ok || val < 9 || val > 16 {
				return fmt.Errorf("max_code_bits must be integer between 9-16")
			}
		}
	case "delta":
		if width, ok := parameters["data_width"]; ok {
			if val, ok := width.(int); !ok || (val != 1 && val != 2 && val != 4 && val != 8) {
				return fmt.Errorf("data_width must be 1, 2, 4, or 8")
			}
		}
	case "bwt":
		if marker, ok := parameters["end_marker"]; ok {
			if val, ok := marker.(string); !ok || len(val) != 1 {
				return fmt.Errorf("end_marker must be single character string")
			}
		}
	}

	return nil
}

// MergeConfigs merges two configurations, with the second taking precedence
func MergeConfigs(base, override *CompressionConfig) *CompressionConfig {
	result := *base

	if override.Version != "" {
		result.Version = override.Version
	}

	if override.DefaultPipeline != "" {
		result.DefaultPipeline = override.DefaultPipeline
	}

	// Merge pipelines
	if result.Pipelines == nil {
		result.Pipelines = make(map[string]PipelineConfig)
	}

	for name, pipeline := range override.Pipelines {
		result.Pipelines[name] = pipeline
	}

	// Merge global settings
	if result.GlobalSettings == nil {
		result.GlobalSettings = make(map[string]interface{})
	}

	for key, value := range override.GlobalSettings {
		result.GlobalSettings[key] = value
	}

	return &result
}
