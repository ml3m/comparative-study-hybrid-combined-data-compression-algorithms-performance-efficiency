// Package pipeline provides the infrastructure for creating and executing
// compression pipelines that chain multiple algorithms together.
package pipeline

import (
	"context"
	"fmt"
	"time"

	"hybrid-compression-study/internal/performance"
	"hybrid-compression-study/pkg/core"
)

// PipelineStage represents a single stage in a compression pipeline
type PipelineStage struct {
	Component  core.CompressionAlgorithm `json:"component"`
	Name       string                    `json:"name"`
	Parameters map[string]interface{}    `json:"parameters"`
	Enabled    bool                      `json:"enabled"`
}

// PipelineResult represents results from executing a compression pipeline
type PipelineResult struct {
	CompressedData         []byte                   `json:"compressed_data"`
	OriginalSize           int64                    `json:"original_size"`
	CompressedSize         int64                    `json:"compressed_size"`
	TotalCompressionRatio  float64                  `json:"total_compression_ratio"`
	TotalCompressionTime   float64                  `json:"total_compression_time"`
	TotalDecompressionTime float64                  `json:"total_decompression_time"`
	StageResults           []map[string]interface{} `json:"stage_results"`
	PipelineName           string                   `json:"pipeline_name"`
	Metadata               map[string]interface{}   `json:"metadata"`
}

// CompressionPercentage calculates total compression percentage
func (pr *PipelineResult) CompressionPercentage() float64 {
	if pr.OriginalSize == 0 {
		return 0.0
	}
	return (1.0 - float64(pr.CompressedSize)/float64(pr.OriginalSize)) * 100.0
}

// TotalTime calculates total processing time
func (pr *PipelineResult) TotalTime() float64 {
	return pr.TotalCompressionTime + pr.TotalDecompressionTime
}

// CompressionPipeline allows chaining multiple compression algorithms
type CompressionPipeline struct {
	Name               string                          `json:"name"`
	Stages             []PipelineStage                 `json:"stages"`
	PerformanceMonitor *performance.StatisticalMonitor `json:"-"`
	Metadata           map[string]interface{}          `json:"metadata"`
}

// NewCompressionPipeline creates a new compression pipeline
func NewCompressionPipeline(name string) (*CompressionPipeline, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}

	return &CompressionPipeline{
		Name:               name,
		Stages:             make([]PipelineStage, 0),
		PerformanceMonitor: monitor,
		Metadata:           make(map[string]interface{}),
	}, nil
}

// AddStage adds a compression stage to the pipeline
func (cp *CompressionPipeline) AddStage(component core.CompressionAlgorithm, name string, parameters map[string]interface{}) *CompressionPipeline {
	if name == "" {
		name = fmt.Sprintf("Stage%d", len(cp.Stages))
	}

	if parameters != nil {
		component.SetParameters(parameters)
	}

	stage := PipelineStage{
		Component:  component,
		Name:       name,
		Parameters: parameters,
		Enabled:    true,
	}

	cp.Stages = append(cp.Stages, stage)
	return cp
}

// RemoveStage removes a stage by index
func (cp *CompressionPipeline) RemoveStage(index int) *CompressionPipeline {
	if index >= 0 && index < len(cp.Stages) {
		cp.Stages = append(cp.Stages[:index], cp.Stages[index+1:]...)
	}
	return cp
}

// ClearStages removes all stages from the pipeline
func (cp *CompressionPipeline) ClearStages() *CompressionPipeline {
	cp.Stages = cp.Stages[:0]
	return cp
}

// EnableStage enables or disables a specific stage
func (cp *CompressionPipeline) EnableStage(index int, enabled bool) *CompressionPipeline {
	if index >= 0 && index < len(cp.Stages) {
		cp.Stages[index].Enabled = enabled
	}
	return cp
}

// Compress compresses data through the entire pipeline
func (cp *CompressionPipeline) Compress(ctx context.Context, data []byte) (*PipelineResult, error) {
	if len(cp.Stages) == 0 {
		return nil, &core.PipelineError{
			Message:      "Pipeline has no stages",
			PipelineName: cp.Name,
			TimestampNs:  time.Now().UnixNano(),
		}
	}

	if len(data) == 0 {
		return &PipelineResult{
			CompressedData:         []byte{},
			OriginalSize:           0,
			CompressedSize:         0,
			TotalCompressionRatio:  1.0,
			TotalCompressionTime:   0.0,
			TotalDecompressionTime: 0.0,
			StageResults:           []map[string]interface{}{},
			PipelineName:           cp.Name,
			Metadata:               make(map[string]interface{}),
		}, nil
	}

	startTime := time.Now()

	currentData := data
	stageResults := make([]map[string]interface{}, 0)
	stageMetadata := make([]map[string]interface{}, 0)
	totalCompressionTime := 0.0

	// Process through each enabled stage
	for i, stage := range cp.Stages {
		if !stage.Enabled {
			continue
		}

		result, err := stage.Component.Compress(ctx, currentData)
		if err != nil {
			return nil, &core.PipelineError{
				Message:      fmt.Sprintf("Error in stage %d (%s): %v", i, stage.Name, err),
				StageName:    stage.Name,
				StageIndex:   i,
				PipelineName: cp.Name,
				TimestampNs:  time.Now().UnixNano(),
			}
		}

		currentData = result.CompressedData

		stageInfo := map[string]interface{}{
			"stage_index":       i,
			"stage_name":        stage.Name,
			"algorithm":         stage.Component.GetName(),
			"input_size":        result.OriginalSize,
			"output_size":       result.CompressedSize,
			"compression_ratio": result.CompressionRatio,
			"compression_time":  result.CompressionTime,
			"metadata":          result.Metadata,
			"precision_metrics": result.PrecisionMetrics,
		}

		stageResults = append(stageResults, stageInfo)
		stageMetadata = append(stageMetadata, result.Metadata)
		totalCompressionTime += result.CompressionTime
	}

	endTime := time.Now()
	actualTotalTime := endTime.Sub(startTime).Seconds()

	// Calculate overall metrics
	totalRatio := float64(len(data)) / float64(len(currentData))
	if len(currentData) == 0 {
		totalRatio = float64(len(data)) // Avoid division by zero
	}

	result := &PipelineResult{
		CompressedData:         currentData,
		OriginalSize:           int64(len(data)),
		CompressedSize:         int64(len(currentData)),
		TotalCompressionRatio:  totalRatio,
		TotalCompressionTime:   totalCompressionTime,
		TotalDecompressionTime: 0.0, // Will be filled during decompression
		StageResults:           stageResults,
		PipelineName:           cp.Name,
		Metadata: map[string]interface{}{
			"pipeline_stages":   len(cp.getEnabledStages()),
			"stage_metadata":    stageMetadata,
			"actual_total_time": actualTotalTime,
			"stage_names":       cp.getEnabledStageNames(),
			"total_stages":      len(cp.Stages),
			"enabled_stages":    len(cp.getEnabledStages()),
		},
	}

	return result, nil
}

// Decompress decompresses data by reversing the pipeline
func (cp *CompressionPipeline) Decompress(ctx context.Context, compressedData []byte, pipelineResult *PipelineResult) ([]byte, error) {
	if len(compressedData) == 0 {
		return []byte{}, nil
	}

	if len(pipelineResult.StageResults) == 0 {
		return nil, &core.PipelineError{
			Message:      "No stage results available for decompression",
			PipelineName: cp.Name,
			TimestampNs:  time.Now().UnixNano(),
		}
	}

	startTime := time.Now()

	currentData := compressedData
	stageMetadata := pipelineResult.Metadata["stage_metadata"].([]map[string]interface{})

	// Process stages in reverse order
	enabledStages := cp.getEnabledStages()

	for i := len(enabledStages) - 1; i >= 0; i-- {
		stage := enabledStages[i]
		metadataIdx := i
		var metadata map[string]interface{}
		if metadataIdx < len(stageMetadata) {
			metadata = stageMetadata[metadataIdx]
		} else {
			metadata = make(map[string]interface{})
		}

		result, err := stage.Component.Decompress(ctx, currentData, metadata)
		if err != nil {
			return nil, &core.PipelineError{
				Message:      fmt.Sprintf("Error decompressing stage %s: %v", stage.Name, err),
				StageName:    stage.Name,
				PipelineName: cp.Name,
				TimestampNs:  time.Now().UnixNano(),
			}
		}

		currentData = result.DecompressedData
	}

	// Update pipeline result with decompression time
	pipelineResult.TotalDecompressionTime = time.Since(startTime).Seconds()

	return currentData, nil
}

// GetStageInfo returns information about all pipeline stages
func (cp *CompressionPipeline) GetStageInfo() []map[string]interface{} {
	info := make([]map[string]interface{}, len(cp.Stages))

	for i, stage := range cp.Stages {
		info[i] = map[string]interface{}{
			"index":          i,
			"name":           stage.Name,
			"component_type": stage.Component.GetName(),
			"enabled":        stage.Enabled,
			"parameters":     stage.Parameters,
			"algorithm_info": stage.Component.GetInfo(),
		}
	}

	return info
}

// ValidatePipeline validates the pipeline configuration
func (cp *CompressionPipeline) ValidatePipeline() []string {
	var issues []string

	if len(cp.Stages) == 0 {
		issues = append(issues, "Pipeline has no stages")
	}

	enabledStages := cp.getEnabledStages()
	if len(enabledStages) == 0 {
		issues = append(issues, "Pipeline has no enabled stages")
	}

	// Check for potential issues in stage ordering
	for i := 0; i < len(enabledStages)-1; i++ {
		currentStage := enabledStages[i]
		nextStage := enabledStages[i+1]

		// Dictionary algorithms work better before entropy coding
		if currentStage.Component.GetCategory() == core.AlgorithmCategoryEntropyCoding &&
			nextStage.Component.GetCategory() == core.AlgorithmCategoryDictionary {
			issues = append(issues, fmt.Sprintf("Consider moving dictionary algorithm (%s) before entropy coding (%s)",
				nextStage.Name, currentStage.Name))
		}
	}

	return issues
}

// Clone creates a copy of this pipeline
func (cp *CompressionPipeline) Clone() (*CompressionPipeline, error) {
	newPipeline, err := NewCompressionPipeline(cp.Name + "_copy")
	if err != nil {
		return nil, err
	}

	for _, stage := range cp.Stages {
		// Note: This creates a shallow copy of the component
		// In a production system, you'd want proper cloning of components
		newPipeline.AddStage(stage.Component, stage.Name, stage.Parameters)
		newPipeline.Stages[len(newPipeline.Stages)-1].Enabled = stage.Enabled
	}

	return newPipeline, nil
}

// getEnabledStages returns only enabled stages
func (cp *CompressionPipeline) getEnabledStages() []PipelineStage {
	var enabled []PipelineStage
	for _, stage := range cp.Stages {
		if stage.Enabled {
			enabled = append(enabled, stage)
		}
	}
	return enabled
}

// getEnabledStageNames returns names of enabled stages
func (cp *CompressionPipeline) getEnabledStageNames() []string {
	var names []string
	for _, stage := range cp.Stages {
		if stage.Enabled {
			names = append(names, stage.Name)
		}
	}
	return names
}

// String returns a string representation of the pipeline
func (cp *CompressionPipeline) String() string {
	enabledStageNames := cp.getEnabledStageNames()
	if len(enabledStageNames) == 0 {
		return fmt.Sprintf("CompressionPipeline('%s', no enabled stages)", cp.Name)
	}

	stages := ""
	for i, name := range enabledStageNames {
		if i > 0 {
			stages += " → "
		}
		stages += name
	}

	return fmt.Sprintf("CompressionPipeline('%s', stages: %s)", cp.Name, stages)
}

// PipelineBuilder provides a fluent interface for building compression pipelines
type PipelineBuilder struct {
	pipeline *CompressionPipeline
}

// NewPipelineBuilder creates a new pipeline builder
func NewPipelineBuilder(name string) (*PipelineBuilder, error) {
	pipeline, err := NewCompressionPipeline(name)
	if err != nil {
		return nil, err
	}

	return &PipelineBuilder{
		pipeline: pipeline,
	}, nil
}

// AddHuffman adds Huffman compression stage
func (pb *PipelineBuilder) AddHuffman() (*PipelineBuilder, error) {
	// This would need to import the algorithms package
	// For now, we'll leave this as a placeholder
	return pb, nil
}

// AddRLE adds RLE compression stage
func (pb *PipelineBuilder) AddRLE(escapeByte int, minRunLength int) (*PipelineBuilder, error) {
	// This would need to import the algorithms package
	// For now, we'll leave this as a placeholder
	return pb, nil
}

// AddLZW adds LZW compression stage
func (pb *PipelineBuilder) AddLZW(maxCodeBits int) (*PipelineBuilder, error) {
	// This would need to import the algorithms package
	// For now, we'll leave this as a placeholder
	return pb, nil
}

// AddCustom adds custom compression stage
func (pb *PipelineBuilder) AddCustom(component core.CompressionAlgorithm, name string, parameters map[string]interface{}) *PipelineBuilder {
	pb.pipeline.AddStage(component, name, parameters)
	return pb
}

// Build builds and returns the pipeline
func (pb *PipelineBuilder) Build() *CompressionPipeline {
	return pb.pipeline
}

// PredefinedPipelines provides collection of predefined compression pipelines
type PredefinedPipelines struct{}

// NewPredefinedPipelines creates a new predefined pipelines factory
func NewPredefinedPipelines() *PredefinedPipelines {
	return &PredefinedPipelines{}
}

// TextOptimized creates a pipeline optimized for text data
func (pp *PredefinedPipelines) TextOptimized() (*CompressionPipeline, error) {
	pipeline, err := NewCompressionPipeline("Text-Optimized")
	if err != nil {
		return nil, err
	}

	// This would be implemented with actual algorithm instances
	// For now, returning empty pipeline
	return pipeline, nil
}

// BinaryOptimized creates a pipeline optimized for binary data
func (pp *PredefinedPipelines) BinaryOptimized() (*CompressionPipeline, error) {
	pipeline, err := NewCompressionPipeline("Binary-Optimized")
	if err != nil {
		return nil, err
	}

	// This would be implemented with actual algorithm instances
	// For now, returning empty pipeline
	return pipeline, nil
}

// HighCompression creates a pipeline focused on maximum compression
func (pp *PredefinedPipelines) HighCompression() (*CompressionPipeline, error) {
	pipeline, err := NewCompressionPipeline("High-Compression")
	if err != nil {
		return nil, err
	}

	// This would be implemented with actual algorithm instances
	// For now, returning empty pipeline
	return pipeline, nil
}

// FastCompression creates a pipeline focused on speed
func (pp *PredefinedPipelines) FastCompression() (*CompressionPipeline, error) {
	pipeline, err := NewCompressionPipeline("Fast-Compression")
	if err != nil {
		return nil, err
	}

	// This would be implemented with actual algorithm instances
	// For now, returning empty pipeline
	return pipeline, nil
}
