package benchmarks

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// MemoryOptimizedBenchmarkExecutor prevents memory leaks by streaming results
type MemoryOptimizedBenchmarkExecutor struct {
	*BenchmarkExecutor
	resultStream *os.File
	tempDir      string
}

// NewMemoryOptimizedBenchmarkExecutor creates a memory-efficient benchmark executor
func NewMemoryOptimizedBenchmarkExecutor(config *BenchmarkConfig) (*MemoryOptimizedBenchmarkExecutor, error) {
	// Create base executor
	baseExecutor, err := NewBenchmarkExecutor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base executor: %w", err)
	}

	// Create temp directory for streaming results
	tempDir := filepath.Join(os.TempDir(), fmt.Sprintf("benchmark_stream_%d", time.Now().Unix()))
	err = os.MkdirAll(tempDir, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}

	// Create result stream file
	streamFile := filepath.Join(tempDir, "results.jsonl")
	resultStream, err := os.Create(streamFile)
	if err != nil {
		return nil, fmt.Errorf("failed to create result stream: %w", err)
	}

	return &MemoryOptimizedBenchmarkExecutor{
		BenchmarkExecutor: baseExecutor,
		resultStream:      resultStream,
		tempDir:           tempDir,
	}, nil
}

// StreamResult writes a result to disk and clears it from memory
func (moe *MemoryOptimizedBenchmarkExecutor) StreamResult(result *TestResult) error {
	// Write result to stream
	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	_, err = moe.resultStream.Write(append(data, '\n'))
	if err != nil {
		return fmt.Errorf("failed to write result: %w", err)
	}

	// Clear compressed data from memory immediately
	if result.CompressionResult != nil {
		result.CompressionResult.CompressedData = nil
		result.CompressionResult.Metadata = nil
	}

	return nil
}

// ForceMemoryCleanup aggressively cleans up memory
func (moe *MemoryOptimizedBenchmarkExecutor) ForceMemoryCleanup() {
	// Clear result slices
	moe.allTestResults = moe.allTestResults[:0]
	moe.allKEIScores = moe.allKEIScores[:0]

	// Force garbage collection
	runtime.GC()
	runtime.GC() // Call twice for better cleanup

	// Give OS time to reclaim memory
	time.Sleep(10 * time.Millisecond)
}

// Close cleans up resources
func (moe *MemoryOptimizedBenchmarkExecutor) Close() error {
	if moe.resultStream != nil {
		moe.resultStream.Close()
	}

	// Clean up temp directory
	if moe.tempDir != "" {
		os.RemoveAll(moe.tempDir)
	}

	return nil
}

// GetMemoryStats returns current memory usage
func GetMemoryStats() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return map[string]interface{}{
		"allocated_mb":       float64(m.Alloc) / 1024 / 1024,
		"total_allocated_mb": float64(m.TotalAlloc) / 1024 / 1024,
		"system_mb":          float64(m.Sys) / 1024 / 1024,
		"num_gc":             m.NumGC,
		"heap_objects":       m.HeapObjects,
	}
}
