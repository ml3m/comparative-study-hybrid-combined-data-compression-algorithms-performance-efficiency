// Package main demonstrates programmatic usage of the hybrid compression study algorithms.
//
// This example shows how to use the compression algorithms directly in Go code
// rather than through the CLI interface.
package main

import (
	"context"
	"fmt"
	"log"

	"hybrid-compression-study/pkg/algorithms"
	"hybrid-compression-study/pkg/core"
)

func main() {
	// Sample data for compression
	testData := []byte("Hello, World! This is a test for aerospace-grade compression analysis with repeated patterns: AAAAAABBBBBBCCCCCCDDDDDD")
	
	fmt.Printf("🚀 Aerospace-Grade Compression Demo\n")
	fmt.Printf("═══════════════════════════════════\n\n")
	fmt.Printf("Original data: %q\n", string(testData))
	fmt.Printf("Original size: %d bytes\n\n", len(testData))
	
	// Test all algorithms
	algorithms := []struct {
		name    string
		creator func() (core.CompressionAlgorithm, error)
	}{
		{"Huffman", func() (core.CompressionAlgorithm, error) { return algorithms.NewHuffmanEncoder() }},
		{"RLE", func() (core.CompressionAlgorithm, error) { return algorithms.NewRLEEncoder() }},
		{"LZW", func() (core.CompressionAlgorithm, error) { return algorithms.NewLZWEncoder() }},
	}
	
	ctx := context.Background()
	
	for _, alg := range algorithms {
		fmt.Printf("📊 Testing %s Algorithm\n", alg.name)
		fmt.Printf("─────────────────────────\n")
		
		// Create algorithm instance
		algorithm, err := alg.creator()
		if err != nil {
			log.Printf("❌ Failed to create %s algorithm: %v\n", alg.name, err)
			continue
		}
		
		// Compress
		result, err := algorithm.Compress(ctx, testData)
		if err != nil {
			log.Printf("❌ Compression failed for %s: %v\n", alg.name, err)
			continue
		}
		
		// Display compression results
		fmt.Printf("Compressed size: %d bytes\n", result.CompressedSize)
		fmt.Printf("Compression ratio: %.3fx\n", result.CompressionRatio)
		fmt.Printf("Space savings: %.1f%%\n", result.CompressionPercentage())
		fmt.Printf("Compression time: %s\n", result.PrecisionMetrics.CompressionTimeFormatted)
		fmt.Printf("Throughput: %.3f MB/s\n", result.PrecisionMetrics.ThroughputMBPS)
		fmt.Printf("Effectiveness: %s\n", map[bool]string{true: "✅ POSITIVE", false: "❌ NEGATIVE"}[result.IsEffective()])
		
		// Test decompression
		decompResult, err := algorithm.Decompress(ctx, result.CompressedData, result.Metadata)
		if err != nil {
			log.Printf("❌ Decompression failed for %s: %v\n", alg.name, err)
			continue
		}
		
		// Verify integrity
		if decompResult.VerifyIntegrity(testData) {
			fmt.Printf("Data integrity: ✅ VERIFIED\n")
		} else {
			fmt.Printf("Data integrity: ❌ FAILED\n")
		}
		
		fmt.Printf("Decompression time: %s\n", decompResult.PrecisionMetrics.DecompressionTimeFormatted)
		fmt.Printf("Total time: %s\n", core.FormatTimePrecision(result.PrecisionMetrics.CompressionTimeNs+decompResult.PrecisionMetrics.DecompressionTimeNs))
		
		// Mission readiness assessment
		fmt.Printf("\n🎯 Mission Readiness:\n")
		fmt.Printf("   Real-time suitable: %v\n", result.PrecisionMetrics.WorstCaseLatencyNs < 1_000_000_000)
		fmt.Printf("   Memory constrained: %v\n", result.PrecisionMetrics.MemoryOverheadRatio < 2.0)
		fmt.Printf("   Determinism score: %.6f\n", result.PrecisionMetrics.DeterminismScore)
		
		fmt.Printf("\n")
	}
	
	fmt.Printf("🎉 Demo completed successfully!\n")
	fmt.Printf("\nThis demonstrates the aerospace-grade precision and comprehensive\n")
	fmt.Printf("performance analysis provided by the hybrid compression study framework.\n")
} 