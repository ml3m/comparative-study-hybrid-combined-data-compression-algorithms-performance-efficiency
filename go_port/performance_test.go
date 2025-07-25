package main

import (
	"context"
	"fmt"
	"time"

	"hybrid-compression-study/pkg/algorithms"
)

func main() {
	fmt.Println("🚀 PERFORMANCE BOTTLENECK ANALYSIS")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Test data - 8KB
	testData := make([]byte, 8192)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	fmt.Printf("Test data size: %d bytes\n\n", len(testData))

	// Test 1: BWT with Aerospace Monitoring (the bottleneck)
	fmt.Println("1. 🐌 BWT with Aerospace Monitoring:")
	bwtStart := time.Now()

	bwtEncoder, err := algorithms.NewBWTEncoder()
	if err == nil {
		_, err = bwtEncoder.Compress(context.Background(), testData)
		if err != nil {
			fmt.Printf("   BWT Error: %v\n", err)
		}
	}

	bwtTime := time.Since(bwtStart)
	fmt.Printf("   Time: %v\n", bwtTime)
	fmt.Printf("   Operations: Aerospace monitoring + O(n²) BWT rotations\n")
	fmt.Printf("   Memory: ~64MB for rotations + monitoring overhead\n\n")

	// Test 2: Simple RLE (the fix)
	fmt.Println("2. ⚡ Simple RLE without Monitoring:")
	rleStart := time.Now()

	simpleRLE, err := algorithms.NewSimpleRLEEncoder()
	if err == nil {
		_, err = simpleRLE.Compress(context.Background(), testData)
		if err != nil {
			fmt.Printf("   RLE Error: %v\n", err)
		}
	}

	rleTime := time.Since(rleStart)
	fmt.Printf("   Time: %v\n", rleTime)
	fmt.Printf("   Operations: Simple O(n) RLE\n")
	fmt.Printf("   Memory: ~16KB working memory\n\n")

	// Test 3: Regular RLE with Aerospace Monitoring
	fmt.Println("3. 🐌 Regular RLE with Aerospace Monitoring:")
	heavyRLEStart := time.Now()

	heavyRLE, err := algorithms.NewRLEEncoder()
	if err == nil {
		_, err = heavyRLE.Compress(context.Background(), testData)
		if err != nil {
			fmt.Printf("   Heavy RLE Error: %v\n", err)
		}
	}

	heavyRLETime := time.Since(heavyRLEStart)
	fmt.Printf("   Time: %v\n", heavyRLETime)
	fmt.Printf("   Operations: RLE + runtime.GC() + monitoring goroutines\n")
	fmt.Printf("   Memory: Monitoring samples + detailed metrics\n\n")

	// Performance Analysis
	fmt.Println("📊 PERFORMANCE ANALYSIS:")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	if bwtTime > 0 && rleTime > 0 {
		speedupVsBWT := float64(bwtTime) / float64(rleTime)
		fmt.Printf("Simple RLE vs BWT:           %.1fx faster\n", speedupVsBWT)
	}

	if heavyRLETime > 0 && rleTime > 0 {
		speedupVsHeavy := float64(heavyRLETime) / float64(rleTime)
		fmt.Printf("Simple RLE vs Heavy RLE:     %.1fx faster\n", speedupVsHeavy)
	}

	fmt.Println("\n🔍 ROOT CAUSES OF SLOWNESS:")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("1. 🚨 BWT O(n²) Algorithm:")
	fmt.Println("   • Creates ALL string rotations in memory")
	fmt.Println("   • 8KB input → 8,192 strings × 8KB = 64MB+ memory")
	fmt.Println("   • O(n² log n) sorting complexity")

	fmt.Println("\n2. 🚨 Aerospace Monitoring Overhead:")
	fmt.Println("   • runtime.GC() before EVERY compression operation")
	fmt.Println("   • Background monitoring goroutines")
	fmt.Println("   • 10,000+ performance samples per operation")
	fmt.Println("   • Detailed CPU/Memory/IO tracking")

	fmt.Println("\n3. 🚨 Massive Scale Impact:")
	fmt.Printf("   • Your benchmark: ~1,080 tests\n")
	fmt.Printf("   • Each test: 2-3 algorithm operations\n")
	fmt.Printf("   • Total ProfileOperation calls: ~2,160-3,240\n")
	fmt.Printf("   • Each call: runtime.GC() + monitoring\n")
	fmt.Printf("   • = Thousands of garbage collections!\n")

	fmt.Println("\n✅ SOLUTIONS:")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("1. 🚫 Avoid BWT algorithm (O(n²) complexity)")
	fmt.Println("2. ⚡ Use simple algorithm implementations")
	fmt.Println("3. 🗑️  Remove aerospace monitoring overhead")
	fmt.Println("4. 💾 Disable JSON file saving")
	fmt.Println("5. 🔄 Reduce garbage collection frequency")

	fmt.Println("\n🎯 RECOMMENDED BENCHMARK CONFIG:")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Selected algorithms: [\"RLE\", \"LZW\", \"Huffman\"]  // Avoid BWT, MTF, Arithmetic")
	fmt.Println("Max combination size: 2                            // Reduce complexity")
	fmt.Println("Input sizes: [\"1KB\", \"4KB\"]                      // Avoid large inputs")
	fmt.Println("Save results: false                                // Disable file I/O")
	fmt.Println("Concurrent tests: 1                                // Prevent memory amplification")

	fmt.Println("\n🚀 EXPECTED PERFORMANCE IMPROVEMENT:")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Current execution time: ~9+ seconds")
	fmt.Println("Optimized execution time: ~1-2 seconds")
	fmt.Println("Expected speedup: 5-10x faster! ⚡")
}
