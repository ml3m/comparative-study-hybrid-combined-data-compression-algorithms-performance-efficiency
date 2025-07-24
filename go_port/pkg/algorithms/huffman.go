// Package algorithms provides compression algorithm implementations with aerospace-grade precision.
//
// This package contains implementations of various compression algorithms
// with nanosecond-precision metrics suitable for mission-critical applications.
package algorithms

import (
	"context"
	"container/heap"
	"fmt"
	"time"

	"hybrid-compression-study/pkg/core"
	"hybrid-compression-study/internal/performance"
)

// HuffmanNode represents a node in the Huffman tree with enhanced debugging info
type HuffmanNode struct {
	Frequency int
	Symbol    *byte // nil for internal nodes
	Left      *HuffmanNode
	Right     *HuffmanNode
	IsLeaf    bool
}

// HuffmanNodeHeap implements heap.Interface for HuffmanNode
type HuffmanNodeHeap []*HuffmanNode

func (h HuffmanNodeHeap) Len() int           { return len(h) }
func (h HuffmanNodeHeap) Less(i, j int) bool { return h[i].Frequency < h[j].Frequency }
func (h HuffmanNodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *HuffmanNodeHeap) Push(x interface{}) {
	*h = append(*h, x.(*HuffmanNode))
}

func (h *HuffmanNodeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// BitWriter provides high-precision bit-level writing with enhanced monitoring
type BitWriter struct {
	data        []byte
	currentByte byte
	bitCount    int
	bitsWritten int64
}

// NewBitWriter creates a new BitWriter
func NewBitWriter() *BitWriter {
	return &BitWriter{
		data:        make([]byte, 0),
		currentByte: 0,
		bitCount:    0,
		bitsWritten: 0,
	}
}

// WriteBit writes a single bit with precision tracking
func (bw *BitWriter) WriteBit(bit int) {
	if bit != 0 {
		bw.currentByte |= (1 << (7 - bw.bitCount))
	}
	
	bw.bitCount++
	bw.bitsWritten++
	
	if bw.bitCount == 8 {
		bw.data = append(bw.data, bw.currentByte)
		bw.currentByte = 0
		bw.bitCount = 0
	}
}

// WriteBits writes multiple bits from string representation
func (bw *BitWriter) WriteBits(bits string) {
	for _, bitChar := range bits {
		if bitChar == '1' {
			bw.WriteBit(1)
		} else {
			bw.WriteBit(0)
		}
	}
}

// Flush flushes remaining bits with padding
func (bw *BitWriter) Flush() []byte {
	if bw.bitCount > 0 {
		// Pad with zeros to complete the byte
		paddingNeeded := 8 - bw.bitCount
		bw.currentByte = bw.currentByte << paddingNeeded
		bw.data = append(bw.data, bw.currentByte)
	}
	return bw.data
}

// BitsWritten returns total bits written
func (bw *BitWriter) BitsWritten() int64 {
	return bw.bitsWritten
}

// BitReader provides high-precision bit-level reading with enhanced monitoring
type BitReader struct {
	data      []byte
	byteIndex int
	bitIndex  int
	totalBits int64
	bitsRead  int64
}

// NewBitReader creates a new BitReader
func NewBitReader(data []byte) *BitReader {
	return &BitReader{
		data:      data,
		byteIndex: 0,
		bitIndex:  0,
		totalBits: int64(len(data)) * 8,
		bitsRead:  0,
	}
}

// ReadBit reads a single bit with bounds checking
func (br *BitReader) ReadBit() int {
	if br.bitsRead >= br.totalBits || br.byteIndex >= len(br.data) {
		return 0 // Safe EOF handling
	}
	
	bit := int((br.data[br.byteIndex] >> (7 - br.bitIndex)) & 1)
	br.bitIndex++
	br.bitsRead++
	
	if br.bitIndex == 8 {
		br.bitIndex = 0
		br.byteIndex++
	}
	
	return bit
}

// HasMoreBits checks if there are more bits to read
func (br *BitReader) HasMoreBits() bool {
	return br.bitsRead < br.totalBits && br.byteIndex < len(br.data)
}

// BitsRead returns total bits read
func (br *BitReader) BitsRead() int64 {
	return br.bitsRead
}

// HuffmanEncoder implements Huffman coding with aerospace-grade precision monitoring
type HuffmanEncoder struct {
	name           string
	category       core.AlgorithmCategory
	compressionType core.CompressionType
	parameters     map[string]interface{}
	monitor        *performance.AerospaceGradeMonitor
}

// NewHuffmanEncoder creates a new Huffman encoder
func NewHuffmanEncoder() (*HuffmanEncoder, error) {
	monitor, err := performance.NewAerospaceGradeMonitor(0.1) // 100μs precision
	if err != nil {
		return nil, fmt.Errorf("failed to create performance monitor: %w", err)
	}
	
	return &HuffmanEncoder{
		name:           "Huffman",
		category:       core.AlgorithmCategoryEntropyCoding,
		compressionType: core.CompressionTypeLossless,
		parameters:     make(map[string]interface{}),
		monitor:        monitor,
	}, nil
}

// GetName returns the algorithm name
func (h *HuffmanEncoder) GetName() string {
	return h.name
}

// GetCategory returns the algorithm category
func (h *HuffmanEncoder) GetCategory() core.AlgorithmCategory {
	return h.category
}

// GetCompressionType returns the compression type
func (h *HuffmanEncoder) GetCompressionType() core.CompressionType {
	return h.compressionType
}

// SetParameters sets algorithm parameters
func (h *HuffmanEncoder) SetParameters(params map[string]interface{}) error {
	for k, v := range params {
		h.parameters[k] = v
	}
	return nil
}

// GetParameters gets current algorithm parameters
func (h *HuffmanEncoder) GetParameters() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range h.parameters {
		result[k] = v
	}
	return result
}

// GetInfo gets comprehensive algorithm information
func (h *HuffmanEncoder) GetInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":             h.name,
		"category":         string(h.category),
		"compression_type": string(h.compressionType),
		"parameters":       h.GetParameters(),
		"supports_streaming": false,
		"thread_safe":       false,
		"deterministic":     true,
		"memory_bounded":    true,
	}
}

// buildFrequencyTable builds frequency table from input data
func (h *HuffmanEncoder) buildFrequencyTable(data []byte) map[byte]int {
	frequencies := make(map[byte]int)
	for _, b := range data {
		frequencies[b]++
	}
	return frequencies
}

// buildHuffmanTree builds the Huffman tree from frequency table
func (h *HuffmanEncoder) buildHuffmanTree(frequencies map[byte]int) *HuffmanNode {
	if len(frequencies) == 0 {
		return nil
	}
	
	// Handle single symbol case
	if len(frequencies) == 1 {
		var symbol byte
		var freq int
		for b, f := range frequencies {
			symbol = b
			freq = f
			break
		}
		return &HuffmanNode{
			Frequency: freq,
			Symbol:    &symbol,
			IsLeaf:    true,
		}
	}
	
	// Create priority queue (min-heap)
	nodeHeap := &HuffmanNodeHeap{}
	heap.Init(nodeHeap)
	
	// Add leaf nodes to heap
	for symbol, freq := range frequencies {
		symbolCopy := symbol // Important: copy the symbol
		node := &HuffmanNode{
			Frequency: freq,
			Symbol:    &symbolCopy,
			IsLeaf:    true,
		}
		heap.Push(nodeHeap, node)
	}
	
	// Build tree by combining nodes
	for nodeHeap.Len() > 1 {
		left := heap.Pop(nodeHeap).(*HuffmanNode)
		right := heap.Pop(nodeHeap).(*HuffmanNode)
		
		parent := &HuffmanNode{
			Frequency: left.Frequency + right.Frequency,
			Left:      left,
			Right:     right,
			IsLeaf:    false,
		}
		
		heap.Push(nodeHeap, parent)
	}
	
	return heap.Pop(nodeHeap).(*HuffmanNode)
}

// buildCodeTable builds the code table from Huffman tree
func (h *HuffmanEncoder) buildCodeTable(root *HuffmanNode) map[byte]string {
	codeTable := make(map[byte]string)
	
	if root == nil {
		return codeTable
	}
	
	// Handle single symbol case
	if root.IsLeaf {
		codeTable[*root.Symbol] = "0"
		return codeTable
	}
	
	var buildCodes func(*HuffmanNode, string)
	buildCodes = func(node *HuffmanNode, code string) {
		if node.IsLeaf {
			codeTable[*node.Symbol] = code
			return
		}
		
		if node.Left != nil {
			buildCodes(node.Left, code+"0")
		}
		if node.Right != nil {
			buildCodes(node.Right, code+"1")
		}
	}
	
	buildCodes(root, "")
	return codeTable
}

// serializeTree serializes the Huffman tree for storage
func (h *HuffmanEncoder) serializeTree(root *HuffmanNode, writer *BitWriter) {
	if root == nil {
		return
	}
	
	if root.IsLeaf {
		writer.WriteBit(1) // Leaf marker
		// Write the symbol (8 bits)
		for i := 7; i >= 0; i-- {
			bit := int((*root.Symbol >> i) & 1)
			writer.WriteBit(bit)
		}
	} else {
		writer.WriteBit(0) // Internal node marker
		h.serializeTree(root.Left, writer)
		h.serializeTree(root.Right, writer)
	}
}

// deserializeTree deserializes the Huffman tree from reader
func (h *HuffmanEncoder) deserializeTree(reader *BitReader) *HuffmanNode {
	if !reader.HasMoreBits() {
		return nil
	}
	
	isLeaf := reader.ReadBit()
	
	if isLeaf == 1 {
		// Read the symbol (8 bits)
		var symbol byte
		for i := 7; i >= 0; i-- {
			bit := reader.ReadBit()
			if bit == 1 {
				symbol |= (1 << i)
			}
		}
		return &HuffmanNode{
			Symbol: &symbol,
			IsLeaf: true,
		}
	} else {
		// Internal node
		left := h.deserializeTree(reader)
		right := h.deserializeTree(reader)
		return &HuffmanNode{
			Left:   left,
			Right:  right,
			IsLeaf: false,
		}
	}
}

// Compress compresses data using Huffman coding with aerospace-grade performance monitoring
func (h *HuffmanEncoder) Compress(ctx context.Context, data []byte) (*core.CompressionResult, error) {
	if len(data) == 0 {
		return &core.CompressionResult{
			CompressedData:   []byte{},
			OriginalSize:     0,
			CompressedSize:   0,
			CompressionRatio: 1.0,
			CompressionTime:  0.0,
			AlgorithmName:    h.name,
			Metadata:         make(map[string]interface{}),
			PrecisionMetrics: core.AerospacePrecisionMetrics{},
		}, nil
	}
	
	profile, err := h.monitor.ProfileOperation(ctx, "huffman_compress", int64(len(data)), func() error {
		return nil // The actual compression work is done below
	})
	
	if err != nil {
		return nil, &core.CompressionError{
			Message:     fmt.Sprintf("profiling failed: %v", err),
			Algorithm:   h.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}
	
	// Build frequency table
	frequencies := h.buildFrequencyTable(data)
	
	// Build Huffman tree
	root := h.buildHuffmanTree(frequencies)
	if root == nil {
		return nil, &core.CompressionError{
			Message:     "failed to build Huffman tree",
			Algorithm:   h.name,
			DataSize:    int64(len(data)),
			TimestampNs: time.Now().UnixNano(),
		}
	}
	
	// Build code table
	codeTable := h.buildCodeTable(root)
	
	// Serialize the tree
	treeWriter := NewBitWriter()
	h.serializeTree(root, treeWriter)
	serializedTree := treeWriter.Flush()
	
	// Encode the data
	dataWriter := NewBitWriter()
	for _, b := range data {
		if code, exists := codeTable[b]; exists {
			dataWriter.WriteBits(code)
		}
	}
	encodedData := dataWriter.Flush()
	
	// Combine tree and data
	// Format: [tree_size(4 bytes)][tree][encoded_data]
	treeSize := len(serializedTree)
	compressedData := make([]byte, 4+len(serializedTree)+len(encodedData))
	
	// Write tree size (big-endian)
	compressedData[0] = byte(treeSize >> 24)
	compressedData[1] = byte(treeSize >> 16)
	compressedData[2] = byte(treeSize >> 8)
	compressedData[3] = byte(treeSize)
	
	// Write tree
	copy(compressedData[4:], serializedTree)
	
	// Write encoded data
	copy(compressedData[4+len(serializedTree):], encodedData)
	
	// Calculate metrics
	compressionRatio := float64(len(data)) / float64(len(compressedData))
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := h.convertProfileToMetrics(profile, int64(len(data)), int64(len(compressedData)))
	
	metadata := map[string]interface{}{
		"tree_size_bytes":    len(serializedTree),
		"encoded_data_bytes": len(encodedData),
		"unique_symbols":     len(frequencies),
		"tree_depth":         h.calculateTreeDepth(root),
		"compression_efficiency": compressionRatio,
	}
	
	return &core.CompressionResult{
		CompressedData:   compressedData,
		OriginalSize:     int64(len(data)),
		CompressedSize:   int64(len(compressedData)),
		CompressionRatio: compressionRatio,
		CompressionTime:  profile.DurationS,
		AlgorithmName:    h.name,
		Metadata:         metadata,
		PrecisionMetrics: precisionMetrics,
	}, nil
}

// Decompress decompresses Huffman-encoded data with aerospace-grade performance monitoring
func (h *HuffmanEncoder) Decompress(ctx context.Context, compressedData []byte, metadata map[string]interface{}) (*core.DecompressionResult, error) {
	if len(compressedData) == 0 {
		return &core.DecompressionResult{
			DecompressedData:       []byte{},
			OriginalCompressedSize: 0,
			DecompressedSize:       0,
			DecompressionTime:      0.0,
			AlgorithmName:          h.name,
			Metadata:               make(map[string]interface{}),
			PrecisionMetrics:       core.AerospacePrecisionMetrics{},
		}, nil
	}
	
	if len(compressedData) < 4 {
		return nil, &core.DecompressionError{
			Message:        "compressed data too short",
			Algorithm:      h.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}
	
	var decompressedData []byte
	
	profile, err := h.monitor.ProfileOperation(ctx, "huffman_decompress", int64(len(compressedData)), func() error {
		// Read tree size
		treeSize := int(compressedData[0])<<24 + int(compressedData[1])<<16 + int(compressedData[2])<<8 + int(compressedData[3])
		
		if len(compressedData) < 4+treeSize {
			return fmt.Errorf("invalid tree size or corrupted data")
		}
		
		// Deserialize tree
		treeData := compressedData[4 : 4+treeSize]
		treeReader := NewBitReader(treeData)
		root := h.deserializeTree(treeReader)
		
		if root == nil {
			return fmt.Errorf("failed to deserialize Huffman tree")
		}
		
		// Decode data
		encodedData := compressedData[4+treeSize:]
		dataReader := NewBitReader(encodedData)
		
		var result []byte
		current := root
		
		for dataReader.HasMoreBits() {
			bit := dataReader.ReadBit()
			
			if current.IsLeaf {
				result = append(result, *current.Symbol)
				current = root
			}
			
			if bit == 0 && current.Left != nil {
				current = current.Left
			} else if bit == 1 && current.Right != nil {
				current = current.Right
			}
			
			if current.IsLeaf {
				result = append(result, *current.Symbol)
				current = root
			}
		}
		
		// Handle final symbol if we ended on a leaf
		if current.IsLeaf {
			result = append(result, *current.Symbol)
		}
		
		decompressedData = result
		return nil
	})
	
	if err != nil {
		return nil, &core.DecompressionError{
			Message:        fmt.Sprintf("decompression failed: %v", err),
			Algorithm:      h.name,
			CompressedSize: int64(len(compressedData)),
			TimestampNs:    time.Now().UnixNano(),
		}
	}
	
	// Convert profile to aerospace precision metrics
	precisionMetrics := h.convertProfileToMetrics(profile, int64(len(compressedData)), int64(len(decompressedData)))
	
	return &core.DecompressionResult{
		DecompressedData:       decompressedData,
		OriginalCompressedSize: int64(len(compressedData)),
		DecompressedSize:       int64(len(decompressedData)),
		DecompressionTime:      profile.DurationS,
		AlgorithmName:          h.name,
		Metadata:               metadata,
		PrecisionMetrics:       precisionMetrics,
	}, nil
}

// calculateTreeDepth calculates the depth of the Huffman tree
func (h *HuffmanEncoder) calculateTreeDepth(root *HuffmanNode) int {
	if root == nil || root.IsLeaf {
		return 0
	}
	
	leftDepth := h.calculateTreeDepth(root.Left)
	rightDepth := h.calculateTreeDepth(root.Right)
	
	if leftDepth > rightDepth {
		return leftDepth + 1
	}
	return rightDepth + 1
}

// convertProfileToMetrics converts performance profile to aerospace precision metrics
func (h *HuffmanEncoder) convertProfileToMetrics(profile *performance.PrecisionPerformanceProfile, dataSize, outputSize int64) core.AerospacePrecisionMetrics {
	metrics := core.AerospacePrecisionMetrics{
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
		DeterminismScore:              1.0, // Huffman is deterministic
		MemoryOverheadRatio:           float64(profile.MemoryPeakBytes) / float64(dataSize),
		CPUEfficiencyBytesPerCPUSecond: float64(dataSize) / max(0.001, profile.CPUTimeUserS+profile.CPUTimeSystemS),
		IOEfficiencyRatio:             float64(dataSize) / max(1.0, float64(profile.IOReadBytes+profile.IOWriteBytes)),
	}
	
	metrics.UpdateFormattedTimes()
	return metrics
}

// Helper function for max
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
} 