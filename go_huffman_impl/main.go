package main

import (
	"fmt"
	"time"
    "huffman_study/compression"
)


func huffmanString(input string) (string, int) {
	root := compression.BuildHuffmanTree(input)
	codes := make(map[rune]string)
	compression.GenerateCodes(root, "", codes)
	huffmanEncoded := compression.Encode(input, codes)
	return huffmanEncoded, MeasureSpace(huffmanEncoded)
}

func main() {
	// Sample input and various test cases
    inputs := loadInputFromFile()

	fmt.Println("Performance and Space comparison for various inputs:")

	for _, in := range inputs {
		fmt.Printf("\nTesting with input size: %d characters\n", len(in))
        fmt.Printf("String:%s\n\n", in)

		// Simple Encoding
		start := time.Now()
		simpleEncoded := compression.SimpleEncode(in)
        TrackTime(start, "Simple Encoding")
		simpleSize := MeasureSpace(simpleEncoded)
		fmt.Printf("Simple Encoded size: %d bits\n", simpleSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, simpleSize))

		// Huffman Encoding
		start = time.Now()
		_, hufSize := huffmanString(in)
        TrackTime(start, "Huffman Encoding")
		fmt.Printf("Huffman Encoded size: %d bits\n", hufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, hufSize))

		// BWT + Huffman Encoding
		start = time.Now()
		_, bwtHufSize := huffmanString(compression.BWTTransform(in))
		TrackTime(start, "BWT + Huffman Encoding")
		fmt.Printf("BWT + Huffman Encoded size: %d bits\n", bwtHufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, bwtHufSize))

		// RLE + Huffman Encoding
		start = time.Now()
		_, rleHufSize := huffmanString(compression.RLECompress(in))
		TrackTime(start, "RLE + Huffman Encoding")
		fmt.Printf("RLE + Huffman Encoded size: %d bits\n", rleHufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, rleHufSize))

		// BWT + RLE + Huffman Encoding
		start = time.Now()
		bwtRle := compression.RLECompress(compression.BWTTransform(in))
		_, bwtRleHufSize := huffmanString(bwtRle)
		TrackTime(start, "BWT + RLE + Huffman Encoding")
		fmt.Printf("BWT + RLE + Huffman Encoded size: %d bits\n", bwtRleHufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, bwtRleHufSize))
        
		// BWT + MTF + RLE + Huffman Encoding
		start = time.Now()
        bwtMtf := compression.MoveToFrontTransform(compression.BWTTransform(in))
        _, bwtMtfRleHufSize := huffmanString(compression.RLECompress(bwtMtf))
		TrackTime(start, "BWT + MTF + RLE + Huffman Encoding")
		fmt.Printf("BWT + MTF + RLE + Huffman Encoding size: %d bits\n", bwtMtfRleHufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, bwtMtfRleHufSize))


/*        
		// BWT + MTF + RLE+ LZW + Huffman Encoding
		start = time.Now()
        bwtMtf = compression.MoveToFrontTransform(compression.BWTTransform(in))
        bwtMtfRleLZW := compression.LZWCompress(compression.RLECompress(bwtMtf))
        _, hufSize = huffmanString(bwtMtfRleLZW)
		TrackTime(start, "BWT + MTF + RLE + LZW + Huffman Encoding")
		fmt.Printf("BWT + MTF + RLE + LZW + Huffman Encoding size: %d bits\n", hufSize)
		fmt.Printf("Compression Ratio: %.5f\n\n", CompressionRatio(len(in)*8, hufSize))
    */


	}

	fmt.Println("*****************************************************")
}
