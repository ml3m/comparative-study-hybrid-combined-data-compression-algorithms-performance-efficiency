package main

import (
	"bufio"
	"os"
	"time"
    "fmt"
)

func TrackTime(start time.Time, name string) {
	elapsed := time.Since(start)
	fmt.Printf("%s took %s\n", name, elapsed)
}

func CompressionRatio(originalSize, compressedSize int) float64 {
	if originalSize == 0 {
		return 0
	}
	return float64(compressedSize) / float64(originalSize)
}

func MeasureSpace(encoded string) int {
	return len(encoded)
}

func getPercentage(algorithmSpace int, simpleSpace int) float32 {
	if algorithmSpace == 0 {
		return 0
	}
	return float32(simpleSpace) / float32(algorithmSpace) * 100
}

func loadInputFromFile() []string {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <input-file>")
		os.Exit(1)
	}
	filepath := os.Args[1]
	readFile, err := os.Open(filepath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	scanner := bufio.NewScanner(readFile)
	scanner.Split(bufio.ScanLines)
	var inputs []string

	for scanner.Scan() {
		inputs = append(inputs, scanner.Text())
	}

	readFile.Close()
	return inputs
}
