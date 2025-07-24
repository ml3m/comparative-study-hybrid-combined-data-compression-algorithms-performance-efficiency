package compression 

import (
    "sort"
)

// Function to perform Burrows-Wheeler Transform
func BWTTransform(input string) string {
    // Add end-of-string marker

    // practically helpful 
    input = input + "$"

    // theoretical correct
    //if input == "" {
    //    return ""
    //}

    // Generate all rotations
    rotations := make([]string, len(input))
    for i := 0; i < len(input); i++ {
        rotations[i] = input[i:] + input[:i]
    }

    // Sort rotations lexicographically
    sort.Strings(rotations)

    // Extract the last column of the sorted rotations
    lastColumn := make([]byte, len(input))
    for i, rotation := range rotations {
        lastColumn[i] = rotation[len(rotation)-1]
    }

    return string(lastColumn)
}


/*
// Function to perform Inverse Burrows-Wheeler Transform
func BWTInverse(lastColumn string) string {
    // Create the first column by sorting the last column
    firstColumn := make([]rune, len(lastColumn))
    copy(firstColumn, lastColumn)
    sort.Slice(firstColumn, func(i, j int) bool {
        return firstColumn[i] < firstColumn[j]
    })

    // Create a map of character counts to track positions
    rank := make(map[rune][]int)
    for i, char := range firstColumn {
        rank[char] = append(rank[char], i)
    }

    // Reconstruct the original string
    result := make([]rune, len(lastColumn))
    index := 0
    for i := 0; i < len(lastColumn); i++ {
        result[i] = rune(lastColumn[index])
        index = rank[result[i]][0]
        rank[result[i]] = rank[result[i]][1:]
    }

    return string(result)
}
*/
