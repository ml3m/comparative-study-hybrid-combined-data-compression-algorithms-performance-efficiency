package compression 

import (
    "fmt"
)

// chars to binary, size: * 8 basically.
func SimpleEncode(input string) string {
	encoded := ""
	for _, char := range input {
		encoded += fmt.Sprintf("%08b", char)
	}
	return encoded
}
