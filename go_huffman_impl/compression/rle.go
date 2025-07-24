package compression

import (
    "fmt"
    "strings"
)

// Run-Length Encoding function
func RLECompress(data string) string {
    var result strings.Builder
    n := len(data)
    if n == 0 {
        return ""
    }

    count := 1
    for i := 1; i < n; i++ {
        if data[i] == data[i-1] {
            count++
        } else {
            result.WriteString(fmt.Sprintf("%d%c", count, data[i-1]))
            count = 1
        }
    }
    result.WriteString(fmt.Sprintf("%d%c", count, data[n-1]))

    return result.String()
}

