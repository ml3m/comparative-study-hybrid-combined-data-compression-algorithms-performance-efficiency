package compression

import (
    "fmt"
    "strings"
)

func LZWCompress(input string) string {
    dict := make(map[string]int)
    var result strings.Builder
    dictSize := 256

    for i := 0; i < 256; i++ {
        dict[string(byte(i))] = i
    }

    w := ""
    for _, char := range input {
        wc := w + string(char)
        if _, ok := dict[wc]; ok {
            w = wc
        } else {
            result.WriteString(fmt.Sprintf("%d ", dict[w]))
            dict[wc] = dictSize
            dictSize++
            w = string(char)
        }
    }
    if w != "" {
        result.WriteString(fmt.Sprintf("%d", dict[w]))
    }
    
    return result.String()
}
