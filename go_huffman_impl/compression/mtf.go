package compression

import "strings"

func MoveToFrontTransform(input string) string {
    list := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    var result strings.Builder
    
    for _, char := range input {
        index := strings.IndexRune(list, char)
        if index == -1 {
            continue
        }
        result.WriteByte(list[index])
        list = string(char) + list[:index] + list[index+1:]
    }
    
    return result.String()
}
