package compression

import (
	"container/heap"
)

type Node struct {
	character rune
	frequency int
	left  *Node
	right *Node
}

// p queue
type HuffmanHeap []*Node

func (h HuffmanHeap) Len() int { 
    return len(h) 
}

func (h HuffmanHeap) Less(i, j int) bool { 
    return h[i].frequency < h[j].frequency 
}

func (h HuffmanHeap) Swap(i, j int) {
    h[i], h[j] = h[j], h[i] 
}

func (h *HuffmanHeap) Push(x interface{}) {
	*h = append(*h, x.(*Node))
}

func (h *HuffmanHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func BuildHuffmanTree(input string) *Node {
    if len(input) == 0 {
        return nil
    }

    // counting the occurences of all chars in the givne input string
	freq := make(map[rune]int)
	for _, char := range input {
		freq[char]++
	}

    // init huffman h
	h := &HuffmanHeap{}
	heap.Init(h)

    // fill p queue with all created on spot nodes of form {char, freq};
    // ascending order freq
	for char, frequency := range freq {
		heap.Push(h, &Node{character: char, frequency: frequency})
	}

   /*
    *Pop the two nodes with the lowest frequencies (the smallest elements).
    *Create a new merged node with the sum of their frequencies.
    *The merged node becomes the parent of the two nodes.
    *Push the merged node back into the heap.
    *This process continues until there is only one node left, 
         which becomes the root of the Huffman tree.
   */
	for h.Len() > 1 {
		left := heap.Pop(h).(*Node)
		right := heap.Pop(h).(*Node)
		merged := &Node{
			character: 0,
			frequency: left.frequency + right.frequency,
			left:      left,
			right:     right,
		}
		heap.Push(h, merged)
	}

    // root
	return heap.Pop(h).(*Node)
}

func GenerateCodes(node *Node, prefix string, codes map[rune]string) {
	if node == nil {
		return
	}

	if node.left == nil && node.right == nil {
        // lone char case
		if prefix == "" {
			codes[node.character] = "0"
		} else {
			codes[node.character] = prefix
		}
	}
	GenerateCodes(node.left, prefix+"0", codes)
	GenerateCodes(node.right, prefix+"1", codes)
}

func Encode(input string, codes map[rune]string) string {
	encoded := ""
	for _, char := range input {
		encoded += codes[char]
	}
	return encoded
}

// a loop as trade for free memory. we take it.
func Decode(encoded string, root *Node) string {
	result := ""
	node := root
	for _, bit := range encoded {
		if bit == '0' {
			node = node.left
		} else {
			node = node.right
		}
		if node.left == nil && node.right == nil {
			result += string(node.character)
			node = root
		}
	}
	return result
}
