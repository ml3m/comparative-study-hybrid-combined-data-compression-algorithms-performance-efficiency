#!/usr/bin/env python3
"""
Script to create various test data files for compression benchmarking.

This script generates different types of data that showcase various
compression algorithm strengths and weaknesses.
"""

import os
import json
import random
import string
from pathlib import Path


def create_text_file(size_kb: int = 10) -> bytes:
    """Create realistic text data."""
    # Sample text with realistic word frequency
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "is", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "we", "him", "been", "has", "had", "which", "more", "when", "who", "may", "said", "each", "other",
        "time", "very", "what", "know", "just", "first", "get", "over", "think", "also", "your", "work", "life"
    ]
    
    words = []
    current_size = 0
    target_size = size_kb * 1024
    
    while current_size < target_size:
        word = random.choice(common_words)
        words.append(word)
        current_size += len(word) + 1  # +1 for space
        
        # Occasionally add punctuation and newlines
        if random.random() < 0.1:
            words.append(random.choice(['.', ',', '!', '?']))
        if random.random() < 0.05:
            words.append('\n')
    
    return ' '.join(words).encode('utf-8')[:target_size]


def create_repetitive_file(size_kb: int = 10) -> bytes:
    """Create highly repetitive data."""
    patterns = [b"AAAA", b"BBBB", b"CCCCCCCC", b"DDDD"]
    data = bytearray()
    target_size = size_kb * 1024
    
    while len(data) < target_size:
        pattern = random.choice(patterns)
        repeat_count = random.randint(5, 50)
        data.extend(pattern * repeat_count)
    
    return bytes(data[:target_size])


def create_structured_file(size_kb: int = 10) -> bytes:
    """Create structured data (JSON)."""
    data = []
    target_size = size_kb * 1024
    current_size = 0
    
    while current_size < target_size:
        record = {
            "id": random.randint(1, 100000),
            "name": f"User{random.randint(1, 1000)}",
            "email": f"user{random.randint(1, 1000)}@example.com",
            "age": random.randint(18, 80),
            "active": random.choice([True, False]),
            "balance": round(random.uniform(0, 10000), 2),
            "tags": random.sample(["premium", "basic", "trial", "vip", "standard"], 
                                 random.randint(1, 3)),
            "metadata": {
                "created": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "source": random.choice(["web", "mobile", "api", "import"])
            }
        }
        data.append(record)
        current_size = len(json.dumps(data).encode('utf-8'))
    
    json_str = json.dumps(data, indent=2)
    return json_str.encode('utf-8')[:target_size]


def create_binary_file(size_kb: int = 10) -> bytes:
    """Create binary data."""
    target_size = size_kb * 1024
    data = bytearray()
    
    # Mix of patterns and random data
    for _ in range(target_size // 100):
        if random.random() < 0.3:
            # Add some pattern
            pattern = bytes([random.randint(0, 255)] * random.randint(4, 16))
            data.extend(pattern)
        else:
            # Add random bytes
            for _ in range(random.randint(10, 50)):
                data.append(random.randint(0, 255))
    
    return bytes(data[:target_size])


def create_sparse_file(size_kb: int = 10) -> bytes:
    """Create sparse data (mostly zeros with occasional non-zero bytes)."""
    target_size = size_kb * 1024
    data = bytearray(target_size)
    
    # Add some non-zero values randomly (about 2% of the data)
    for _ in range(target_size // 50):
        pos = random.randint(0, target_size - 1)
        data[pos] = random.randint(1, 255)
    
    return bytes(data)


def create_english_text_file(size_kb: int = 10) -> bytes:
    """Create English text with realistic letter frequency."""
    # English letter frequencies (approximate)
    letters = "etaoinshrdlcumwfgypbvkjxqz"
    weights = [12.0, 9.1, 8.1, 7.5, 7.0, 6.9, 6.3, 6.1, 5.9, 4.3, 4.0, 3.4, 2.8, 
               2.4, 2.4, 2.1, 1.9, 1.5, 0.95, 0.15, 0.074, 0.019, 0.015, 0.0074]
    
    target_size = size_kb * 1024
    text = []
    
    for _ in range(target_size):
        if random.random() < 0.15:  # Space probability
            text.append(' ')
        elif random.random() < 0.02:  # Newline probability
            text.append('\n')
        else:
            char = random.choices(letters, weights=weights)[0]
            # Randomly capitalize
            if random.random() < 0.1:
                char = char.upper()
            text.append(char)
    
    return ''.join(text).encode('utf-8')[:target_size]


def main():
    """Generate all test files."""
    print("🔧 Creating test data files...")
    
    # File generators
    generators = [
        ("text.txt", create_text_file, 50),
        ("repetitive.bin", create_repetitive_file, 20),
        ("structured.json", create_structured_file, 100),
        ("binary.bin", create_binary_file, 30),
        ("sparse.bin", create_sparse_file, 50),
        ("english.txt", create_english_text_file, 75),
        
        # Different sizes
        ("small_text.txt", lambda: create_text_file(5), None),
        ("medium_text.txt", lambda: create_text_file(25), None),
        ("large_text.txt", lambda: create_text_file(100), None),
        
        ("small_binary.bin", lambda: create_binary_file(5), None),
        ("medium_binary.bin", lambda: create_binary_file(25), None),
        ("large_binary.bin", lambda: create_binary_file(100), None),
    ]
    
    base_dir = Path(__file__).parent
    
    for filename, generator, size_kb in generators:
        filepath = base_dir / filename
        
        try:
            if size_kb is not None:
                data = generator(size_kb)
            else:
                data = generator()
            
            with open(filepath, 'wb') as f:
                f.write(data)
            
            print(f"✅ Created {filename} ({len(data):,} bytes)")
            
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    # Create a README file
    readme_content = """# Test Data Files

This directory contains various test files for compression benchmarking:

## File Types

- **text.txt**: Realistic text with common English words
- **repetitive.bin**: Highly repetitive binary data (good for RLE)
- **structured.json**: JSON data with realistic structure
- **binary.bin**: Mixed binary data with patterns
- **sparse.bin**: Sparse data (mostly zeros)
- **english.txt**: Text with realistic English letter frequencies

## Size Variants

- **small_***: ~5KB files for quick testing
- **medium_***: ~25KB files for moderate testing  
- **large_***: ~100KB files for comprehensive testing

## Usage

These files are automatically used by the benchmarking suite when you specify
the test data directory:

```bash
hcs benchmark --data-dir data/test_files
```

You can also use them individually:

```bash
hcs compress data/test_files/text.txt compressed_output --algorithm huffman
```
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README.md")
    print(f"📁 All test files saved to: {base_dir}")
    print(f"📊 Total files created: {len(generators) + 1}")


if __name__ == "__main__":
    main() 