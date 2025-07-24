"""
LZ77 (Lempel-Ziv 1977) compression algorithm implementation.

LZ77 is a dictionary-based compression algorithm that uses a sliding window
to find repeated substrings and replaces them with references to earlier
occurrences (distance, length) pairs.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from ..core.base import (
    CompressionAlgorithm,
    CompressionResult,
    DecompressionResult,
    AlgorithmCategory,
    CompressionError,
    DecompressionError
)


class LZ77Token:
    """Represents an LZ77 token (distance, length, literal)."""
    
    def __init__(self, distance: int = 0, length: int = 0, literal: int = 0):
        self.distance = distance  # Distance back to start of match
        self.length = length      # Length of match
        self.literal = literal    # Next literal byte after match
    
    def is_literal(self) -> bool:
        """Check if this token represents a literal (no match)."""
        return self.distance == 0 and self.length == 0
    
    def __repr__(self) -> str:
        if self.is_literal():
            return f"Literal({self.literal})"
        return f"Match(dist={self.distance}, len={self.length}, next={self.literal})"


class LZ77Encoder(CompressionAlgorithm):
    """
    LZ77 compression implementation.
    
    Uses a sliding window approach to find matches in previously seen data
    and encodes them as (distance, length, next_literal) tuples.
    """
    
    def __init__(self, window_size: int = 4096, lookahead_size: int = 18):
        """
        Initialize LZ77 encoder.
        
        Args:
            window_size: Size of the sliding window (search buffer)
            lookahead_size: Size of the lookahead buffer
        """
        super().__init__("LZ77", AlgorithmCategory.DICTIONARY)
        
        if window_size <= 0 or window_size > 32768:
            raise ValueError("Window size must be between 1 and 32768")
        if lookahead_size <= 0 or lookahead_size > 258:
            raise ValueError("Lookahead size must be between 1 and 258")
        
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        
        self.set_parameters(
            window_size=window_size,
            lookahead_size=lookahead_size
        )
    
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress data using LZ77 algorithm.
        
        Args:
            data: Raw bytes to compress
            
        Returns:
            CompressionResult with compressed data and metadata
        """
        if not data:
            return CompressionResult(
                compressed_data=b'',
                original_size=0,
                compressed_size=0,
                compression_ratio=1.0,
                compression_time=0.0,
                algorithm_name=self.name,
                metadata={'tokens': 0, 'matches': 0, 'literals': 0}
            )
        
        start_time = time.perf_counter()
        
        try:
            tokens = self._encode_lz77(data)
            compressed_data = self._serialize_tokens(tokens)
            
            compression_time = time.perf_counter() - start_time
            
            # Calculate statistics
            matches = sum(1 for token in tokens if not token.is_literal())
            literals = len(tokens) - matches
            
            return CompressionResult(
                compressed_data=compressed_data,
                original_size=len(data),
                compressed_size=len(compressed_data),
                compression_ratio=len(data) / len(compressed_data) if compressed_data else float('inf'),
                compression_time=compression_time,
                algorithm_name=self.name,
                metadata={
                    'tokens': len(tokens),
                    'matches': matches,
                    'literals': literals,
                    'match_ratio': matches / len(tokens) if tokens else 0,
                    'window_size': self.window_size,
                    'lookahead_size': self.lookahead_size
                }
            )
            
        except Exception as e:
            raise CompressionError(f"LZ77 compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> DecompressionResult:
        """
        Decompress LZ77-compressed data.
        
        Args:
            compressed_data: Compressed bytes
            metadata: Metadata from compression operation
            
        Returns:
            DecompressionResult with original data
        """
        if not compressed_data:
            return DecompressionResult(
                decompressed_data=b'',
                original_compressed_size=0,
                decompressed_size=0,
                decompression_time=0.0,
                algorithm_name=self.name,
                metadata={}
            )
        
        start_time = time.perf_counter()
        
        try:
            tokens = self._deserialize_tokens(compressed_data)
            decompressed_data = self._decode_lz77(tokens)
            
            decompression_time = time.perf_counter() - start_time
            
            return DecompressionResult(
                decompressed_data=decompressed_data,
                original_compressed_size=len(compressed_data),
                decompressed_size=len(decompressed_data),
                decompression_time=decompression_time,
                algorithm_name=self.name,
                metadata={'tokens_processed': len(tokens)}
            )
            
        except Exception as e:
            raise DecompressionError(f"LZ77 decompression failed: {str(e)}")
    
    def _encode_lz77(self, data: bytes) -> List[LZ77Token]:
        """
        Core LZ77 encoding logic.
        
        Returns:
            List of LZ77 tokens
        """
        tokens = []
        pos = 0
        
        while pos < len(data):
            # Find the longest match in the sliding window
            match = self._find_longest_match(data, pos)
            
            if match and match[1] >= 3:  # Only use matches of length 3 or more
                distance, length = match
                
                # Get the next literal after the match (if exists)
                next_pos = pos + length
                next_literal = data[next_pos] if next_pos < len(data) else 0
                
                token = LZ77Token(distance, length, next_literal)
                tokens.append(token)
                
                # Advance position past the match and literal
                pos = next_pos + 1
            else:
                # No match found, emit literal
                token = LZ77Token(0, 0, data[pos])
                tokens.append(token)
                pos += 1
        
        return tokens
    
    def _find_longest_match(self, data: bytes, pos: int) -> Optional[Tuple[int, int]]:
        """
        Find the longest match in the sliding window.
        
        Args:
            data: Input data
            pos: Current position
            
        Returns:
            Tuple of (distance, length) or None if no match found
        """
        if pos >= len(data):
            return None
        
        # Define search window bounds
        search_start = max(0, pos - self.window_size)
        search_end = pos
        
        # Define lookahead bounds
        lookahead_end = min(len(data), pos + self.lookahead_size)
        
        best_distance = 0
        best_length = 0
        
        # Search for matches in the sliding window
        for search_pos in range(search_start, search_end):
            match_length = 0
            
            # Count matching bytes
            while (pos + match_length < lookahead_end and
                   search_pos + match_length < pos and
                   data[search_pos + match_length] == data[pos + match_length]):
                match_length += 1
            
            # Update best match if this one is longer
            if match_length > best_length:
                best_length = match_length
                best_distance = pos - search_pos
        
        return (best_distance, best_length) if best_length > 0 else None
    
    def _serialize_tokens(self, tokens: List[LZ77Token]) -> bytes:
        """
        Serialize LZ77 tokens to bytes.
        
        Format for each token:
        - If literal: [0][literal_byte]
        - If match: [distance_high][distance_low|length][literal_byte]
        """
        result = bytearray()
        
        for token in tokens:
            if token.is_literal():
                # Literal: 0x00 followed by the literal byte
                result.extend([0x00, token.literal])
            else:
                # Match: encode distance (12 bits) and length (4 bits) + literal
                if token.distance > 4095:  # 12-bit limit
                    raise ValueError(f"Distance too large: {token.distance}")
                if token.length > 15:  # 4-bit limit  
                    raise ValueError(f"Length too large: {token.length}")
                
                # Pack distance (12 bits) and length (4 bits) into 2 bytes
                distance_high = (token.distance >> 4) & 0xFF
                distance_low_and_length = ((token.distance & 0x0F) << 4) | (token.length & 0x0F)
                
                result.extend([distance_high, distance_low_and_length, token.literal])
        
        return bytes(result)
    
    def _deserialize_tokens(self, data: bytes) -> List[LZ77Token]:
        """
        Deserialize bytes back to LZ77 tokens.
        """
        tokens = []
        i = 0
        
        while i < len(data):
            if i + 1 >= len(data):
                break
            
            if data[i] == 0x00:
                # Literal token
                literal = data[i + 1]
                tokens.append(LZ77Token(0, 0, literal))
                i += 2
            else:
                # Match token
                if i + 2 >= len(data):
                    break
                
                distance_high = data[i]
                distance_low_and_length = data[i + 1]
                literal = data[i + 2]
                
                # Unpack distance and length
                distance = (distance_high << 4) | ((distance_low_and_length >> 4) & 0x0F)
                length = distance_low_and_length & 0x0F
                
                tokens.append(LZ77Token(distance, length, literal))
                i += 3
        
        return tokens
    
    def _decode_lz77(self, tokens: List[LZ77Token]) -> bytes:
        """
        Decode LZ77 tokens back to original data.
        """
        result = bytearray()
        
        for token in tokens:
            if token.is_literal():
                result.append(token.literal)
            else:
                # Copy from earlier position
                start_pos = len(result) - token.distance
                
                if start_pos < 0:
                    raise DecompressionError(f"Invalid distance: {token.distance} at position {len(result)}")
                
                # Copy bytes (may overlap with current position)
                for _ in range(token.length):
                    if start_pos >= len(result):
                        raise DecompressionError("Copy position beyond current data")
                    result.append(result[start_pos])
                    start_pos += 1
                
                # Add the literal byte
                if token.literal != 0 or len(result) == 0:  # Handle case where literal is meaningful
                    result.append(token.literal)
        
        return bytes(result)


class OptimizedLZ77Encoder(LZ77Encoder):
    """
    Optimized LZ77 with hash table for faster string matching.
    """
    
    def __init__(self, window_size: int = 4096, lookahead_size: int = 18):
        super().__init__(window_size, lookahead_size)
        self.name = "LZ77-Optimized"
        self._hash_table: Dict[int, List[int]] = {}
    
    def _find_longest_match(self, data: bytes, pos: int) -> Optional[Tuple[int, int]]:
        """
        Optimized match finding using hash table.
        """
        if pos + 2 >= len(data):  # Need at least 3 bytes for hashing
            return super()._find_longest_match(data, pos)
        
        # Create hash of next 3 bytes
        hash_key = (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2]
        
        # Get potential match positions
        candidates = self._hash_table.get(hash_key, [])
        
        # Filter candidates within sliding window
        search_start = max(0, pos - self.window_size)
        valid_candidates = [c for c in candidates if search_start <= c < pos]
        
        best_distance = 0
        best_length = 0
        lookahead_end = min(len(data), pos + self.lookahead_size)
        
        # Check each candidate
        for candidate_pos in valid_candidates[-10:]:  # Limit candidates for performance
            match_length = 0
            
            # Count matching bytes
            while (pos + match_length < lookahead_end and
                   candidate_pos + match_length < pos and
                   data[candidate_pos + match_length] == data[pos + match_length]):
                match_length += 1
            
            # Update best match
            if match_length > best_length:
                best_length = match_length
                best_distance = pos - candidate_pos
        
        # Add current position to hash table
        if hash_key not in self._hash_table:
            self._hash_table[hash_key] = []
        self._hash_table[hash_key].append(pos)
        
        # Keep hash table size reasonable
        if len(self._hash_table[hash_key]) > 50:
            self._hash_table[hash_key] = self._hash_table[hash_key][-25:]
        
        return (best_distance, best_length) if best_length >= 3 else None 