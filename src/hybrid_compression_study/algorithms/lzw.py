"""
LZW (Lempel-Ziv-Welch) compression algorithm implementation.

LZW is a dictionary-based compression algorithm that builds a dictionary
of strings dynamically during compression, replacing repeated strings
with shorter codes.
"""

import time
from typing import Dict, Any, List, Union, Optional
from ..core.base import (
    CompressionAlgorithm,
    CompressionResult,
    DecompressionResult,
    AlgorithmCategory,
    CompressionError,
    DecompressionError
)


class LZWEncoder(CompressionAlgorithm):
    """
    LZW compression implementation.
    
    Builds a dictionary of strings dynamically and replaces them with codes.
    Uses variable-length codes that grow as the dictionary expands.
    """
    
    def __init__(self, max_code_bits: int = 12, initial_dict_size: int = 256):
        """
        Initialize LZW encoder.
        
        Args:
            max_code_bits: Maximum number of bits for codes (8-16)
            initial_dict_size: Initial dictionary size (typically 256 for bytes)
        """
        super().__init__("LZW", AlgorithmCategory.DICTIONARY)
        
        if not 8 <= max_code_bits <= 16:
            raise ValueError("max_code_bits must be between 8 and 16")
        if initial_dict_size < 1 or initial_dict_size > (1 << max_code_bits):
            raise ValueError("initial_dict_size must be between 1 and 2^max_code_bits")
        
        self.max_code_bits = max_code_bits
        self.initial_dict_size = initial_dict_size
        self.max_dict_size = (1 << max_code_bits) - 1
        
        self.set_parameters(
            max_code_bits=max_code_bits,
            initial_dict_size=initial_dict_size,
            max_dict_size=self.max_dict_size
        )
    
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress data using LZW algorithm.
        
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
                metadata={'codes_used': 0, 'dict_size': self.initial_dict_size}
            )
        
        start_time = time.perf_counter()
        
        try:
            codes, final_dict_size = self._encode_lzw(data)
            compressed_data = self._serialize_codes(codes)
            
            compression_time = time.perf_counter() - start_time
            
            return CompressionResult(
                compressed_data=compressed_data,
                original_size=len(data),
                compressed_size=len(compressed_data),
                compression_ratio=len(data) / len(compressed_data) if compressed_data else float('inf'),
                compression_time=compression_time,
                algorithm_name=self.name,
                metadata={
                    'codes_used': len(codes),
                    'dict_size': final_dict_size,
                    'max_code_bits': self.max_code_bits,
                    'initial_dict_size': self.initial_dict_size,
                    'compression_efficiency': len(codes) / len(data) if data else 0
                }
            )
            
        except Exception as e:
            raise CompressionError(f"LZW compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> DecompressionResult:
        """
        Decompress LZW-compressed data.
        
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
            codes = self._deserialize_codes(compressed_data)
            decompressed_data = self._decode_lzw(codes)
            
            decompression_time = time.perf_counter() - start_time
            
            return DecompressionResult(
                decompressed_data=decompressed_data,
                original_compressed_size=len(compressed_data),
                decompressed_size=len(decompressed_data),
                decompression_time=decompression_time,
                algorithm_name=self.name,
                metadata={'codes_processed': len(codes)}
            )
            
        except Exception as e:
            raise DecompressionError(f"LZW decompression failed: {str(e)}")
    
    def _encode_lzw(self, data: bytes) -> tuple[List[int], int]:
        """
        Core LZW encoding logic.
        
        Returns:
            Tuple of (codes_list, final_dictionary_size)
        """
        # Initialize dictionary with single-byte strings
        dictionary = {}
        for i in range(self.initial_dict_size):
            dictionary[bytes([i])] = i
        
        next_code = self.initial_dict_size
        codes = []
        current_string = b""
        
        for byte in data:
            current_byte = bytes([byte])
            extended_string = current_string + current_byte
            
            if extended_string in dictionary:
                # String is in dictionary, continue building
                current_string = extended_string
            else:
                # String not in dictionary
                # Output code for current_string
                if current_string:
                    codes.append(dictionary[current_string])
                
                # Add new string to dictionary if there's room
                if next_code <= self.max_dict_size:
                    dictionary[extended_string] = next_code
                    next_code += 1
                
                # Start new string with current byte
                current_string = current_byte
        
        # Output code for remaining string
        if current_string:
            codes.append(dictionary[current_string])
        
        return codes, len(dictionary)
    
    def _decode_lzw(self, codes: List[int]) -> bytes:
        """
        Core LZW decoding logic.
        """
        if not codes:
            return b''
        
        # Initialize dictionary with single-byte strings
        dictionary = {}
        for i in range(self.initial_dict_size):
            dictionary[i] = bytes([i])
        
        next_code = self.initial_dict_size
        result = bytearray()
        
        # Process first code
        old_code = codes[0]
        if old_code >= len(dictionary):
            raise DecompressionError(f"Invalid first code: {old_code}")
        
        string = dictionary[old_code]
        result.extend(string)
        
        # Process remaining codes
        for code in codes[1:]:
            if code in dictionary:
                # Code is in dictionary
                string = dictionary[code]
            elif code == next_code:
                # Special case: code not yet in dictionary
                # This happens when we're referring to a string we just added
                string = dictionary[old_code] + dictionary[old_code][:1]
            else:
                raise DecompressionError(f"Invalid code: {code} (next_code: {next_code})")
            
            result.extend(string)
            
            # Add new entry to dictionary
            if next_code <= self.max_dict_size:
                new_entry = dictionary[old_code] + string[:1]
                dictionary[next_code] = new_entry
                next_code += 1
            
            old_code = code
        
        return bytes(result)
    
    def _serialize_codes(self, codes: List[int]) -> bytes:
        """
        Serialize codes to bytes using variable-length encoding.
        """
        if not codes:
            return b''
        
        # Determine required bits based on maximum code value
        max_code = max(codes) if codes else 0
        bits_needed = max(8, (max_code.bit_length()))
        bits_needed = min(bits_needed, self.max_code_bits)
        
        result = bytearray()
        # Store the number of bits used (first byte)
        result.append(bits_needed)
        
        # Pack codes into bytes
        bit_buffer = 0
        bit_count = 0
        
        for code in codes:
            # Add code to bit buffer
            bit_buffer = (bit_buffer << bits_needed) | code
            bit_count += bits_needed
            
            # Extract complete bytes
            while bit_count >= 8:
                bit_count -= 8
                byte_value = (bit_buffer >> bit_count) & 0xFF
                result.append(byte_value)
        
        # Add remaining bits if any
        if bit_count > 0:
            bit_buffer <<= (8 - bit_count)
            result.append(bit_buffer & 0xFF)
        
        return bytes(result)
    
    def _deserialize_codes(self, data: bytes) -> List[int]:
        """
        Deserialize bytes back to codes.
        """
        if not data:
            return []
        
        # Extract number of bits per code
        bits_per_code = data[0]
        if bits_per_code < 8 or bits_per_code > self.max_code_bits:
            raise DecompressionError(f"Invalid bits per code: {bits_per_code}")
        
        codes = []
        bit_buffer = 0
        bit_count = 0
        
        # Process data bytes
        for byte_value in data[1:]:
            bit_buffer = (bit_buffer << 8) | byte_value
            bit_count += 8
            
            # Extract codes
            while bit_count >= bits_per_code:
                bit_count -= bits_per_code
                code = (bit_buffer >> bit_count) & ((1 << bits_per_code) - 1)
                codes.append(code)
        
        return codes
    
    def _calculate_optimal_parameters(self, data: bytes) -> Dict[str, int]:
        """
        Analyze data to suggest optimal parameters.
        """
        if not data:
            return {'max_code_bits': 12, 'initial_dict_size': 256}
        
        # Analyze data characteristics
        unique_bytes = len(set(data))
        data_length = len(data)
        
        # Estimate dictionary growth
        estimated_patterns = min(data_length // 4, 4096)  # Conservative estimate
        
        # Calculate required bits
        total_symbols = self.initial_dict_size + estimated_patterns
        required_bits = max(8, total_symbols.bit_length())
        optimal_bits = min(required_bits, 16)
        
        return {
            'max_code_bits': optimal_bits,
            'initial_dict_size': min(256, unique_bytes + 1),
            'estimated_dict_size': total_symbols
        }


class AdaptiveLZWEncoder(LZWEncoder):
    """
    Adaptive LZW that resets dictionary when compression efficiency drops.
    """
    
    def __init__(self, max_code_bits: int = 12, reset_threshold: float = 1.1):
        """
        Initialize adaptive LZW encoder.
        
        Args:
            max_code_bits: Maximum number of bits for codes
            reset_threshold: Reset dictionary when compression ratio drops below this
        """
        super().__init__(max_code_bits)
        self.name = "LZW-Adaptive"
        self.reset_threshold = reset_threshold
        self.set_parameters(reset_threshold=reset_threshold)
    
    def _encode_lzw(self, data: bytes) -> tuple[List[int], int]:
        """
        Adaptive LZW encoding with dictionary reset capability.
        """
        # Initialize dictionary
        dictionary = {}
        for i in range(self.initial_dict_size):
            dictionary[bytes([i])] = i
        
        next_code = self.initial_dict_size
        codes = []
        current_string = b""
        
        # Statistics for adaptive behavior
        bytes_processed = 0
        codes_generated = 0
        last_reset_pos = 0
        
        for i, byte in enumerate(data):
            current_byte = bytes([byte])
            extended_string = current_string + current_byte
            bytes_processed += 1
            
            if extended_string in dictionary:
                current_string = extended_string
            else:
                # Output code for current_string
                if current_string:
                    codes.append(dictionary[current_string])
                    codes_generated += 1
                
                # Check if we should reset dictionary
                if (next_code > self.max_dict_size and 
                    self._should_reset_dictionary(bytes_processed, codes_generated, last_reset_pos)):
                    
                    # Reset dictionary
                    dictionary = {}
                    for j in range(self.initial_dict_size):
                        dictionary[bytes([j])] = j
                    next_code = self.initial_dict_size
                    last_reset_pos = i
                    
                    # Signal dictionary reset with special code
                    codes.append(self.initial_dict_size + 1)  # Reset marker
                
                # Add new string to dictionary if there's room
                if next_code <= self.max_dict_size:
                    dictionary[extended_string] = next_code
                    next_code += 1
                
                current_string = current_byte
        
        # Output code for remaining string
        if current_string:
            codes.append(dictionary[current_string])
        
        return codes, len(dictionary)
    
    def _should_reset_dictionary(self, bytes_processed: int, codes_generated: int, last_reset: int) -> bool:
        """
        Determine if dictionary should be reset based on compression efficiency.
        """
        if bytes_processed - last_reset < 1000:  # Don't reset too frequently
            return False
        
        if codes_generated == 0:
            return False
        
        # Calculate recent compression ratio
        recent_ratio = (bytes_processed - last_reset) / codes_generated
        return recent_ratio < self.reset_threshold 