"""
Enhanced Huffman Coding implementation with aerospace-grade precision monitoring.

This implementation provides nanosecond-precision metrics suitable for
mission-critical applications where every byte and microsecond matters.
"""

import heapq
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..core.base import (
    CompressionAlgorithm,
    CompressionResult,
    DecompressionResult,
    AlgorithmCategory,
    CompressionError,
    DecompressionError
)


@dataclass
class HuffmanNode:
    """Node in the Huffman tree with enhanced debugging info."""
    frequency: int
    symbol: Optional[int] = None
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def __post_init__(self):
        self.is_leaf = self.symbol is not None
    
    def __lt__(self, other):
        return self.frequency < other.frequency


class BitWriter:
    """High-precision bit-level writer with enhanced monitoring."""
    
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bit_count = 0
        self._bits_written = 0
    
    def write_bit(self, bit: int) -> None:
        """Write a single bit with precision tracking."""
        if bit:
            self.current_byte |= (1 << (7 - self.bit_count))
        
        self.bit_count += 1
        self._bits_written += 1
        
        if self.bit_count == 8:
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0
    
    def write_bits(self, bits: str) -> None:
        """Write multiple bits from string representation."""
        for bit_char in bits:
            self.write_bit(1 if bit_char == '1' else 0)
    
    def flush(self) -> bytes:
        """Flush remaining bits with padding."""
        if self.bit_count > 0:
            # Pad with zeros to complete the byte
            padding_needed = 8 - self.bit_count
            self.current_byte = self.current_byte << padding_needed
            self.data.append(self.current_byte)
        
        return bytes(self.data)
    
    @property
    def bits_written(self) -> int:
        """Get total bits written."""
        return self._bits_written


class BitReader:
    """High-precision bit-level reader with enhanced monitoring."""
    
    def __init__(self, data: bytes):
        self.data = data
        self.byte_index = 0
        self.bit_index = 0
        self.total_bits = len(data) * 8
        self.bits_read = 0
    
    def read_bit(self) -> int:
        """Read a single bit with bounds checking."""
        if self.bits_read >= self.total_bits:
            return 0  # Safe EOF handling
        
        if self.byte_index >= len(self.data):
            return 0  # Safe EOF handling
        
        bit = (self.data[self.byte_index] >> (7 - self.bit_index)) & 1
        
        self.bit_index += 1
        self.bits_read += 1
        
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        
        return bit
    
    def has_more_bits(self) -> bool:
        """Check if more bits are available."""
        return self.bits_read < self.total_bits and self.byte_index < len(self.data)


class HuffmanEncoder(CompressionAlgorithm):
    """
    Enhanced Huffman coding implementation with aerospace-grade monitoring.
    
    Provides nanosecond-precision performance metrics suitable for
    mission-critical space applications.
    """
    
    def __init__(self):
        super().__init__("Huffman", AlgorithmCategory.ENTROPY_CODING)
        self._debug_enabled = False
    
    def enable_debug(self, enabled: bool = True) -> None:
        """Enable detailed debug logging."""
        self._debug_enabled = enabled
    
    def compress(self, data: bytes) -> CompressionResult:
        """
        Compress data using Huffman coding with aerospace-grade monitoring.
        
        Args:
            data: Raw bytes to compress
            
        Returns:
            CompressionResult with nanosecond-precision metrics
        """
        if not data:
            # Handle empty data case
            return self._create_compression_result(
                compressed_data=b'',
                original_size=0,
                profile=self._create_empty_profile(),
                metadata={'empty_data': True}
            )
        
        # Use aerospace-grade monitoring
        with self._monitor.profile_operation("huffman_compress", len(data)) as profile:
            try:
                if len(set(data)) == 1:
                    # Single symbol optimization
                    compressed_data = self._compress_single_symbol(data)
                    metadata = {
                        'single_symbol': True,
                        'symbol': data[0],
                        'count': len(data),
                        'tree_size': 0
                    }
                else:
                    # Multi-symbol compression
                    compressed_data, metadata = self._compress_multi_symbol(data)
                
                return self._create_compression_result(
                    compressed_data=compressed_data,
                    original_size=len(data),
                    profile=profile,
                    metadata=metadata
                )
                
            except Exception as e:
                raise CompressionError(
                    f"Huffman compression failed: {str(e)}",
                    algorithm=self.name,
                    data_size=len(data),
                    error_context={'symbols_count': len(set(data))}
                )
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> DecompressionResult:
        """
        Decompress Huffman-compressed data with aerospace-grade monitoring.
        
        Args:
            compressed_data: Compressed bytes
            metadata: Compression metadata
            
        Returns:
            DecompressionResult with nanosecond-precision metrics
        """
        if not compressed_data:
            return self._create_decompression_result(
                decompressed_data=b'',
                compressed_size=0,
                profile=self._create_empty_profile(),
                metadata={}
            )
        
        with self._monitor.profile_operation("huffman_decompress", len(compressed_data)) as profile:
            try:
                if metadata.get('single_symbol', False):
                    # Single symbol decompression
                    decompressed_data = self._decompress_single_symbol(compressed_data, metadata)
                else:
                    # Multi-symbol decompression
                    decompressed_data = self._decompress_multi_symbol(compressed_data, metadata)
                
                return self._create_decompression_result(
                    decompressed_data=decompressed_data,
                    compressed_size=len(compressed_data),
                    profile=profile,
                    metadata=metadata
                )
                
            except Exception as e:
                raise DecompressionError(
                    f"Huffman decompression failed: {str(e)}",
                    algorithm=self.name,
                    compressed_size=len(compressed_data),
                    error_context=metadata
                )
    
    def _compress_single_symbol(self, data: bytes) -> bytes:
        """Optimized compression for single-symbol data."""
        symbol = data[0]
        count = len(data)
        
        # Encode as: [symbol][count_bytes][count_value]
        result = bytearray()
        result.append(symbol)
        
        # Encode count as variable length
        count_bytes = count.to_bytes((count.bit_length() + 7) // 8, 'big')
        result.append(len(count_bytes))
        result.extend(count_bytes)
        
        return bytes(result)
    
    def _decompress_single_symbol(self, compressed_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Optimized decompression for single-symbol data."""
        if len(compressed_data) < 3:
            raise DecompressionError("Invalid single symbol compressed data")
        
        symbol = compressed_data[0]
        count_len = compressed_data[1]
        
        if len(compressed_data) < 2 + count_len:
            raise DecompressionError("Incomplete single symbol compressed data")
        
        count = int.from_bytes(compressed_data[2:2+count_len], 'big')
        return bytes([symbol] * count)
    
    def _compress_multi_symbol(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Compress multi-symbol data using full Huffman coding."""
        # Step 1: Build frequency table
        frequencies = Counter(data)
        
        if self._debug_enabled:
            print(f"🔍 Frequencies: {dict(frequencies)}")
        
        # Step 2: Build Huffman tree
        root = self._build_huffman_tree(frequencies)
        
        # Step 3: Generate codes
        codes = {}
        self._generate_codes(root, "", codes)
        
        if self._debug_enabled:
            print(f"🔍 Generated codes: {codes}")
        
        # Step 4: Encode data
        encoded_bits = self._encode_data(data, codes)
        
        # Step 5: Serialize tree and combine with encoded data
        tree_data = self._serialize_tree(root)
        compressed_data = self._combine_tree_and_data(tree_data, encoded_bits, len(data))
        
        # Calculate metrics
        avg_code_length = self._calculate_average_code_length(codes, frequencies)
        entropy = self._calculate_entropy(frequencies, len(data))
        
        metadata = {
            'single_symbol': False,
            'tree_size': len(tree_data),
            'encoded_bits': len(encoded_bits),
            'codes_count': len(codes),
            'average_code_length': avg_code_length,
            'entropy': entropy,
            'efficiency': entropy / avg_code_length if avg_code_length > 0 else 0
        }
        
        return compressed_data, metadata
    
    def _decompress_multi_symbol(self, compressed_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Decompress multi-symbol data using full Huffman decoding."""
        # Extract tree and encoded data
        tree_data, encoded_bits, original_length = self._extract_tree_and_data(compressed_data)
        
        # Deserialize tree
        root = self._deserialize_tree(tree_data)
        
        # Decode data
        return self._decode_data(encoded_bits, root, original_length)
    
    def _build_huffman_tree(self, frequencies: Counter) -> HuffmanNode:
        """Build Huffman tree from frequency table."""
        if len(frequencies) == 1:
            # Special case: single symbol
            symbol = list(frequencies.keys())[0]
            return HuffmanNode(frequencies[symbol], symbol)
        
        # Create priority queue with leaf nodes
        heap = []
        for symbol, freq in frequencies.items():
            node = HuffmanNode(freq, symbol)
            heapq.heappush(heap, node)
        
        # Build tree bottom-up
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(left.frequency + right.frequency)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def _generate_codes(self, node: HuffmanNode, code: str, codes: Dict[int, str]) -> None:
        """Generate Huffman codes recursively."""
        if node is None:
            return
        
        if node.is_leaf:
            # Handle single symbol case
            codes[node.symbol] = code if code else "0"
            return
        
        self._generate_codes(node.left, code + "0", codes)
        self._generate_codes(node.right, code + "1", codes)
    
    def _encode_data(self, data: bytes, codes: Dict[int, str]) -> str:
        """Encode data using Huffman codes."""
        encoded_bits = []
        for byte in data:
            if byte in codes:
                encoded_bits.append(codes[byte])
            else:
                raise CompressionError(f"Symbol {byte} not in code table")
        
        return ''.join(encoded_bits)
    
    def _decode_data(self, encoded_bits: str, root: HuffmanNode, original_length: int) -> bytes:
        """Decode data using Huffman tree with enhanced safety."""
        if not encoded_bits or original_length == 0:
            return b''
        
        if root.is_leaf:
            # Single symbol case
            return bytes([root.symbol] * original_length)
        
        result = bytearray()
        current_node = root
        decoded_count = 0
        
        for bit_char in encoded_bits:
            if decoded_count >= original_length:
                break  # Prevent over-decoding
            
            if bit_char == '0':
                if current_node.left is not None:
                    current_node = current_node.left
                else:
                    raise DecompressionError("Invalid Huffman tree traversal: null left child")
            elif bit_char == '1':
                if current_node.right is not None:
                    current_node = current_node.right
                else:
                    raise DecompressionError("Invalid Huffman tree traversal: null right child")
            else:
                raise DecompressionError(f"Invalid bit character: {bit_char}")
            
            if current_node.is_leaf:
                result.append(current_node.symbol)
                decoded_count += 1
                current_node = root
                
                # Safety check
                if decoded_count > original_length:
                    raise DecompressionError("Decoded more symbols than expected")
        
        return bytes(result)
    
    def _serialize_tree(self, node: HuffmanNode) -> bytes:
        """Serialize Huffman tree to bytes."""
        if node is None:
            return b''
        
        result = bytearray()
        
        if node.is_leaf:
            result.append(1)  # Leaf marker
            result.append(node.symbol)
        else:
            result.append(0)  # Internal node marker
            result.extend(self._serialize_tree(node.left))
            result.extend(self._serialize_tree(node.right))
        
        return bytes(result)
    
    def _deserialize_tree(self, data: bytes) -> HuffmanNode:
        """Deserialize Huffman tree from bytes."""
        if not data:
            raise DecompressionError("Empty tree data")
        
        index = [0]  # Use list for reference semantics
        
        def deserialize_recursive() -> HuffmanNode:
            if index[0] >= len(data):
                raise DecompressionError("Incomplete tree data")
            
            marker = data[index[0]]
            index[0] += 1
            
            if marker == 1:  # Leaf node
                if index[0] >= len(data):
                    raise DecompressionError("Incomplete leaf node data")
                symbol = data[index[0]]
                index[0] += 1
                return HuffmanNode(0, symbol)
            else:  # Internal node
                left = deserialize_recursive()
                right = deserialize_recursive()
                node = HuffmanNode(0)
                node.left = left
                node.right = right
                return node
        
        return deserialize_recursive()
    
    def _combine_tree_and_data(self, tree_data: bytes, encoded_bits: str, original_length: int) -> bytes:
        """Combine serialized tree and encoded data."""
        result = bytearray()
        
        # Header: tree_size (4 bytes) + original_length (4 bytes)
        result.extend(len(tree_data).to_bytes(4, 'big'))
        result.extend(original_length.to_bytes(4, 'big'))
        
        # Tree data
        result.extend(tree_data)
        
        # Encoded data
        writer = BitWriter()
        writer.write_bits(encoded_bits)
        encoded_bytes = writer.flush()
        result.extend(encoded_bytes)
        
        return bytes(result)
    
    def _extract_tree_and_data(self, compressed_data: bytes) -> Tuple[bytes, str, int]:
        """Extract tree data and encoded bits from compressed data."""
        if len(compressed_data) < 8:
            raise DecompressionError("Compressed data too short")
        
        # Read header
        tree_size = int.from_bytes(compressed_data[0:4], 'big')
        original_length = int.from_bytes(compressed_data[4:8], 'big')
        
        # Extract tree data
        if len(compressed_data) < 8 + tree_size:
            raise DecompressionError("Incomplete compressed data")
        
        tree_data = compressed_data[8:8+tree_size]
        encoded_data = compressed_data[8+tree_size:]
        
        # Convert encoded bytes back to bit string
        reader = BitReader(encoded_data)
        encoded_bits = []
        
        while reader.has_more_bits():
            bit = reader.read_bit()
            encoded_bits.append('1' if bit else '0')
        
        return tree_data, ''.join(encoded_bits), original_length
    
    def _calculate_average_code_length(self, codes: Dict[int, str], frequencies: Counter) -> float:
        """Calculate weighted average code length."""
        if not codes or not frequencies:
            return 0.0
        
        total_bits = sum(len(codes[symbol]) * freq for symbol, freq in frequencies.items() if symbol in codes)
        total_symbols = sum(frequencies.values())
        
        return total_bits / total_symbols if total_symbols > 0 else 0.0
    
    def _calculate_entropy(self, frequencies: Counter, total_count: int) -> float:
        """Calculate Shannon entropy of the data."""
        if total_count == 0:
            return 0.0
        
        entropy = 0.0
        for freq in frequencies.values():
            if freq > 0:
                prob = freq / total_count
                entropy -= prob * (prob.bit_length() - 1)  # log2 approximation
        
        return entropy
    
    def _create_empty_profile(self):
        """Create an empty performance profile for edge cases."""
        from ..utils.performance import PrecisionPerformanceProfile
        
        return PrecisionPerformanceProfile(
            operation_name="empty_operation",
            data_size_bytes=0,
            start_time_ns=time.time_ns(),
            end_time_ns=time.time_ns(),
            duration_ns=0,
            duration_us=0.0,
            duration_ms=0.0,
            duration_s=0.0,
            memory_before_bytes=0,
            memory_after_bytes=0,
            memory_peak_bytes=0,
            memory_delta_bytes=0,
            memory_peak_delta_bytes=0,
            tracemalloc_current_mb=0.0,
            tracemalloc_peak_mb=0.0,
            tracemalloc_diff_mb=0.0,
            cpu_percent_avg=0.0,
            cpu_percent_peak=0.0,
            cpu_time_user_s=0.0,
            cpu_time_system_s=0.0,
            cpu_freq_avg_mhz=0.0,
            io_read_bytes=0,
            io_write_bytes=0,
            io_read_ops=0,
            io_write_ops=0,
            io_read_time_ms=0.0,
            io_write_time_ms=0.0,
            page_faults=0,
            context_switches=0,
            threads_created=0,
            gc_collections=[],
            throughput_mbps=0.0,
            throughput_ops_per_sec=0.0,
            bytes_per_cpu_cycle=0.0,
            memory_efficiency_ratio=0.0,
            compression_efficiency=0.0,
            time_per_byte_ns=0.0
        ) 