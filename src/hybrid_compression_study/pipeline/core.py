"""
Core pipeline system for chaining compression algorithms.

This module provides the infrastructure for creating and executing
compression pipelines that chain multiple algorithms together.
"""

import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from ..core.base import (
    CompressionAlgorithm,
    PipelineComponent,
    CompressionResult,
    DecompressionResult,
    PipelineError,
    CompressionError,
    DecompressionError
)
from ..utils.performance import PerformanceMonitor


@dataclass
class PipelineStage:
    """Represents a single stage in a compression pipeline."""
    component: Union[CompressionAlgorithm, PipelineComponent]
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        if self.parameters:
            if hasattr(self.component, 'set_parameters'):
                self.component.set_parameters(**self.parameters)


@dataclass
class PipelineResult:
    """Results from executing a compression pipeline."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    total_compression_ratio: float
    total_compression_time: float
    total_decompression_time: float
    stage_results: List[Dict[str, Any]]
    pipeline_name: str
    metadata: Dict[str, Any]
    
    @property
    def compression_percentage(self) -> float:
        """Calculate total compression percentage."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100
    
    @property
    def total_time(self) -> float:
        """Calculate total processing time."""
        return self.total_compression_time + self.total_decompression_time


class CompressionPipeline:
    """
    A pipeline for chaining multiple compression algorithms.
    
    Allows creating complex compression schemes by combining different
    algorithms in sequence, such as BWT → MTF → Huffman.
    """
    
    def __init__(self, name: str = "CompressionPipeline"):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.performance_monitor = PerformanceMonitor()
        self._metadata: Dict[str, Any] = {}
    
    def add_stage(self, component: Union[CompressionAlgorithm, PipelineComponent], 
                  name: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> 'CompressionPipeline':
        """
        Add a compression stage to the pipeline.
        
        Args:
            component: Compression algorithm or pipeline component
            name: Optional name for the stage
            parameters: Optional parameters for the component
            
        Returns:
            Self for method chaining
        """
        if name is None:
            name = getattr(component, 'name', f"Stage{len(self.stages)}")
        
        stage = PipelineStage(
            component=component,
            name=name,
            parameters=parameters or {}
        )
        
        self.stages.append(stage)
        return self
    
    def remove_stage(self, index: int) -> 'CompressionPipeline':
        """Remove a stage by index."""
        if 0 <= index < len(self.stages):
            self.stages.pop(index)
        return self
    
    def clear_stages(self) -> 'CompressionPipeline':
        """Remove all stages from the pipeline."""
        self.stages.clear()
        return self
    
    def enable_stage(self, index: int, enabled: bool = True) -> 'CompressionPipeline':
        """Enable or disable a specific stage."""
        if 0 <= index < len(self.stages):
            self.stages[index].enabled = enabled
        return self
    
    def compress(self, data: bytes) -> PipelineResult:
        """
        Compress data through the entire pipeline.
        
        Args:
            data: Raw bytes to compress
            
        Returns:
            PipelineResult with comprehensive results
        """
        if not self.stages:
            raise PipelineError("Pipeline has no stages")
        
        if not data:
            return PipelineResult(
                compressed_data=b'',
                original_size=0,
                compressed_size=0,
                total_compression_ratio=1.0,
                total_compression_time=0.0,
                total_decompression_time=0.0,
                stage_results=[],
                pipeline_name=self.name,
                metadata={}
            )
        
        start_time = time.perf_counter()
        
        try:
            current_data = data
            stage_results = []
            stage_metadata = []
            total_compression_time = 0.0
            
            # Process through each enabled stage
            for i, stage in enumerate(self.stages):
                if not stage.enabled:
                    continue
                
                stage_start = time.perf_counter()
                
                try:
                    if isinstance(stage.component, CompressionAlgorithm):
                        # Use compression algorithm
                        result = stage.component.compress(current_data)
                        current_data = result.compressed_data
                        
                        stage_info = {
                            'stage_index': i,
                            'stage_name': stage.name,
                            'algorithm': stage.component.name,
                            'input_size': result.original_size,
                            'output_size': result.compressed_size,
                            'compression_ratio': result.compression_ratio,
                            'compression_time': result.compression_time,
                            'metadata': result.metadata
                        }
                        
                        stage_metadata.append(result.metadata)
                        total_compression_time += result.compression_time
                        
                    elif isinstance(stage.component, PipelineComponent):
                        # Use pipeline component
                        prev_metadata = stage_metadata[-1] if stage_metadata else {}
                        processed_data, new_metadata = stage.component.process(current_data, prev_metadata)
                        current_data = processed_data
                        
                        stage_time = time.perf_counter() - stage_start
                        
                        stage_info = {
                            'stage_index': i,
                            'stage_name': stage.name,
                            'component': stage.component.__class__.__name__,
                            'input_size': len(current_data),
                            'output_size': len(processed_data),
                            'processing_time': stage_time,
                            'metadata': new_metadata
                        }
                        
                        stage_metadata.append(new_metadata)
                        total_compression_time += stage_time
                    
                    else:
                        raise PipelineError(f"Invalid component type in stage {i}: {type(stage.component)}")
                    
                    stage_results.append(stage_info)
                    
                except Exception as e:
                    raise PipelineError(f"Error in stage {i} ({stage.name}): {str(e)}")
            
            end_time = time.perf_counter()
            actual_total_time = end_time - start_time
            
            # Calculate overall metrics
            total_ratio = len(data) / len(current_data) if current_data else float('inf')
            
            return PipelineResult(
                compressed_data=current_data,
                original_size=len(data),
                compressed_size=len(current_data),
                total_compression_ratio=total_ratio,
                total_compression_time=total_compression_time,
                total_decompression_time=0.0,  # Will be filled during decompression
                stage_results=stage_results,
                pipeline_name=self.name,
                metadata={
                    'pipeline_stages': len([s for s in self.stages if s.enabled]),
                    'stage_metadata': stage_metadata,
                    'actual_total_time': actual_total_time,
                    'stage_names': [s.name for s in self.stages if s.enabled]
                }
            )
            
        except Exception as e:
            raise PipelineError(f"Pipeline compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, pipeline_result: PipelineResult) -> bytes:
        """
        Decompress data by reversing the pipeline.
        
        Args:
            compressed_data: Compressed data
            pipeline_result: Result from compression operation
            
        Returns:
            Original decompressed data
        """
        if not compressed_data:
            return b''
        
        if not pipeline_result.stage_results:
            raise PipelineError("No stage results available for decompression")
        
        start_time = time.perf_counter()
        
        try:
            current_data = compressed_data
            stage_metadata = pipeline_result.metadata.get('stage_metadata', [])
            
            # Process stages in reverse order
            enabled_stages = [(i, stage) for i, stage in enumerate(self.stages) if stage.enabled]
            
            for stage_idx, (original_idx, stage) in enumerate(reversed(enabled_stages)):
                metadata_idx = len(enabled_stages) - 1 - stage_idx
                metadata = stage_metadata[metadata_idx] if metadata_idx < len(stage_metadata) else {}
                
                try:
                    if isinstance(stage.component, CompressionAlgorithm):
                        # Decompress using algorithm
                        result = stage.component.decompress(current_data, metadata)
                        current_data = result.decompressed_data
                        
                    elif isinstance(stage.component, PipelineComponent):
                        # Reverse process using component
                        current_data = stage.component.reverse_process(current_data, metadata)
                    
                except Exception as e:
                    raise PipelineError(f"Error decompressing stage {original_idx} ({stage.name}): {str(e)}")
            
            # Update pipeline result with decompression time
            pipeline_result.total_decompression_time = time.perf_counter() - start_time
            
            return current_data
            
        except Exception as e:
            raise PipelineError(f"Pipeline decompression failed: {str(e)}")
    
    def get_stage_info(self) -> List[Dict[str, Any]]:
        """Get information about all pipeline stages."""
        return [
            {
                'index': i,
                'name': stage.name,
                'component_type': type(stage.component).__name__,
                'enabled': stage.enabled,
                'parameters': stage.parameters
            }
            for i, stage in enumerate(self.stages)
        ]
    
    def validate_pipeline(self) -> List[str]:
        """
        Validate the pipeline configuration.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        if not self.stages:
            issues.append("Pipeline has no stages")
        
        enabled_stages = [s for s in self.stages if s.enabled]
        if not enabled_stages:
            issues.append("Pipeline has no enabled stages")
        
        # Check for potential issues in stage ordering
        for i, stage in enumerate(enabled_stages[:-1]):
            next_stage = enabled_stages[i + 1]
            
            # Warn about potentially inefficient orderings
            if (hasattr(stage.component, 'category') and hasattr(next_stage.component, 'category')):
                from ..core.base import AlgorithmCategory
                
                # Dictionary algorithms work better before entropy coding
                if (stage.component.category == AlgorithmCategory.ENTROPY_CODING and
                    next_stage.component.category == AlgorithmCategory.DICTIONARY):
                    issues.append(f"Consider moving dictionary algorithm (stage {i+1}) before entropy coding (stage {i})")
        
        return issues
    
    def clone(self) -> 'CompressionPipeline':
        """Create a copy of this pipeline."""
        new_pipeline = CompressionPipeline(f"{self.name}_copy")
        
        for stage in self.stages:
            # Create new component instance if possible
            try:
                if hasattr(stage.component, '__class__'):
                    # Try to create new instance with same parameters
                    component_class = stage.component.__class__
                    if hasattr(stage.component, 'get_parameters'):
                        params = stage.component.get_parameters()
                        new_component = component_class(**params)
                    else:
                        new_component = component_class()
                else:
                    new_component = stage.component
                
                new_pipeline.add_stage(new_component, stage.name, stage.parameters.copy())
                new_pipeline.stages[-1].enabled = stage.enabled
                
            except Exception:
                # If we can't clone, use the same instance (not ideal but workable)
                new_pipeline.add_stage(stage.component, stage.name, stage.parameters.copy())
                new_pipeline.stages[-1].enabled = stage.enabled
        
        return new_pipeline
    
    def __repr__(self) -> str:
        enabled_stages = [s.name for s in self.stages if s.enabled]
        return f"CompressionPipeline('{self.name}', stages: {' → '.join(enabled_stages)})"


class PipelineBuilder:
    """
    Helper class for building compression pipelines with fluent interface.
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.pipeline = CompressionPipeline(name)
    
    def add_rle(self, escape_byte: int = 0x00, min_run_length: int = 3) -> 'PipelineBuilder':
        """Add RLE compression stage."""
        from ..algorithms.rle import RunLengthEncoder
        rle = RunLengthEncoder(escape_byte, min_run_length)
        self.pipeline.add_stage(rle, "RLE")
        return self
    
    def add_huffman(self) -> 'PipelineBuilder':
        """Add Huffman compression stage."""
        from ..algorithms.huffman import HuffmanEncoder
        huffman = HuffmanEncoder()
        self.pipeline.add_stage(huffman, "Huffman")
        return self
    
    def add_lz77(self, window_size: int = 4096, lookahead_size: int = 18) -> 'PipelineBuilder':
        """Add LZ77 compression stage."""
        from ..algorithms.lz77 import LZ77Encoder
        lz77 = LZ77Encoder(window_size, lookahead_size)
        self.pipeline.add_stage(lz77, "LZ77")
        return self
    
    def add_lzw(self, max_code_bits: int = 12) -> 'PipelineBuilder':
        """Add LZW compression stage."""
        from ..algorithms.lzw import LZWEncoder
        lzw = LZWEncoder(max_code_bits)
        self.pipeline.add_stage(lzw, "LZW")
        return self
    
    def add_custom(self, component: Union[CompressionAlgorithm, PipelineComponent], 
                   name: str, parameters: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """Add custom compression stage."""
        self.pipeline.add_stage(component, name, parameters)
        return self
    
    def build(self) -> CompressionPipeline:
        """Build and return the pipeline."""
        return self.pipeline


# Predefined pipeline configurations
class PredefinedPipelines:
    """Collection of predefined compression pipelines."""
    
    @staticmethod
    def deflate_like() -> CompressionPipeline:
        """Create a Deflate-like pipeline (LZ77 + Huffman)."""
        return (PipelineBuilder("Deflate-like")
                .add_lz77(window_size=32768, lookahead_size=258)
                .add_huffman()
                .build())
    
    @staticmethod
    def text_optimized() -> CompressionPipeline:
        """Create a pipeline optimized for text data."""
        return (PipelineBuilder("Text-Optimized")
                .add_rle(min_run_length=2)
                .add_lzw(max_code_bits=14)
                .add_huffman()
                .build())
    
    @staticmethod
    def binary_optimized() -> CompressionPipeline:
        """Create a pipeline optimized for binary data."""
        return (PipelineBuilder("Binary-Optimized")
                .add_lz77(window_size=8192, lookahead_size=32)
                .add_huffman()
                .build())
    
    @staticmethod
    def high_compression() -> CompressionPipeline:
        """Create a pipeline focused on maximum compression."""
        return (PipelineBuilder("High-Compression")
                .add_lz77(window_size=32768, lookahead_size=258)
                .add_lzw(max_code_bits=16)
                .add_huffman()
                .build())
    
    @staticmethod
    def fast_compression() -> CompressionPipeline:
        """Create a pipeline focused on speed."""
        return (PipelineBuilder("Fast-Compression")
                .add_rle()
                .add_huffman()
                .build()) 