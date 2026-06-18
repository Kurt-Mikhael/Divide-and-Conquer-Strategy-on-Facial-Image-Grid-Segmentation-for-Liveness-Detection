"""
Abstract interfaces for the liveness detection system.
Follows Interface Segregation Principle (ISP) - small, focused interfaces.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from liveness_detection.models import GridResult, ProcessingResult


class IVarianceCalculator(ABC):
    """Interface for variance/texture calculation strategies.
    
    Allows different algorithms to be plugged in without changing
    the segmentation logic (Open/Closed Principle).
    """
    
    @abstractmethod
    def calculate(self, image_region: np.ndarray) -> float:
        """Calculate variance or texture metric for an image region.
        
        Args:
            image_region: BGR image region (numpy array)
            
        Returns:
            float: Calculated variance/texture score
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the calculator strategy."""
        pass


class ISegmentationStrategy(ABC):
    """Interface for image segmentation strategies.
    
    Decouples the segmentation algorithm from the detection logic.
    """
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> List[GridResult]:
        """Segment image into regions and calculate variance for each.
        
        Args:
            image: Input BGR image
            
        Returns:
            List[GridResult]: List of grid analysis results
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the segmentation strategy."""
        pass


class ILivenessDetector(ABC):
    """Interface for liveness detection algorithms.
    
    Allows different detection algorithms to be used interchangeably
    (Liskov Substitution Principle).
    """
    
    @abstractmethod
    def detect(self, grid_results: List[GridResult]) -> ProcessingResult:
        """Analyze grid results and determine liveness.
        
        Args:
            grid_results: List of grid variance results
            
        Returns:
            ProcessingResult: Detection result with confidence score
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the detector."""
        pass


class IImageProcessor(ABC):
    """Interface for image pre-processing operations.
    
    Segregates pre-processing concerns from segmentation.
    """
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply pre-processing to image.
        
        Args:
            image: Raw input image
            
        Returns:
            np.ndarray: Pre-processed image
        """
        pass


class IBenchmarkLogger(ABC):
    """Interface for benchmark logging.
    
    Allows different logging implementations (console, file, CSV, etc.)
    """
    
    @abstractmethod
    def log_execution(self, image_path: str, execution_time: float, 
                     grid_count: int, result: str) -> None:
        """Log execution metrics.
        
        Args:
            image_path: Path to processed image
            execution_time: Execution time in seconds
            grid_count: Number of grids processed
            result: Detection result string
        """
        pass
    
    @abstractmethod
    def log_comparison(self, method_name: str, total_time: float, 
                      avg_time: float, accuracy: float) -> None:
        """Log comparison between methods.
        
        Args:
            method_name: Name of the method
            total_time: Total execution time
            avg_time: Average execution time per image
            accuracy: Detection accuracy
        """
        pass
    
    @abstractmethod
    def save_results(self, output_path: str) -> None:
        """Save logged results to file.
        
        Args:
            output_path: Path to save results
        """
        pass
