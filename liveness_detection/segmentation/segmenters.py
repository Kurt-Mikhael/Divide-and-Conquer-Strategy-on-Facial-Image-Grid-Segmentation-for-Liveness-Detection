
import cv2
import numpy as np
from typing import List
from liveness_detection.interfaces import ISegmentationStrategy, IVarianceCalculator
from liveness_detection.models import GridResult
from liveness_detection.strategies.variance_calculators import LaplacianVarianceCalculator


class DivideAndConquerSegmenter(ISegmentationStrategy):

    
    def __init__(self, 
                 min_grid_size: int = 64,
                 variance_calculator: IVarianceCalculator = None,
                 skip_background: bool = True,
                 background_threshold: float = 0.15):

        self.min_grid_size = min_grid_size
        self.variance_calculator = variance_calculator or LaplacianVarianceCalculator()
        self.skip_background = skip_background
        self.background_threshold = background_threshold
    
    def segment(self, image: np.ndarray) -> List[GridResult]:

        return self._divide_and_conquer(image, position=(0, 0))
    
    def _divide_and_conquer(self, image: np.ndarray, position: tuple) -> List[GridResult]:

        h, w = image.shape[:2]
        
        if h <= self.min_grid_size or w <= self.min_grid_size:
            variance = self.variance_calculator.calculate(image)
            
            if self.skip_background and self._is_background(image):
                return []
            
            return [GridResult(variance=variance,position=position,size=(w, h),is_face_region=True,region=image.copy() if h > 32 else None
            )]
        
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [(image[0:mid_h, 0:mid_w], (position[0], position[1])),(image[0:mid_h, mid_w:w], (position[0] + mid_w, position[1])),(image[mid_h:h, 0:mid_w], (position[0], position[1] + mid_h)),(image[mid_h:h, mid_w:w], (position[0] + mid_w, position[1] + mid_h))
        ]
        
        results = []
        for quadrant, quad_pos in quadrants:
            if quadrant.size > 0:  
                results.extend(self._divide_and_conquer(quadrant, quad_pos))
        
        return results
    
    def _is_background(self, image: np.ndarray) -> bool:

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        std = np.std(gray)
        mean = np.mean(gray)
        
        is_uniform = std < 10
        is_extreme = mean < 30 or mean > 225
        
        return is_uniform and is_extreme
    
    def get_name(self) -> str:
        return f"DivideAndConquer(min_grid={self.min_grid_size})"


class NaiveFullProcessor(ISegmentationStrategy):

    
    def __init__(self, variance_calculator: IVarianceCalculator = None):
        self.variance_calculator = variance_calculator or LaplacianVarianceCalculator()
    
    def segment(self, image: np.ndarray) -> List[GridResult]:

        h, w = image.shape[:2]
        variance = self.variance_calculator.calculate(image)
        
        return [GridResult(
            variance=variance,
            position=(0, 0),
            size=(w, h),
            is_face_region=True,
            region=image.copy()
        )]
    
    def get_name(self) -> str:
        return "NaiveFullProcess"


class SlidingWindowSegmenter(ISegmentationStrategy):

    
    def __init__(self,
                 window_size: int = 64,
                 stride: int = 32,
                 variance_calculator: IVarianceCalculator = None):
        self.window_size = window_size
        self.stride = stride
        self.variance_calculator = variance_calculator or LaplacianVarianceCalculator()
    
    def segment(self, image: np.ndarray) -> List[GridResult]:

        h, w = image.shape[:2]
        results = []
        
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                window = image[y:y + self.window_size, x:x + self.window_size]
                variance = self.variance_calculator.calculate(window)
                
                results.append(GridResult(
                    variance=variance,
                    position=(x, y),
                    size=(self.window_size, self.window_size),
                    is_face_region=True,
                    region=None
                ))
        
        return results
    
    def get_name(self) -> str:
        return f"SlidingWindow(size={self.window_size},stride={self.stride})"
