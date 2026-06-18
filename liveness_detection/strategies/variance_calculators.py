
import cv2
import numpy as np
from liveness_detection.interfaces import IVarianceCalculator


class LaplacianVarianceCalculator(IVarianceCalculator):

    
    def __init__(self, ksize: int = 3):
        self.ksize = ksize
    
    def calculate(self, image_region: np.ndarray) -> float:
        """Calculate Laplacian variance."""
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        return cv2.Laplacian(gray, cv2.CV_64F, ksize=self.ksize).var()
    
    def get_name(self) -> str:
        return f"Laplacian(k={self.ksize})"


class SobelVarianceCalculator(IVarianceCalculator):

    
    def __init__(self, ksize: int = 3):
        self.ksize = ksize
    
    def calculate(self, image_region: np.ndarray) -> float:
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return float(np.var(gradient_magnitude))
    
    def get_name(self) -> str:
        return f"Sobel(k={self.ksize})"


class CombinedVarianceCalculator(IVarianceCalculator):

    
    def __init__(self, calculators: list = None, weights: list = None):
        self.calculators = calculators or [
            LaplacianVarianceCalculator(),
            SobelVarianceCalculator()
        ]
        self.weights = weights or [0.6, 0.4]
        
        if len(self.calculators) != len(self.weights):
            raise ValueError("Number of calculators must match number of weights")
    
    def calculate(self, image_region: np.ndarray) -> float:
        scores = [calc.calculate(image_region) for calc in self.calculators]
        return sum(s * w for s, w in zip(scores, self.weights))
    
    def get_name(self) -> str:
        names = [calc.get_name() for calc in self.calculators]
        return f"Combined({', '.join(names)})"


class LocalBinaryPatternVariance(IVarianceCalculator):

    
    def __init__(self, radius: int = 1, n_points: int = 8):
        self.radius = radius
        self.n_points = n_points
    
    def calculate(self, image_region: np.ndarray) -> float:
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        # Simple LBP implementation
        height, width = gray.shape
        lbp = np.zeros((height - 2 * self.radius, width - 2 * self.radius), dtype=np.uint8)
        
        for i in range(self.n_points):
            angle = 2 * np.pi * i / self.n_points
            x = int(self.radius * np.cos(angle))
            y = int(self.radius * np.sin(angle))
            
            shifted = gray[self.radius + y:height - self.radius + y,
                          self.radius + x:width - self.radius + x]
            center = gray[self.radius:height - self.radius,
                       self.radius:width - self.radius]
            
            lbp |= ((shifted >= center) << i).astype(np.uint8)
        
        # Calculate histogram variance
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        return float(np.var(hist))
    
    def get_name(self) -> str:
        return f"LBP(r={self.radius},p={self.n_points})"
