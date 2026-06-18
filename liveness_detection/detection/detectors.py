
import numpy as np
from typing import List
from liveness_detection.interfaces import ILivenessDetector
from liveness_detection.models import GridResult, ProcessingResult


class ThresholdLivenessDetector(ILivenessDetector):

    
    def __init__(self, 
                 variance_threshold: float = 100.0,
                 anomaly_threshold: float = 3.0):

        self.variance_threshold = variance_threshold
        self.anomaly_threshold = anomaly_threshold
    
    def detect(self, grid_results: List[GridResult]) -> ProcessingResult:

        if not grid_results:
            return ProcessingResult(
                is_live=False,
                confidence=0.0,
                max_variance=0.0,
                min_variance=0.0,
                avg_variance=0.0,
                anomaly_score=0.0,
                grid_count=0,
                execution_time=0.0,
                details={"error": "No grid results available"}
            )
        
        variances = [g.variance for g in grid_results]
        
        max_var = np.max(variances)
        min_var = np.min(variances)
        avg_var = np.mean(variances)
        
        anomaly_score = max_var / (avg_var + 1e-7) if avg_var > 0 else 0
        
        is_live = (avg_var >= self.variance_threshold and 
                  anomaly_score <= self.anomaly_threshold)
        
        if is_live:
            confidence = min(1.0, avg_var / (self.variance_threshold * 2))
        else:
            confidence = min(1.0, self.anomaly_threshold / (anomaly_score + 1e-7))
        
        details = {
            "threshold": self.variance_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "variance_std": float(np.std(variances))
        }
        
        return ProcessingResult(
            is_live=is_live,
            confidence=confidence,
            max_variance=max_var,
            min_variance=min_var,
            avg_variance=avg_var,
            anomaly_score=anomaly_score,
            grid_count=len(grid_results),
            execution_time=0.0,  
            details=details
        )
    
    def get_name(self) -> str:
        return f"Threshold(thresh={self.variance_threshold})"


class AnomalyLivenessDetector(ILivenessDetector):

    
    def __init__(self,
                 variance_ratio_threshold: float = 3.0,
                 uniformity_threshold: float = 0.3,
                 min_grids: int = 4):

        self.variance_ratio_threshold = variance_ratio_threshold
        self.uniformity_threshold = uniformity_threshold
        self.min_grids = min_grids
    
    def detect(self, grid_results: List[GridResult]) -> ProcessingResult:

        if len(grid_results) < self.min_grids:
            return ProcessingResult(
                is_live=False,
                confidence=0.5,
                max_variance=0.0,
                min_variance=0.0,
                avg_variance=0.0,
                anomaly_score=0.0,
                grid_count=len(grid_results),
                execution_time=0.0,
                details={"error": f"Insufficient grids ({len(grid_results)} < {self.min_grids})"}
            )
        
        variances = np.array([g.variance for g in grid_results])
        
        max_var = float(np.max(variances))
        min_var = float(np.min(variances))
        avg_var = float(np.mean(variances))
        std_var = float(np.std(variances))
        
        variance_ratio = max_var / (avg_var + 1e-7)
        coefficient_of_variation = std_var / (avg_var + 1e-7)
        

        
        is_spoof_by_ratio = variance_ratio > self.variance_ratio_threshold
        is_spoof_by_uniformity = coefficient_of_variation < self.uniformity_threshold
        
        spoof_score = 0.0
        
        if is_spoof_by_ratio:
            spoof_score += 0.6
        if is_spoof_by_uniformity:
            spoof_score += 0.4
        
        is_live = spoof_score < 0.5
        confidence = 1.0 - spoof_score if is_live else spoof_score
        
        anomaly_score = variance_ratio
        
        details = {
            "variance_ratio": variance_ratio,
            "coefficient_of_variation": coefficient_of_variation,
            "is_spoof_by_ratio": is_spoof_by_ratio,
            "is_spoof_by_uniformity": is_spoof_by_uniformity,
            "spoof_score": spoof_score
        }
        
        return ProcessingResult(
            is_live=is_live,
            confidence=confidence,
            max_variance=max_var,
            min_variance=min_var,
            avg_variance=avg_var,
            anomaly_score=anomaly_score,
            grid_count=len(grid_results),
            execution_time=0.0,
            details=details
        )
    
    def get_name(self) -> str:
        return f"Anomaly(ratio={self.variance_ratio_threshold})"


class VarianceBasedDetector(ILivenessDetector):
    """Detector based on average variance with configurable thresholds.
    
    Uses average variance across all grids to classify live vs spoof.
    More robust than simple threshold for real-world datasets.
    """
    
    def __init__(self,
                 min_variance: float = 100.0,
                 max_variance_ratio: float = 15.0):
        """Initialize detector.
        
        Args:
            min_variance: Minimum average variance for live detection
            max_variance_ratio: Maximum allowed max/avg variance ratio
        """
        self.min_variance = min_variance
        self.max_variance_ratio = max_variance_ratio
    
    def detect(self, grid_results: List[GridResult]) -> ProcessingResult:
        """Detect liveness based on average variance.
        
        Args:
            grid_results: Grid variance results
            
        Returns:
            ProcessingResult: Detection result
        """
        if not grid_results:
            return ProcessingResult(
                is_live=False,
                confidence=0.0,
                max_variance=0.0,
                min_variance=0.0,
                avg_variance=0.0,
                anomaly_score=0.0,
                grid_count=0,
                execution_time=0.0,
                details={"error": "No grid results available"}
            )
        
        variances = np.array([g.variance for g in grid_results])
        
        max_var = float(np.max(variances))
        min_var = float(np.min(variances))
        avg_var = float(np.mean(variances))
        std_var = float(np.std(variances))
        
        variance_ratio = max_var / (avg_var + 1e-7)
        
        # Decision: Live if average variance is high enough and not too anomalous
        is_live = (avg_var >= self.min_variance and 
                  variance_ratio <= self.max_variance_ratio)
        
        # Confidence based on distance from threshold
        if is_live:
            confidence = min(1.0, avg_var / (self.min_variance * 2))
        else:
            confidence = min(1.0, self.min_variance / (avg_var + 1e-7))
        
        details = {
            "min_variance_threshold": self.min_variance,
            "max_ratio_threshold": self.max_variance_ratio,
            "variance_ratio": variance_ratio,
            "std_variance": std_var
        }
        
        return ProcessingResult(
            is_live=is_live,
            confidence=confidence,
            max_variance=max_var,
            min_variance=min_var,
            avg_variance=avg_var,
            anomaly_score=variance_ratio,
            grid_count=len(grid_results),
            execution_time=0.0,
            details=details
        )
    
    def get_name(self) -> str:
        return f"VarianceBased(min={self.min_variance})"


class EnsembleLivenessDetector(ILivenessDetector):

    
    def __init__(self, detectors: List[ILivenessDetector] = None):

        self.detectors = detectors or [
            VarianceBasedDetector(),
            ThresholdLivenessDetector()
        ]
    
    def detect(self, grid_results: List[GridResult]) -> ProcessingResult:

        results = [detector.detect(grid_results) for detector in self.detectors]
        
        live_score = 0.0
        total_weight = 0.0
        
        for r in results:
            weight = r.confidence
            if r.is_live:
                live_score += weight
            total_weight += weight
        
        is_live = live_score / (total_weight + 1e-7) > 0.5
        confidence = abs(live_score - (total_weight - live_score)) / (total_weight + 1e-7)
        
        max_var = np.max([r.max_variance for r in results])
        min_var = np.min([r.min_variance for r in results])
        avg_var = np.mean([r.avg_variance for r in results])
        anomaly_score = np.max([r.anomaly_score for r in results])
        
        details = {
            "detector_results": [
                {
                    "detector": d.get_name(),
                    "is_live": r.is_live,
                    "confidence": r.confidence
                }
                for d, r in zip(self.detectors, results)
            ],
            "ensemble_vote": live_score / (total_weight + 1e-7)
        }
        
        return ProcessingResult(
            is_live=is_live,
            confidence=confidence,
            max_variance=max_var,
            min_variance=min_var,
            avg_variance=avg_var,
            anomaly_score=anomaly_score,
            grid_count=len(grid_results),
            execution_time=0.0,
            details=details
        )
    
    def get_name(self) -> str:
        names = [d.get_name() for d in self.detectors]
        return f"Ensemble({', '.join(names)})"
