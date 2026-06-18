
import os
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from liveness_detection.interfaces import (
    ISegmentationStrategy, 
    ILivenessDetector, 
    IBenchmarkLogger
)
from liveness_detection.models import ProcessingResult, BenchmarkResult
from liveness_detection.utils.visualizer import ConsoleLogger
from liveness_detection.segmentation.segmenters import NaiveFullProcessor


class DatasetProcessor:

    
    def __init__(self,
                 segmenter: ISegmentationStrategy,
                 detector: ILivenessDetector,
                 logger: IBenchmarkLogger = None):

        self.segmenter = segmenter
        self.detector = detector
        self.logger = logger or ConsoleLogger()
    
    def process_image(self, image_path: str) -> Tuple[ProcessingResult, np.ndarray]:

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        start_time = time.time()
        grid_results = self.segmenter.segment(img)
        
        result = self.detector.detect(grid_results)
        result.execution_time = time.time() - start_time
        
        self.logger.log_execution(image_path, result.execution_time,len(grid_results),"LIVE" if result.is_live else "SPOOF")
        
        return result, img
    
    def process_dataset(self, dataset_path: str, label: str = "unknown") -> List[ProcessingResult]:

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in dataset_path.iterdir()if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"Warning: No images found in {dataset_path}")
            return []
        
        print(f"Processing {len(image_files)} images from {dataset_path}...")
        
        results = []
        for i, img_path in enumerate(image_files, 1):
            try:
                result, _ = self.process_image(str(img_path))
                result.details["ground_truth"] = label
                result.details["file_name"] = img_path.name
                results.append(result)
                
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(image_files)} processed")
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
        
        return results


class BenchmarkRunner:
    def __init__(self, logger: IBenchmarkLogger = None):
        self.logger = logger or ConsoleLogger()
    
    def compare_methods(self,
                       real_dataset: str,
                       spoof_dataset: str,
                       methods: List[Tuple[str, ISegmentationStrategy, ILivenessDetector]]) -> Dict[str, BenchmarkResult]:
        results = {}
        
        for name, segmenter, detector in methods:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}")
            print(f"{'='*60}")
            
            benchmark = self._benchmark_method(
                real_dataset, spoof_dataset, segmenter, detector
            )
            results[name] = benchmark
            
            self.logger.log_comparison(
                name,
                benchmark.total_time,
                benchmark.avg_time,
                benchmark.accuracy
            )
        
        return results
    
    def _benchmark_method(self,
                         real_dataset: str,
                         spoof_dataset: str,
                         segmenter: ISegmentationStrategy,
                         detector: ILivenessDetector) -> BenchmarkResult:
        processor = DatasetProcessor(segmenter, detector)
        
        real_results = processor.process_dataset(real_dataset, "real")
        
        spoof_results = processor.process_dataset(spoof_dataset, "spoof")
        
        all_results = real_results + spoof_results
        total_time = sum(r.execution_time for r in all_results)
        total_grids = sum(r.grid_count for r in all_results)
        
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for r in real_results:
            if r.is_live:
                tp += 1
            else:
                fn += 1
        
        for r in spoof_results:
            if not r.is_live:
                tn += 1
            else:
                fp += 1
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        execution_times = [r.execution_time for r in all_results]
        
        return BenchmarkResult(
            method_name=segmenter.get_name(),
            total_time=total_time,
            avg_time=total_time / len(all_results) if all_results else 0,
            min_time=min(execution_times) if execution_times else 0,
            max_time=max(execution_times) if execution_times else 0,
            total_images=len(all_results),
            total_grids=total_grids,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn
        )
    
    def generate_report(self, results: Dict[str, BenchmarkResult], output_path: str = None):
        lines = []
        lines.append("="*80)
        lines.append("LIVENESS DETECTION BENCHMARK REPORT")
        lines.append("="*80)
        lines.append("")
        
        for name, result in results.items():
            lines.append(f"\nMethod: {name}")
            lines.append("-"*80)
            lines.append(f"  Total Images:     {result.total_images}")
            lines.append(f"  Total Grids:      {result.total_grids}")
            lines.append(f"  Total Time:       {result.total_time:.4f}s")
            lines.append(f"  Avg Time/Image:   {result.avg_time:.4f}s")
            lines.append(f"  Min Time:         {result.min_time:.4f}s")
            lines.append(f"  Max Time:         {result.max_time:.4f}s")
            lines.append(f"  Accuracy:         {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
            lines.append(f"  Precision:        {result.precision:.4f}")
            lines.append(f"  Recall:           {result.recall:.4f}")
            lines.append(f"  F1 Score:         {result.f1_score:.4f}")
            lines.append(f"  True Positives:   {result.true_positives}")
            lines.append(f"  True Negatives:   {result.true_negatives}")
            lines.append(f"  False Positives:  {result.false_positives}")
            lines.append(f"  False Negatives:  {result.false_negatives}")
        
        lines.append("\n" + "="*80)
        
        report = "\n".join(lines)
        print(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")


class SingleImageAnalyzer:

    def __init__(self,
                 segmenter: ISegmentationStrategy,
                 detector: ILivenessDetector):
        self.segmenter = segmenter
        self.detector = detector
    
    def analyze(self, image_path: str) -> Dict:

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        start_time = time.time()
        grid_results = self.segmenter.segment(img)
        segment_time = time.time() - start_time
        
        start_time = time.time()
        result = self.detector.detect(grid_results)
        detect_time = time.time() - start_time
        
        grid_map = np.zeros(img.shape[:2], dtype=np.float32)
        for grid in grid_results:
            x, y = grid.position
            w, h = grid.size
            grid_map[y:y+h, x:x+w] = grid.variance
        
        return {
            "result": result,
            "grid_results": grid_results,
            "grid_map": grid_map,
            "segment_time": segment_time,
            "detect_time": detect_time,
            "total_time": segment_time + detect_time,
            "image_shape": img.shape
        }
