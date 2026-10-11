import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'WaferDefectX/python')

from preprocessing import Preprocessor
from defect_localization import DefectLocalizer
from features import FeatureExtractor

def create_linkedin_image1():
    """Create before/after defect detection image"""
    # Load a scratch image for more visual impact
    img_path = "WaferDefectX/data/synthetic/wafer_0033_scratch.png"
    image = cv2.imread(img_path)
    
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    
    # Process
    enhanced, _ = preprocessor.process_pipeline(image)
    loc_results = localizer.localize(enhanced)
    
    # Create detection output
    output_img = image.copy()
    for (x, y, w, h) in loc_results['bboxes']:
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(output_img, "DEFECT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Wafer Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Defect Localization Output", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle("WaferDefectX: Real-time Defect Detection", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("WaferDefectX/results/linkedin_image1_detection.png", dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_image1_detection.png")

def create_linkedin_image2():
    """Create feature extraction / classification pipeline visualization"""
    img_path = "WaferDefectX/data/synthetic/wafer_0019_particle.png"
    image = cv2.imread(img_path)
    
    preprocessor = Preprocessor()
    localizer = DefectLocalizer()
    extractor = FeatureExtractor()
    
    # Process
    enhanced, thresh = preprocessor.process_pipeline(image)
    loc_results = localizer.localize(enhanced)
    
    # Create multi-stage visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Pipeline stages
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Input Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title("2. Preprocessing (CLAHE)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(loc_results['edges'], cmap='gray')
    axes[0, 2].set_title("3. Edge Detection (Canny)", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Localization and Features
    axes[1, 0].imshow(loc_results['mask'], cmap='gray')
    axes[1, 0].set_title("4. Defect Mask", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Feature visualization
    if loc_results['contours']:
        cnt = max(loc_results['contours'], key=cv2.contourArea)
        feats = extractor.extract_features(enhanced, cnt)
        feature_names = ['Area', 'Perimeter', 'Aspect\nRatio', 'Rect.', 'Circular.', 'Mean\nIntensity', 'Std Dev']
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
        bars = axes[1, 1].bar(feature_names, feats / (np.max(feats) + 1e-6), color=colors)
        axes[1, 1].set_title("5. Feature Extraction", fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Classification result
    output_img = image.copy()
    for (x, y, w, h) in loc_results['bboxes']:
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(output_img, "Class: PARTICLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    axes[1, 2].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Classification Output", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle("WaferDefectX: Feature-based Classification Pipeline", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("WaferDefectX/results/linkedin_image2_pipeline.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_image2_pipeline.png")

def create_linkedin_image3():
    """Create performance benchmark chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency comparison
    implementations = ['Python\n(OpenCV)\nunverified', 'C++\n(OpenCV)\nESTIMATE, not measured']
    latencies = [8.6, 2.1]  # ESTIMATES: neither value is backed by a committed benchmark run
    colors = ['#3498db', '#e74c3c']
    
    bars1 = axes[0].bar(implementations, latencies, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Per-Image Latency (illustrative estimates, not measured)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 12)
    
    # Add value labels
    for bar, val in zip(bars1, latencies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val} ms', ha='center', fontsize=12, fontweight='bold')
    
    # Speedup annotation intentionally omitted
    # (speedup annotation removed: the C++ figure is an estimate, not a measurement)
    
    # Throughput derived from the estimated latencies above
    fps = [116, 476]  # 1000/latency (estimates)
    bars2 = axes[1].bar(implementations, fps, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Throughput (FPS)', fontsize=12, fontweight='bold')
    axes[1].set_title('Implied Throughput (from estimates)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 550)
    
    for bar, val in zip(bars2, fps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{val} FPS', ha='center', fontsize=12, fontweight='bold')
    
    # Add target line
    axes[1].axhline(y=60, color='green', linestyle='--', linewidth=2, label='Real-time Target (60 FPS)')
    axes[1].legend(loc='upper right')
    
    plt.suptitle("WaferDefectX: Python vs C++ latency (ESTIMATES, not a benchmark)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("WaferDefectX/results/linkedin_image3_benchmark.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: linkedin_image3_benchmark.png")

if __name__ == "__main__":
    create_linkedin_image1()
    create_linkedin_image2()
    create_linkedin_image3()
    print("\nAll LinkedIn images created in WaferDefectX/results/")
