"""
Script to measure and visualize gaze estimation accuracy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_accuracy(csv_file="results/GazeTracking.csv"):
    """Analyze gaze tracking accuracy"""
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Calculate error between predicted and actual position
    if 'set_x' in df.columns and 'Sgaze_x' in df.columns:
        # Error in pixels
        error_x = df['Sgaze_x'] - df['set_x']
        error_y = df['Sgaze_y'] - df['set_y']
        
        # Euclidean distance error
        euclidean_error = np.sqrt(error_x**2 + error_y**2)
        
        # Statistics
        print("=" * 50)
        print("GAZE ESTIMATION ACCURACY REPORT")
        print("=" * 50)
        print(f"\nMean error X: {error_x.mean():.2f} pixels")
        print(f"Mean error Y: {error_y.mean():.2f} pixels")
        print(f"\nStd dev X: {error_x.std():.2f} pixels")
        print(f"Std dev Y: {error_y.std():.2f} pixels")
        print(f"\nMean Euclidean error: {euclidean_error.mean():.2f} pixels")
        print(f"Median Euclidean error: {euclidean_error.median():.2f} pixels")
        print(f"Max error: {euclidean_error.max():.2f} pixels")
        print(f"Min error: {euclidean_error.min():.2f} pixels")
        
        # Accuracy percentage (within threshold)
        thresholds = [50, 100, 150, 200]
        print("\n" + "=" * 50)
        print("ACCURACY BY THRESHOLD")
        print("=" * 50)
        for threshold in thresholds:
            accuracy = (euclidean_error < threshold).sum() / len(euclidean_error) * 100
            print(f"Within {threshold} pixels: {accuracy:.1f}%")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Error distribution
        axes[0, 0].hist(euclidean_error, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(euclidean_error.mean(), color='red', linestyle='--', 
                          label=f'Mean: {euclidean_error.mean():.2f}px')
        axes[0, 0].axvline(euclidean_error.median(), color='green', linestyle='--', 
                          label=f'Median: {euclidean_error.median():.2f}px')
        axes[0, 0].set_xlabel('Error (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error over time
        axes[0, 1].plot(euclidean_error, alpha=0.6, linewidth=0.5)
        axes[0, 1].plot(euclidean_error.rolling(window=50).mean(), 
                       color='red', linewidth=2, label='Moving average (50)')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Error (pixels)')
        axes[0, 1].set_title('Error Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. X vs Y error scatter
        axes[1, 0].scatter(error_x, error_y, alpha=0.5, s=10)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('X Error (pixels)')
        axes[1, 0].set_ylabel('Y Error (pixels)')
        axes[1, 0].set_title('X vs Y Error')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # 4. Gaze heatmap (predicted vs actual)
        axes[1, 1].scatter(df['set_x'], df['set_y'], 
                          c='red', marker='x', s=100, label='Target', alpha=0.7)
        axes[1, 1].scatter(df['Sgaze_x'], df['Sgaze_y'], 
                          c='blue', marker='.', s=20, label='Predicted', alpha=0.5)
        axes[1, 1].set_xlabel('X Position (pixels)')
        axes[1, 1].set_ylabel('Y Position (pixels)')
        axes[1, 1].set_title('Predicted vs Target Positions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()  # Screen coordinates
        
        plt.tight_layout()
        output_file = os.path.join(os.path.dirname(csv_file), "accuracy_analysis.png")
        plt.savefig(output_file, dpi=150)
        print(f"\nVisualization saved to: {output_file}")
        plt.show()
        
    else:
        print("CSV file doesn't have required columns for accuracy analysis")
        print(f"Available columns: {df.columns.tolist()}")

def compare_calibrations():
    """Compare accuracy with different calibration settings"""
    print("\n" + "=" * 50)
    print("CALIBRATION COMPARISON")
    print("=" * 50)
    print("\nTo compare different calibration settings:")
    print("1. Run calibration with 4 points (current)")
    print("2. Run calibration with 9 points (recommended)")
    print("3. Compare the accuracy results")

if __name__ == "__main__":
    print("Gaze Estimation Accuracy Analyzer")
    print("=" * 50)
    
    # Analyze latest tracking data
    analyze_accuracy("results/GazeTracking.csv")
    
    # Show comparison guide
    compare_calibrations()
