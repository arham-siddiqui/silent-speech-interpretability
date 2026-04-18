"""
Radar-Laser Synchronization Analysis

This script synchronizes radar amplitudes and laser vibration frequencies
for sentence data and provides visualization tools to check synchronization.

Author: Generated for SilentSpeech project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class RadarLaserSyncAnalyzer:
    """
    A class to analyze synchronization between radar amplitudes and laser vibration frequencies.
    """
    
    def __init__(self, data_root: str = "src/data/rvtall/processed_cut_data"):
        """
        Initialize the analyzer with data root path.
        
        Args:
            data_root: Root path to the processed data directory
        """
        self.data_root = Path(data_root)
        self.radar_path = self.data_root / "radar_processed"
        self.laser_path = self.data_root / "laser_processed"
        
    def load_radar_data(self, subject_id: int, sentence_id: int, sample_id: int) -> np.ndarray:
        """
        Load radar data for a specific subject, sentence, and sample.
        
        Args:
            subject_id: Subject identifier (1-20)
            sentence_id: Sentence identifier (1-10)
            sample_id: Sample identifier (1-7)
            
        Returns:
            Radar amplitude data as numpy array
        """
        radar_file = self.radar_path / str(subject_id) / f"sentences{sentence_id}" / f"{subject_id}_3_{sentence_id}_sample{sample_id}.npy"
        
        if not radar_file.exists():
            raise FileNotFoundError(f"Radar data file not found: {radar_file}")
            
        return np.load(radar_file)
    
    def load_laser_data(self, subject_id: int, sentence_id: int, sample_id: int) -> np.ndarray:
        """
        Load laser data for a specific subject, sentence, and sample.
        
        Args:
            subject_id: Subject identifier (1-20)
            sentence_id: Sentence identifier (1-10)
            sample_id: Sample identifier (4-7, as per data structure)
            
        Returns:
            Laser vibration frequency data as numpy array
        """
        laser_file = self.laser_path / str(subject_id) / f"sentences{sentence_id}" / f"Laser_data_person_sample{sample_id}.npy"
        
        if not laser_file.exists():
            raise FileNotFoundError(f"Laser data file not found: {laser_file}")
            
        return np.load(laser_file)
    
    def create_time_frames(self, radar_data: np.ndarray, laser_data: np.ndarray, 
                          radar_sampling_rate: float = 1000.0, 
                          laser_sampling_rate: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time frames for both radar and laser data.
        
        Args:
            radar_data: Radar amplitude data
            laser_data: Laser vibration frequency data
            radar_sampling_rate: Sampling rate for radar data (Hz)
            laser_sampling_rate: Sampling rate for laser data (Hz)
            
        Returns:
            Tuple of (radar_time, laser_time) arrays
        """
        # For radar data, we'll use the first dimension as time
        radar_time = np.arange(radar_data.shape[0]) / radar_sampling_rate
        
        # For laser data, it's 1D so we use its length
        laser_time = np.arange(len(laser_data)) / laser_sampling_rate
        
        return radar_time, laser_time
    
    def extract_radar_amplitudes(self, radar_data: np.ndarray) -> np.ndarray:
        """
        Extract radar amplitudes from the radar data.
        For now, we'll use the mean across all channels as the amplitude.
        
        Args:
            radar_data: Raw radar data
            
        Returns:
            Radar amplitude time series
        """
        # Take mean across all channels (second dimension)
        return np.mean(radar_data, axis=1)
    
    def plot_radar_amplitudes(self, radar_amplitudes: np.ndarray, time: np.ndarray, 
                             title: str = "Radar Amplitudes vs Time") -> None:
        """
        Plot radar amplitudes over time.
        
        Args:
            radar_amplitudes: Radar amplitude data
            time: Time array
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(time, radar_amplitudes, 'b-', linewidth=1, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Radar Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_laser_frequencies(self, laser_data: np.ndarray, time: np.ndarray,
                              title: str = "Laser Vibration Frequencies vs Time") -> None:
        """
        Plot laser vibration frequencies over time.
        
        Args:
            laser_data: Laser vibration frequency data
            time: Time array
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(time, laser_data, 'r-', linewidth=1, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Laser Vibration Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_synchronized_data(self, radar_amplitudes: np.ndarray, laser_data: np.ndarray,
                              radar_time: np.ndarray, laser_time: np.ndarray,
                              title: str = "Synchronized Radar and Laser Data") -> None:
        """
        Plot radar amplitudes and laser frequencies overlaid for synchronization check.
        
        Args:
            radar_amplitudes: Radar amplitude data
            laser_data: Laser vibration frequency data
            radar_time: Time array for radar data
            laser_time: Time array for laser data
            title: Plot title
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot radar amplitudes
        ax1.plot(radar_time, radar_amplitudes, 'b-', linewidth=1, alpha=0.8, label='Radar Amplitudes')
        ax1.set_ylabel('Radar Amplitude')
        ax1.set_title('Radar Amplitudes')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot laser frequencies
        ax2.plot(laser_time, laser_data, 'r-', linewidth=1, alpha=0.8, label='Laser Frequencies')
        ax2.set_ylabel('Laser Vibration Frequency')
        ax2.set_title('Laser Vibration Frequencies')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Overlay both signals (normalized)
        radar_norm = (radar_amplitudes - np.min(radar_amplitudes)) / (np.max(radar_amplitudes) - np.min(radar_amplitudes))
        laser_norm = (laser_data - np.min(laser_data)) / (np.max(laser_data) - np.min(laser_data))
        
        ax3.plot(radar_time, radar_norm, 'b-', linewidth=1, alpha=0.8, label='Radar (normalized)')
        ax3.plot(laser_time, laser_norm, 'r-', linewidth=1, alpha=0.8, label='Laser (normalized)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Normalized Amplitude')
        ax3.set_title('Overlaid Radar and Laser Data (Normalized)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def analyze_synchronization(self, subject_id: int, sentence_id: int, 
                              radar_sample_id: int = 1, laser_sample_id: int = 5) -> Dict:
        """
        Perform complete synchronization analysis for a given subject and sentence.
        
        Args:
            subject_id: Subject identifier
            sentence_id: Sentence identifier
            radar_sample_id: Radar sample identifier
            laser_sample_id: Laser sample identifier
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing synchronization for Subject {subject_id}, Sentence {sentence_id}")
        print(f"Radar Sample: {radar_sample_id}, Laser Sample: {laser_sample_id}")
        
        try:
            # Load data
            radar_data = self.load_radar_data(subject_id, sentence_id, radar_sample_id)
            laser_data = self.load_laser_data(subject_id, sentence_id, laser_sample_id)
            
            print(f"Radar data shape: {radar_data.shape}")
            print(f"Laser data shape: {laser_data.shape}")
            
            # Extract radar amplitudes
            radar_amplitudes = self.extract_radar_amplitudes(radar_data)
            
            # Create time frames
            radar_time, laser_time = self.create_time_frames(radar_data, laser_data)
            
            print(f"Radar time range: {radar_time[0]:.3f}s to {radar_time[-1]:.3f}s")
            print(f"Laser time range: {laser_time[0]:.3f}s to {laser_time[-1]:.3f}s")
            
            # Plot individual signals
            self.plot_radar_amplitudes(radar_amplitudes, radar_time, 
                                     f"Subject {subject_id}, Sentence {sentence_id} - Radar Amplitudes")
            
            self.plot_laser_frequencies(laser_data, laser_time,
                                      f"Subject {subject_id}, Sentence {sentence_id} - Laser Frequencies")
            
            # Plot synchronized data
            self.plot_synchronized_data(radar_amplitudes, laser_data, radar_time, laser_time,
                                      f"Subject {subject_id}, Sentence {sentence_id} - Synchronized Analysis")
            
            # Calculate some basic statistics
            radar_stats = {
                'mean': np.mean(radar_amplitudes),
                'std': np.std(radar_amplitudes),
                'min': np.min(radar_amplitudes),
                'max': np.max(radar_amplitudes)
            }
            
            laser_stats = {
                'mean': np.mean(laser_data),
                'std': np.std(laser_data),
                'min': np.min(laser_data),
                'max': np.max(laser_data)
            }
            
            print(f"\nRadar Statistics:")
            print(f"  Mean: {radar_stats['mean']:.2f}")
            print(f"  Std: {radar_stats['std']:.2f}")
            print(f"  Range: {radar_stats['min']:.2f} to {radar_stats['max']:.2f}")
            
            print(f"\nLaser Statistics:")
            print(f"  Mean: {laser_stats['mean']:.2f}")
            print(f"  Std: {laser_stats['std']:.2f}")
            print(f"  Range: {laser_stats['min']:.2f} to {laser_stats['max']:.2f}")
            
            return {
                'radar_data': radar_data,
                'laser_data': laser_data,
                'radar_amplitudes': radar_amplitudes,
                'radar_time': radar_time,
                'laser_time': laser_time,
                'radar_stats': radar_stats,
                'laser_stats': laser_stats
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None

def main():
    """
    Main function to demonstrate the radar-laser synchronization analysis.
    """
    # Initialize analyzer
    analyzer = RadarLaserSyncAnalyzer()
    
    # Analyze synchronization for Subject 1, Sentence 1
    print("=" * 60)
    print("RADAR-LASER SYNCHRONIZATION ANALYSIS")
    print("=" * 60)
    
    # You can modify these parameters to analyze different subjects/sentences
    subject_id = 1
    sentence_id = 5
    radar_sample_id = 1
    laser_sample_id = 5  # Based on available data structure
    
    results = analyzer.analyze_synchronization(
        subject_id=subject_id,
        sentence_id=sentence_id,
        radar_sample_id=radar_sample_id,
        laser_sample_id=laser_sample_id
    )
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Check the generated plots to assess synchronization between radar and laser data.")
    else:
        print("Analysis failed. Please check the data paths and file availability.")

if __name__ == "__main__":
    main()
