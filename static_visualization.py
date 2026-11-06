#!/usr/bin/env python3
"""
Static Icosphere Visualization

This module renders an icosphere with facet intensities (gloom) based on emotion scores.
Uses PyVista for 3D visualization.
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Optional
import os
import json


class StaticVisualizer:
    """Render static icosphere visualization with emotion facet intensities."""

    def __init__(self, icosahedron_radius: float = 1.0, icosahedron_nsub: int = 3):
        """
        Initialize the visualizer.

        Args:
            icosahedron_radius: Radius of the icosphere
            icosahedron_nsub: Subdivision level of the icosphere
        """
        self.radius = icosahedron_radius
        self.nsub = icosahedron_nsub
        self.cell_count = None
        self.mesh = None
        
        # Create icosphere mesh
        self.create_icosphere()

    def create_icosphere(self):
        """Create and store the icosphere mesh."""
        print(f"Creating icosahedron with nsub={self.nsub}, radius={self.radius}")
        self.mesh = pv.Icosphere(radius=self.radius, nsub=self.nsub)
        self.cell_count = self.mesh.n_cells
        print(f"  Created mesh with {self.cell_count} cells (triangles)")

    def assign_cell_intensities(self, cell_intensities: np.ndarray):
        """
        Assign cell intensities to the mesh.

        Args:
            cell_intensities: Array of intensities for each cell (0.0 to 1.0)
        """
        if self.mesh is None:
            raise ValueError("Mesh not created yet. Call create_icosphere() first.")
            
        if len(cell_intensities) != self.cell_count:
            raise ValueError(f"Cell intensities length {len(cell_intensities)} doesn't match mesh cell count {self.cell_count}")
            
        # Assign intensities to cell data
        self.mesh.cell_data["intensity"] = cell_intensities
        print(f"  Assigned {len(cell_intensities)} cell intensities")

    def render(self, output_path: Optional[str] = None):
        """
        Render the icosphere with current cell intensities.
        
        Args:
            output_path: Optional path to save image
        """
        if self.mesh is None:
            raise ValueError("No mesh to render. Create icosphere first.")
            
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Add icosphere with intensity coloring on cells
        plotter.add_mesh(
            self.mesh,
            scalars="intensity",
            cmap="viridis",
            show_edges=True,
            edge_color="black",
            show_scalar_bar=True,
        )
        
        # Set camera view
        plotter.camera_position = 'iso'
        
        # Add title
        plotter.add_text("Emotion Visualization on Icosphere", font_size=16)
        
        # Save or show
        if output_path:
            plotter.screenshot(output_path)
            print(f"  Rendered image saved to {output_path}")
        else:
            plotter.show()
            print("  Rendered image displayed (closing window will exit)")

    def visualize_example(self, output_path: str = "static_visualization_example.png"):
        """Render an example with sample cell intensities."""
        print("\n=== Rendering Example Visualization ===")
        
        # Create sample cell intensities (8 emotions, mapped to cells)
        sample_scores = {
            "joy": 0.9,
            "sadness": 0.7,
            "fear": 0.8,
            "anger": 0.6,
            "surprise": 0.5,
            "disgust": 0.2,
            "calm": 0.9,
            "love": 0.8
        }
        
        # Map emotions to cell clusters (simplified)
        # Each cell index (0-1279) gets mapped to emotion regions
        num_cells = self.cell_count
        cell_intensities = np.zeros(num_cells)
        
        # Distribute cells across emotions (equal distribution for demo)
        emotions = list(sample_scores.keys())
        cells_per_emotion = num_cells // len(emotions)
        
        for i, emotion in enumerate(emotions):
            score = sample_scores[emotion]
            start_idx = i * cells_per_emotion
            end_idx = start_idx + cells_per_emotion if i < len(emotions) - 1 else num_cells
            for cell_idx in range(start_idx, end_idx):
                cell_intensities[cell_idx] = score
        
        print(f"  Cell distribution: {start_idx} - {end_idx-1} per emotion")
        
        # Assign intensities
        self.assign_cell_intensities(cell_intensities)
        
        # Render
        self.render(output_path)


def main():
    """Main entry point for static visualization."""
    print("=" * 60)
    print("Static Icosphere Visualization")
    print("=" * 60)
    
    # Create visualizer
    visualizer = StaticVisualizer(icosahedron_radius=1.0, icosahedron_nsub=3)
    
    # Render example
    visualizer.visualize_example("static_visualization_example.png")
    
    print("\n=== Static Visualization Complete ===")


if __name__ == "__main__":
    main()