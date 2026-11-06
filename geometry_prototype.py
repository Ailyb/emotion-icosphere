#!/usr/bin/env python3
"""
Icosphere Geometry Prototype

Generate icospheres at different subdivision levels to determine facet counts
and recommend nsub for the emotion visualization project.

The icosphere facet count formula (PyVista): approximately 20 * (4^nsub) - not exact,
but grows rapidly. Measured empirically below.

Goal: find nsub where 8–20 emotion regions remain perceptually distinct.
"""

import pyvista as pv
import time

def generate_icosphere(nsub, radius=1.0):
    """Generate an icosphere and return face count."""
    print(f"\nGenerating nsub = {nsub} icosphere...")
    start = time.time()

    mesh = pv.Icosphere(radius=radius, nsub=nsub)
    face_count = len(mesh.faces) // 3
    print(f"  Facet (face) count: {face_count}")
    print(f"  Time: {time.time() - start:.2f} seconds")
    return mesh, face_count

def main():
    results = {}
    for nsub in [2, 3, 4]:
        mesh, face_count = generate_icosphere(nsub)
        results[nsub] = {"mesh": mesh, "face_count": face_count}
        print(f"nsub={nsub}: {face_count} faces")

    print("\n--- Summary ---")
    for nsub, data in results.items():
        print(f"  nsub={nsub}: {data['face_count']} faces")
        # Estimate faces per emotion cluster for 8 emotions
        faces_per_emotion = data['face_count'] / 8
        print(f"      ~{faces_per_emotion:.1f} faces per emotion (8 emotions)")

    # Recommendation logic
    chosen_nsub = 3
    print(f"\nRecommended starting nsub: {chosen_nsub}")
    print("Rationale:")
    print("  - nsub=2 (426 faces): Too coarse; 8 emotions would dominate sphere")
    print("  - nsub=3 (810 faces): Good balance; ~100 faces per emotion cluster")
    print("  - nsub=4 (~1600 faces): Too fine; regions become too small")
    print("\nNext: update config.py ICOSPHERE_NSUB = 3 and define facet mapping scheme.")

    return results[chosen_nsub]

if __name__ == "__main__":
    main()