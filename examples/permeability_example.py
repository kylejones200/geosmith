"""Example: Permeability calculations from porosity and water saturation.

Demonstrates various permeability estimation models.
"""

import numpy as np

from geosmith.primitives.petrophysics import (
    calculate_permeability_kozeny_carman,
    calculate_permeability_porosity_only,
    calculate_permeability_timur,
)


def main():
    """Run permeability calculation example."""
    print("=" * 60)
    print("Permeability Calculations Example")
    print("=" * 60)

    # Create sample porosity and water saturation data
    print("\n1. Creating sample data...")
    phi = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
    sw = np.array([0.4, 0.5, 0.6, 0.7, 0.8])

    print(f"Porosity range: {phi.min():.2f} - {phi.max():.2f}")
    print(f"Water saturation range: {sw.min():.2f} - {sw.max():.2f}")

    # Kozeny-Carman
    print("\n2. Kozeny-Carman permeability...")
    k_kc = calculate_permeability_kozeny_carman(
        phi, sw, grain_size_microns=100.0
    )
    print("Results:")
    for i, (p, s, k) in enumerate(zip(phi, sw, k_kc)):
        print(f"  φ={p:.2f}, Sw={s:.2f} → k={k:.2f} mD")

    # Timur
    print("\n3. Timur permeability...")
    k_timur = calculate_permeability_timur(phi, sw)
    print("Results:")
    for i, (p, s, k) in enumerate(zip(phi, sw, k_timur)):
        print(f"  φ={p:.2f}, Sw={s:.2f} → k={k:.2f} mD")

    # Porosity-only (simple power law)
    print("\n4. Porosity-only permeability (power law)...")
    k_phi = calculate_permeability_porosity_only(phi)
    print("Results:")
    for i, (p, k) in enumerate(zip(phi, k_phi)):
        print(f"  φ={p:.2f} → k={k:.2f} mD")

    # Compare methods
    print("\n5. Comparison of methods:")
    print(f"  Kozeny-Carman: {k_kc.mean():.2f} ± {k_kc.std():.2f} mD")
    print(f"  Timur: {k_timur.mean():.2f} ± {k_timur.std():.2f} mD")
    print(f"  Porosity-only: {k_phi.mean():.2f} ± {k_phi.std():.2f} mD")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


