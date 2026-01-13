"""Example: Petrophysics calculations using Archie's equation.

Demonstrates water saturation calculations and Pickett plot isolines.
"""

import numpy as np

from geosmith.primitives.petrophysics import (
    ArchieParams,
    calculate_bulk_volume_water,
    calculate_water_saturation,
    pickett_isolines,
)


def main():
    """Run petrophysics example."""
    print("=" * 60)
    print("Petrophysics Calculations Example")
    print("=" * 60)

    # Create Archie parameters
    print("\n1. Setting up Archie parameters...")
    params = ArchieParams(a=1.0, m=2.0, n=2.0, rw=0.1)
    print(f"Tortuosity factor (a): {params.a}")
    print(f"Cementation exponent (m): {params.m}")
    print(f"Saturation exponent (n): {params.n}")
    print(f"Water resistivity (Rw): {params.rw} ohm·m")

    # Calculate water saturation
    print("\n2. Calculating water saturation...")
    rt = np.array([5.0, 10.0, 20.0, 50.0])  # True resistivity (ohm·m)
    phi = np.array([0.15, 0.20, 0.25, 0.30])  # Porosity (fraction)

    sw = calculate_water_saturation(rt, phi, params)
    print("Results:")
    for i, (r, p, s) in enumerate(zip(rt, phi, sw)):
        print(f"  Rt={r:.1f} ohm·m, φ={p:.2f} → Sw={s:.2%}")

    # Calculate bulk volume water
    print("\n3. Calculating bulk volume water (BVW)...")
    bvw = calculate_bulk_volume_water(phi, sw)
    print("Results:")
    for i, (p, s, b) in enumerate(zip(phi, sw, bvw)):
        print(f"  φ={p:.2f}, Sw={s:.2%} → BVW={b:.3f}")

    # Generate Pickett plot isolines
    print("\n4. Generating Pickett plot isolines...")
    phi_vals = np.array([0.1, 0.3])
    sw_vals = np.array([0.5, 0.7, 1.0])

    isolines = pickett_isolines(phi_vals, sw_vals, params, num_points=50)
    print(f"Generated {len(isolines)} isolines:")
    for phi, rt, label in isolines:
        print(f"  {label}: {len(phi)} points, "
              f"Rt range: {rt.min():.2f} - {rt.max():.2f} ohm·m")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


