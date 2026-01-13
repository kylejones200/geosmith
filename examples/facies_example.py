"""Example: Facies classification using machine learning.

Demonstrates facies classification with SVM and Random Forest.
"""

import numpy as np
import pandas as pd

from geosmith import GeoTable
from geosmith.tasks import FaciesTask


def main():
    """Run facies classification example."""
    print("=" * 60)
    print("Facies Classification Example")
    print("=" * 60)

    # Create synthetic well log data with facies
    print("\n1. Creating synthetic well log data...")
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic log data
    data = {
        "DEPTH": np.linspace(1000, 3000, n_samples),
        "GR": np.random.randn(n_samples) * 30 + 50,
        "NPHI": np.random.randn(n_samples) * 0.05 + 0.20,
        "RHOB": np.random.randn(n_samples) * 0.1 + 2.4,
        "PE": np.random.randn(n_samples) * 0.5 + 3.0,
    }

    # Create facies based on log values
    facies = []
    for i in range(n_samples):
        if data["GR"][i] < 40:
            facies.append("Sandstone")
        elif data["GR"][i] < 60:
            facies.append("Shale")
        else:
            facies.append("Limestone")

    data["Facies"] = facies
    df = pd.DataFrame(data)

    print(f"Created {len(df)} samples")
    print(f"Facies distribution:")
    print(df["Facies"].value_counts())

    # Create GeoTable
    geotable = GeoTable(data=df)

    # Train facies classifier
    print("\n2. Training facies classifier (Random Forest)...")
    task = FaciesTask()
    result = task.train_and_predict(
        data=geotable,
        feature_cols=["GR", "NPHI", "RHOB", "PE"],
        target_col="Facies",
        model_type="RF",
        test_size=0.2,
        random_state=42,
    )

    print(f"Model: {result.model_name}")
    print(f"Classes: {result.classes_}")
    if result.report:
        print("\nClassification Report:")
        print(result.report)

    # Add predictions to dataframe
    df["Predicted"] = result.y_pred
    df["Probability"] = result.proba.max(axis=1)

    # Show accuracy
    accuracy = (df["Facies"] == df["Predicted"]).mean()
    print(f"\n3. Classification accuracy: {accuracy:.2%}")

    # Show sample predictions
    print("\n4. Sample predictions:")
    for i in range(0, min(10, len(df)), 5):
        row = df.iloc[i]
        print(
            f"  Depth={row['DEPTH']:.0f}m: "
            f"True={row['Facies']}, Pred={row['Predicted']}, "
            f"Prob={row['Probability']:.2f}"
        )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


