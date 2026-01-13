"""Example: Chronostratigraphic Entity Recognition with Modern NLP.

Replicates geosemantics functionality using spaCy/transformers instead of Amazon Comprehend.

This example demonstrates:
1. Loading entity list and training data
2. Training a custom NER model
3. Extracting chronostratigraphic entities from geological text
4. Comparing results across different model types
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from geosmith.workflows.nlp import (
        extract_entities_from_file,
        load_entity_list,
        train_custom_ner_model,
    )
    from geosmith.tasks.nlptask import extract_chronostrat_entities

    NLP_AVAILABLE = True
except ImportError:
    print("❌ NLP dependencies not available.")
    print("Install with: pip install geosmith[nlp]")
    print("Or: pip install spacy transformers torch")
    NLP_AVAILABLE = False


def main():
    """Run chronostratigraphic entity recognition example."""
    if not NLP_AVAILABLE:
        return

    # Example text with chronostratigraphic entities
    example_text = """
    Sedimentary rocks of Silurian age occur in a series of inliers along the 
    southern margin of the Midland Valley. The inliers lie on opposite sides 
    of the NE-trending Kerse Loch Fault which clearly was active in late 
    Silurian times. It has been suggested that in Lower Palaeozoic times the 
    Midland Valley and Southern Uplands depositional basins lay some distance 
    apart. The highly irregular middle zone is underlain by Permian and 
    Triassic rocks: the Permian strata include Upper Permian Zechstein 
    sedimentary rocks. The Carboniferous rocks of the Midland Valley include 
    Cambrian formations.
    """

    print("=" * 70)
    print("Chronostratigraphic Entity Recognition Example")
    print("=" * 70)
    print("\nExample text:")
    print(example_text)
    print("\n" + "=" * 70)

    # Entity list (from geosemantics project)
    entity_list = [
        "Cambrian",
        "Silurian",
        "Permian",
        "Triassic",
        "Carboniferous",
        "Palaeozoic",
        "Lower Palaeozoic",
        "Late Silurian",
        "Upper Permian",
    ]

    print(f"\nEntity list ({len(entity_list)} terms):")
    print(", ".join(entity_list[:10]) + "...")

    # Method 1: Quick extraction using convenience function
    print("\n" + "-" * 70)
    print("Method 1: Quick extraction (spaCy)")
    print("-" * 70)

    try:
        result = extract_chronostrat_entities(
            example_text, entity_list, model_type="spacy", confidence_threshold=0.5
        )

        print(f"\n✅ Found {result.n_entities} entities:")
        for entity in result.entities:
            print(
                f"  - '{entity.text}' ({entity.label}) "
                f"at position {entity.start}-{entity.end} "
                f"(confidence: {entity.confidence:.2f})"
            )
    except ImportError as e:
        print(f"⚠️ spaCy not available: {e}")
        print("Install with: pip install spacy && python -m spacy download en_core_web_sm")

    # Method 2: Using model class directly
    print("\n" + "-" * 70)
    print("Method 2: Custom model (spaCy)")
    print("-" * 70)

    try:
        from geosmith.tasks.nlptask import ChronostratNER

        ner = ChronostratNER(model_type="spacy", confidence_threshold=0.5)
        ner.fit(entity_list=entity_list)

        result = ner.predict(example_text)
        print(f"\n✅ Found {result.n_entities} entities:")
        for entity in result.entities:
            print(f"  - '{entity.text}' at position {entity.start}-{entity.end}")

    except ImportError as e:
        print(f"⚠️ spaCy not available: {e}")

    # Method 3: Full workflow (file-based, like Amazon Comprehend)
    print("\n" + "-" * 70)
    print("Method 3: File-based workflow (replaces Amazon Comprehend)")
    print("-" * 70)

    # Check if geosemantics data files exist
    geosemantics_path = Path("/Users/kylejonespatricia/geosemantics")
    entity_file = geosemantics_path / "bgs-geo-entity-list.txt"
    test_file = geosemantics_path / "bgs-geo-testing-data.txt"

    if entity_file.exists() and test_file.exists():
        print(f"\nFound geosemantics data files:")
        print(f"  Entity list: {entity_file}")
        print(f"  Test data: {test_file}")

        try:
            # Extract entities from file (main workflow)
            results_df = extract_entities_from_file(
                text_file=test_file,
                entity_file=entity_file,
                model_type="spacy",
                format="one_per_line",
                confidence_threshold=0.5,
            )

            print(f"\n✅ Extraction complete!")
            print(f"  Total entities: {len(results_df)}")
            print(f"  Unique documents: {results_df['document_id'].nunique()}")
            print(f"  Unique entities: {results_df['text'].nunique()}")
            print(f"\nTop 10 entities found:")
            print(results_df["text"].value_counts().head(10))

        except Exception as e:
            print(f"⚠️ Error during file extraction: {e}")
    else:
        print("\n⚠️ Geosemantics data files not found.")
        print(f"  Expected: {entity_file}")
        print(f"  Expected: {test_file}")
        print("\nRunning with example data instead...")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()


