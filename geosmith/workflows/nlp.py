"""NLP workflows for geoscience text processing.

Layer 4: Workflows - Public entry points for NLP operations.
Handles file I/O and provides user-friendly API for entity recognition.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from geosmith.tasks.nlptask import (
    ChronostratNER,
    NERResult,
    extract_chronostrat_entities,
)

logger = logging.getLogger(__name__)


def load_entity_list(entity_file: Union[str, Path]) -> List[str]:
    """Load entity list from CSV file.

    Expected format: CSV with 'Text,Type' header or simple text file with one
        entity per line.

    Args:
        entity_file: Path to entity list file (CSV or text).

    Returns:
        List of entity names.

    Example:
        >>> from geosmith.workflows.nlp import load_entity_list
        >>> entities = load_entity_list('bgs-geo-entity-list.txt')
        >>> print(f"Loaded {len(entities)} entities")
    """
    entity_file = Path(entity_file)
    if not entity_file.exists():
        raise FileNotFoundError(f"Entity file not found: {entity_file}")

    entities = []

    try:
        # Try CSV format (Text,Type)
        df = pd.read_csv(entity_file)
        if "Text" in df.columns:
            entities = df["Text"].tolist()
        elif "Type" in df.columns:
            # Assume first column is entity names
            entities = df.iloc[:, 0].tolist()
        else:
            # Use first column
            entities = df.iloc[:, 0].tolist()
    except Exception:
        # Fallback: treat as plain text file (one entity per line)
        with open(entity_file, "r", encoding="utf-8") as f:
            entities = [line.strip() for line in f if line.strip()]

    # Filter out header if present
    entities = [e for e in entities if e.lower() not in ["text", "type", "chronostrat"]]

    logger.info(f"Loaded {len(entities)} entities from {entity_file}")
    return entities


def load_text_documents(
    text_file: Union[str, Path], format: str = "one_per_line"
) -> List[str]:
    """Load text documents from file.

    Args:
        text_file: Path to text file.
        format: File format ('one_per_line' or 'plain').
                'one_per_line': One document per line.
                'plain': Single document.

    Returns:
        List of text documents.

    Example:
        >>> from geosmith.workflows.nlp import load_text_documents
        >>> texts = load_text_documents(
        ...     'bgs-geo-testing-data.txt', format='one_per_line'
        ... )
        >>> print(f"Loaded {len(texts)} documents")
    """
    text_file = Path(text_file)
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")

    if format == "one_per_line":
        with open(text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:  # plain
        with open(text_file, "r", encoding="utf-8") as f:
            texts = [f.read()]

    logger.info(f"Loaded {len(texts)} documents from {text_file}")
    return texts


def extract_entities_from_file(
    text_file: Union[str, Path],
    entity_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    model_type: str = "spacy",
    format: str = "one_per_line",
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """Extract chronostratigraphic entities from text file.

    This is the main workflow function that replaces Amazon Comprehend functionality.

    Args:
        text_file: Path to text file with documents.
        entity_file: Path to entity list file (CSV or text).
        output_file: Optional path to save results CSV.
        model_type: Model type ('spacy', 'transformers', or 'hybrid').
        format: Text file format ('one_per_line' or 'plain').
        confidence_threshold: Minimum confidence score (0-1).

    Returns:
        DataFrame with columns: document_id, text, entity, label, start, end,
            confidence.

    Example:
        >>> from geosmith.workflows.nlp import extract_entities_from_file
        >>>
        >>> results = extract_entities_from_file(
        ...     'bgs-geo-testing-data.txt',
        ...     'bgs-geo-entity-list.txt',
        ...     output_file='results.csv'
        ... )
        >>> print(
        ...     f"Extracted {len(results)} entities from "
        ...     f"{results['document_id'].nunique()} documents"
        ... )
    """
    logger.info(f"Starting entity extraction workflow")
    logger.info(f"Text file: {text_file}")
    logger.info(f"Entity file: {entity_file}")
    logger.info(f"Model type: {model_type}")

    # Load entity list
    entities = load_entity_list(entity_file)

    # Load text documents
    texts = load_text_documents(text_file, format=format)

    # Initialize NER model
    ner = ChronostratNER(
        model_type=model_type, confidence_threshold=confidence_threshold
    )
    ner.fit(entity_list=entities)

    # Extract entities from all documents
    all_results = []
    for doc_id, text in enumerate(texts):
        result = ner.predict(text)

        for entity in result.entities:
            all_results.append(
                {
                    "document_id": doc_id,
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                }
            )

    results_df = pd.DataFrame(all_results)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path} ({len(results_df)} entities)")

    logger.info(
        f"Extraction complete: {len(results_df)} entities from {len(texts)} documents"
    )

    return results_df


def train_custom_ner_model(
    training_file: Union[str, Path],
    entity_file: Union[str, Path],
    model_output_dir: Optional[Union[str, Path]] = None,
    model_type: str = "spacy",
    n_iter: int = 20,
    validation_split: float = 0.2,
) -> ChronostratNER:
    """Train a custom NER model on geoscience text.

    Replaces Amazon Comprehend training workflow with modern NLP tools.

    Args:
        training_file: Path to training text file.
        entity_file: Path to entity list file.
        model_output_dir: Optional directory to save trained model.
        model_type: Model type ('spacy', 'transformers', or 'hybrid').
        n_iter: Number of training iterations (default 20).
        validation_split: Fraction of data for validation (default 0.2).

    Returns:
        Trained ChronostratNER model.

    Example:
        >>> from geosmith.workflows.nlp import train_custom_ner_model
        >>>
        >>> model = train_custom_ner_model(
        ...     'bgs-geo-training-data.txt',
        ...     'bgs-geo-entity-list.txt',
        ...     model_output_dir='./models'
        ... )
        >>>
        >>> # Use trained model
        >>> result = model.predict("Sedimentary rocks of Silurian age occur.")
        >>> print(f"Found: {[e.text for e in result.entities]}")
    """
    logger.info(f"Starting NER model training")
    logger.info(f"Training file: {training_file}")
    logger.info(f"Entity file: {entity_file}")
    logger.info(f"Model type: {model_type}")

    # Load entity list
    entities = load_entity_list(entity_file)

    # Load training texts
    training_texts = load_text_documents(training_file, format="one_per_line")

    # Initialize model
    ner = ChronostratNER(model_type=model_type)
    ner.fit(
        entity_list=entities,
        training_texts=training_texts,
        n_iter=n_iter,
        validation_split=validation_split,
    )

    # Save model if requested
    if model_output_dir:
        output_path = Path(model_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if model_type == "spacy":
            # Save spaCy model
            model_path = output_path / "ner_model"
            if ner.nlp:
                ner.nlp.to_disk(model_path)
                logger.info(f"Saved spaCy model to {model_path}")
        else:
            # For transformers, save tokenizer and model
            # (would need additional implementation)
            logger.warning(
                "Transformers model saving requires additional implementation. "
                "Model loaded but not saved."
            )

    logger.info("Model training complete")

    return ner
