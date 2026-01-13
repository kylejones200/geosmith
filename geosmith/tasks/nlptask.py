"""Named Entity Recognition (NER) for geoscience text.

Migrated from geosemantics project, replacing Amazon Comprehend with modern NLP tools.
Layer 3: Tasks - User intent translation.

Uses spaCy + transformers for chronostratigraphic entity recognition.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from geosmith.primitives.base import BaseEstimator

logger = logging.getLogger(__name__)

# Optional spaCy (preferred - fast and accurate)
try:
    import spacy
    from spacy import displacy
    from spacy.tokens import Doc

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore
    displacy = None  # type: ignore
    Doc = None  # type: ignore

# Optional transformers (for better accuracy)
try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )
    from transformers.pipelines import TokenClassificationPipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore
    TokenClassificationPipeline = None  # type: ignore


@dataclass
class EntityMatch:
    """A single entity match from NER."""

    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None


@dataclass
class NERResult:
    """Results from Named Entity Recognition."""

    entities: List[EntityMatch]
    model_type: str
    n_entities: int
    unique_labels: List[str]


class ChronostratNER(BaseEstimator):
    """Named Entity Recognition for chronostratigraphic terms.

    Extracts geological time periods (CHRONOSTRAT) from text using modern NLP.
    Supports both spaCy (fast) and transformers (accurate) backends.

    Example:
        >>> from geosmith.tasks.nlptask import ChronostratNER
        >>>
        >>> ner = ChronostratNER(model_type='spacy')
        >>> ner.fit(entity_list=['Cambrian', 'Silurian', 'Permian'])
        >>>
        >>> text = "Sedimentary rocks of Silurian age occur in a series of inliers."
        >>> result = ner.predict(text)
        >>> print(
        ...     f"Found {result.n_entities} entities: "
        ...     f"{[e.text for e in result.entities]}"
        ... )
    """

    def __init__(
        self,
        model_type: str = "spacy",
        entity_type: str = "CHRONOSTRAT",
        use_transformer: bool = False,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize chronostratigraphic NER model.

        Args:
            model_type: Model type ('spacy', 'transformers', or 'hybrid').
                       'spacy' is fast, 'transformers' is accurate,
                       'hybrid' combines both.
            entity_type: Entity type label (default: 'CHRONOSTRAT').
            use_transformer: If True, use transformer-based model for better accuracy.
            model_name: Pre-trained model name (e.g., 'en_core_web_sm' for
                spaCy, 'dbmdz/bert-large-cased-finetuned-conll03-english' for
                transformers). If None, uses default model.
            confidence_threshold: Minimum confidence score for entity detection (0-1).

        Raises:
            ImportError: If required NLP libraries are not available.
        """
        self.model_type = model_type.lower()
        self.entity_type = entity_type
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Model components (set during fit)
        self.nlp = None
        self.ner_pipeline = None
        self.entity_list = None
        self._estimator_type = "transformer"

        # Initialize model backend
        self._initialize_model()

    def _initialize_model(self):
        """Initialize NLP model backend."""
        if self.model_type == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is required for NER. "
                    "Install with: pip install spacy && "
                    "python -m spacy download en_core_web_sm"
                )

            model_name = self.model_name or "en_core_web_sm"
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                raise ImportError(
                    f"spaCy model '{model_name}' not found. "
                    f"Download with: python -m spacy download {model_name}"
                )

            logger.info(f"Loaded spaCy model: {model_name}")

        elif self.model_type == "transformers" or self.use_transformer:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is required for transformer-based NER. "
                    "Install with: pip install transformers torch"
                )

            model_name = (
                self.model_name
                or "dbmdz/bert-large-cased-finetuned-conll03-english"
                # Pre-trained NER model
            )
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
            )
            logger.info(f"Loaded transformer model: {model_name}")

        elif self.model_type == "hybrid":
            # Use both spaCy and transformers
            if not SPACY_AVAILABLE or not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Hybrid mode requires both spaCy and transformers. "
                    "Install with: pip install spacy transformers torch"
                )
            self._initialize_model()  # Initialize spaCy
            # Transformers pipeline initialized on-demand

    def fit(
        self,
        entity_list: List[str],
        training_texts: Optional[List[str]] = None,
        validation_split: float = 0.2,
        n_iter: int = 10,
    ) -> "ChronostratNER":
        """Train or fine-tune NER model on chronostratigraphic entities.

        Args:
            entity_list: List of known chronostratigraphic terms
                (e.g., ['Cambrian', 'Silurian']).
            training_texts: Optional list of training texts with entities labeled.
                           If None, uses entity_list for rule-based matching.
            validation_split: Fraction of data for validation (default 0.2).
            n_iter: Number of training iterations for spaCy (default 10).

        Returns:
            Self for method chaining.

        Example:
            >>> ner = ChronostratNER(model_type='spacy')
            >>> ner.fit(entity_list=['Cambrian', 'Silurian', 'Permian', 'Triassic'])
            >>> ner.fit(entity_list=entities, training_texts=labeled_texts, n_iter=20)
        """
        if entity_list is None or len(entity_list) == 0:
            raise ValueError("entity_list cannot be empty")

        self.entity_list = [entity.lower() for entity in entity_list]

        if self.model_type == "spacy":
            # Add custom NER patterns to spaCy model
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": self.entity_type, "pattern": entity}
                for entity in self.entity_list
            ]
            ruler.add_patterns(patterns)

            logger.info(f"Added {len(patterns)} entity patterns to spaCy model")

            # If training texts provided, fine-tune model
            if training_texts:
                self._train_spacy_model(training_texts, validation_split, n_iter)

        elif self.model_type == "transformers" or self.use_transformer:
            # For transformers, we use pre-trained model as-is
            # Fine-tuning would require more setup (tokenization, labels, etc.)
            logger.info(
                "Using pre-trained transformer model. "
                "Fine-tuning requires labeled training data in specific format."
            )

        return self

    def _train_spacy_model(
        self,
        training_texts: List[str],
        validation_split: float,
        n_iter: int,
    ):
        """Train spaCy NER model on labeled texts.

        Note: Full training requires properly formatted training data.
        For now, this is a placeholder for future implementation.
        """
        logger.warning(
            "Full spaCy training requires labeled training data in spaCy format. "
            "Using rule-based entity ruler for now."
        )
        # TODO: Implement full spaCy training with labeled data
        # Would require converting training texts to spaCy training format
        pass

    def predict(self, text: Union[str, List[str]]) -> Union[NERResult, List[NERResult]]:
        """Extract chronostratigraphic entities from text.

        Args:
            text: Input text (str) or list of texts (List[str]).

        Returns:
            NERResult with extracted entities, or List[NERResult] if input is list.

        Example:
            >>> ner = ChronostratNER(model_type='spacy')
            >>> ner.fit(entity_list=['Silurian', 'Cambrian'])
            >>>
            >>> result = ner.predict(
            ...     "Sedimentary rocks of Silurian age occur in inliers."
            ... )
            >>> print(f"Found: {[e.text for e in result.entities]}")
        """
        if self.entity_list is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if isinstance(text, str):
            return self._predict_single(text)
        else:
            return [self._predict_single(t) for t in text]

    def _predict_single(self, text: str) -> NERResult:
        """Extract entities from a single text."""
        if self.model_type == "spacy":
            return self._predict_spacy(text)
        elif self.model_type == "transformers" or self.use_transformer:
            return self._predict_transformers(text)
        elif self.model_type == "hybrid":
            # Combine both methods
            spacy_result = self._predict_spacy(text)
            trans_result = self._predict_transformers(text)
            return self._merge_results(spacy_result, trans_result)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _predict_spacy(self, text: str) -> NERResult:
        """Extract entities using spaCy."""
        if self.nlp is None:
            raise ValueError("spaCy model not initialized")

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ == self.entity_type:
                # Check confidence if available
                confidence = (
                    ent._.confidence
                    if hasattr(ent, "_") and hasattr(ent._, "confidence")
                    else 1.0
                )

                if confidence >= self.confidence_threshold:
                    entities.append(
                        EntityMatch(
                            text=ent.text,
                            label=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=confidence,
                        )
                    )

        # Also check entity ruler matches (case-insensitive)
        text_lower = text.lower()
        for entity in self.entity_list:
            entity_lower = entity.lower()
            idx = text_lower.find(entity_lower)
            if idx >= 0:
                # Check if not already captured by NER
                if not any(
                    e.start <= idx < e.end
                    for e in entities
                    if e.text.lower() == entity_lower
                ):
                    entities.append(
                        EntityMatch(
                            text=text[idx : idx + len(entity)],
                            label=self.entity_type,
                            start=idx,
                            end=idx + len(entity),
                            confidence=1.0,  # Rule-based match
                        )
                    )

        unique_labels = sorted(set(e.label for e in entities))

        return NERResult(
            entities=entities,
            model_type="spacy",
            n_entities=len(entities),
            unique_labels=unique_labels,
        )

    def _predict_transformers(self, text: str) -> NERResult:
        """Extract entities using transformers."""
        if self.ner_pipeline is None:
            raise ValueError("Transformer pipeline not initialized")

        # Run NER pipeline
        predictions = self.ner_pipeline(text)

        entities = []
        for pred in predictions:
            label = pred.get("entity_group", pred.get("label", "O"))
            score = pred.get("score", 1.0)

            # Filter by entity type and confidence
            if (
                label == self.entity_type
                or label.startswith("B-")
                or label.startswith("I-")
            ):
                if score >= self.confidence_threshold:
                    # Check if entity matches our entity list (case-insensitive)
                    entity_text = pred["word"].strip()
                    if any(
                        entity.lower() in entity_text.lower()
                        for entity in self.entity_list
                    ):
                        entities.append(
                            EntityMatch(
                                text=entity_text,
                                label=self.entity_type,
                                start=pred.get("start", 0),
                                end=pred.get("end", len(entity_text)),
                                confidence=score,
                            )
                        )

        unique_labels = sorted(set(e.label for e in entities))

        return NERResult(
            entities=entities,
            model_type="transformers",
            n_entities=len(entities),
            unique_labels=unique_labels,
        )

    def _merge_results(self, result1: NERResult, result2: NERResult) -> NERResult:
        """Merge results from multiple models, removing duplicates."""
        # Combine entities, removing overlaps
        merged_entities = result1.entities.copy()

        for ent2 in result2.entities:
            # Check for overlap with existing entities
            overlaps = False
            for ent1 in merged_entities:
                if (
                    ent1.start <= ent2.start < ent1.end
                    or ent1.start < ent2.end <= ent1.end
                    or (ent2.start <= ent1.start and ent2.end >= ent1.end)
                ):
                    overlaps = True
                    # Keep entity with higher confidence
                    if ent2.confidence and ent1.confidence:
                        if ent2.confidence > ent1.confidence:
                            merged_entities.remove(ent1)
                            merged_entities.append(ent2)
                    break

            if not overlaps:
                merged_entities.append(ent2)

        # Sort by start position
        merged_entities.sort(key=lambda x: x.start)

        unique_labels = sorted(set(e.label for e in merged_entities))

        return NERResult(
            entities=merged_entities,
            model_type="hybrid",
            n_entities=len(merged_entities),
            unique_labels=unique_labels,
        )


def extract_chronostrat_entities(
    text: Union[str, List[str]],
    entity_list: List[str],
    model_type: str = "spacy",
    confidence_threshold: float = 0.5,
) -> Union[NERResult, List[NERResult]]:
    """Convenience function for extracting chronostratigraphic entities.

    Args:
        text: Input text or list of texts.
        entity_list: List of chronostratigraphic terms to recognize.
        model_type: Model type ('spacy', 'transformers', or 'hybrid').
        confidence_threshold: Minimum confidence score (0-1).

    Returns:
        NERResult or List[NERResult] with extracted entities.

    Example:
        >>> from geosmith.tasks.nlptask import extract_chronostrat_entities
        >>>
        >>> entities = ['Cambrian', 'Silurian', 'Permian', 'Triassic']
        >>> text = "Sedimentary rocks of Silurian age occur in Permian strata."
        >>> result = extract_chronostrat_entities(text, entities)
        >>> print(f"Found: {[e.text for e in result.entities]}")
    """
    ner = ChronostratNER(
        model_type=model_type, confidence_threshold=confidence_threshold
    )
    ner.fit(entity_list=entity_list)
    return ner.predict(text)
