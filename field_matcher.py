# Enables postponed evaluation of type annotations (PEP 563).
# Allows forward references without quotes (e.g., -> FieldMatcher instead of -> "FieldMatcher")
# and improves import performance by storing annotations as strings until needed.
from __future__ import annotations

import config  # Must be first - configures environment before ML libraries load
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the default target fields JSON file
# __file__ = this script's path, .parent = its directory, .resolve() = absolute path
DEFAULT_TARGET_FIELDS_PATH: Path = Path(__file__).parent.resolve() / "target_fields.json"


# =============================================================================
# DATA MODELS
# =============================================================================
# Pydantic models define the structure of our data with automatic validation.
# They can be easily converted to/from JSON for API integration.

class Field(BaseModel):
    """
    Represents a field from a system schema.

    Attributes:
        fieldHandle: The programmatic identifier (e.g., "productSku")
        fieldLabel: Human-readable name (e.g., "Product SKU")
        fieldType: Data type (e.g., "string", "int", "number", "date", "boolean")
        fieldDescription: Optional explanation of what the field contains
    """
    fieldHandle: str  # Required: the field's code/API name
    fieldLabel: str  # Required: the field's display name
    fieldType: str  # Required: the field's data type
    fieldDescription: Optional[str] = None  # Optional: defaults to None if not provided

    # Decorator: Apply this validator to multiple fields
    @field_validator("fieldHandle", "fieldLabel", "fieldType")
    # classmethod: This method belongs to the class, not an instance
    @classmethod
    def must_not_be_empty(field_class, value: str, field_info) -> str:
        """
        Validate that required fields are not empty strings.

        Args:
            field_class: The Field class itself (not an instance) - provided by @classmethod.
                         Required by Pydantic but unused in this validator.
            value: The value being validated
            field_info: Pydantic's ValidationInfo with metadata about the field

        Returns:
            The stripped (trimmed) value if valid

        Raises:
            ValueError: If the value is empty or only whitespace
        """
        # Check if value is falsy (None, empty string) or only whitespace
        if not value or not value.strip():
            # field_info.field_name gives us which field failed (e.g., "fieldHandle")
            raise ValueError(f"{field_info.field_name} is required and cannot be empty")
        # Return the trimmed value (removes leading/trailing whitespace)
        return value.strip()


class MatchResult(BaseModel):
    """
    Represents a matching result with the matched field and similarity score.

    Attributes:
        field: The target field that matched
        score: Similarity score from 0.0 (no match) to 1.0 (perfect match)
    """
    field: Field  # The matched target field
    score: float  # Cosine similarity score (0.0 to 1.0)


# =============================================================================
# CORE MATCHING LOGIC
# =============================================================================

def _convert_field_to_text(field: Field) -> str:
    """
    Convert a Field object into a natural language text representation.

    The embedding model works best when given descriptive text rather than
    just keywords. This function creates a readable string that captures
    the field's meaning.

    Args:
        field: The Field object to convert

    Returns:
        A text string like "Handle: sku | Label: Product SKU | Type: string"
    """
    # Build the base text with handle, label, and type
    text_representation = (
        f"Handle: {field.fieldHandle} | "  # Programmatic name
        f"Label: {field.fieldLabel} | "  # Human-readable name
        f"Type: {field.fieldType}"  # Data type
    )
    # Append description if provided (adds more semantic context)
    if field.fieldDescription:
        text_representation += f" | Description: {field.fieldDescription}"
    return text_representation


def load_target_fields(path: Optional[Path] = None) -> list[Field]:
    """
    Load target fields from a JSON file.

    Args:
        path: Absolute path to JSON file. If None, uses DEFAULT_TARGET_FIELDS_PATH.

    Returns:
        List of Field objects parsed from the JSON file.

    Example JSON format:
        [
            {"fieldHandle": "sku", "fieldLabel": "SKU", "fieldType": "string"},
            {"fieldHandle": "price", "fieldLabel": "Price", "fieldType": "number"}
        ]
    """
    # Use provided path or fall back to default
    target_fields_path = path or DEFAULT_TARGET_FIELDS_PATH
    # Open file with UTF-8 encoding (handles international characters)
    with open(target_fields_path, "r", encoding="utf-8") as json_file:
        # Parse JSON into Python list of dictionaries
        fields_data = json.load(json_file)
    # Convert each dictionary to a Field object using ** unpacking
    # **field_dict passes dict keys as keyword arguments: Field(fieldHandle="sku", ...)
    return [Field(**field_dict) for field_dict in fields_data]


class FieldMatcher:
    """
    Matches an input field against target fields using semantic similarity.

    This class loads a pre-trained embedding model that converts text into
    numeric vectors. Similar text produces similar vectors, allowing us to
    find the best matching fields without hardcoded rules.

    Usage:
        # Create matcher (loads model once)
        matcher = FieldMatcher()

        # Match an input field against targets
        results = matcher.match(input_field, target_fields, top_k=3)

        # Access results
        best_match = results[0].field
        confidence = results[0].score
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the matcher by loading the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default is all-MiniLM-L6-v2 (fast, good quality).

        The model is downloaded on first use and cached locally.
        """
        # Import utilities for suppressing console output
        import contextlib  # Context managers for redirecting output
        import io  # In-memory text streams

        # Suppress MLX/transformers load report that prints to console
        # redirect_stdout/stderr temporarily capture all console output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # Load the pre-trained model (downloads ~80MB on first run)
            self.model = SentenceTransformer(model_name)

    def match(
        self,
        input_field: Field,
        target_fields: list[Field],
        top_k: int = 3,
    ) -> list[MatchResult]:
        """
        Find the best matching target fields for an input field.

        This method:
        1. Converts fields to text descriptions
        2. Encodes text as numeric vectors (embeddings)
        3. Calculates cosine similarity between input and all targets
        4. Returns top matches sorted by similarity score

        Args:
            input_field: The field to find matches for
            target_fields: List of possible matching fields
            top_k: Number of top matches to return (default: 3)

        Returns:
            List of MatchResult objects, sorted by score (highest first)

        Raises:
            ValueError: If target_fields is empty
        """
        # Validate we have targets to match against
        if not target_fields:
            raise ValueError("target_fields is empty; nothing to match against.")

        # Convert input field to descriptive text
        input_field_text = _convert_field_to_text(input_field)
        # Convert all target fields to descriptive text
        target_field_texts = [_convert_field_to_text(target_field) for target_field in target_fields]

        # Encode input text as a vector (list of ~384 numbers)
        # Returns 2D array, so we pass a list with one item
        input_embedding = self.model.encode([input_field_text])
        # Encode all target texts as vectors (batch processing is faster)
        target_embeddings = self.model.encode(target_field_texts)

        # Calculate cosine similarity between input and each target
        # Returns 2D array [[score1, score2, ...]], we take first row [0]
        similarity_scores = cosine_similarity(input_embedding, target_embeddings)[0]

        # Get indices sorted by score in descending order (highest first)
        # sorted() with key=lambda and reverse=True sorts high to low
        ranked_indices = sorted(
            range(len(target_fields)),
            key=lambda index: similarity_scores[index],
            reverse=True
        )

        # Build result list with top_k matches
        match_results: list[MatchResult] = []
        # Slice to get top_k indices, ensuring at least 1 result
        for field_index in ranked_indices[: max(1, top_k)]:
            # Create MatchResult with the target field and its similarity score
            match_results.append(MatchResult(
                field=target_fields[field_index],
                score=float(similarity_scores[field_index])
            ))

        return match_results


# =============================================================================
# INTERACTIVE CLI (Command Line Interface)
# =============================================================================

def _prompt_required_input(prompt_text: str) -> str:
    """
    Prompt user for input, re-prompting if they enter an empty value.

    Args:
        prompt_text: The text to display (e.g., "  Handle: ")

    Returns:
        The user's non-empty input, with whitespace trimmed
    """
    # Loop until we get a valid (non-empty) response
    while True:
        # Display prompt and read user input, strip whitespace
        user_input = input(prompt_text).strip()
        # If user_input is truthy (not empty), return it
        if user_input:
            return user_input
        # Otherwise, show error and loop again
        print("    This field is required.")


def _prompt_for_input_field() -> Field:
    """
    Interactively prompt the user to enter all field details.

    Returns:
        A validated Field object with user-provided values
    """
    print("\nEnter input field details:")

    # Prompt for required fields (will re-prompt if empty)
    field_handle = _prompt_required_input("  Handle: ")
    field_label = _prompt_required_input("  Label: ")
    field_type = _prompt_required_input("  Type (string/int/number/date/boolean): ")

    # Prompt for optional description (empty is OK)
    field_description = input("  Description (optional, press Enter to skip): ").strip()

    # Create and return Field object
    # Pydantic will validate the values match our schema
    return Field(
        fieldHandle=field_handle,
        fieldLabel=field_label,
        fieldType=field_type,
        # Use None if description is empty, otherwise use the value
        fieldDescription=field_description if field_description else None,
    )


def main(default_target_fields_path: Optional[Path] = None) -> None:
    """
    Run the interactive command-line interface.

    This function:
    1. Prompts for target fields JSON file
    2. Prompts for input field details
    3. Loads the ML model and finds matches
    4. Displays ranked results

    Args:
        default_target_fields_path: Custom default path for targets.
                                    Uses DEFAULT_TARGET_FIELDS_PATH if None.
    """
    # Use provided default or fall back to module-level default
    resolved_default_path = default_target_fields_path or DEFAULT_TARGET_FIELDS_PATH

    # Print welcome header
    print("\n🔗 Field Matcher")
    print("=" * 40)  # Print 40 equals signs as a separator

    # Prompt for target fields file
    print(f"\nDefault target fields: {resolved_default_path}")
    user_path_input = input("Target fields JSON path (press Enter for default): ").strip()

    # Use user input if provided, otherwise use default
    if user_path_input:
        target_fields_path = Path(user_path_input)  # Convert string to Path object
    else:
        target_fields_path = resolved_default_path

    # Validate the file exists before trying to load it
    if not target_fields_path.exists():
        print(f"Error: File not found: {target_fields_path}")
        return  # Exit early if file doesn't exist

    # Load and parse the target fields from JSON
    target_fields = load_target_fields(target_fields_path)
    print(f"Loaded {len(target_fields)} target fields.")

    # Prompt user to enter the input field they want to match
    input_field = _prompt_for_input_field()

    # Load the ML model (this may take a few seconds on first run)
    print("\nLoading model...")
    field_matcher = FieldMatcher()

    # Perform the matching
    print("Matching...")
    match_results = field_matcher.match(input_field, target_fields, top_k=3)

    # Display results
    print("\n" + "=" * 40)  # Separator line
    print("Results:")
    print("-" * 40)  # Separator line

    # enumerate() gives us (rank_position, match_result) pairs, starting from 1
    for rank_position, match_result in enumerate(match_results, 1):
        # :20 pads the handle to 20 characters for alignment
        # :.3f formats the score with 3 decimal places
        print(f"  {rank_position}. {match_result.field.fieldHandle:20} (score: {match_result.score:.3f})")
        print(f"     Label: {match_result.field.fieldLabel}")
        print(f"     Type: {match_result.field.fieldType}")
        # Only print description if it exists
        if match_result.field.fieldDescription:
            print(f"     Description: {match_result.field.fieldDescription}")
        print()  # Empty line between results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This block only runs when the script is executed directly (python field_matcher.py)
# It does NOT run when the module is imported (from field_matcher import FieldMatcher)

if __name__ == "__main__":
    main()
