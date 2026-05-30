"""
Schema validator for Auto-HKG LLM extraction output.

This module validates that the JSON object returned by the language model
conforms to the required Auto-HKG schema before it is passed to the
GraphBuilder. Validation catches structural errors early and prevents
malformed data from entering the knowledge graph.

Validated fields:
    topic_coarse  : Non-empty string representing the broad subject category.
    topic_fine    : Non-empty string representing the specific sub-topic.
    concepts      : Non-empty list of strings (1-4 items).
    methods       : List of strings (may be empty).
    bloom_level   : String starting with C1 through C6.
    prerequisites : List of strings (may be empty).
    successors    : List of strings (may be empty).
    difficulty    : Integer or float in range [1, 5].
"""

REQUIRED_KEYS = {
    "topic_coarse",
    "topic_fine",
    "concepts",
    "methods",
    "bloom_level",
    "prerequisites",
    "successors",
    "difficulty",
}

BLOOM_PREFIXES = [f"C{i}" for i in range(1, 7)]  # C1 through C6

LIST_FIELDS = ("concepts", "methods", "prerequisites", "successors")
STRING_FIELDS = ("topic_coarse", "topic_fine")


class SchemaError(Exception):
    """
    Raised when the LLM output does not conform to the Auto-HKG extraction schema.
    The error message describes which field failed and why.
    """
    pass


def validate_extraction(data: dict) -> None:
    """
    Validate a parsed JSON extraction dict against the Auto-HKG schema.

    Raises SchemaError with a descriptive message if any field is invalid.
    Returns None if all validations pass.

    Args:
        data: Parsed Python dict from the LLM JSON output.

    Raises:
        SchemaError: If any required field is missing, has wrong type,
                     is empty when it must not be, or contains an out-of-range value.
    """
    # 1. Check all required keys are present
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise SchemaError(
            f"Missing required fields: {sorted(missing)}. "
            f"Received keys: {sorted(data.keys())}"
        )

    # 2. All list fields must be actual Python lists
    for key in LIST_FIELDS:
        if not isinstance(data[key], list):
            raise SchemaError(
                f"Field '{key}' must be a list. "
                f"Received type: {type(data[key]).__name__}, value: {repr(data[key])}"
            )

    # 3. 'concepts' must not be empty — at least one concept must be extracted
    if len(data["concepts"]) == 0:
        raise SchemaError(
            "Field 'concepts' must contain at least one item. "
            "Every question must map to at least one knowledge concept."
        )

    # 4. 'difficulty' must be a number in [1, 5]
    if not isinstance(data["difficulty"], (int, float)):
        raise SchemaError(
            f"Field 'difficulty' must be a number. "
            f"Received type: {type(data['difficulty']).__name__}, value: {repr(data['difficulty'])}"
        )
    if not (1 <= data["difficulty"] <= 5):
        raise SchemaError(
            f"Field 'difficulty' must be between 1 and 5 inclusive. "
            f"Received: {data['difficulty']}"
        )

    # 5. 'bloom_level' must start with a recognized prefix (C1 through C6)
    bloom = str(data["bloom_level"]).strip()
    if not any(bloom.startswith(prefix) for prefix in BLOOM_PREFIXES):
        raise SchemaError(
            f"Field 'bloom_level' must start with one of {BLOOM_PREFIXES}. "
            f"Received: '{bloom}'"
        )

    # 6. String fields must not be empty or whitespace-only
    for key in STRING_FIELDS:
        if not str(data[key]).strip():
            raise SchemaError(
                f"Field '{key}' must not be an empty string. "
                f"Received: {repr(data[key])}"
            )

    # 7. All list items in concepts, prerequisites, successors must be non-empty strings
    for key in ("concepts", "prerequisites", "successors"):
        for i, item in enumerate(data[key]):
            if not isinstance(item, str) or not item.strip():
                raise SchemaError(
                    f"All items in '{key}' must be non-empty strings. "
                    f"Item at index {i} is invalid: {repr(item)}"
                )
