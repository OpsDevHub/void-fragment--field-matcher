# Void Fragment: Field Matcher

**Field Matcher** is a [Void Fragment](https://github.com/OpsDevHub/void-fragments/tree/main/fragments) that matches fields between two systems
based on *semantic meaning* --- not hardcoded rules.

Given an input field (handle, label, type, and optional description),
it finds the best matching field from a list of target fields.

This project explores how **embedding-based similarity** can be used to
reduce manual schema mapping work in integrations, ERPs, and data
pipelines.

------------------------------------------------------------------------

## What problem does this solve?

When integrating two systems (ERP ↔ ecommerce, CMS ↔ warehouse, etc.),
field mapping is often:

-   manual
-   error-prone
-   repetitive
-   dependent on naming conventions

**Field Matcher** answers:

> "Which field in the target system *means the same thing* as this input field?"

It does this **without rules**, **without training**, and **without
APIs** --- using a small local AI model.

------------------------------------------------------------------------

## How it works (high level)

1.  **Input**
    -   An input field:
        -   `fieldHandle` (required)
        -   `fieldLabel` (required)
        -   `fieldType` (required)
        -   `fieldDescription` (optional)
    -   A list of target fields with the same attributes
    -   Validation ensures required fields are non-empty
2.  **Semantic encoding**
    -   Each field is converted into a short text description
    -   A local embedding model converts that text into vectors (numbers
        representing meaning)
3.  **Similarity scoring**
    -   The input vector is compared against all target vectors
    -   Cosine similarity measures how close the meanings are
4.  **Result**
    -   The best matching target field is returned
    -   Top matches and confidence scores are shown for inspection

------------------------------------------------------------------------

## Example session

```
$ python run_app.py

🔗 Field Matcher
========================================

Default target fields: /path/to/field-match/target_fields.json
Target fields JSON path (press Enter for default):
Loaded 8 target fields.

Enter input field details:
  Handle: qtyavail
  Label: Quantity Available
  Type (string/int/number/date/boolean): int
  Description (optional, press Enter to skip):

Loading model...
Matching...

========================================
Results:
----------------------------------------
  1. availableQuantity      (score: 0.810)
     Label: Available Quantity
     Type: int

  2. onHandQuantity         (score: 0.728)
     Label: On Hand Quantity
     Type: int

  3. price                  (score: 0.590)
     Label: Price
     Type: number
```

------------------------------------------------------------------------

## The model

This project uses:

    sentence-transformers/all-MiniLM-L6-v2

Why this model? - Runs **locally** - CPU-friendly - Fast after first
download - Strong general semantic understanding - No API keys required

------------------------------------------------------------------------

## Project structure

    field-matcher/
    ├── field_matcher.py         # Main module (importable + CLI)
    ├── config.py                # Environment config (must import first)
    ├── target_fields.json       # Default target fields
    ├── run_app.py               # Run the app (python run_app.py)
    ├── run_tests.py             # Run tests (python run_tests.py)
    ├── test_field_matcher.py    # Tests
    ├── pyproject.toml           # Project config and dependencies
    └── README.md

------------------------------------------------------------------------

## Running the project

### Requirements

-   Python 3.10+
-   pip

### Install dependencies

``` bash
pip install sentence-transformers scikit-learn pydantic
```

### Optional: Hugging Face authentication

For higher download rate limits, set your Hugging Face token:

``` bash
export HF_TOKEN=your_token_here
```

Get your token at https://huggingface.co/settings/tokens

### Run interactively

``` bash
python run_app.py
```

### Run tests

``` bash
pip install pytest
python run_tests.py
```

### Use programmatically

``` python
from field_matcher import FieldMatcher, Field, MatchResult, load_target_fields

# Load target fields from default JSON file
target_fields = load_target_fields()

# Or define your own target fields
target_fields = [
    Field(fieldHandle="amount", fieldLabel="Amount", fieldType="int"),
    Field(fieldHandle="count", fieldLabel="Item Count", fieldType="int"),
]

# Create input field (handle, label, type are required and validated)
input_field = Field(
    fieldHandle="qty",
    fieldLabel="Quantity",
    fieldType="int",
    fieldDescription="Number of items"  # optional
)

# Create matcher and find matches
matcher = FieldMatcher()
results: list[MatchResult] = matcher.match(input_field, target_fields, top_k=3)

# Access results
for result in results:
    print(f"{result.field.fieldHandle}: {result.score:.3f}")
```

------------------------------------------------------------------------

## Void Fragments

**Field Matcher** is part of [**Void Fragments**](https://github.com/OpsDevHub/void-fragments/tree/main/fragments) --- a catalog of isolated
capabilities extracted from Void Crystal and released as independent
building blocks. Each fragment solves one specific problem and nothing more.
