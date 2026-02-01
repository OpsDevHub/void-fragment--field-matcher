"""
Tests for the Field Matcher module.

Run with: python run_tests.py
"""

import pytest
from field_matcher import Field, FieldMatcher, load_target_fields


def test_sales_description_matches_product_description():
    """Test that salesDescription input matches productDescription as top result."""
    # Load default target fields from JSON
    target_fields = load_target_fields()

    # Create input field to match
    input_field = Field(
        fieldHandle="salesDescription",
        fieldLabel="Sales Description",
        fieldType="string",
        fieldDescription="Sales oriented description about the product",
    )

    # Create matcher and find matches
    field_matcher = FieldMatcher()
    match_results = field_matcher.match(input_field, target_fields, top_k=3)

    # Assert we got results and top match is productDescription
    assert len(match_results) > 0, "Expected at least one match result"
    assert match_results[0].field.fieldHandle == "productDescription", (
        f"Expected 'productDescription' as top match, got '{match_results[0].field.fieldHandle}'"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
