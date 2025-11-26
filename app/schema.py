"""
JSON Schema for MOF synthesis data extraction and validation.
"""

from typing import Dict, Any

# Core MOF synthesis schema matching the exact column names specified
TABLE_SCHEMA: Dict[str, Any] = {
    "name": "mof_row",
    "schema": {
        "type": "object",
        "properties": {
            "Target": {
                "type": "string",
                "description": "Target MOF name or identifier"
            },
            "Metal source (mmol)": {
                "type": "string",
                "description": "Metal source and amount in mmol"
            },
            "Linker(s) (mmol)": {
                "type": "string", 
                "description": "Organic linker(s) and amount(s) in mmol"
            },
            "Solvent(s) (mL)": {
                "type": "string",
                "description": "Solvent(s) and volume(s) in mL"
            },
            "Modulator / Additive": {
                "type": "string",
                "description": "Modulator or additive compounds"
            },
            "Temp (°C)": {
                "type": "string",
                "description": "Temperature in degrees Celsius"
            },
            "Time (h)": {
                "type": "string", 
                "description": "Reaction time in hours"
            },
            "Pressure": {
                "type": "string",
                "enum": ["", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"],
                "description": "Pressure conditions"
            },
            "Method": {
                "type": "string",
                "description": "Synthesis method"
            },
            "Wash / Activation": {
                "type": "string",
                "description": "Washing and activation procedure"
            }
        },
        "required": [
            "Target", "Metal source (mmol)", "Linker(s) (mmol)", 
            "Solvent(s) (mL)", "Modulator / Additive", "Temp (°C)",
            "Time (h)", "Pressure", "Method", "Wash / Activation"
        ],
        "additionalProperties": False
    },
    "strict": True
}

# Response format for OpenAI Structured Outputs
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": TABLE_SCHEMA
}

# Valid pressure values for validation
VALID_PRESSURE_VALUES = ["", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"]

# Column names for easy reference
COLUMN_NAMES = [
    "Target", "Metal source (mmol)", "Linker(s) (mmol)", 
    "Solvent(s) (mL)", "Modulator / Additive", "Temp (°C)",
    "Time (h)", "Pressure", "Method", "Wash / Activation"
]

def validate_mof_row(row: Dict[str, Any]) -> bool:
    """
    Validate that a MOF row conforms to the schema.
    
    Args:
        row: Dictionary containing MOF synthesis data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(row, dict):
        return False
    
    # Check all required fields are present
    for field in COLUMN_NAMES:
        if field not in row:
            return False
    
    # Check pressure value is valid
    if row.get("Pressure") not in VALID_PRESSURE_VALUES:
        return False
    
    return True

def clean_mof_row(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Clean and normalize a MOF row, ensuring all values are strings.
    
    Args:
        row: Raw MOF synthesis data
        
    Returns:
        Dict[str, str]: Cleaned row with string values
    """
    cleaned = {}
    for field in COLUMN_NAMES:
        value = row.get(field, "")
        # Convert to string and strip whitespace
        cleaned[field] = str(value).strip() if value is not None else ""
    
    return cleaned
