"""
Unit tests for MOF Synthesis Assistant.
"""

import unittest
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from schema import validate_mof_row, clean_mof_row, COLUMN_NAMES, VALID_PRESSURE_VALUES
from utils import extract_numerical_value, sanitize_input

class TestSchemaValidation(unittest.TestCase):
    """Test schema validation functions."""
    
    def test_validate_mof_row_valid(self):
        """Test validation of valid MOF row."""
        valid_row = {
            "Target": "MOF-5",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            "Linker(s) (mmol)": "H2BDC (0.5)",
            "Solvent(s) (mL)": "DMF (10)",
            "Modulator / Additive": "H2O (0.5)",
            "Temp (°C)": "120",
            "Time (h)": "24",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, chloroform"
        }
        self.assertTrue(validate_mof_row(valid_row))
    
    def test_validate_mof_row_invalid_pressure(self):
        """Test validation with invalid pressure."""
        invalid_row = {
            "Target": "MOF-5",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            "Linker(s) (mmol)": "H2BDC (0.5)",
            "Solvent(s) (mL)": "DMF (10)",
            "Modulator / Additive": "H2O (0.5)",
            "Temp (°C)": "120",
            "Time (h)": "24",
            "Pressure": "invalid_pressure",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, chloroform"
        }
        self.assertFalse(validate_mof_row(invalid_row))
    
    def test_validate_mof_row_missing_field(self):
        """Test validation with missing field."""
        incomplete_row = {
            "Target": "MOF-5",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            # Missing other required fields
        }
        self.assertFalse(validate_mof_row(incomplete_row))
    
    def test_clean_mof_row(self):
        """Test cleaning of MOF row."""
        dirty_row = {
            "Target": "  MOF-5  ",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            "Linker(s) (mmol)": "H2BDC (0.5)",
            "Solvent(s) (mL)": "DMF (10)",
            "Modulator / Additive": "H2O (0.5)",
            "Temp (°C)": "120",
            "Time (h)": "24",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, chloroform"
        }
        
        cleaned = clean_mof_row(dirty_row)
        self.assertEqual(cleaned["Target"], "MOF-5")
        self.assertEqual(len(cleaned), len(COLUMN_NAMES))

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_extract_numerical_value(self):
        """Test numerical value extraction."""
        self.assertEqual(extract_numerical_value("120"), 120.0)
        self.assertEqual(extract_numerical_value("120.5"), 120.5)
        self.assertEqual(extract_numerical_value("120°C"), 120.0)
        self.assertEqual(extract_numerical_value(""), None)
        self.assertEqual(extract_numerical_value("no numbers"), None)
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test API key removal
        text_with_key = "sk-1234567890abcdef"
        sanitized = sanitize_input(text_with_key)
        self.assertIn("[API_KEY]", sanitized)
        self.assertNotIn("sk-1234567890abcdef", sanitized)
        
        # Test length limiting
        long_text = "a" * 1000
        sanitized = sanitize_input(long_text)
        self.assertEqual(len(sanitized), 500)

if __name__ == '__main__':
    unittest.main()
