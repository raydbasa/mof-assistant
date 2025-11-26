"""
Prompt templates for MOF synthesis data extraction and suggestion.
"""

# System prompt for MOF extraction
SYSTEM_PROMPT = """You are a chemistry assistant specialized in Metal-Organic Framework (MOF) synthesis. 
Your task is to extract structured synthesis information from free text descriptions.

Always return only a single JSON object matching the provided schema. 
If a field is missing in the text, set it to an empty string "".
Be precise and extract only information that is explicitly mentioned in the text.
For numerical values, preserve the original units and format."""

# Extraction prompt template
EXTRACTION_PROMPT = """Extract MOF synthesis information from the following text and return it as a structured JSON object.

Text to analyze:
{text}

Return a JSON object with the following fields:
- Target: MOF name or identifier
- Metal source (mmol): Metal compound and amount in mmol
- Linker(s) (mmol): Organic linker(s) and amount(s) in mmol  
- Solvent(s) (mL): Solvent(s) and volume(s) in mL
- Modulator / Additive: Any modulator or additive compounds
- Temp (°C): Temperature in degrees Celsius
- Time (h): Reaction time in hours
- Pressure: One of: "", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"
- Method: Synthesis method used
- Wash / Activation: Washing and activation procedure

If any information is not mentioned, use an empty string ""."""

# Suggestion prompt template  
SUGGESTION_PROMPT = """Based on the following cues, suggest a complete MOF synthesis protocol.

Cues:
{cues}

Generate a complete synthesis protocol as a JSON object with the following fields:
- Target: Suggested MOF name
- Metal source (mmol): Metal compound and amount in mmol
- Linker(s) (mmol): Organic linker(s) and amount(s) in mmol
- Solvent(s) (mL): Solvent(s) and volume(s) in mL
- Modulator / Additive: Modulator or additive compounds
- Temp (°C): Temperature in degrees Celsius
- Time (h): Reaction time in hours
- Pressure: One of: "", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"
- Method: Synthesis method
- Wash / Activation: Washing and activation procedure

Provide realistic values based on common MOF synthesis practices. 
If specific information is not provided in the cues, make reasonable assumptions."""

def format_extraction_prompt(text: str) -> str:
    """
    Format the extraction prompt with the input text.
    
    Args:
        text: Input text to extract MOF synthesis data from
        
    Returns:
        str: Formatted prompt
    """
    return EXTRACTION_PROMPT.format(text=text)

def format_suggestion_prompt(cues: str) -> str:
    """
    Format the suggestion prompt with the input cues.
    
    Args:
        cues: Input cues for MOF synthesis suggestion
        
    Returns:
        str: Formatted prompt
    """
    return SUGGESTION_PROMPT.format(cues=cues)
