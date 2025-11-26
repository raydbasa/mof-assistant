# MOF Synthesis Assistant 

A minimal-viable product (MVP) for extracting, predicting, and suggesting Metal-Organic Framework (MOF) synthesis protocols using AI and machine learning.

## Features

- ⚗️ **Complete**: Generate full synthesis protocols from minimal inputs (metal + linker)
- 🔍 **Extract**: Extract structured synthesis data from free text using OpenAI's Structured Outputs
- 🔮 **Predict**: Predict missing synthesis parameters using XGBoost ML models
- 💡 **Suggest**: Generate complete synthesis protocols from partial cues
- 📊 **Export**: Download results as CSV or JSON 

## Project Structure

```
mof-assistant/
├── app/
│   ├── app.py              # Main Streamlit application
│   ├── schema.py           # JSON schema for MOF data
│   ├── prompts.py          # AI prompt templates
│   └── utils.py            # Utility functions
├── data/
│   ├── raw/                # Raw data (placeholder)
│   ├── interim/            # Intermediate data (placeholder)
│   └── processed/
│       └── mof_runs.csv    # Sample MOF synthesis dataset
├── models/                 # Trained ML models (joblib files) 
├── notebooks/
│   ├── 01_unify_data.ipynb # Data preparation notebook
│   ├── 02_train_tabular.ipynb # Model training notebook
│   └── 03_eval_tabular.ipynb # Model evaluation notebook
├── evaluation/
│   └── prompts/            # Sample prompts for testing
├── requirements.txt         # Python dependencies
├── env.example            # Environment variables template
└── README.md              # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your API key
# Required: OPENAI_API_KEY
```

### 3. Prepare Sample Data and Models

```bash
# Run data preparation notebook
jupyter notebook notebooks/01_unify_data.ipynb

# Train ML models
jupyter notebook notebooks/02_train_tabular.ipynb
```

### 4. Launch Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

### Complete Tab
- Input metal precursor and linker (minimal requirements)
- Optionally add hints like solvent, pressure, method
- Click "Generate Complete Protocol" to get a full synthesis protocol
- Download results as CSV or JSON

### Extract Tab
- Paste MOF synthesis text descriptions
- Click "Extract Row" to get structured data
- Download results as CSV or JSON

### Predict Tab
- Input known synthesis parameters
- Select which fields to predict (Temperature, Time, Method)
- Click "Predict Missing Fields" to get ML predictions
- Download complete synthesis protocols

### Suggest Tab
- Enter synthesis cues (e.g., "Gold-based MOF, room temperature")
- Click "Generate Protocol" to get complete synthesis suggestions
- Download generated protocols

## MOF Data Schema

The application uses a standardized schema with these fields:

- **Target**: MOF name or identifier
- **Metal source (mmol)**: Metal compound and amount
- **Linker(s) (mmol)**: Organic linker(s) and amount(s)
- **Solvent(s) (mL)**: Solvent(s) and volume(s)
- **Modulator / Additive**: Modulator or additive compounds
- **Temp (°C)**: Temperature in degrees Celsius
- **Time (h)**: Reaction time in hours
- **Pressure**: One of: "", "ambient", "autogenous", "solvothermal", "hydrothermal", "microwave", "other"
- **Method**: Synthesis method
- **Wash / Activation**: Washing and activation procedure

## Machine Learning Models

The application includes three XGBoost models:

1. **Temperature Model**: Predicts synthesis temperature (°C)
2. **Time Model**: Predicts reaction time (hours)
3. **Method Model**: Predicts synthesis method (classification)

Models are trained on the sample dataset and saved as joblib files in the `models/` directory.

## Development

### Training New Models

1. **Prepare Data**: Use `notebooks/01_unify_data.ipynb` to clean and standardize your dataset
2. **Train Models**: Run `notebooks/02_train_tabular.ipynb` to train new models
3. **Evaluate**: Use `notebooks/03_eval_tabular.ipynb` to assess model performance
4. **Deploy**: Copy new model files to the `models/` directory

### Adding New Features

- **New Tabs**: Add new tabs in `app/app.py`
- **New Models**: Extend the ML pipeline in the training notebooks
- **New Schemas**: Update `app/schema.py` and retrain models
- **New Prompts**: Modify `app/prompts.py` for different AI behaviors

## Privacy and Cost Control

- **Local Processing**: All ML predictions run locally
- **API Calls**: Only extraction, suggestion, and completion use OpenAI API
- **Caching**: Consider implementing caching for repeated queries
- **Environment Variables**: API keys stored securely in .env file

## Troubleshooting

### Common Issues

1. **Missing Models**: Run the training notebooks first
2. **API Errors**: Check your OpenAI API key in `.env`
3. **Import Errors**: Ensure all dependencies are installed
4. **Port Conflicts**: Streamlit runs on port 8501 by default

### Getting Help

- Check the Streamlit logs for error messages
- Review the notebook outputs for model training issues
- Ensure all environment variables are correctly set

## License

This project is provided as-is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- OpenAI for the GPT API and Structured Outputs
- XGBoost for machine learning capabilities
- Streamlit for the web interface
- scikit-learn for preprocessing utilities
