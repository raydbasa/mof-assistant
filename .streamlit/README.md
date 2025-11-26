# Streamlit Configuration

This directory contains Streamlit-specific configuration files.

## Files

### `config.toml`
Non-sensitive Streamlit configuration (theme, server settings, etc.). This file is safe to commit to version control.

### `secrets.toml` (NOT included)
**CRITICAL**: This file contains your API keys and should NEVER be committed to git.

For local development:
1. Copy `secrets.toml.example` to `secrets.toml`
2. Add your actual API key
3. Verify it's gitignored: `git status --ignored`

For Streamlit Cloud:
1. Go to your app dashboard at share.streamlit.io
2. Click Settings → Secrets
3. Paste your secrets there

### `secrets.toml.example`
Template showing the required secrets format. Safe to commit.

## Quick Setup

**Local:**
```bash
cd .streamlit
cp secrets.toml.example secrets.toml
# Edit secrets.toml with your API key
```

**Cloud:**
```toml
# In Streamlit Cloud Settings → Secrets, add:
OPENAI_API_KEY = "sk-your-key-here"
```

## Security Notes

✅ DO:
- Use secrets.toml for local development
- Use Streamlit Cloud Secrets for production
- Keep secrets.toml gitignored
- Rotate keys regularly

❌ DON'T:
- Commit secrets.toml to git
- Share your secrets.toml file
- Hardcode API keys in code
- Use production keys in development

## Troubleshooting

**"Missing OPENAI_API_KEY" error:**
- Local: Ensure secrets.toml exists with your key
- Cloud: Add key in Streamlit Cloud Secrets

**Key not working:**
- Verify format: `OPENAI_API_KEY = "sk-..."`
- Check for extra spaces or quotes
- Ensure key is valid at platform.openai.com
