# 🚀 Streamlit Cloud Deployment Guide

This guide will help you securely deploy the MOF Synthesis Assistant to Streamlit Cloud.

## 📋 Prerequisites

1. A GitHub account
2. An OpenAI API key (get one at https://platform.openai.com/api-keys)
3. A Streamlit Cloud account (sign up at https://share.streamlit.io/)

## 🔐 Security Checklist

Before deploying, ensure:

- [ ] ✅ `.env` file is in `.gitignore` (already configured)
- [ ] ✅ `.streamlit/secrets.toml` is in `.gitignore` (already configured)
- [ ] ✅ No API keys are hardcoded in any Python files
- [ ] ✅ Repository is pushed to GitHub

## 📤 Deployment Steps

### 1. Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files (secrets will be excluded automatically)
git add .

# Commit
git commit -m "Initial commit - MOF Synthesis Assistant"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click **"New app"**
3. Select your repository: `YOUR-USERNAME/YOUR-REPO-NAME`
4. Set **Main file path**: `app/app.py`
5. Click **"Advanced settings"** (optional)
   - Python version: 3.10 or higher
6. Click **"Deploy"**

### 3. Configure Secrets (IMPORTANT!)

After deployment starts (it will fail initially without the API key):

1. Go to your app's dashboard on Streamlit Cloud
2. Click **"Settings"** (⚙️ icon)
3. Click **"Secrets"**
4. Paste the following (replace with your actual API key):

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

5. Click **"Save"**
6. The app will automatically redeploy with the secrets

### 4. Verify Deployment

1. Wait for the app to finish deploying (usually 2-5 minutes)
2. Open your app URL: `https://YOUR-APP-NAME.streamlit.app`
3. Check the sidebar - you should see "✅ OPENAI_API_KEY configured"
4. Test one of the tabs to ensure the OpenAI integration works

## 🔒 Security Best Practices

### API Key Management

- **NEVER** commit API keys to version control
- **NEVER** share your `.env` file or `secrets.toml` file
- **ALWAYS** use Streamlit secrets for cloud deployments
- **ROTATE** your API keys periodically

### Rate Limiting

The app includes basic input validation and sanitization:

- Text inputs are limited to reasonable lengths
- API keys are automatically masked in logs
- Environment validation prevents running without proper configuration

### Monitoring

Monitor your OpenAI usage:
1. Go to https://platform.openai.com/usage
2. Check your API usage regularly
3. Set up billing alerts if available

## 🛠️ Local Development

For local testing:

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## 🔄 Updating Your Deployment

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy your app.

## ⚠️ Troubleshooting

### "Missing OPENAI_API_KEY" Error

**Solution**: Add your API key to Streamlit Cloud Secrets (see Step 3 above)

### App Not Loading

**Solution**: 
1. Check the app logs in Streamlit Cloud dashboard
2. Verify `requirements.txt` has all dependencies
3. Ensure `app/app.py` is the correct main file path

### API Key Not Working

**Solution**:
1. Verify the API key is valid at https://platform.openai.com/api-keys
2. Check that there are no extra spaces in the secrets.toml file
3. Ensure the key starts with `sk-`

## 📊 Optional: ML Models

The ML prediction feature requires model files in the `models/` directory:
- `preproc.joblib`
- `xgb_temp.joblib`
- `xgb_time.joblib`
- `xgb_method.joblib`

These files are gitignored by default (large binary files). If you want prediction features:
1. Train the models locally using the notebooks
2. Upload them to a cloud storage (e.g., Google Drive, S3)
3. Download them in your app startup (see Streamlit documentation)

## 📞 Support

For issues:
- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- OpenAI API: https://platform.openai.com/docs
- This project: Open an issue on GitHub

## 🎉 Success!

Once deployed, your MOF Synthesis Assistant will be available at:
`https://YOUR-APP-NAME.streamlit.app`

Share the link with your team and start generating MOF synthesis protocols!
