# PhishGuard — Banking URL Phishing Detection

Streamlit deployment for the thesis Random Forest model that classifies banking-related URLs as Legitimate or Phishing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `phishing_rf_model.pkl`, and `requirements.txt`.
3. In Streamlit Community Cloud, create a new app.
4. Select the repository and branch.
5. Set the main file to `app.py`.
6. Deploy.
