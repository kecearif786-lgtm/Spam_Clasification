# Spam Detector AI — Shared Cloud History

A Streamlit application for the existing spam-classification model. This version stores the analysis history in a shared Supabase database, so every visitor sees the same records and the data remains available after Streamlit Community Cloud restarts.

The original model artifacts are used as-is:

- `spam_model.pkl` — trained classifier (Multinomial Naive Bayes)
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer

**Do not retrain, rename, overwrite, or modify these two files.** Put them in the same directory as `app.py`.

## Features

- Dashboard statistics and charts based on the global Supabase history.
- Manual text classification using the original TF-IDF and Naive Bayes artifacts.
- Screenshot OCR with Tesseract Indonesian and English (`ind+eng`).
- Image preprocessing and editable OCR output.
- Prediction confidence when the existing model supports `predict_proba()`.
- Shared persistent history, search, SPAM/HAM filter, CSV export, and refresh.
- Optional admin-password-protected action to clear the entire shared history.
- Privacy acknowledgement before a text or OCR result is added to the shared database.

## Important privacy behavior

This deployment is designed as a **shared public application without individual user accounts**:

- Every visitor can see the shared analysis history, including message text, classification, source, confidence, and timestamp.
- New analyses are saved to Supabase. The uploaded screenshot image itself is **not** stored in the database; only the corrected OCR text and its result are stored when the user chooses to analyze it.
- The app asks users to acknowledge that their message text will be visible in the shared history before it saves a result.
- Do not enter OTPs, PINs, passwords, payment details, private conversations, or other sensitive/personal information.
- The Supabase Secret key remains server-side in Streamlit Secrets. It must never be placed in the repository or browser-facing code.
- Global history deletion is disabled unless `history_admin_password` is configured. Only people who know that password can use the delete control inside the app.

There is no per-user separation or private history in this configuration. Add Supabase Auth and user-based Row Level Security before using the app for private or sensitive user data.

## 1. Folder structure

```text
spam_detector_ai/
├── app.py
├── requirements.txt
├── packages.txt
├── spam_model.pkl                 # your original file
├── tfidf_vectorizer.pkl           # your original file
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example       # template only; not a real secret file
└── supabase/
    └── schema.sql
```

The `.pkl` files are not included in this package because you must use your original trained artifacts.

## 2. Create the Supabase database

1. Open [Supabase](https://supabase.com/) and create a project (or use an existing project).
2. Open **SQL Editor** in the Supabase dashboard.
3. Open `supabase/schema.sql` from this project, copy the complete SQL, paste it into the SQL Editor, and run it once.
4. Confirm that the table `public.spam_analysis_history` appears in **Table Editor**.
5. In **Project Settings → API Keys**, copy the project's **Secret key** (`sb_secret_...`). If your project only shows the legacy `service_role` key, the app accepts that key too.
6. Copy the Project URL from the Supabase project settings.

The SQL script enables Row Level Security (RLS), removes table access from public `anon` and `authenticated` roles, and grants the server-side `service_role` permission to read, insert, and delete rows. The Streamlit server uses the Supabase Secret key, which bypasses RLS. Keep this key private and only store it in Streamlit Secrets.

## 3. Configure Streamlit Community Cloud

Your app is already deployed on Streamlit Community Cloud. Update the files in the GitHub repository connected to that app:

1. Replace the existing `app.py` with this version.
2. Replace `requirements.txt` and add `packages.txt`.
3. Add the `supabase/` folder and `.streamlit/secrets.toml.example` (the example file itself contains placeholders only).
4. Keep your original `spam_model.pkl` and `tfidf_vectorizer.pkl` in the repository or app directory, unchanged.
5. Push/commit the changes to the connected GitHub branch. Community Cloud will detect the code and dependency changes and rebuild the app.
6. Open the deployed app, select **Manage app → Settings → Secrets**, then paste the following TOML with your real values:

```toml
[supabase]
url = "https://YOUR_PROJECT_REF.supabase.co"
secret_key = "sb_secret_YOUR_REAL_SECRET_KEY"

[app]
history_admin_password = "YOUR_LONG_RANDOM_ADMIN_PASSWORD"
```

Use the URL and Secret key from your own Supabase project. Use a long, randomly generated admin password (at least 32 characters). The `[app]` section is optional; omit it to keep the global delete action disabled.

7. Click **Save** in Streamlit Secrets. The app will restart/reload with the new settings.
8. Open **History** in the deployed app. The sidebar should show **Shared history connected**. Run one test analysis and verify that a row appears in Supabase Table Editor and remains visible after refreshing/restarting the app.

Streamlit Community Cloud stores app secrets separately from the GitHub repository. **Never commit `.streamlit/secrets.toml`**. The included `.gitignore` excludes it. For the official Streamlit instructions, see [Community Cloud secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) and [app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## 4. Local development

Copy the example secrets file and edit it with your own Supabase credentials:

### Windows PowerShell

```powershell
cd path\to\spam_detector_ai
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd /path/to/spam_detector_ai
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Edit `.streamlit/secrets.toml` and replace the placeholder URL and keys before launching the app. Do not upload this real secrets file to GitHub.

## 5. Tesseract OCR on Streamlit Community Cloud

The included `packages.txt` tells Community Cloud to install the Tesseract executable and the Indonesian/English language data:

```text
tesseract-ocr
tesseract-ocr-eng
tesseract-ocr-ind
```

The `pytesseract` Python package is only a wrapper; the native executable and language packs are installed through `packages.txt` on Community Cloud. On a local Ubuntu/Debian machine, install them with:

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind
```

Verify the language packs with:

```bash
tesseract --list-langs
```

The output should include `ind` and `eng`.

## 6. Model compatibility

Use the same Python and scikit-learn version that was used to train and serialize your existing model files whenever possible. Pickle/joblib artifacts can be incompatible across scikit-learn versions. If Community Cloud shows a model-loading error after deployment, check the training environment and pin the matching `scikit-learn` version in `requirements.txt` rather than retraining or modifying the `.pkl` files.

## 7. History behavior and troubleshooting

- `spam_analysis_history` is the single shared table for all visitors.
- History is fetched in pages of 500 rows and cached for 20 seconds to reduce repeated database requests. Use **Refresh shared history** on the History page to force a refresh.
- Dashboard statistics are calculated from the rows fetched from Supabase.
- CSV export includes the shared history currently available to the app.
- New analysis records use the database timestamp and are displayed in WIB (`Asia/Jakarta`).
- If predictions work but saving fails, check that the SQL script ran successfully, `SUPABASE_URL` is correct, the `sb_secret_...` key (or legacy `service_role` key) belongs to the same project, and Streamlit Secrets was saved.
- If the sidebar says **Database belum terhubung**, check the app logs and confirm the `[supabase]` section is present in Streamlit Cloud Secrets.
- If history is empty immediately after deploying, this is expected: the prior implementation stored history only in browser session memory. It did not have a permanent database to migrate from. New analyses will persist after the Supabase setup is complete.

## Notes

- Original label mapping remains supported: numeric `1` = SPAM and `0` = HAM; string labels `SPAM` and `HAM` are also supported.
- Confidence is shown only if the existing model exposes `predict_proba()`.
- This application predicts classifications; it does not block messages, scan URLs, or guarantee that a message is safe.
