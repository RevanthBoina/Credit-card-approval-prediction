# Deploying to IBM Cloud

This Flask app can be deployed to IBM Cloud two ways. **Code Engine (Option A)**
is recommended because the container image handles the heavy ML stack
(xgboost, scikit-learn, scipy) more reliably than the Cloud Foundry buildpack.

Prerequisites:
- An IBM Cloud account and the IBM Cloud CLI: https://cloud.ibm.com/docs/cli
- Log in: `ibmcloud login` (add `--sso` if your org uses SSO)

---

## Option A — Code Engine (container, recommended)

Uses the included `Dockerfile`.

```bash
# 1. Install the Code Engine plugin
ibmcloud plugin install code-engine

# 2. Target a resource group and region (example)
ibmcloud target -g Default -r us-south

# 3. Create (or select) a project
ibmcloud ce project create --name credit-card-approval
ibmcloud ce project select --name credit-card-approval

# 4. Deploy straight from source — Code Engine builds the Dockerfile for you
ibmcloud ce application create \
  --name credit-card-approval \
  --build-source . \
  --port 8080 \
  --cpu 1 --memory 2G \
  --env STORAGE_DIR=/tmp/storage \
  --env SECRET_KEY="$(openssl rand -hex 32)"

# 5. Get the public URL
ibmcloud ce application get --name credit-card-approval --output url
```

To redeploy after changes: `ibmcloud ce application update --name credit-card-approval --build-source .`

---

## Option B — Cloud Foundry (buildpack)

Uses the included `manifest.yml`, `Procfile`, and `runtime.txt`.

```bash
# 1. Target your CF org and space
ibmcloud target --cf

# 2. Push (reads manifest.yml)
ibmcloud cf push

# 3. Set a real secret key
ibmcloud cf set-env credit-card-approval-prediction SECRET_KEY "$(openssl rand -hex 32)"
ibmcloud cf restage credit-card-approval-prediction
```

> Note: Cloud Foundry apps need enough memory/disk for the ML libraries.
> `manifest.yml` requests 1G memory / 2G disk. Increase if a push fails on size.

---

## Environment variables

| Variable      | Purpose                                              | Recommended value    |
| ------------- | ---------------------------------------------------- | -------------------- |
| `PORT`        | Port the server binds to (injected by IBM Cloud)     | auto                 |
| `STORAGE_DIR` | Writable dir for prediction history                  | `/tmp/storage`       |
| `SECRET_KEY`  | Flask session secret                                 | a long random string |
| `FLASK_DEBUG` | Enable debug mode locally (never in production)      | unset / `0`          |

## Local run

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
# or production-style:
gunicorn --bind 0.0.0.0:5000 app:app
```
