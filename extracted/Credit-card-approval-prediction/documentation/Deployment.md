# Deployment Guide

## Local Deployment

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train the model.

```bash
python -m models.train
```

3. Start Flask.

```bash
python app.py
```

## IBM Cloud / Watson Ready Deployment

This project includes:

- `requirements.txt`
- `runtime.txt`
- `Procfile`
- `manifest.yml`

### Cloud Foundry Deployment

1. Log in to IBM Cloud.

```bash
ibmcloud login
```

2. Target an organization and space.

```bash
ibmcloud target --cf
```

3. Push the app.

```bash
ibmcloud cf push
```

The app starts with:

```bash
gunicorn app:app
```

## Deployment Checklist

- Verify `dataset/` contains both CSV files before training.
- Run `python -m models.train` before deploying or include saved artifacts.
- Set a strong `SECRET_KEY` environment variable in production.
- Keep raw datasets private if they contain sensitive or licensed data.
- Confirm `models/saved/` contains trained artifacts.

