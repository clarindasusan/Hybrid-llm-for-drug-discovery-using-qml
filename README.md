---
title: Drug Predictor API
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 💊 Drug Predictor API

A quantum machine learning model for predicting drug-likeness from SMILES strings.

## 🚀 Quick Start

### API Documentation
Visit `/docs` for interactive API documentation (Swagger UI).

### Health Check
```bash
GET /
GET /health
```

### Predict Drug Potential
```bash
POST /predict
```

**Request Body:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

**Response:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "score": 0.8542,
  "is_promising": true,
  "confidence": "high"
}
```

## 📝 Example Usage

### Python
```python
import requests

API_URL = "https://huggingface.co/spaces/YOUR-USERNAME/drug-predictor-api"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Predict
response = requests.post(
    f"{API_URL}/predict",
    json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}  # Aspirin
)
print(response.json())
```

### cURL
```bash
curl -X POST "YOUR-SPACE-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### JavaScript
```javascript
const response = await fetch('YOUR-SPACE-URL/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({smiles: 'CCO'})
});
const data = await response.json();
console.log(data);
```

## 🧪 Test Molecules

- **Aspirin:** `CC(=O)OC1=CC=CC=C1C(=O)O`
- **Caffeine:** `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- **Ibuprofen:** `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
- **Paracetamol:** `CC(=O)NC1=CC=C(C=C1)O`

## 🔧 Model Architecture

This API uses a hybrid quantum-classical machine learning model trained on molecular fingerprints and descriptors to predict drug-likeness.

**Features:**
- Morgan fingerprints (2048 bits)
- Molecular descriptors (5 features)
- Total: 2053 input features

## 📊 Response Fields

- `smiles`: The input SMILES string
- `score`: Probability score (0-1) indicating drug-likeness
- `is_promising`: Boolean indicating if score >= 0.5
- `confidence`: "low", "medium", or "high" based on score distance from threshold

## ⚠️ Limitations

- Only accepts valid SMILES strings
- Predictions are based on structural features only
- Not a substitute for actual drug development processes