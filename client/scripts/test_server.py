import sys
from dotenv import load_dotenv
import os
from server.nlp.mcp_sentiment_server import analyze_sentiment


# Charger automatiquement le .env
load_dotenv()

# Test d'écriture dans le log
log_file = os.getenv("LOG_FILE")

try:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Test de démarrage ===\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Path: {sys.executable}\n")
        
        # Test imports
        f.write("\nTest imports...\n")
        import torch
        f.write(f"✓ torch {torch.__version__}\n")
        
        import transformers
        f.write(f"✓ transformers {transformers.__version__}\n")
        
        import numpy as np
        f.write(f"✓ numpy {np.__version__}\n")
        
        # Test chargement modèle
        f.write("\nTest chargement modèle...\n")
        MODEL_PATH = r"C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\models\twitter-xlm-roberta"
        f.write(f"Modèle path: {MODEL_PATH}\n")
        f.write(f"Existe: {os.path.exists(MODEL_PATH)}\n")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        f.write("✓ Tokenizer chargé\n")
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        f.write("✓ Modèle chargé\n")
        
        f.write("\n=== SUCCÈS ===\n")
        
except Exception as e:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n!!! ERREUR: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())