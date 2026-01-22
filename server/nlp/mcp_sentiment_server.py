import sys
import json
import os
from dotenv import load_dotenv
from datetime import datetime

# Charger automatiquement le .env
load_dotenv()

# Configuration du logging
LOG_FILE = os.getenv("LOG_FILE")
MODEL_PATH = os.path.abspath(os.getenv("MODEL_PATH"))


def log(message):
    """Écrire dans le fichier log"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except:
        pass

log("=== Démarrage du serveur MCP ===")
log(f"Python: {sys.version}")
log(f"Executable: {sys.executable}")

try:
    log("Import torch...")
    import torch
    log(f"✓ torch {torch.__version__}")
    
    log("Import numpy...")
    import numpy as np
    log(f"✓ numpy {np.__version__}")
    
    log("Import re, unicodedata...")
    import re
    import unicodedata
    log("✓ re, unicodedata")
    
    log("Import transformers...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    log("✓ transformers")
    
    log(f"Chemin modèle: {MODEL_PATH}")
    log(f"Existe: {os.path.exists(MODEL_PATH)}")
    
    log("Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    log("✓ Tokenizer chargé")
    
    log("Chargement modèle...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    log("✓ Modèle chargé et en mode eval")
    
    LABELS = ["negative", "neutral", "positive"]
    
except Exception as e:
    log(f"!!! ERREUR lors de l'initialisation: {str(e)}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)


def clean_text(text: str) -> str:

    """Nettoyage du texte"""

    # RèGLE 1 : garantir que l'entrée est bien une chaîne de caractères
    if not isinstance(text, str):
        text = str(text)

    # RÈGLE 2 : suppression des balises HTML éventuelles
    text = re.sub(r'<[^>]+>', ' ', text)

    # RÈGLE 3 : suppression des caractères non linguistiques
    text = re.sub(r'[^\w\s.,!?]', ' ', text)

    # RÈGLE 4 : normalisation des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    # RÈGLE 5 : passage en minuscules (réduction de la variabilité lexicale)
    text = text.lower()

    # RÈGLE 6 : suppression des accents (normalisation Unicode)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    return text


def analyze_sentiment(text: str) -> dict:
    """Analyse de sentiment"""
    log(f"Analyse sentiment pour: {str(text)[:50]}...")

    
    # Application des règles de nettoyage
    text = clean_text(text)

    # RÈGLE DE VALIDATION : refuser un texte vide après nettoyage
    if not text or not isinstance(text, str):
        raise ValueError("Text cleaning failed: empty or invalid result")

    # Tokenisation complète du texte pour mesurer sa longueur réelle
    full_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    # RÈGLE STRUCTURELLE : longueur maximale supportée par le modèle
    max_len = 512

    if len(full_tokens) <= max_len:
        # RÈGLE : le texte tient dans la limite du modèle
        # → analyse directe sans perte d'information
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len
        )
        with torch.no_grad():
            outputs = model(**inputs)

        # Transformation des logits en probabilités
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    else:
        # RÈGLE ANTI-BIAIS :
        # Le texte est découpé en segments pour éviter
        # la perte d'information due à la troncature
        chunks_scores = []

        # RÈGLE : pas de chevauchement (simplicité + traçabilité)
        step = max_len
        for i in range(0, len(full_tokens), step):
            # Extraction d'un segment de tokens
            chunk_tokens = full_tokens[i:i+step]

            # Reconstruction du texte correspondant au segment
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Analyse du sentiment pour chaque segment
            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len
            )
            with torch.no_grad():
                outputs = model(**inputs)

            # Conversion logits → probabilités
            chunk_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            chunks_scores.append(chunk_scores)

        # RÈGLE D’AGRÉGATION :
            # La prédiction finale est la moyenne des scores de tous les segments analysés
        scores = np.mean(chunks_scores, axis=0)

    # RÈGLE DE DÉCISION :
        # la classe ayant la probabilité maximale est sélectionnée
    idx = int(np.argmax(scores))
    result = {
        "sentiment": LABELS[idx],
        # RÈGLE DE PRÉSENTATION :
            # arrondi pour améliorer la lisibilité
        "confidence": round(float(scores[idx]), 3)
    }
    
    log(f"Résultat: {result}")
    return result

# Gestionnaire principal des requêtes MCP (Model Context Protocol).
def handle_mcp_request(request: dict) -> dict:

    """Gestionnaire des requêtes MCP"""
    # RÈGLE DE ROUTAGE :
        # Chaque requête MCP doit spécifier une méthode
    method = request.get("method")
    log(f"Méthode appelée: {method}")
    
    # =============================
    # MÉTHODE 1 : initialize
    # =============================
    # RÈGLE PROTOCOLE MCP :
        # Cette méthode est appelée au démarrage pour :
            # - annoncer la version du protocole supportée
            # - déclarer les capacités du serveur
            # - fournir les métadonnées du serveur
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                # RÈGLE : déclaration explicite des outils disponibles
                "tools": {}
            },
            "serverInfo": {
                "name": "sentiment-analyzer",
                "version": "1.0.0"
            }
        }
    
    # =============================
    # MÉTHODE 2 : tools/list
    # =============================
    # RÈGLE :
        # Le client peut demander la liste des outils exposés
        # par le serveur MCP
    elif method == "tools/list":
        return {
            "tools": [
                {
                    # Nom unique de l'outil
                    "name": "analyze_sentiment",

                    # Description textuelle destinée au client
                    "description": "Analyze sentiment of French text (positive, neutral, negative)",
                    
                    # SCHÉMA D'ENTRÉE (contrat strict)
                        # RÈGLE : toute requête doit respecter ce schéma
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            ]
        }
    
    # =============================
    # MÉTHODE 3 : tools/call
    # =============================
    # RÈGLE :
        # Permet l'appel effectif d'un outil déclaré précédemment
    elif method == "tools/call":

        # RÈGLE DE SÉLECTION :
            # le nom de l'outil doit correspondre à un outil exposé
        tool_name = request["params"]["name"]


        if tool_name == "analyze_sentiment":
            # Extraction contrôlée de l'argument requis
            text = request["params"]["arguments"]["text"]

             # Appel de la fonction métier (NLP)
            result = analyze_sentiment(text)

            # RÈGLE MCP :
                # la réponse doit être encapsulée dans un tableau "content"
            return {
                "content": [
                    {
                        "type": "text",

                        # Sérialisation JSON pour compatibilité protocolaire
                        "text": json.dumps(result, ensure_ascii=False)
                    }
                ]
            }
    # RÈGLE DE SÉCURITÉ :
        # si la méthode n'est pas reconnue, on retourne une réponse vide
    return {}


def main():
    """
    Boucle principale du serveur MCP.

    FONCTIONNEMENT :
    - Lecture des requêtes depuis stdin (entrée standard)
    - Traitement séquentiel des requêtes JSON-RPC
    - Envoi des réponses sur stdout
    """

    
    log("Démarrage de la boucle principale")
    
    try:

        # RÈGLE MCP :
        # Le serveur reste actif tant qu'il reçoit des requêtes
        for line in sys.stdin:
            try:
                # Journalisation de la requête reçue (limite à 100 caractères)
                log(f"Requête reçue: {line.strip()[:100]}")

                # Désérialisation JSON
                request = json.loads(line)

                # Traitement de la requête
                response = handle_mcp_request(request)
                
                # RÈGLE JSON-RPC :
                    # chaque réponse doit inclure :
                        # - jsonrpc version
                        # - id de la requête
                        # - result
                output = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": response
                }
                
                # Sérialisation de la réponse
                output_str = json.dumps(output)

                # Envoi de la réponse au client
                print(output_str, flush=True)
                log(f"Réponse envoyée: {output_str[:100]}")
                
            except Exception as e:
                # GESTION DES ERREURS D'EXÉCUTION
                log(f"Erreur dans la boucle: {str(e)}")
                import traceback
                log(traceback.format_exc())
                
                # RÈGLE JSON-RPC :
                # Les erreurs doivent être renvoyées avec un code standard
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if 'request' in locals() else None,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)
                
    except Exception as e:
        # ERREUR FATALE : arrêt du serveur
        log(f"Erreur fatale: {str(e)}")
        import traceback
        log(traceback.format_exc())


if __name__ == "__main__":
    log("Appel de main()")
    main()