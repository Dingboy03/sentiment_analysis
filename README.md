# 🎯 Système d'Analyse de Sentiment avec MCP (Model Context Protocol)

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du système](#architecture-du-système)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [API Reference](#api-reference)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Dépannage](#dépannage)
- [Contribution](#contribution)

---

## 🌟 Vue d'ensemble

Ce projet implémente un **système complet d'analyse de sentiment** pour textes en français, basé sur le modèle **XLM-RoBERTa** fine-tuné sur Twitter. Le système propose trois interfaces d'utilisation :

1. **Serveur MCP** : Intégration directe avec Claude Desktop via le Model Context Protocol
2. **API REST FastAPI** : Service HTTP pour applications web/mobiles
3. **Notebooks Jupyter** : Analyse batch et expérimentation

### 🎯 Cas d'usage principaux

- Analyse de sentiment de commentaires sur réseaux sociaux
- Évaluation d'articles et leurs commentaires associés
- Détection d'opinions (positive, neutre, négative) avec score de confiance
- Intégration IA conversationnelle (Claude Desktop)

### 🔑 Caractéristiques clés

✅ Support de textes longs (découpage automatique en chunks)  
✅ Nettoyage et normalisation avancés du texte  
✅ Logging détaillé pour debugging  
✅ API REST avec documentation Swagger automatique  
✅ Intégration MCP pour Claude Desktop  
✅ Analyse batch via notebooks Jupyter  

---

## 🏗️ Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACES UTILISATEUR                    │
├──────────────────┬──────────────────┬──────────────────────┤
│  Claude Desktop  │   API FastAPI    │  Jupyter Notebooks   │
│   (MCP Client)   │  (HTTP REST)     │   (Batch Analysis)   │
└────────┬─────────┴────────┬─────────┴──────────┬───────────┘
         │                  │                    │
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              COUCHE TRAITEMENT (Core Engine)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         mcp_sentiment_server.py                      │   │
│  │  • clean_text() : Nettoyage & normalisation          │   │
│  │  • analyze_sentiment() : Analyse NLP                 │   │
│  │  • handle_mcp_request() : Routage MCP                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  MODÈLE NLP (Deep Learning)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  twitter-xlm-roberta-base (XLM-RoBERTa)             │   │
│  │  • Tokenizer : AutoTokenizer                         │   │
│  │  • Model : AutoModelForSequenceClassification        │   │
│  │  • Classes : [negative, neutral, positive]           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 🔄 Flux de données

```
Texte brut → Nettoyage → Tokenisation → Découpage (si >512 tokens)
  ↓
Analyse par chunks → Agrégation → Softmax → Prédiction finale
  ↓
{sentiment: "positive", confidence: 0.92}
```

---

## 📦 Prérequis

### Système d'exploitation
- **Windows** 10/11 (adapté, mais portable sur Linux/macOS)

### Logiciels
- **Python** 3.9.25 (recommandé, testé avec cette version)
- **Conda** ou **Miniconda** (gestion d'environnement)
- **Claude Desktop** (optionnel, pour l'interface MCP)

### Hardware recommandé
- **RAM** : 8 GB minimum (16 GB recommandé)
- **Espace disque** : 2 GB pour le modèle + dépendances
- **CPU** : Processeur multi-cœurs (le modèle tourne en CPU par défaut)

---

## 🚀 Installation

### 1️⃣ Cloner le projet

```bash
git clone <votre-repo>
cd NLP/MCP
```

### 2️⃣ Créer l'environnement Conda

#### Option A : Depuis le fichier `environment.yml`

```bash
conda env create -f environment.yml
conda activate nlp
```

#### Option B : Installation manuelle

```bash
# Créer l'environnement
conda create -n nlp python=3.9.25

# Activer l'environnement
conda activate nlp

# Installer les dépendances principales
pip install torch==2.8.0 transformers==4.57.3 numpy==2.0.2

# Installer les dépendances API
pip install fastapi==0.128.0 uvicorn==0.39.0 pydantic==2.12.5

# Installer les dépendances notebooks
pip install jupyter ipykernel pandas matplotlib seaborn

# Autres dépendances utiles
pip install requests tqdm nltk
```

### 3️⃣ Télécharger le modèle

Le modèle doit être placé dans :
```
C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\models\twitter-xlm-roberta
```

**Téléchargement depuis Hugging Face** :

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SAVE_PATH = r"C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\models\twitter-xlm-roberta"

# Télécharger et sauvegarder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)
```

### 4️⃣ Vérifier l'installation

Exécutez le script de test :

```bash
python test_server.py
```

Vérifiez le fichier `mcp_server.log` - vous devriez voir :
```
=== SUCCÈS ===
```

---

## ⚙️ Configuration

### Configuration MCP pour Claude Desktop

**Fichier** : `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sentiment-analyzer": {
      "command": "C:\\Users\\HP ZBOOK\\anaconda3\\envs\\nlp\\python.exe",
      "args": [
        "C:\\Users\\HP ZBOOK\\Desktop\\ETUDES\\2024-2025\\NLP\\MCP\\mcp_sentiment_server.py"
      ]
    }
  }
}
```

**⚠️ Important** :
- Utilisez des doubles backslashes `\\` dans les chemins Windows
- Vérifiez que le chemin Python pointe vers l'environnement `nlp`
- Redémarrez Claude Desktop après modification

### Configuration API FastAPI

L'API ne nécessite pas de configuration spéciale, mais vous pouvez modifier :

**Port** (dans `mcp_api_server.py`) :
```python
# Par défaut : 8000
# Pour changer : uvicorn mcp_api_server:app --port 8080
```

**CORS** (si nécessaire) :
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Logging

Les logs sont écrits dans :
```
C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\MCP\mcp_server.log
```

Pour changer l'emplacement, modifiez `LOG_FILE` dans `mcp_sentiment_server.py`.

---

## 💻 Utilisation

### 1️⃣ Serveur MCP (Claude Desktop)

#### Démarrage

Le serveur MCP démarre automatiquement lors du lancement de Claude Desktop.

#### Vérification de la connexion

1. Ouvrez Claude Desktop
2. Cliquez sur l'icône 🔨 (outils) en bas à gauche
3. Vérifiez que `sentiment-analyzer` apparaît avec une pastille verte

#### Exemples de requêtes

```
Utilisateur : Analyse le sentiment de "Je suis super content !"

Claude : [Utilise analyze_sentiment]
Résultat : {
  "sentiment": "positive",
  "confidence": 0.94
}
```

```
Utilisateur : Quel est le sentiment de cet avis : "Le service est lent et le produit est défectueux"

Claude : [Utilise analyze_sentiment]
Résultat : {
  "sentiment": "negative",
  "confidence": 0.88
}
```

### 2️⃣ API FastAPI

#### Démarrage du serveur

```bash
# Activer l'environnement
conda activate nlp

# Lancer le serveur
uvicorn mcp_api_server:app --reload --host 0.0.0.0 --port 8000
```

#### Accéder à la documentation

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

#### Tester l'API

**Avec curl** :
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Ce produit est excellent !"}'
```

**Avec Python** :
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "Ce produit est excellent !"}
)

print(response.json())
# {"sentiment": "positive", "confidence": 0.92}
```

### 3️⃣ Notebooks Jupyter

#### Lancer Jupyter

```bash
conda activate nlp
jupyter notebook
```

#### Notebooks disponibles

1. **run_analysis_with_mcp.ipynb** : Analyse batch d'articles + commentaires
2. **test_mcp_api.ipynb** : Tests de l'API REST

---

## 📁 Structure du projet

```
NLP/MCP/
│
├── mcp_sentiment_server.py      # Serveur MCP principal
├── mcp_api_server.py             # API REST FastAPI
├── test_server.py                # Script de test/diagnostic
│
├── run_analysis_with_mcp.ipynb  # Notebook analyse batch
├── test_mcp_api.ipynb           # Notebook tests API
│
├── environment.yml              # Configuration Conda
├── mcp_server.log               # Fichier de logs
│
├── articles_commentaires_final.json        # Données d'entrée
└── analyse_sentiments_result.json          # Résultats d'analyse
```

### Description des fichiers

| Fichier | Description |
|---------|-------------|
| `mcp_sentiment_server.py` | Cœur du système : analyse NLP + serveur MCP |
| `mcp_api_server.py` | API REST avec endpoints `/analyze` et `/analyze_article` |
| `test_server.py` | Script de diagnostic pour vérifier l'installation |
| `run_analysis_with_mcp.ipynb` | Analyse batch de fichiers JSON |
| `test_mcp_api.ipynb` | Tests unitaires de l'API |
| `environment.yml` | Définition de l'environnement Conda |
| `mcp_server.log` | Logs d'exécution et debugging |

---

## 📚 API Reference

### Serveur MCP

#### Méthode : `analyze_sentiment`

**Description** : Analyse le sentiment d'un texte en français.

**Input Schema** :
```json
{
  "text": "string (required)"
}
```

**Output** :
```json
{
  "sentiment": "positive" | "neutral" | "negative",
  "confidence": 0.0-1.0
}
```

**Exemple** :
```python
# Depuis Claude Desktop
"Analyse ce texte : 'Le film était fantastique !'"

# Réponse
{
  "sentiment": "positive",
  "confidence": 0.95
}
```

---

### API REST FastAPI

#### `POST /analyze`

**Description** : Analyse le sentiment d'un texte unique.

**Request Body** :
```json
{
  "text": "string"
}
```

**Response** :
```json
{
  "sentiment": "positive" | "neutral" | "negative",
  "confidence": 0.92
}
```

**Exemple curl** :
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Je déteste ce produit"}'
```

**Réponse** :
```json
{
  "sentiment": "negative",
  "confidence": 0.87
}
```

---

#### `POST /analyze_article`

**Description** : Analyse un article et ses commentaires, avec distribution des sentiments.

**Parameters** :
- `article_text` (string, required) : Texte de l'article
- `article_author` (string, optional) : Auteur de l'article (défaut: "Inconnu")
- `commentaires` (array, optional) : Liste des commentaires

**Commentaire Schema** :
```json
{
  "auteur": "string (optional, défaut: Anonyme)",
  "content": "string (required)"
}
```

**Response** :
```json
{
  "post": {
    "type": "article",
    "author": "John Doe",
    "content": "Article text...",
    "sentiment": "positive",
    "confidence": 0.85
  },
  "commentaires": [
    {
      "type": "commentaire",
      "author": "Alice",
      "content": "Super article !",
      "sentiment": "positive",
      "confidence": 0.92
    }
  ],
  "distribution": {
    "positive": 0.6,
    "neutral": 0.3,
    "negative": 0.1
  }
}
```

**Exemple Python** :
```python
import requests

payload = {
    "article_text": "Nouvel iPhone sortie aujourd'hui",
    "article_author": "Tech News",
    "commentaires": [
        {"auteur": "Alice", "content": "Trop cher !"},
        {"auteur": "Bob", "content": "J'adore le design"},
        {"content": "Bof, rien de nouveau"}
    ]
}

response = requests.post("http://localhost:8000/analyze_article", json=payload)
print(response.json())
```

---

#### `GET /`

**Description** : Message de bienvenue et informations de l'API.

**Response** :
```json
{
  "message": "Bienvenue sur l'API MCP Sentiment. POST /analyze ou /analyze_article"
}
```

---

## 🧪 Exemples d'utilisation

### Exemple 1 : Analyse simple (API)

```python
import requests

url = "http://localhost:8000/analyze"

# Texte positif
response = requests.post(url, json={"text": "J'adore ce restaurant !"})
print(response.json())
# {"sentiment": "positive", "confidence": 0.94}

# Texte négatif
response = requests.post(url, json={"text": "Service horrible"})
print(response.json())
# {"sentiment": "negative", "confidence": 0.89}
```

### Exemple 2 : Analyse d'article avec commentaires

```python
import requests

url = "http://localhost:8000/analyze_article"

payload = {
    "article_text": "Le nouveau smartphone est sorti avec de nouvelles fonctionnalités",
    "article_author": "TechBlog",
    "commentaires": [
        {"auteur": "User1", "content": "Super, j'ai hâte de l'acheter !"},
        {"auteur": "User2", "content": "Trop cher pour ce qu'il propose"},
        {"content": "Intéressant mais je vais attendre les avis"}
    ]
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Article : {result['post']['sentiment']} ({result['post']['confidence']})")
print(f"\nDistribution des commentaires :")
for sentiment, pct in result['distribution'].items():
    print(f"  {sentiment}: {pct*100:.1f}%")
```

### Exemple 3 : Analyse batch (Notebook)

```python
import json
from mcp_sentiment_server import analyze_sentiment

# Charger données
with open("articles_commentaires_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for item in data:
    article = item["article"]
    
    # Analyser l'article
    article_result = analyze_sentiment({"text": article["contenu"]})
    
    # Analyser chaque commentaire
    for comment in item["commentaires"]:
        comment_result = analyze_sentiment({"text": comment["content"]})
        results.append({
            "article_id": article.get("id"),
            "comment_author": comment.get("auteur"),
            "sentiment": comment_result["sentiment"],
            "confidence": comment_result["confidence"]
        })

# Sauvegarder
with open("batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### Exemple 4 : Utilisation avec Claude Desktop

```
Vous : Analyse le sentiment de ces avis clients et dis-moi le pourcentage de chaque catégorie :
1. "Produit parfait, livraison rapide"
2. "Déçu de la qualité"
3. "Correct pour le prix"
4. "Je recommande vivement !"
5. "Service client inexistant"

Claude : [Utilise analyze_sentiment 5 fois]

Voici l'analyse des sentiments :

✅ Positifs (40%) :
- Avis 1 : positive (0.95)
- Avis 4 : positive (0.92)

⚠️ Neutres (20%) :
- Avis 3 : neutral (0.78)

❌ Négatifs (40%) :
- Avis 2 : negative (0.85)
- Avis 5 : negative (0.91)
```

---

## 🔧 Dépannage

### Problème : Serveur MCP "failed" dans Claude Desktop

**Solution** :

1. Vérifiez le fichier de log :
   ```
   C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\MCP\mcp_server.log
   ```

2. Vérifiez que le chemin Python est correct :
   ```bash
   where python
   # Utilisez ce chemin dans claude_desktop_config.json
   ```

3. Testez manuellement :
   ```bash
   conda activate nlp
   python test_server.py
   ```

4. Vérifiez les permissions :
   - Le dossier doit être accessible en lecture/écriture
   - Désactivez temporairement l'antivirus si nécessaire

### Problème : API FastAPI ne démarre pas

**Erreur** : `ModuleNotFoundError: No module named 'fastapi'`

**Solution** :
```bash
conda activate nlp
pip install fastapi uvicorn
```

**Erreur** : `Port already in use`

**Solution** :
```bash
# Utiliser un autre port
uvicorn mcp_api_server:app --port 8001
```

### Problème : Modèle non trouvé

**Erreur** : `OSError: Can't load tokenizer`

**Solution** :

1. Vérifiez le chemin :
   ```python
   import os
   MODEL_PATH = r"C:\Users\HP ZBOOK\Desktop\ETUDES\2024-2025\NLP\models\twitter-xlm-roberta"
   print(os.path.exists(MODEL_PATH))  # Doit afficher True
   ```

2. Re-téléchargez le modèle :
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   
   tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
   model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
   
   tokenizer.save_pretrained(MODEL_PATH)
   model.save_pretrained(MODEL_PATH)
   ```

### Problème : Texte trop long

**Erreur** : `Token indices sequence length is longer than the maximum sequence length`

**Ce n'est normalement PAS un problème** car le système découpe automatiquement. Si cela se produit :

1. Vérifiez que la fonction `analyze_sentiment` contient bien la logique de chunking
2. Vérifiez les logs pour voir où l'erreur se produit
3. Limitez manuellement la longueur :
   ```python
   text = text[:5000]  # Limite à ~5000 caractères
   ```

### Problème : Résultats incohérents

**Causes possibles** :

1. **Texte mal formaté** : Vérifiez le nettoyage
   ```python
   from mcp_sentiment_server import clean_text
   print(clean_text("Votre texte"))
   ```

2. **Langue incorrecte** : Le modèle est optimisé pour le français
   ```python
   # Évitez les textes en anglais, espagnol, etc.
   ```

3. **Texte trop court** : Minimum 3-5 mots recommandés
   ```python
   if len(text.split()) < 3:
       print("Texte trop court pour une analyse fiable")
   ```

---

## 🛠️ Développement

### Ajouter une nouvelle fonctionnalité MCP

1. Modifiez `handle_mcp_request` dans `mcp_sentiment_server.py`
2. Ajoutez le schéma dans `tools/list`
3. Implémentez la logique dans `tools/call`
4. Redémarrez Claude Desktop

**Exemple** : Ajouter une fonction de traduction

```python
def handle_mcp_request(request: dict) -> dict:
    method = request.get("method")
    
    if method == "tools/list":
        return {
            "tools": [
                # ... outil existant ...
                {
                    "name": "translate_and_analyze",
                    "description": "Translate text to French and analyze sentiment",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_lang": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                }
            ]
        }
    
    elif method == "tools/call":
        tool_name = request["params"]["name"]
        
        if tool_name == "translate_and_analyze":
            # Implémenter la traduction + analyse
            pass
```

### Tests unitaires

Créez `test_sentiment.py` :

```python
import pytest
from mcp_sentiment_server import analyze_sentiment, clean_text

def test_clean_text():
    assert clean_text("  HELLO  ") == "hello"
    assert clean_text("<p>Test</p>") == "test"

def test_analyze_positive():
    result = analyze_sentiment({"text": "J'adore ce produit !"})
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5

def test_analyze_negative():
    result = analyze_sentiment({"text": "C'est horrible"})
    assert result["sentiment"] == "negative"

# Exécuter : pytest test_sentiment.py
```

---

## 📊 Performance

### Benchmarks

| Taille du texte | Temps d'analyse | Mémoire |
|-----------------|-----------------|---------|
| < 100 mots      | ~0.5s           | ~500 MB |
| 100-500 mots    | ~1.5s           | ~600 MB |
| > 500 mots      | ~3-5s           | ~800 MB |

### Optimisations possibles

1. **Cache des résultats** :
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def analyze_sentiment_cached(text):
       return analyze_sentiment({"text": text})
   ```

2. **Batch processing** :
   ```python
   # Analyser plusieurs textes en un seul appel
   def analyze_batch(texts):
       inputs = tokenizer(texts, return_tensors="pt", padding=True)
       with torch.no_grad():
           outputs = model(**inputs)
       # ...
   ```

3. **GPU** (si disponible) :
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

---

## 🤝 Contribution

### Comment contribuer

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

### Standards de code

- **PEP 8** pour Python
- Docstrings pour toutes les fonctions
- Tests unitaires pour les nouvelles features
- Logging pour debugging

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👥 Auteurs

- **Votre Nom** - Développement initial

---

## 🙏 Remerciements

- [Hugging Face](https://huggingface.co/) pour les modèles transformers
- [Cardiff NLP](https://cardiffnlp.github.io/) pour le modèle XLM-RoBERTa
- [Anthropic](https://www.anthropic.com/) pour Claude et le MCP

---

## 📞 Support

Pour toute question ou problème :

- 📧 Email : votre.email@example.com
- 🐛 Issues : [GitHub Issues](https://github.com/votre-repo/issues)
- 📖 Documentation : [Wiki](https://github.com/votre-repo/wiki)

---

**Version** : 1.0.0  
**Dernière mise à jour** : Janvier 2026