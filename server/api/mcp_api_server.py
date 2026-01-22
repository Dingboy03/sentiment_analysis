# fichier: mcp_api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from server.nlp.mcp_sentiment_server import analyze_sentiment


app = FastAPI(title="MCP Sentiment API", version="1.0")

# -------------------------------
# MODELES DE DONNÉES
# -------------------------------
class CommentairePayload(BaseModel):
    auteur: Optional[str] = "Anonyme"
    content: str

class MCPRequest(BaseModel):
    text: str

class MCPResponse(BaseModel):
    sentiment: str
    confidence: float

class ArticleResponse(BaseModel):
    type: str = "article"
    author: str
    content: str
    sentiment: str
    confidence: float

class CommentaireResponse(BaseModel):
    type: str = "commentaire"
    author: str
    content: str
    sentiment: str
    confidence: float

class MCPFullResponse(BaseModel):
    post: ArticleResponse
    commentaires: List[CommentaireResponse]
    distribution: dict

# -------------------------------
# ROUTES
# -------------------------------

@app.post("/analyze", response_model=MCPResponse)
def analyze_text(payload: MCPRequest):
    """Analyse sentiment d'un texte unique"""
    try:
        
        result = analyze_sentiment(payload.text)

        print("RESULT:", result, type(result))

        return result
    except Exception as e:
        print("ERROR:", e)

        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze_article", response_model=MCPFullResponse)
def analyze_article(article_text: str, article_author: Optional[str] = "Inconnu", commentaires: Optional[List[CommentairePayload]] = []):
    """Analyse sentiment d'un article + commentaires"""
    # Article
    post_result = analyze_sentiment(article_text)
    post_info = {
        "type": "article",
        "author": article_author,
        "content": article_text,
        "sentiment": post_result["sentiment"],
        "confidence": post_result["confidence"]
    }

    # Commentaires
    results = []
    for c in commentaires:
        res = analyze_sentiment(c.content)
        comment_info = {
            "type": "commentaire",
            "author": c.auteur,
            "content": c.content,
            "sentiment": res["sentiment"],
            "confidence": res["confidence"]
        }
        results.append(comment_info)

    # Distribution globale
    from collections import Counter
    sentiments = [r["sentiment"] for r in results]
    distribution = Counter(sentiments)
    distribution_pct = {k: round(v / len(results), 3) if results else 0 for k, v in distribution.items()}

    return {
        "post": post_info,
        "commentaires": results,
        "distribution": distribution_pct
    }

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MCP Sentiment. POST /analyze ou /analyze_article"}
