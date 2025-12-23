# Configuration du Chatbot RAG

## Configuration de l'API Gemini

Pour activer le chatbot avec l'IA Gemini, vous devez configurer votre clé API.

### Option 1: Variable d'environnement (Recommandé)

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="votre_cle_api_ici"

# Windows CMD
set GEMINI_API_KEY=votre_cle_api_ici

# Linux/Mac
export GEMINI_API_KEY="votre_cle_api_ici"
```

### Option 2: Fichier .env

Créez un fichier `.env` à la racine du projet:

```
GEMINI_API_KEY=votre_cle_api_ici
```

Puis installez python-dotenv:

```bash
pip install python-dotenv
```

Et ajoutez dans `settings.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Obtenir une clé API Gemini

1. Visitez: https://makersuite.google.com/app/apikey
2. Connectez-vous avec votre compte Google
3. Créez une nouvelle clé API
4. Copiez la clé et configurez-la comme indiqué ci-dessus

## Installation des dépendances

```bash
pip install -r requirements.txt
```

## Fonctionnalités du Chatbot

Le chatbot utilise RAG (Retrieval-Augmented Generation) pour répondre aux questions sur:

- 💼 Les emplois disponibles dans le dataset
- 💰 Les salaires par poste et localisation
- 🎓 Les compétences requises pour différents rôles
- 🏢 Les entreprises et leurs offres
- 📊 Les statistiques sur le marché de l'emploi dans l'IA

## Utilisation

1. Cliquez sur l'icône du chatbot en bas à droite de l'écran
2. Posez votre question en français ou en anglais
3. Le chatbot recherchera dans le dataset et vous donnera une réponse basée sur les données réelles

## Exemples de questions

- "Quelle entreprise offre le salaire le plus élevé pour un ingénieur IA?"
- "Quelles sont les compétences clés requises pour un data scientist?"
- "Quel est le salaire médian pour un ingénieur en machine learning?"
- "Quels sont les types d'emploi disponibles en remote?"
