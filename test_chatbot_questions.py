"""
Script de test complet du chatbot
"""
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_project.settings')
django.setup()

from ml_app.chatbot_module import get_chatbot_instance

print("="*80)
print("TEST CHATBOT - QUESTIONS MULTIPLES")
print("="*80)

api_key = "AIzaSyASA62pPJt-fF2mjvnDSvK-9-BhVMRwF5Q"

questions = [
    "c'est quoi le meilleur métier en france en terme de salaire",
    "quelles compétences sont requises pour un data scientist",
    "quel est le salaire moyen dans le secteur IA",
]

try:
    chatbot = get_chatbot_instance(api_key=api_key)
    chatbot.initialize()
    print("✓ Chatbot initialized\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print(f"{'='*80}")
        
        result = chatbot.answer_question(question)
        
        if result['success']:
            print(f"✓ Success")
            print(f"\nRéponse:")
            print(result['answer'])
        else:
            print(f"✗ Failed: {result['answer']}")
    
    print(f"\n{'='*80}")
    print("✓ ALL TESTS COMPLETED")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    traceback.print_exc()
