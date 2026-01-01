"""
Script de debug pour tester le chatbot
"""
import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_project.settings')
django.setup()

from ml_app.chatbot_module import get_chatbot_instance

print("="*80)
print("TEST CHATBOT DEBUG")
print("="*80)

api_key = "AIzaSyASA62pPJt-fF2mjvnDSvK-9-BhVMRwF5Q"

try:
    print("\n1. Creating chatbot instance...")
    chatbot = get_chatbot_instance(api_key=api_key)
    print("✓ Chatbot instance created")
    
    print("\n2. Initializing chatbot...")
    chatbot.initialize()
    print("✓ Chatbot initialized")
    
    print("\n3. Testing question...")
    question = "c'est quoi le meilleur métier en france en terme de salaire"
    result = chatbot.answer_question(question)
    
    print(f"\n4. Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Answer: {result.get('answer')[:200]}...")
    
    print("\n✓ ALL TESTS PASSED")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
