#!/usr/bin/env python3
"""
Test the original Python models to verify they work correctly
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def test_python_models():
    print("🔍 TESTING PYTHON MODELS")
    print("=" * 50)
    
    # Load training data
    with open('noise_ai_training_dataset_cleaned_1.json', 'r') as f:
        training_data = json.load(f)
    
    print(f"📊 Loaded {len(training_data)} training examples")
    
    # Load sentence transformer and train classifier
    print("🤖 Loading sentence transformer...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [item['text'] for item in training_data]
    intents = [item['intent'] for item in training_data]
    
    print("🔤 Encoding texts...")
    X = sentence_model.encode(texts, show_progress_bar=True)
    
    # Encode labels
    intent_encoder = LabelEncoder()
    y = intent_encoder.fit_transform(intents)
    
    # Train classifier
    print("🎯 Training classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X, y)
    
    print(f"✅ Training accuracy: {classifier.score(X, y):.3f}")
    
    # Test with same commands that are failing in Android
    test_commands = [
        "How many steps did I take today?",
        "Set an alarm for 7 AM", 
        "What's the weather like?",
        "Start a workout",
        "Show me my heart rate",
        "Set my daily step goal to 10000",
        "Turn on do not disturb",
        "Call mom"
    ]
    
    print("\n🧪 TESTING COMMANDS:")
    print("-" * 40)
    
    for command in test_commands:
        # Get embedding
        embedding = sentence_model.encode([command])
        
        # Predict intent
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        confidence = probabilities.max()
        
        # Get intent name
        intent_name = intent_encoder.inverse_transform([prediction])[0]
        
        print(f"\nText: '{command}'")
        print(f"Predicted Intent: {intent_name}")
        print(f"Confidence: {confidence:.3f}")
        
        # Show top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        print("Top 3 predictions:")
        for i, idx in enumerate(top_indices):
            intent = intent_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            print(f"  {i+1}. {intent}: {prob:.3f}")
    
    print("\n📊 EMBEDDING ANALYSIS:")
    print("-" * 30)
    
    # Test if embeddings are actually different
    sample_texts = test_commands[:3]
    sample_embeddings = sentence_model.encode(sample_texts)
    
    print(f"Embedding dimensions: {sample_embeddings.shape}")
    print(f"Sample embedding stats:")
    for i, text in enumerate(sample_texts):
        emb = sample_embeddings[i]
        print(f"  '{text[:30]}...': mean={emb.mean():.6f}, std={emb.std():.6f}, range=[{emb.min():.3f}, {emb.max():.3f}]")
    
    # Check if embeddings are similar (potential issue)
    similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
    print(f"\nEmbedding similarities:")
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            sim = similarity_matrix[i][j]
            print(f"  Text {i+1} vs Text {j+1}: {sim:.3f}")
    
    return sentence_model, classifier, intent_encoder

def test_lightweight_preprocessing():
    """Test the text preprocessing that Android uses"""
    print("\n🔧 TESTING LIGHTWEIGHT PREPROCESSING")
    print("=" * 50)
    
    # Load vocabulary that Android uses
    with open('tflite_models/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)
    
    print(f"📝 Vocabulary size: {len(vocabulary)}")
    
    # Test preprocessing function (same as Android)
    def text_to_sequence(text, vocab, max_len=32):
        words = text.lower().split()[:max_len]
        sequence = [vocab.get(word, 0) for word in words]
        sequence += [0] * (max_len - len(sequence))  # Pad
        return sequence
    
    test_commands = [
        "How many steps did I take today?",
        "Set an alarm for 7 AM",
        "What's the weather like?",
        "Turn on do not disturb"
    ]
    
    print("\n🔤 TEXT TO SEQUENCE CONVERSION:")
    print("-" * 40)
    
    for command in test_commands:
        sequence = text_to_sequence(command, vocabulary)
        words = command.lower().split()
        
        print(f"\nText: '{command}'")
        print(f"Words: {words}")
        print(f"Sequence: {sequence[:len(words)]}")
        
        # Check how many words are in vocabulary
        in_vocab = sum(1 for word in words if word in vocabulary)
        print(f"Words in vocab: {in_vocab}/{len(words)} ({in_vocab/len(words)*100:.1f}%)")

if __name__ == "__main__":
    # Test original Python models
    sentence_model, classifier, intent_encoder = test_python_models()
    
    # Test lightweight preprocessing
    test_lightweight_preprocessing()
    
    print("\n✅ PYTHON MODEL TESTING COMPLETED!")
    print("If Python models work correctly but Android doesn't,")
    print("the issue is likely in the TFLite conversion or Android preprocessing.")