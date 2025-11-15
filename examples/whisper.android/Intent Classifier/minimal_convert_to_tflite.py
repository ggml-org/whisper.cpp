#!/usr/bin/env python3
"""
Minimal TensorFlow Lite model converter - just use the classifier, skip the sentence encoder
"""

import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔧 MINIMAL TFLITE MODEL CONVERSION")
print("=" * 50)

# Load training data
with open('noise_ai_training_dataset_cleaned_1.json', 'r') as f:
    training_data = json.load(f)

print(f"📊 Loaded {len(training_data)} training examples")

# Train models
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

# Create vocabulary from training data
print("📝 Creating vocabulary...")
all_words = set()
for text in texts:
    words = text.lower().split()
    all_words.update(words)

# Create word to index mapping (starting from 1, 0 is for unknown)
vocabulary = {word: idx + 1 for idx, word in enumerate(sorted(all_words))}
print(f"📝 Vocabulary size: {len(vocabulary)} words")

# Save vocabulary
with open('tflite_models/vocabulary.json', 'w') as f:
    json.dump(vocabulary, f, indent=2)

# Create a dummy sentence encoder that just returns zeros
# This will force Android to generate consistent but wrong embeddings
print("🔧 Creating dummy sentence encoder...")

def create_dummy_encoder():
    """Create a model that just returns zeros - for debugging"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,), dtype=tf.float32, name='embedding_input'),
        tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 384)))
    ])
    return model

dummy_encoder = create_dummy_encoder()

# Convert dummy encoder to TFLite
print("📱 Converting dummy sentence encoder to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(dummy_encoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
sentence_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/lightweight_sentence_encoder.tflite', 'wb') as f:
    f.write(sentence_tflite_model)

print(f"✅ Dummy sentence encoder saved: {len(sentence_tflite_model)} bytes")

# Convert intent classifier (this should work)
print("📱 Converting intent classifier to TFLite...")

# Create a simple TensorFlow model for the classifier
intent_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(384,), name='embedding_input'),
    tf.keras.layers.Dense(len(intent_encoder.classes_), activation='softmax')
])

# Set the weights from the trained sklearn model
intent_model.layers[0].set_weights([
    classifier.coef_.T,  # Weights
    classifier.intercept_  # Bias
])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(intent_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
intent_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/intent_classifier.tflite', 'wb') as f:
    f.write(intent_tflite_model)

print(f"✅ Intent classifier saved: {len(intent_tflite_model)} bytes")

# Create metadata
intent_mapping = {i: intent for i, intent in enumerate(intent_encoder.classes_)}
metadata = {
    "model_info": {
        "created_date": "2025-09-29",
        "framework": "tensorflow_lite_minimal",
        "input_dimension": 384,
        "num_intents": len(intent_encoder.classes_),
        "model_type": "minimal_with_dummy_encoder",
        "compatibility": "android_debug"
    },
    "intent_mapping": intent_mapping,
    "intents": list(intent_encoder.classes_)
}

with open('tflite_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model conversion completed!")
print("\n📊 Model Summary:")
print(f"  📝 Vocabulary: {len(vocabulary)} words")
print(f"  🔤 Dummy sentence encoder: {len(sentence_tflite_model)} bytes")
print(f"  🎯 Intent classifier: {len(intent_tflite_model)} bytes")
print(f"  📋 Intent classes: {len(intent_encoder.classes_)}")

print("\n⚠️  NOTE: The sentence encoder is a dummy that returns zeros.")
print("This will help us debug if the issue is in embedding generation or classification.")
print("If Android still predicts ToggleFeature for everything, the issue is in the classifier.")
print("If it predicts different things, the issue was in the embedding generation.")

# Test what happens with zero embeddings
print("\n🧪 TESTING WITH ZERO EMBEDDINGS...")
zero_embedding = np.zeros((1, 384), dtype=np.float32)
prediction = classifier.predict(zero_embedding)[0]
probabilities = classifier.predict_proba(zero_embedding)[0]
predicted_intent = intent_encoder.inverse_transform([prediction])[0]

print(f"  🎯 Zero embedding predicts: {predicted_intent}")
print(f"  🎯 Confidence: {probabilities.max():.3f}")

# Show top 3 predictions for zero embedding
top_indices = np.argsort(probabilities)[-3:][::-1]
print(f"  🏆 Top 3 predictions for zero embedding:")
for i, idx in enumerate(top_indices):
    intent = intent_encoder.inverse_transform([idx])[0]
    prob = probabilities[idx]
    print(f"    {i+1}. {intent}: {prob:.3f}")