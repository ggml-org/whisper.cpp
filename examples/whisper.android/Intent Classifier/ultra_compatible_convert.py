#!/usr/bin/env python3
"""
Create ultra-compatible TensorFlow Lite models for Android
Uses only basic operations supported by older TFLite versions
"""

import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔧 ULTRA-COMPATIBLE TFLITE MODEL CONVERSION")
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

# Create an ultra-simple encoder using TensorFlow 1.x compatible operations
print("🔧 Creating ultra-compatible sentence encoder...")

def create_ultra_compatible_encoder():
    """Create a model using only TF 1.x compatible operations"""
    
    # Use TF 1.x compatible approach
    tf.compat.v1.disable_v2_behavior()
    
    # Create a simple model that just returns a constant embedding
    # This ensures compatibility but gives consistent results for debugging
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,), dtype=tf.float32, name='embedding_input'),
        # Just return a constant vector for all inputs (ultra-compatible)
        tf.keras.layers.Lambda(lambda x: tf.ones((tf.shape(x)[0], 384)) * 0.1)
    ])
    
    return model

# Actually, let's use an even simpler approach - create the model manually
def create_manual_encoder():
    """Create a model manually to ensure compatibility"""
    
    # Input layer
    inputs = tf.keras.Input(shape=(32,), dtype=tf.float32, name='embedding_input')
    
    # Create a constant output (all 0.1 values)
    # This ensures the model works but gives consistent predictions
    constant_output = tf.keras.layers.Lambda(
        lambda x: tf.fill([tf.shape(x)[0], 384], 0.1),
        name='constant_embedding'
    )(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=constant_output)
    return model

encoder_model = create_manual_encoder()

# Convert sentence encoder to TFLite with maximum compatibility
print("📱 Converting ultra-compatible sentence encoder to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(encoder_model)

# Use maximum compatibility settings
converter.optimizations = []  # No optimizations for maximum compatibility
converter.target_spec.supported_types = [tf.float32]
converter.experimental_new_converter = False  # Use old converter

sentence_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/ultra_compatible_encoder.tflite', 'wb') as f:
    f.write(sentence_tflite_model)

print(f"✅ Ultra-compatible sentence encoder saved: {len(sentence_tflite_model)} bytes")

# Create an ultra-compatible intent classifier manually
print("📱 Creating ultra-compatible intent classifier...")

def create_manual_classifier():
    """Create a classifier manually using only basic operations"""
    
    # Input layer
    inputs = tf.keras.Input(shape=(384,), name='embedding_input')
    
    # Manual matrix multiplication (most compatible)
    # We'll create the weights as constants in the model
    weights = tf.constant(classifier.coef_.T, dtype=tf.float32, name='classifier_weights')
    bias = tf.constant(classifier.intercept_, dtype=tf.float32, name='classifier_bias')
    
    # Manual dense layer computation
    logits = tf.add(tf.matmul(inputs, weights), bias)
    
    # Manual softmax
    outputs = tf.nn.softmax(logits, name='probabilities')
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

classifier_model = create_manual_classifier()

# Convert intent classifier to TFLite with maximum compatibility
print("📱 Converting ultra-compatible intent classifier to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(classifier_model)

# Use maximum compatibility settings
converter.optimizations = []  # No optimizations
converter.target_spec.supported_types = [tf.float32]
converter.experimental_new_converter = False  # Use old converter

intent_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/compatible_intent_classifier.tflite', 'wb') as f:
    f.write(intent_tflite_model)

print(f"✅ Ultra-compatible intent classifier saved: {len(intent_tflite_model)} bytes")

# Create metadata
intent_mapping = {i: intent for i, intent in enumerate(intent_encoder.classes_)}
metadata = {
    "model_info": {
        "created_date": "2025-09-29",
        "framework": "tensorflow_lite_ultra_compatible",
        "input_dimension": 384,
        "num_intents": len(intent_encoder.classes_),
        "model_type": "ultra_compatible_intent_classification",
        "compatibility": "android_ultra_compatible"
    },
    "intent_mapping": intent_mapping,
    "intents": list(intent_encoder.classes_)
}

with open('tflite_models/compatible_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Ultra-compatible model conversion completed!")
print("\n📊 Model Summary:")
print(f"  📝 Vocabulary: {len(vocabulary)} words")
print(f"  🔤 Ultra-compatible sentence encoder: {len(sentence_tflite_model)} bytes")
print(f"  🎯 Ultra-compatible intent classifier: {len(intent_tflite_model)} bytes")
print(f"  📋 Intent classes: {len(intent_encoder.classes_)}")

print("\n⚠️  NOTE: These models use ultra-compatible operations.")
print("The sentence encoder returns constant values (0.1) for all inputs.")
print("This should help us test if the TFLite runtime works at all.")

# Test what happens with constant embeddings (all 0.1)
print("\n🧪 TESTING WITH CONSTANT EMBEDDINGS (all 0.1)...")
constant_embedding = np.full((1, 384), 0.1, dtype=np.float32)
prediction = classifier.predict(constant_embedding)[0]
probabilities = classifier.predict_proba(constant_embedding)[0]
predicted_intent = intent_encoder.inverse_transform([prediction])[0]

print(f"  🎯 Constant embedding (0.1) predicts: {predicted_intent}")
print(f"  🎯 Confidence: {probabilities.max():.3f}")

# Show top 3 predictions for constant embedding
top_indices = np.argsort(probabilities)[-3:][::-1]
print(f"  🏆 Top 3 predictions for constant embedding:")
for i, idx in enumerate(top_indices):
    intent = intent_encoder.inverse_transform([idx])[0]
    prob = probabilities[idx]
    print(f"    {i+1}. {intent}: {prob:.3f}")

print("\nIf Android predicts the same intent as above, the models work!")
print("If it still fails to load, we need to downgrade TensorFlow Lite further.")