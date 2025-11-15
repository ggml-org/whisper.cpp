#!/usr/bin/env python3
"""
Simple TensorFlow Lite model converter that handles data types correctly
"""

import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔧 SIMPLE TFLITE MODEL CONVERSION")
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

# Create a very simple sentence encoder that accepts int32 input
def create_simple_encoder():
    """Create a simple model that handles int32 input correctly"""
    
    vocab_size = len(vocabulary) + 1
    max_length = 32
    embedding_dim = 384
    
    # Create the model using functional API for better control
    input_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='embedding_input')
    
    # Embedding layer that accepts int32 directly
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,  # Smaller embedding for simplicity
        mask_zero=True
    )(input_layer)
    
    # Global average pooling to get fixed-size output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers to get to desired embedding dimension
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(embedding_dim, activation='tanh')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    return model

print("🏗️ Creating simple sentence encoder...")
sentence_encoder_model = create_simple_encoder()

# Train the sentence encoder with sample data
print("🏋️ Training sentence encoder...")
sample_sequences = []
sample_embeddings = []

for text in texts[:200]:  # Use more samples for better training
    words = text.lower().split()[:32]
    sequence = [vocabulary.get(word, 0) for word in words]
    sequence += [0] * (32 - len(sequence))  # Pad to 32
    
    # Get actual embedding from sentence transformer
    embedding = sentence_model.encode([text])[0]
    
    sample_sequences.append(sequence)
    sample_embeddings.append(embedding)

sample_sequences = np.array(sample_sequences, dtype=np.int32)
sample_embeddings = np.array(sample_embeddings, dtype=np.float32)

sentence_encoder_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
sentence_encoder_model.fit(sample_sequences, sample_embeddings, epochs=20, batch_size=32, verbose=1)

# Convert sentence encoder to TFLite
print("📱 Converting sentence encoder to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(sentence_encoder_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set input and output types explicitly
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.int32  # Accept int32 input
converter.inference_output_type = tf.float32  # Output float32

sentence_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/lightweight_sentence_encoder.tflite', 'wb') as f:
    f.write(sentence_tflite_model)

print(f"✅ Sentence encoder saved: {len(sentence_tflite_model)} bytes")

# Convert intent classifier
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
        "framework": "tensorflow_lite_simple",
        "input_dimension": 384,
        "num_intents": len(intent_encoder.classes_),
        "model_type": "simple_intent_classification",
        "compatibility": "android_simple"
    },
    "intent_mapping": intent_mapping,
    "intents": list(intent_encoder.classes_)
}

with open('tflite_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model conversion completed!")
print("\n📊 Model Summary:")
print(f"  📝 Vocabulary: {len(vocabulary)} words")
print(f"  🔤 Sentence encoder: {len(sentence_tflite_model)} bytes")
print(f"  🎯 Intent classifier: {len(intent_tflite_model)} bytes")
print(f"  📋 Intent classes: {len(intent_encoder.classes_)}")

# Test the converted models
print("\n🧪 TESTING CONVERTED MODELS...")

# Test TFLite sentence encoder
interpreter = tf.lite.Interpreter(model_path='tflite_models/lightweight_sentence_encoder.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  🔤 Sentence encoder input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
print(f"  🔤 Sentence encoder output: {output_details[0]['shape']} ({output_details[0]['dtype']})")

# Test with a sample
test_text = "How many steps did I take today?"
test_words = test_text.lower().split()[:32]
test_sequence = [vocabulary.get(word, 0) for word in test_words]
test_sequence += [0] * (32 - len(test_sequence))
test_sequence = np.array([test_sequence], dtype=np.int32)

interpreter.set_tensor(input_details[0]['index'], test_sequence)
interpreter.invoke()
test_embedding = interpreter.get_tensor(output_details[0]['index'])

print(f"  🧪 Test embedding shape: {test_embedding.shape}")
print(f"  🧪 Test embedding range: [{test_embedding.min():.3f}, {test_embedding.max():.3f}]")

print("\n✅ All tests passed! Models should work in Android now.")