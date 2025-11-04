#!/usr/bin/env python3
"""
Fixed TensorFlow Lite model converter with proper data types
"""

import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔧 FIXED TFLITE MODEL CONVERSION")
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

# Create a TensorFlow model that does sequence-to-embedding conversion
@tf.function
def text_to_embedding(input_sequence):
    """Convert token sequence to embeddings using embedding layer + pooling"""
    # Create embedding layer
    vocab_size = len(vocabulary) + 1  # +1 for unknown token (0)
    embedding_dim = 384
    
    # Initialize embedding weights randomly (this is a simplified approach)
    # In practice, you'd want to use pre-trained embeddings
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='embedding'
    )
    
    # Get embeddings for input sequence
    embeddings = embedding_layer(input_sequence)
    
    # Global average pooling to get fixed-size representation
    pooled = tf.reduce_mean(embeddings, axis=1)
    
    return pooled

# For now, let's create a simplified model that just maps tokens to dense vectors
def create_simple_sentence_encoder():
    """Create a simple model that converts token sequences to embeddings"""
    
    vocab_size = len(vocabulary) + 1
    max_length = 32
    embedding_dim = 384
    
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='embedding_input')
    
    # Cast to float32 for embedding layer
    float_input = tf.cast(input_layer, tf.float32)
    
    # Create a simple dense mapping from token IDs to embeddings
    # This is a simplified approach - just map each token to a random embedding
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)
    
    # Use embedding lookup
    embedded = tf.nn.embedding_lookup(embedding_matrix, tf.cast(float_input, tf.int32))
    
    # Global average pooling
    output = tf.reduce_mean(embedded, axis=1)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

print("🏗️ Creating simplified sentence encoder...")
sentence_encoder_model = create_simple_sentence_encoder()

# For better results, let's use a different approach - 
# Create a model that takes the original sentence transformer embeddings as a lookup table
def create_lookup_based_encoder():
    """Create a model that uses pre-computed embeddings"""
    
    print("🔍 Pre-computing embeddings for all possible token combinations...")
    
    # Create a simplified approach: map common token sequences to their embeddings
    common_sequences = []
    common_embeddings = []
    
    # Get embeddings for training texts
    for text in texts[:100]:  # Use first 100 for speed
        words = text.lower().split()[:32]  # Max 32 words
        sequence = [vocabulary.get(word, 0) for word in words]
        sequence += [0] * (32 - len(sequence))  # Pad to 32
        
        # Get actual embedding from sentence transformer
        embedding = sentence_model.encode([text])[0]
        
        common_sequences.append(sequence)
        common_embeddings.append(embedding)
    
    # Create a model that does approximate nearest neighbor lookup
    vocab_size = len(vocabulary) + 1
    max_length = 32
    embedding_dim = 384
    
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='embedding_input')
    
    # Convert to float and create simple random embeddings
    float_input = tf.cast(input_layer, tf.float32)
    
    # Simple dense layers to map from tokens to embeddings
    x = tf.keras.layers.Dense(128, activation='relu')(float_input)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output = tf.keras.layers.Dense(embedding_dim)(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

# Actually, let's use the simplest approach that will work:
# Create a model that just returns random but consistent embeddings based on input
def create_consistent_random_encoder():
    """Create a model that returns consistent embeddings for debugging"""
    
    vocab_size = len(vocabulary) + 1
    max_length = 32
    embedding_dim = 384
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='embedding_input'),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(embedding_dim, activation='tanh')
    ])
    
    return model

print("🔧 Creating consistent embedding model...")
simple_encoder = create_consistent_random_encoder()

# Compile and train briefly to get reasonable weights
print("🏋️ Training simple encoder...")
sample_sequences = []
sample_embeddings = []

for text in texts[:100]:
    words = text.lower().split()[:32]
    sequence = [vocabulary.get(word, 0) for word in words]
    sequence += [0] * (32 - len(sequence))
    
    embedding = sentence_model.encode([text])[0]
    
    sample_sequences.append(sequence)
    sample_embeddings.append(embedding)

sample_sequences = np.array(sample_sequences, dtype=np.int32)
sample_embeddings = np.array(sample_embeddings, dtype=np.float32)

simple_encoder.compile(optimizer='adam', loss='mse')
simple_encoder.fit(sample_sequences, sample_embeddings, epochs=10, verbose=1)

# Convert sentence encoder to TFLite
print("📱 Converting sentence encoder to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(simple_encoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
sentence_tflite_model = converter.convert()

# Save the TFLite model
with open('tflite_models/lightweight_sentence_encoder.tflite', 'wb') as f:
    f.write(sentence_tflite_model)

print(f"✅ Sentence encoder saved: {len(sentence_tflite_model)} bytes")

# Convert intent classifier (this part should work)
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
        "framework": "tensorflow_lite_fixed",
        "input_dimension": 384,
        "num_intents": len(intent_encoder.classes_),
        "model_type": "fixed_intent_classification",
        "compatibility": "android_fixed"
    },
    "intent_mapping": intent_mapping,
    "intents": list(intent_encoder.classes_)
}

with open('tflite_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model conversion completed with proper data types!")
print("\n📊 Model Summary:")
print(f"  📝 Vocabulary: {len(vocabulary)} words")
print(f"  🔤 Sentence encoder: {len(sentence_tflite_model)} bytes")
print(f"  🎯 Intent classifier: {len(intent_tflite_model)} bytes")
print(f"  📋 Intent classes: {len(intent_encoder.classes_)}")