import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, GRU, Input, DropOut
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=== SMS SPAM DETECTOR WITH NEURAL NETWORK ===\n")

# 1. LOAD AND PREPARE DATA
print("📥 Loading dataset...")
url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding="latin-1")

# Keep only relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"✅ Loaded {len(df)} messages")
print(f"📊 {df['label'].sum()} spam messages, {len(df) - df['label'].sum()} normal messages")

# 2. TEXT PREPROCESSING
print("\n🔧 Preprocessing text data...")

X = df['message'].values  # Message texts
Y = df['label'].values    # Labels (0 or 1)

# Create tokenizer - converts words to numbers
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)  # Learn vocabulary from all messages

# Convert texts to sequences of numbers
sequences = tokenizer.texts_to_sequences(X)

# Make all sequences same length (50 words)
maxlen = 50
x_padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

print(f"✅ Text processing complete")
print(f"📏 All sequences padded to length: {maxlen}")

# 3. SPLIT DATA FOR TRAINING AND TESTING
print("\n🎯 Splitting data into training and test sets...")

x_train, x_test, y_train, y_test = train_test_split(
    x_padded, Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y  # Keep same spam ratio in both sets
)

print(f"📚 Training set: {len(x_train)} messages")
print(f"🧪 Test set: {len(x_test)} messages")

# 4. BUILD NEURAL NETWORK
print("\n🧠 Building neural network model...")

def create_model():
    # Input layer - accepts sequences of 50 numbers
    inputs = Input(shape=(maxlen,))
    
    # Embedding layer - learns word representations
    x = Embedding(input_dim=1000, output_dim=32)(inputs)
    
    # GRU layer - understands sequence patterns in text
    x = GRU(64)(x)
    
    # Dense layers - learn to classify based on GRU output
    x = Dense(16, activation='relu')(x)

    x = DropOut(0.2)(x)
    
    # Output layer - single number between 0-1 (probability of spam)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = create_model()

# Compile model with appropriate settings for binary classification
model.compile(
    loss="binary_crossentropy",  # Standard loss for yes/no problems
    optimizer="adam",            # Popular and effective optimizer
    metrics=['accuracy']         # Track accuracy during training
)

print("✅ Model architecture:")
model.summary()

# 5. TRAIN THE MODEL
print("\n🎯 Training the model...")

history = model.fit(
    x_train, y_train,
    batch_size=32,      # Process 32 messages at a time
    epochs=10,          # Train for 10 complete passes through data
    validation_data=(x_test, y_test),  # Check performance on test set
    verbose=1           # Show progress bars
)

# 6. TEST THE MODEL
print("\n🧪 Testing the model on example messages...")

def predict_spam(model, text, tokenizer):
    """Predict if a message is spam or not"""
    # Convert text to numbers (same process as training)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
    
    # Get prediction from model
    prediction = model.predict(padded, verbose=0)[0][0]
    
    # Return formatted result
    if prediction > 0.5:
        return f"SPAM 🔴 (confidence: {prediction * 100:.1f}%)"
    else:
        return f"NOT SPAM 🟢 (confidence: {(1 - prediction) * 100:.1f}%)"

# Test messages - mix of spam and normal
test_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://prize.com to claim your prize",
    "Hey, are we still meeting for lunch today?",
    "URGENT: Your bank account has been suspended. Click here to verify: http://bank-security.com",
    "Hi mom, I'll be home late tonight. Don't wait for me for dinner",
    "FREE iPhone for you! Just reply YES to get your premium smartphone"
]

print("\n📱 PREDICTION RESULTS:")
print("-" * 60)

for i, message in enumerate(test_messages, 1):
    result = predict_spam(model, message, tokenizer)
    # Show first 50 characters of message and prediction
    print(f"{i}. {message[:50]}...")
    print(f"   → {result}\n")

# 7. FINAL EVALUATION
print("\n📈 FINAL MODEL PERFORMANCE:")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"🎯 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

print("\n" + "="*50)
print("✅ SPAM DETECTOR READY FOR USE!")

print("="*50)
