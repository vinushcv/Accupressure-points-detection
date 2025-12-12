"""
Deep Learning Training Pipeline for Acupressure Point Detection
Generates synthetic hand landmark data and trains a regression model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("DEEP LEARNING ACUPRESSURE POINT DETECTOR - TRAINING PIPELINE")
print("=" * 70)
print()

# ============================================================================
# STEP 1: CREATE CANONICAL HAND SKELETON (21 MediaPipe Landmarks)
# ============================================================================

def create_canonical_hand():
    """
    Create a normalized canonical hand skeleton based on MediaPipe Hand topology.
    Returns: numpy array of shape (21, 2) representing (x, y) coordinates.
    
    MediaPipe Hand Landmark Indices:
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    
    # Normalized hand skeleton (centered at origin, scale 0-1)
    canonical = np.array([
        # Wrist
        [0.5, 0.9],     # 0: WRIST
        
        # Thumb (extends from wrist, curves inward)
        [0.35, 0.75],   # 1: THUMB_CMC
        [0.25, 0.60],   # 2: THUMB_MCP
        [0.20, 0.45],   # 3: THUMB_IP
        [0.18, 0.30],   # 4: THUMB_TIP
        
        # Index finger
        [0.40, 0.65],   # 5: INDEX_MCP
        [0.38, 0.45],   # 6: INDEX_PIP
        [0.37, 0.30],   # 7: INDEX_DIP
        [0.36, 0.15],   # 8: INDEX_TIP
        
        # Middle finger
        [0.50, 0.65],   # 9: MIDDLE_MCP
        [0.50, 0.43],   # 10: MIDDLE_PIP
        [0.50, 0.25],   # 11: MIDDLE_DIP
        [0.50, 0.08],   # 12: MIDDLE_TIP
        
        # Ring finger
        [0.60, 0.65],   # 13: RING_MCP
        [0.62, 0.45],   # 14: RING_PIP
        [0.63, 0.30],   # 15: RING_DIP
        [0.64, 0.15],   # 16: RING_TIP
        
        # Pinky finger
        [0.70, 0.68],   # 17: PINKY_MCP
        [0.73, 0.52],   # 18: PINKY_PIP
        [0.75, 0.40],   # 19: PINKY_DIP
        [0.77, 0.28],   # 20: PINKY_TIP
    ], dtype=np.float32)
    
    return canonical


# ============================================================================
# STEP 2: DATA AUGMENTATION - GENERATE SYNTHETIC SAMPLES
# ============================================================================

def rotate_points(points, angle_degrees):
    """Rotate 2D points around center by given angle."""
    angle_rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Center points
    center = points.mean(axis=0)
    centered = points - center
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Apply rotation and recenter
    rotated = centered @ rotation_matrix.T
    return rotated + center


def augment_hand(canonical_hand, rotation_range=180, scale_range=(0.8, 1.2), noise_std=0.01):
    """
    Apply random augmentations to canonical hand.
    
    Args:
        canonical_hand: Original hand landmarks (21, 2)
        rotation_range: Max rotation in degrees (±)
        scale_range: (min_scale, max_scale)
        noise_std: Standard deviation for Gaussian noise
    
    Returns:
        Augmented hand landmarks (21, 2)
    """
    hand = canonical_hand.copy()
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    hand = rotate_points(hand, angle)
    
    # Random scaling
    scale = np.random.uniform(*scale_range)
    center = hand.mean(axis=0)
    hand = (hand - center) * scale + center
    
    # Random translation
    translation = np.random.uniform(-0.1, 0.1, size=2)
    hand += translation
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, hand.shape)
    hand += noise
    
    # Clip to valid range [0, 1]
    hand = np.clip(hand, 0, 1)
    
    return hand


def generate_dataset(canonical_hand, num_samples=50000):
    """
    Generate synthetic dataset of augmented hands.
    
    Returns:
        X: Input features (num_samples, 42) - flattened landmarks
        y: Ground truth labels (num_samples, 6) - 3 acupressure points
    """
    print(f"Generating {num_samples:,} synthetic hand samples...")
    
    X = np.zeros((num_samples, 42), dtype=np.float32)  # 21 landmarks * 2 coords
    y = np.zeros((num_samples, 6), dtype=np.float32)   # 3 points * 2 coords
    
    for i in range(num_samples):
        # Generate augmented hand
        augmented = augment_hand(canonical_hand)
        
        # Flatten for input
        X[i] = augmented.flatten()
        
        # Calculate ground truth acupressure points
        y[i] = calculate_acupressure_points(augmented)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1:,} / {num_samples:,} samples")
    
    print("✓ Dataset generation complete!\n")
    return X, y


# ============================================================================
# STEP 3: GROUND TRUTH CALCULATION (LABELS)
# ============================================================================

def calculate_acupressure_points(landmarks):
    """
    Calculate the 3 acupressure point coordinates from hand landmarks.
    
    Args:
        landmarks: Hand landmarks array of shape (21, 2)
    
    Returns:
        Flattened array of 6 values: [x1, y1, x2, y2, x3, y3]
    """
    
    # Point 1: LI-4 (Hegu) - For Cold/Headache
    # Midpoint between Thumb CMC (1) and Index MCP (5)
    point1 = (landmarks[1] + landmarks[5]) / 2.0
    
    # Point 2: PC-8 (Laogong) - For Stress
    # Center of palm: weighted average of Wrist (0) and Middle MCP (9)
    point2 = landmarks[0] * 0.35 + landmarks[9] * 0.65
    
    # Point 3: HT-7 (Shen Men) - For Anxiety
    # On wrist crease, shifted towards Pinky MCP (17)
    point3 = landmarks[0] * 0.7 + landmarks[17] * 0.3
    
    # Flatten to 1D array: [x1, y1, x2, y2, x3, y3]
    return np.array([
        point1[0], point1[1],
        point2[0], point2[1],
        point3[0], point3[1]
    ], dtype=np.float32)


# ============================================================================
# STEP 4: BUILD NEURAL NETWORK MODEL
# ============================================================================

def build_model():
    """
    Build a deep neural network for acupressure point regression.
    
    Architecture:
        Input: 42 features (21 landmarks × 2 coordinates)
        Hidden Layers: Multiple Dense layers with ReLU activation
        Output: 6 values (3 points × 2 coordinates)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(42,), name='hand_landmarks'),
        
        # Hidden layers with dropout for regularization
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(256, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu', name='dense_3'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Dense(128, activation='relu', name='dense_4'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Dense(64, activation='relu', name='dense_5'),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation='relu', name='dense_6'),
        
        # Output layer (no activation for regression)
        layers.Dense(6, activation='linear', name='acupressure_points')
    ], name='AcupressurePointDetector')
    
    return model


# ============================================================================
# STEP 5: TRAINING PIPELINE
# ============================================================================

def train_model():
    """Main training pipeline."""
    
    # 1. Create canonical hand
    print("Creating canonical hand skeleton...")
    canonical = create_canonical_hand()
    print(f"✓ Canonical hand shape: {canonical.shape}\n")
    
    # 2. Generate synthetic dataset
    X, y = generate_dataset(canonical, num_samples=50000)
    
    # 3. Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples\n")
    
    # 4. Build model
    print("Building neural network model...")
    model = build_model()
    model.summary()
    print()
    
    # 5. Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    # 6. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'acupressure_model_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 7. Train model
    print("=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Save final model
    model.save('acupressure_model.keras')
    print("\n✓ Model saved as 'acupressure_model.keras'")
    
    # 9. Plot training history
    plot_training_history(history)
    
    # 10. Evaluate model
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (MSE): {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")
    
    # Test on a few samples
    print("\nTesting on sample predictions...")
    test_samples = X_val[:5]
    predictions = model.predict(test_samples, verbose=0)
    
    for i in range(5):
        print(f"\nSample {i + 1}:")
        print(f"  Ground Truth: {y_val[i]}")
        print(f"  Prediction:   {predictions[i]}")
        print(f"  Error:        {np.abs(y_val[i] - predictions[i])}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now run 'main_app.py' to use the trained model.")


def plot_training_history(history):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss During Training')
    plt.legend()
    plt.grid(True)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model MAE During Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training pipeline
    train_model()
