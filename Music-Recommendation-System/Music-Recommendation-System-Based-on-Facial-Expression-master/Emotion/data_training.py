import os
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sys

def load_and_preprocess_data():
    """Load and preprocess all data files"""
    is_init = False
    X = None
    y = None
    label = []
    dictionary = {}
    c = 0

    print("Loading data files...")
    
    for i in os.listdir():
        if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
            try:
                data = np.load(i)
                
                
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                elif len(data.shape) > 2:
                    data = data.reshape(data.shape[0], -1)
                
                print(f"Loaded {i} with shape: {data.shape}")
                
                size = data.shape[0]
                
                if not(is_init):
                    is_init = True 
                    X = data
                    y = np.array([i.split('.')[0]]*size).reshape(-1,1)
                else:
                    try:
                        X = np.concatenate((X, data))
                        y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
                    except ValueError as e:
                        print(f"Error concatenating {i}: {str(e)}")
                        continue

                label.append(i.split('.')[0])
                dictionary[i.split('.')[0]] = c
                c += 1
                
            except Exception as e:
                print(f"Error loading {i}: {str(e)}")
                continue
    
    return X, y, label, dictionary

def create_model(input_shape, num_classes):
    """Create and compile the model using Sequential API"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    try:
       
        print("Starting data loading and preprocessing...")
        X, y, label, dictionary = load_and_preprocess_data()
        
        if X is None or len(X) == 0:
            raise ValueError("No valid data files found!")
            
        print("\nData loading complete!")
        print(f"Total samples: {X.shape[0]}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Number of classes: {len(label)}")
        print(f"Classes: {', '.join(label)}")
        
        
        for i in range(y.shape[0]):
            y[i, 0] = dictionary[y[i, 0]]
        y = np.array(y, dtype="int32")
        
       
        y = to_categorical(y)
        
        
        print("\nShuffling data...")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        
        print("\nCreating and training model...")
        model = create_model(X.shape[1], y.shape[1])
        
        
        print("\nModel Architecture:")
        model.summary()
        
        
        print("\nTraining model...")
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model and labels
        print("\nSaving model and labels...")
        model.save("model.h5", save_format='h5')
        np.save("labels.npy", np.array(label))
        
        
        print("\nTraining complete!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        if 'val_accuracy' in history.history:
            print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all your .npy files contain data in the correct format")
        print("2. Check if all your data files have the same number of features")
        print("3. Verify that you have enough memory for training")
        print("4. Make sure you have collected data for at least two different emotions")
        sys.exit(1)

if __name__ == "__main__":
    main()