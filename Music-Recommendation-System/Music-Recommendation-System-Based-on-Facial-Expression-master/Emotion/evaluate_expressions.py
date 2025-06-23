import os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_test_data():
    """Load and prepare test data from .npy files with proper validation and reshaping"""
    is_init = False
    X = None
    label = []
    dictionary = {}
    c = 0
    expected_features = None
    
    print("Loading and validating data files...")
    
   
    for i in os.listdir():
        if i.endswith(".npy") and not i.startswith("labels"):
            try:
                data = np.load(i)
                if data.size == 0:
                    print(f"Skipping empty file: {i}")
                    continue
                    
                
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif len(data.shape) > 2:
                    data = data.reshape(data.shape[0], -1)
                
                if expected_features is None:
                    expected_features = data.shape[1]
                    print(f"Expected features per sample: {expected_features}")
                
                print(f"Loaded {i} with shape: {data.shape}")
                
            except Exception as e:
                print(f"Error inspecting {i}: {str(e)}")
                continue
    
    if expected_features is None:
        raise ValueError("No valid data files found to determine feature size!")
    
   
    for i in os.listdir():
        if i.endswith(".npy") and not i.startswith("labels"):
            try:
                data = np.load(i)
                
               
                if data.size == 0:
                    continue
                    
               
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif len(data.shape) > 2:
                    data = data.reshape(data.shape[0], -1)
                
                
                if data.shape[1] != expected_features:
                    print(f"Warning: Skipping {i} - expected {expected_features} features but got {data.shape[1]}")
                    continue
                
                print(f"Processing {i} with {data.shape[0]} samples")
                
                if not is_init:
                    is_init = True 
                    X = data
                    y = np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)
                else:
                    X = np.concatenate((X, data), axis=0)
                    y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))

                label.append(i.split('.')[0])
                dictionary[i.split('.')[0]] = c
                c += 1
                
            except Exception as e:
                print(f"Error loading {i}: {str(e)}")
                continue
    
    if X is None or len(X) == 0:
        raise ValueError("No valid data was loaded!")
        
    
    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]
    y = np.array(y, dtype="int32")
    y_cat = np.zeros((y.shape[0], len(dictionary)))
    for i in range(y.shape[0]):
        y_cat[i, y[i]] = 1
    
    print(f"\nSuccessfully loaded {X.shape[0]} samples with {X.shape[1]} features each")
    print(f"Found {len(label)} classes: {', '.join(label)}")
    
    return X, y_cat, label

def evaluate_expressions():
    try:
        
        print("Loading model and data...")
        model = load_model("model.h5")
        X, y, labels = load_test_data()
        
       
        print("Making predictions...")
        y_pred = model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y, axis=1)
        
        
        print("\nCalculating accuracies...")
        class_accuracies = {}
        for i in range(len(labels)):
            mask = (y_true_classes == i)
            if np.sum(mask) > 0:  
                class_accuracy = np.mean(y_pred_classes[mask] == i)
                class_accuracies[labels[i]] = class_accuracy * 100
        
        
        print("\nAccuracy per Expression:")
        print("=" * 50)
        for expression, accuracy in class_accuracies.items():
            print(f"{expression}: {accuracy:.2f}%")
        
        
        print("\nGenerating confusion matrix...")
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for i in range(len(y_true_classes)):
            cm[y_true_classes[i]][y_pred_classes[i]] += 1
        
       
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        
        total_accuracy = np.mean(y_pred_classes == y_true_classes) * 100
        print(f"\nOverall Model Accuracy: {total_accuracy:.2f}%")
        
        
        with open("expression_metrics.txt", "w") as f:
            f.write("Expression Recognition Metrics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {total_accuracy:.2f}%\n\n")
            f.write("Per-Expression Accuracies:\n")
            for expression, accuracy in class_accuracies.items():
                f.write(f"{expression}: {accuracy:.2f}%\n")
                
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all your .npy files contain data in the correct format")
        print("2. Check if your data files were created with the same feature extraction process")
        print("3. Verify that model.h5 and labels.npy exist in the current directory")
        print("4. Ensure you have at least two different emotion classes for evaluation")

if __name__ == "__main__":
    evaluate_expressions()