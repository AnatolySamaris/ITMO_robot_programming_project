import torch
from torch.utils.data import DataLoader

import os
import sys
from pathlib import Path

# Для правильного импорта из родительской директории
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# sys.path.append(os.path.join(Path(__file__).parent.parent.parent))

from gesture_recognition.model import ModelFactory
from utils import DataPreprocessor, GestureDataset, ModelTrainer

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

MODEL_SAVE_PATH = os.path.join(
    Path(__file__).parent.parent,
    'gesture_classifier.pt'
)

MAPPING_SAVE_PATH = os.path.join(
    Path(__file__).parent.parent,
    'gesture_mapping.json'
)

DATASET_PATH = os.path.join(
    Path(__file__).parent,
    'dataset'
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    print(f"Using device: {DEVICE}")
    
    preprocessor = DataPreprocessor(DATASET_PATH)
    X, y, class_names = preprocessor.load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_val_test_split(
        X, y, test_size=0.2, val_size=0.2
    )
    
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    test_dataset = GestureDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = X.shape[1]
    num_classes = len(class_names)
    
    model = ModelFactory.create_model(input_size, num_classes)
    
    trainer = ModelTrainer(model, DEVICE)
    best_val_acc = trainer.train(
        train_loader, val_loader, 
        num_epochs=EPOCHS, 
        learning_rate=LR
    )
    
    test_acc = trainer.evaluate(test_loader)
    print(f"Test accuracy: {test_acc:.3f}")
    
    ModelFactory.save_model(model, MODEL_SAVE_PATH)
    
    label_mapping = {i: label for i, label in enumerate(class_names)}
    ModelFactory.save_mapping(label_mapping, MAPPING_SAVE_PATH)


if __name__ == "__main__":
    main()