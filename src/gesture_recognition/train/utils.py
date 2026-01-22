from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import pandas as pd

import os
import sys
from pathlib import Path
from tqdm import tqdm

# для правильных импортов
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gesture_recognition.model import HandLandmarkDetector


class DataCollector:    
    def __init__(self, gesture_name, dataset_path: str):
        self.gesture_name = gesture_name
        self.detector = HandLandmarkDetector()
        self.data = []
        self.is_collecting = False
        self.counter = 0
        self.dataset_path = dataset_path
        Path(dataset_path).mkdir(exist_ok=True)
        
    def toggle_collection(self):    # Переключение режима сбора данных
        self.is_collecting = not self.is_collecting
        if not self.is_collecting and self.data:
            self.save_data()
            
    def process_frame(self, frame):
        landmarks = None
        
        if self.is_collecting:
            landmarks = self.detector.extract_landmarks(frame)
            if landmarks is not None:
                self.data.append(landmarks)
                self.counter += 1
        
        if landmarks is not None:
            frame = self.detector.draw_landmarks(frame, landmarks)
            
        return frame, landmarks
    
    def save_data(self):
        if not self.data:
            return
        
        df = pd.DataFrame(self.data)
        df['label'] = self.gesture_name
        
        data_filepath = os.path.join(self.dataset_path, f"{self.gesture_name}.csv")
        df.to_csv(data_filepath, index=False)
        print(f"Saved {len(self.data)} samples to {data_filepath}")
        
        self.data = []
        self.counter = 0


class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataPreprocessor:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self):
        all_data = []

        csv_files = list(Path(self.dataset_path).glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.dataset_path}")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        X = combined_df.drop('label', axis=1).values
        y = combined_df['label'].values
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded, self.label_encoder.classes_
    
    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.2):
        assert test_size + val_size < 1.0, "Invalid test and val size"

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, 
            random_state=42, stratify=y_train_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    

class ModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in tqdm(range(num_epochs), desc="Model train"):
            _, train_acc = self.train_epoch(train_loader, optimizer)
            val_acc = self.evaluate(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]:')
                print(f'\tTrain Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f} (Best Val Acc: {best_val_acc:.3f})')
        
        print(f"\nBest validation accuracy: {best_val_acc:.3f}")
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return best_val_acc
