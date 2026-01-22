import cv2
import mediapipe as mp

import numpy as np

import torch
import torch.nn as nn

import json


class HandLandmarkDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks and results.multi_hand_landmarks[0]:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []  # Приводим к виду [x1, y1, x2, y2, ..., x21, y21]
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            return np.array(landmarks)
        
        return None
    
    def draw_landmarks(self, image, landmarks):
        if landmarks is not None:
            h, w = image.shape[:2]
            for i in range(0, len(landmarks), 2):
                x, y = int(landmarks[i] * w), int(landmarks[i+1] * h)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image


class GestureClassifier(nn.Module):
    def __init__(self, input_size: int = 42, num_classes: int = 5):
        super(GestureClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)
    

class ModelFactory:
    @staticmethod
    def create_model(input_size: int = 42, num_classes: int = 5) -> GestureClassifier:
        return GestureClassifier(input_size, num_classes)
    
    @staticmethod
    def load_model(model_path: str, input_size: int = 42, num_classes: int = 5) -> GestureClassifier:
        model = GestureClassifier(input_size, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    @staticmethod
    def save_model(model: GestureClassifier, model_path: str) -> None:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    @staticmethod
    def load_mapping(mapping_path: str) -> dict:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping

    @staticmethod
    def save_mapping(mapping: dict, mapping_path: str) -> None:
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)


class GesturePredictor:
    def __init__(self, classifier_path: str, mapping_path: str, input_size: int = 42):

        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_factory = ModelFactory()
        self.mapping = self.model_factory.load_mapping(mapping_path)
        self.classifier = self.model_factory.load_model(
            classifier_path, input_size, len(self.mapping)
        ).to(self.device)
        self.detector = HandLandmarkDetector()
    
    def predict(self, image):
        landmarks = self.detector.extract_landmarks(image)
        
        # Если нет руки в кадре или рука в кадре частично
        if landmarks is None or len(landmarks) != self.input_size:
            return None, "NO CONTROL"
        
        features_tensor = torch.FloatTensor(landmarks).unsqueeze(0)
        with torch.no_grad():
            outputs = self.classifier(features_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            
            gesture_id = predicted.item()
            
            gesture_name = self.mapping.get(str(gesture_id), "UNDEFINED")
            
            return gesture_id, gesture_name
    
    def process_frame(self, frame):
        landmarks = self.detector.extract_landmarks(frame)
        if landmarks is not None:
            frame = self.detector.draw_landmarks(frame, landmarks)

        gesture_id, gesture_name = self.predict(frame)
        
        return frame, gesture_id, gesture_name
