import cv2

import os
import argparse
from pathlib import Path

from utils import DataCollector


DATASET_PATH = os.path.join(
    Path(__file__).parent,
    'dataset'
)


def main():
    parser = argparse.ArgumentParser(description='Collect gesture data')
    parser.add_argument('--gesture', type=str, required=True,
                       help='Name of the gesture to collect')
    args = parser.parse_args()
    
    collector = DataCollector(args.gesture, DATASET_PATH)
    
    cap = cv2.VideoCapture(0)
    
    print("Press SPACE to start/stop collecting data")
    print("Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        processed_frame, _ = collector.process_frame(frame)
        
        mode_text = "PROCESS" if collector.is_collecting else "STOP"
        color = (0, 255, 0) if collector.is_collecting else (0, 0, 255)
        
        cv2.putText(processed_frame, f"MODE: {mode_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(processed_frame, f"Gesture: {args.gesture}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Samples: {collector.counter}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, "Press SPACE to toggle", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Data Collection', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            if collector.is_collecting:
                collector.toggle_collection()
            break
        elif key == 32:  # SPACE
            collector.toggle_collection()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()