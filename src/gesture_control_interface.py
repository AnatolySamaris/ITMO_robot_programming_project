import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2

from gesture_recognition.model import GesturePredictor


@dataclass
class GestureCommand:
    gesture_id: Optional[int] = None
    gesture_name: str = "NO CONTROL"
    timestamp: float = 0.0


class GestureControlWindow:
    def __init__(self, predictor: GesturePredictor, video_source=0):
        self.predictor = predictor
        self.video_source = video_source
        self.current_command = GestureCommand()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_camera, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_command(self) -> GestureCommand:
        with self._lock:
            return self.current_command

    def _run_camera(self):
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            self._running = False
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра
            processed_frame, gesture_id, gesture_name = self.predictor.process_frame(frame)

            with self._lock:
                self.current_command = GestureCommand(
                    gesture_id=gesture_id,
                    gesture_name=gesture_name,
                    timestamp=time.time()
                )

            cv2.imshow("Gesture Control", cv2.flip(processed_frame, 1))
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
