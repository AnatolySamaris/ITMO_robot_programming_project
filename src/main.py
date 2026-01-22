import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from gesture_control_interface import GestureControlWindow, GestureCommand
from gesture_recognition.model import GesturePredictor


PREDICTOR_INPUT_SIZE = 21 * 2
ROBOT_MAX_SPEED = 50.0

GESTURE_CLASSIFIER_PATH = os.path.join(
    Path(__file__).parent,
    'gesture_recognition',
    'gesture_classifier.pt'
)

GESTURE_MAPPING_PATH = os.path.join(
    Path(__file__).parent,
    'gesture_recognition',
    'gesture_mapping.json'
)

ROBOT_SIMULATION_PATH = os.path.join(
    Path(__file__).parent,
    'robot_model',
    'wheeled_robot.xml'
)

# Маппинг жестов в команды. Имена команд заданы для читаемости
# gesture_id: (command_name, left_wheel_speed, right_wheel_speed)
GESTURE_TO_COMMAND = {
    4: ("forward", ROBOT_MAX_SPEED, ROBOT_MAX_SPEED),       # thumbsup
    0: ("backward", -ROBOT_MAX_SPEED, -ROBOT_MAX_SPEED),    # five
    2: ("left", ROBOT_MAX_SPEED, -ROBOT_MAX_SPEED),         # one
    5: ("right", -ROBOT_MAX_SPEED, ROBOT_MAX_SPEED),        # two
}


def main():
    predictor = GesturePredictor(
        classifier_path=GESTURE_CLASSIFIER_PATH,
        mapping_path=GESTURE_MAPPING_PATH,
        input_size=PREDICTOR_INPUT_SIZE
    )

    gesture_source = GestureControlWindow(predictor, video_source=0)
    gesture_source.start()

    try:
        model = mujoco.MjModel.from_xml_path(ROBOT_SIMULATION_PATH)
        data = mujoco.MjData(model)

        left_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_motor')
        right_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_motor')

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                cmd: GestureCommand = gesture_source.get_command()

                wheel_left, wheel_right = 0.0, 0.0
                if cmd.gesture_id in GESTURE_TO_COMMAND:
                    _, wheel_left, wheel_right = GESTURE_TO_COMMAND[cmd.gesture_id]

                data.ctrl[left_idx] = wheel_left
                data.ctrl[right_idx] = wheel_right

                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(0.01)

    finally:
        gesture_source.stop()


if __name__ == "__main__":
    main()
