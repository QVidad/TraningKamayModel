import csv
import copy
import argparse
import itertools

import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int)
    parser.add_argument("--height", help='cap height', type=int)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()
    return args

def main():
    # Argument parsing
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # Load the model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    # Initializing tensorflow lite model for the classification
    keypoint_classifier = KeyPointClassifier()
    # Initializing csv file(for the label)
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    # When collecting keypoints input
    mode = 0

    while True:
        fps = cvFpsCalc.get()
        # ESC = end
        key = cv.waitKey(10)
        if key == 27:  # ESC key
            break
        number, mode = select_mode(key, mode)
        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Hand detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # For the hand tracking
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.putText(debug_image, "Translation: ", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

# For pressing selected keys
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9 keys
        number = key - 48
    if key == 97:  # a key
        mode = 2
    if key == 113:  # q key
        mode = 1
    if key == 119:  # w key
        mode = 4
    if key == 115:  # s key
        mode = 5
    if key == 122:  # z key
        mode = 3
    if key == 120:  # x key
        mode = 6
    return number, mode

# For the mediapipe API
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

# For the keypoints
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Drawing the lines and keypoints
def draw_landmarks(image, landmark_point):
    # The Lines
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (61, 172, 242), 6)
        # Index
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (61, 172, 242), 6)
        # Middle
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (61, 172, 242), 6)
        # Ring
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (61, 172, 242), 6)
        # Pinky
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (61, 172, 242), 6)
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (61, 172, 242), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (61, 172, 242), 6)
    # The Keypoints
    for index, landmark in enumerate(landmark_point):
        # Thumb
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        # Index
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        # Middle
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        # Ring
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        # Pinky
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (53, 70, 123), -1)
    return image

# Display the translation
def draw_info_text(image, handedness, hand_sign_text):
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image,hand_sign_text, (200, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    return image

# Saving keypoints
def logging_csv(number, mode, landmark_list):
    if mode >= 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        if mode == 1:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number # change according to number
                writer.writerow([number, *landmark_list])
        elif mode == 2:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number + 10 # change according to number
                writer.writerow([number, *landmark_list])
        elif mode == 3:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number + 20 # change according to number
                writer.writerow([number, *landmark_list])
        elif mode == 4:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number + 30 # change according to number
                writer.writerow([number, *landmark_list])
        elif mode == 5:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number + 40 # change according to number
                writer.writerow([number, *landmark_list])
        elif mode == 6:
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                number = number + 50 # change according to number
                writer.writerow([number, *landmark_list])
    return

# Display collection information
def draw_info(image, fps, mode, number):
    if mode >=1:
        if mode == 1:
            cv.putText(image, "MODE: Collecting Keypoints (0+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        elif mode == 2:
            cv.putText(image, "MODE: Collecting Keypoints (10+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        elif mode == 3:
            cv.putText(image, "MODE: Collecting Keypoints (20+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        elif mode == 4:
            cv.putText(image, "MODE: Collecting Keypoints (30+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        elif mode == 5:
            cv.putText(image, "MODE: Collecting Keypoints (40+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        elif mode == 6:
            cv.putText(image, "MODE: Collecting Keypoints (50+)", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        if 0 <= number <= 9:
            if mode == 1:
                first_num = 0
            elif mode == 2:
                first_num = 1
            elif mode == 3:
                first_num = 2
            elif mode == 4:
                first_num = 3
            elif mode == 5:
                first_num = 4
            elif mode == 6:
                first_num = 5
            cv.putText(image, "NUM:"+str(first_num) + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
