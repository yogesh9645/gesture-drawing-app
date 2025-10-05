import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

shape = "None"
shapes_list = []
holding_shape = False
shape_selected = False
shape_size = 20
buttons = {
    "Circle": (180, 10, 230, 50),
    "Square": (280, 10, 330, 50),
    "Triangle": (380, 10, 430, 50),
}
clear_button = (500, 10, 580, 50)
held_shape_position = None  


screen_width = 1280  
screen_height = 720  
cv2.namedWindow("Resize with Pinch", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resize with Pinch", screen_width, screen_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    for btn_text, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.putText(frame, btn_text[0], (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.rectangle(frame, (clear_button[0], clear_button[1]), (clear_button[2], clear_button[3]), (200, 50, 50), -1)
    cv2.putText(frame, "Clear", (clear_button[0] + 15, clear_button[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        right_hand = results.multi_hand_landmarks[0]  
        left_hand = results.multi_hand_landmarks[1] if hand_count > 1 else None  

        mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)
        index_finger = right_hand.landmark[8]
        middle_finger = right_hand.landmark[12]
        ix, iy = int(index_finger.x * w), int(index_finger.y * h)
        mx, my = int(middle_finger.x * w), int(middle_finger.y * h)

        if not holding_shape:
            for btn_text, (x1, y1, x2, y2) in buttons.items():
                if x1 < ix < x2 and y1 < iy < y2:
                    shape = btn_text  
                    shape_selected = True  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), -1)
                    cv2.putText(frame, btn_text[0], (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

  
        if clear_button[0] < ix < clear_button[2] and clear_button[1] < iy < clear_button[3]:
            shapes_list.clear()  

        finger_distance = abs(ix - mx) + abs(iy - my)
        if finger_distance < 20 and shape_selected:
            holding_shape = True
            held_shape_position = (ix, iy)  

        elif holding_shape and finger_distance > 30:
            shapes_list.append((shape, held_shape_position, shape_size))
            holding_shape = False
            shape_selected = False
            held_shape_position = None
            shape = "None"
            
        if left_hand:
            mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
            thumb_tip = left_hand.landmark[4]
            index_tip = left_hand.landmark[8]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            pinch_distance = np.linalg.norm([thumb_x - index_x, thumb_y - index_y])

            shape_size = int(pinch_distance / 2)  
            shape_size = max(10, min(shape_size, 100))  
    for shape_type, pos, size in shapes_list:
        if shape_type == "Circle":
            cv2.circle(frame, pos, size, (255, 255, 255), -1)
        elif shape_type == "Square":
            cv2.rectangle(frame, (pos[0] - size, pos[1] - size),
                          (pos[0] + size, pos[1] + size), (255, 255, 255), -1)
        elif shape_type == "Triangle":
            pts = np.array([[pos[0], pos[1] - size],
                            [pos[0] - size, pos[1] + size],
                            [pos[0] + size, pos[1] + size]], np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))

    if holding_shape and held_shape_position:
        if shape == "Circle":
            cv2.circle(frame, held_shape_position, shape_size, (0, 255, 255), 2)
        elif shape == "Square":
            cv2.rectangle(frame, (held_shape_position[0] - shape_size, held_shape_position[1] - shape_size),
                          (held_shape_position[0] + shape_size, held_shape_position[1] + shape_size), (0, 255, 255), 2)
        elif shape == "Triangle":
            pts = np.array([[held_shape_position[0], held_shape_position[1] - shape_size],
                            [held_shape_position[0] - shape_size, held_shape_position[1] + shape_size],
                            [held_shape_position[0] + shape_size, held_shape_position[1] + shape_size]], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

    cv2.imshow("Resize with Pinch", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
