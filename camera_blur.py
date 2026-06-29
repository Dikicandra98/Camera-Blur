import cv2
import mediapipe as mp
import numpy as np
import sys

mp_hands   = mp.solutions.hands

BLUR_KERNEL          = 35         
FINGER_COUNT_TRIGGER = 2
DETECTION_CONF       = 0.7
TRACKING_CONF        = 0.6

FINGERS = [
    (8,  7,  5),   
    (12, 11, 9),   
    (16, 15, 13),  
    (20, 19, 17),  
]

THUMB = (4, 3, 2)

def is_palm_facing_camera(lm) -> bool:
    """
    Deteksi apakah telapak menghadap kamera menggunakan cross product
    dari vektor tangan (lebih akurat dari z saja).
    Landmark: wrist=0, index_mcp=5, pinky_mcp=17
    """
    wrist     = np.array([lm[0].x,  lm[0].y,  lm[0].z])
    index_mcp = np.array([lm[5].x,  lm[5].y,  lm[5].z])
    pinky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z])

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    normal = np.cross(v1, v2)

    return normal[2] < 0

def count_fingers(lm, palm_facing: bool) -> int:
    """
    Hitung jari terangkat dengan logika berbeda sesuai orientasi telapak.
    - Palm facing (telapak ke kamera): ujung jari lebih tinggi dari PIP (y lebih kecil)
    - Back facing (punggung ke kamera): kebalikannya
    """
    count = 0

    for tip, pip_, mcp in FINGERS:
        tip_y = lm[tip].y
        pip_y = lm[pip_].y
        mcp_y = lm[mcp].y

        if palm_facing:
            extended = tip_y < pip_y
        else:
            extended = tip_y > pip_y

        if extended:
            count += 1

    tip_id, ip_id, mcp_id = THUMB

    mcp_pt = np.array([lm[mcp_id].x, lm[mcp_id].y])
    ip_pt  = np.array([lm[ip_id].x,  lm[ip_id].y])
    tip_pt = np.array([lm[tip_id].x, lm[tip_id].y])

    v1 = ip_pt  - mcp_pt
    v2 = tip_pt - ip_pt

    thumb_angle = v1[0] * v2[1] - v1[1] * v2[0]

    if palm_facing:
        extended = thumb_angle > 0.002
    else:
        extended = thumb_angle < -0.002

    if extended:
        count += 1

    return count
    
def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,           
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("=" * 50)
    print("  Camera Blur App — Siap digunakan")
    print("  Angkat 2 jari untuk mengaktifkan blur")
    print("  Bekerja untuk telapak depan maupun dibalik")
    print("  Tekan Q atau ESC untuk keluar")
    print("=" * 50)

    cv2.namedWindow("Camera Blur App", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result    = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        display       = frame.copy()
        total_fingers = 0
        hand_detected = False

        if result.multi_hand_landmarks:
            hand_detected = True
            hand_lm       = result.multi_hand_landmarks[0]
            lm            = hand_lm.landmark

            palm_facing   = is_palm_facing_camera(lm)
            total_fingers = count_fingers(lm, palm_facing)

        blur_active = hand_detected and (total_fingers == FINGER_COUNT_TRIGGER)

        if blur_active:
            display = cv2.GaussianBlur(display, (BLUR_KERNEL, BLUR_KERNEL), 0)

        cv2.imshow("Camera Blur App", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\nAplikasi ditutup.")
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
