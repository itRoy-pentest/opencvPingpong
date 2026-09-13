import cv2
import math
import urllib.request
from pathlib import Path

import mediapipe as mp


# ============================================================
# НАСТРОЙКИ
# ============================================================

CAMERA_ID = 0

MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

BALL_RADIUS = 12

GRAVITY = 0.45

BOUNCE_STRENGTH = 0.9

HORIZONTAL_SPEED = 0.0

MODEL_PATH = Path("hand_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


# ============================================================
# ЗАГРУЗКА МОДЕЛИ MEDIAPIPE
# ============================================================

def download_model():
    """
    Загружает модель MediaPipe Hand Landmarker,
    если её ещё нет рядом с main.py.
    """

    if MODEL_PATH.exists():
        return

    print("MediaPipe model not found.")
    print("Downloading hand_landmarker.task...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        raise RuntimeError(
            "Не удалось скачать модель MediaPipe.\n"
            f"Ошибка: {error}\n\n"
            "Скачай hand_landmarker.task вручную и положи "
            "рядом с main.py."
        )

    print("Model downloaded successfully.")


# ============================================================
# ИНИЦИАЛИЗАЦИЯ MEDIAPIPE
# ============================================================

def initHandelObject():
    """
    Создаёт Hand Landmarker через новый MediaPipe Tasks API.
    """

    download_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    hand_landmarker = HandLandmarker.create_from_options(options)

    return hand_landmarker


# ============================================================
# КООРДИНАТЫ ПАЛЬЦА
# ============================================================

def getFingerCoord(hand_landmarks, frame):
    """
    Получает координаты:
        8 — кончик указательного пальца
        5 — основание указательного пальца
    """

    height, width = frame.shape[:2]

    finger_tip = hand_landmarks[8]
    finger_base = hand_landmarks[5]

    x1 = int(finger_tip.x * width)
    y1 = int(finger_tip.y * height)

    x2 = int(finger_base.x * width)
    y2 = int(finger_base.y * height)

    return x1, y1, x2, y2


# ============================================================
# РИСОВАНИЕ РУКИ
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


def drawHand(frame, landmarks):
    """
    Рисует точки и соединения руки.
    """

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )


# ============================================================
# СОЗДАНИЕ МЯЧА
# ============================================================

def createBall(frame_width, frame_height):
    """
    Начальное состояние мяча.
    """

    ball_pos_x = frame_width // 2
    ball_pos_y = 80

    ball_radius = BALL_RADIUS

    speed_x = HORIZONTAL_SPEED
    speed_y = 0.0

    return (
        float(ball_pos_x),
        float(ball_pos_y),
        ball_radius,
        speed_x,
        speed_y,
    )


# ============================================================
# РАССТОЯНИЕ ОТ ТОЧКИ ДО ОТРЕЗКА
# ============================================================

def distancePointToSegment(
    px,
    py,
    x1,
    y1,
    x2,
    y2
):
    """
    Расстояние от точки до отрезка.
    Также возвращает ближайшую точку на отрезке.
    """

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        distance = math.hypot(px - x1, py - y1)

        return distance, x1, y1

    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / (dx * dx + dy * dy)

    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    distance = math.hypot(
        px - closest_x,
        py - closest_y
    )

    return distance, closest_x, closest_y


# ============================================================
# СТОЛКНОВЕНИЕ МЯЧА С РАКЕТКОЙ
# ============================================================

def checkBallLineCollision(
    ball_x,
    ball_y,
    ball_radius,
    line_x1,
    line_y1,
    line_x2,
    line_y2
):
    """
    Проверяет столкновение мяча с отрезком,
    соединяющим два указательных пальца.
    """

    distance, closest_x, closest_y = distancePointToSegment(
        ball_x,
        ball_y,
        line_x1,
        line_y1,
        line_x2,
        line_y2
    )

    if distance <= ball_radius:
        return True, closest_x, closest_y

    return False, closest_x, closest_y


# ============================================================
# ОТСКОК ОТ ЛИНИИ
# ============================================================

def calculateBounce(
    ball_x,
    ball_y,
    speed_x,
    speed_y,
    line_x1,
    line_y1,
    line_x2,
    line_y2
):
    """
    Рассчитывает отражение скорости мяча
    относительно нормали к линии.
    """

    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1

    line_length = math.hypot(
        line_dx,
        line_dy
    )

    if line_length == 0:
        return speed_x, -abs(speed_y)

    # Единичный вектор линии
    tangent_x = line_dx / line_length
    tangent_y = line_dy / line_length

    # Нормаль к линии
    normal_x = -tangent_y
    normal_y = tangent_x

    # Скорость
    velocity_x = speed_x
    velocity_y = speed_y

    # Скалярное произведение скорости на нормаль
    velocity_normal = (
        velocity_x * normal_x +
        velocity_y * normal_y
    )

    # Отражение
    reflected_x = (
        velocity_x -
        2 * velocity_normal * normal_x
    )

    reflected_y = (
        velocity_y -
        2 * velocity_normal * normal_y
    )

    reflected_x *= BOUNCE_STRENGTH
    reflected_y *= BOUNCE_STRENGTH

    # Не позволяем мячу после столкновения
    # лететь слишком медленно.
    speed = math.hypot(
        reflected_x,
        reflected_y
    )

    minimum_speed = 5.0

    if speed < minimum_speed:
        reflected_x *= minimum_speed / max(speed, 0.001)
        reflected_y *= minimum_speed / max(speed, 0.001)

    return reflected_x, reflected_y


# ============================================================
# СТЕНЫ
# ============================================================

def handleWallCollisions(
    ball_x,
    ball_y,
    ball_radius,
    speed_x,
    speed_y,
    frame_width,
    frame_height
):
    """
    Обработка столкновений мяча с границами камеры.
    """

    # Левая стена
    if ball_x - ball_radius <= 0:

        ball_x = ball_radius
        speed_x = abs(speed_x)

    # Правая стена
    if ball_x + ball_radius >= frame_width:

        ball_x = frame_width - ball_radius
        speed_x = -abs(speed_x)

    # Верх
    if ball_y - ball_radius <= 0:

        ball_y = ball_radius
        speed_y = abs(speed_y)

    # Низ
    if ball_y + ball_radius >= frame_height:

        # Возвращаем мяч наверх
        ball_y = ball_radius + 5

        speed_y = 0

    return (
        ball_x,
        ball_y,
        speed_x,
        speed_y
    )


# ============================================================
# ОПРЕДЕЛЕНИЕ СТОРОНЫ РУКИ
# ============================================================

def getHandSide(hand_landmarks):
    """
    Определяет положение руки относительно изображения.

    Это НЕ определение Left/Right от MediaPipe.
    Нам здесь важно только понять,
    какая рука находится слева/справа на экране.
    """

    wrist = hand_landmarks[0]

    return wrist.x


# ============================================================
# ОСНОВНОЙ ЦИКЛ КАМЕРЫ
# ============================================================

def mainWhileCam(hand_landmarker):

    cam = cv2.VideoCapture(CAMERA_ID)

    if not cam.isOpened():
        print("ERROR: Не удалось открыть камеру.")
        return

    # Разрешение камеры
    frame_width = int(
        cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if frame_width <= 0:
        frame_width = 640

    if frame_height <= 0:
        frame_height = 480

    print(
        f"Camera resolution: "
        f"{frame_width}x{frame_height}"
    )

    # ========================================================
    # МЯЧ
    # ========================================================

    (
        ball_x,
        ball_y,
        ball_radius,
        speed_x,
        speed_y,
    ) = createBall(
        frame_width,
        frame_height
    )

    # ========================================================
    # ОКНО
    # ========================================================

    window_name = "OpenCV Ping Pong"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        1280,
        720
    )

    # Чтобы не было повторных столкновений
    # с линией в каждом кадре.
    was_colliding = False

    while cam.isOpened():

        ret, frame = cam.read()

        if not ret:
            print("ERROR: Не удалось получить кадр.")
            break

        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(frame, 1)

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = hand_landmarker.detect(
            mp_image
        )

        # ====================================================
        # РУКИ
        # ====================================================

        hand_lines = []

        if results.hand_landmarks:

            for landmarks in results.hand_landmarks:

                # Рисуем руку
                drawHand(
                    frame,
                    landmarks
                )

                # Координаты указательного пальца
                (
                    finger_x,
                    finger_y,
                    base_x,
                    base_y
                ) = getFingerCoord(
                    landmarks,
                    frame
                )

                # Рисуем линию указательного пальца
                cv2.line(
                    frame,
                    (finger_x, finger_y),
                    (base_x, base_y),
                    (255, 0, 0),
                    4
                )

                # Точка кончика
                cv2.circle(
                    frame,
                    (finger_x, finger_y),
                    8,
                    (0, 0, 255),
                    -1
                )

                # Добавляем координаты
                hand_lines.append(
                    (
                        finger_x,
                        finger_y
                    )
                )

        # ====================================================
        # РАКЕТКА ИЗ ДВУХ УКАЗАТЕЛЬНЫХ ПАЛЬЦЕВ
        # ====================================================

        if len(hand_lines) >= 2:

            point1 = hand_lines[0]
            point2 = hand_lines[1]

            line_x1, line_y1 = point1
            line_x2, line_y2 = point2

            # Рисуем основную линию
            cv2.line(
                frame,
                point1,
                point2,
                (0, 0, 255),
                6
            )

            # Точки
            cv2.circle(
                frame,
                point1,
                10,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                frame,
                point2,
                10,
                (0, 255, 255),
                -1
            )

            # =================================================
            # СТОЛКНОВЕНИЕ
            # =================================================

            (
                collision,
                closest_x,
                closest_y
            ) = checkBallLineCollision(
                ball_x,
                ball_y,
                ball_radius,
                line_x1,
                line_y1,
                line_x2,
                line_y2
            )

            # Столкновение обрабатываем только один раз,
            # чтобы мяч не получил несколько отскоков
            # подряд.
            if collision and not was_colliding:

                # Небольшой сдвиг мяча от линии
                dx = ball_x - closest_x
                dy = ball_y - closest_y

                distance = math.hypot(dx, dy)

                if distance > 0:

                    push = (
                        ball_radius - distance + 2
                    )

                    ball_x += (
                        dx / distance * push
                    )

                    ball_y += (
                        dy / distance * push
                    )

                # Рассчитываем отражение
                (
                    speed_x,
                    speed_y
                ) = calculateBounce(
                    ball_x,
                    ball_y,
                    speed_x,
                    speed_y,
                    line_x1,
                    line_y1,
                    line_x2,
                    line_y2
                )

                # Если мяч после столкновения
                # продолжает двигаться вниз,
                # принудительно отправляем его вверх.
                if speed_y > 0:

                    speed_y = -abs(speed_y)

            was_colliding = collision

        else:

            was_colliding = False

        # ====================================================
        # ФИЗИКА МЯЧА
        # ====================================================

        speed_y += GRAVITY

        ball_x += speed_x
        ball_y += speed_y

        # ====================================================
        # СТЕНЫ
        # ====================================================

        (
            ball_x,
            ball_y,
            speed_x,
            speed_y
        ) = handleWallCollisions(
            ball_x,
            ball_y,
            ball_radius,
            speed_x,
            speed_y,
            frame_width,
            frame_height
        )

        # ====================================================
        # МЯЧ
        # ====================================================

        cv2.circle(
            frame,
            (
                int(ball_x),
                int(ball_y)
            ),
            ball_radius,
            (0, 255, 0),
            -1
        )

        # Обводка мяча
        cv2.circle(
            frame,
            (
                int(ball_x),
                int(ball_y)
            ),
            ball_radius,
            (255, 255, 255),
            2
        )

        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            "Q - quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Ball: {int(ball_x)}, {int(ball_y)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Speed: {speed_x:.1f}, {speed_y:.1f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # ПОКАЗ КАДРА
        # ====================================================

        cv2.imshow(
            window_name,
            frame
        )

        # Q — выход
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # ========================================================
    # ЗАКРЫТИЕ
    # ========================================================

    cam.release()

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Starting OpenCV Ping Pong...")

    try:

        hand_landmarker = initHandelObject()

        mainWhileCam(
            hand_landmarker
        )

    except KeyboardInterrupt:

        print("\nProgram stopped.")

    except Exception as error:

        print("\nERROR:")
        print(error)

    finally:

        try:
            hand_landmarker.close()
        except Exception:
            pass

