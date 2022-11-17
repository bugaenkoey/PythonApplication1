
import cv2
from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp                  # библиотека mediapipe (распознавание рук)


camera = cv2.VideoCapture(0)            # получаем изображение с камеры (0 - порядковый номер камеры в системе)
mpHands = mp.solutions.hands            # подключаем раздел распознавания рук
hands = mpHands.Hands()                 # создаем объект класса "руки"
mpDraw = mp.solutions.drawing_utils     # подключаем инструменты для рисования

while True:
    good, img = camera.read()
    imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    
    results = hands.process(imgRGB)                             # получаем результат распознавания
    if results.multi_hand_landmarks:                            # если обнаружили точки руки
        for handLms in results.multi_hand_landmarks:            # получаем координаты каждой точки

            # при помощи инструмента рисования проводим линии между точками
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

             # работаем с каждой точкой по отдельности
            # создаем список от 0 до 21 с координатами точек
            for id, point in enumerate(handLms.landmark):
                # получаем размеры изображения с камеры и масштабируем
                width, height, color = img.shape
                width, height = int(point.x * height), int(point.y * width)

             #   p[id] = height           # заполняем массив высотой каждой точки
                if id == 8:              # выбираем нужную точку
                    # рисуем нужного цвета кружок вокруг выбранной точки
                    cv2.circle(img, (width, height), 15, (255, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (width, height), 15, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Imge",img)
    if cv2.waitKey(1) == ord('q'):
        break

    pass