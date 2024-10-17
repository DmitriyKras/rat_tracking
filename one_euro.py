import math
from yolo import YOLOPoseTRT
import cv2



def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, dt, x=None):
        """Compute the filtered signal."""
        t_e = dt
        if x is None:
            x = self.x_prev + self.dx_prev * dt
        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return self.x_prev
    


yolo = YOLOPoseTRT('yolo/yolov8m-rat.engine', (640, 384), n_kpts=13)

cap = cv2.VideoCapture('rec_2024_07_09_13_42_01_up.mp4')
_, frame = cap.read()


n = 2
dt = 1 / 13.7
i = 1

x, y = yolo(frame)[0][:2]
x_filter = OneEuroFilter(0, x, min_cutoff=10, beta=0.)
y_filter = OneEuroFilter(0, y, min_cutoff=10, beta=0.)

while cap.isOpened():
    ret, frame = cap.read()


    if not ret or cv2.waitKey(0) == ord('q'):
        break
    i += 1
    boxes = yolo(frame)
    

    if len(boxes):
        if i == n:
            x, y = boxes[0][:2]
            x_f = x_filter(dt, x)
            y_f = y_filter(dt, y)
            i = 0
        else:
            x_f = x_filter(dt, None)
            y_f = y_filter(dt, None)

        frame = cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        frame = cv2.circle(frame, (int(x_f), int(y_f)), 3, (0, 0, 255), -1)

    cv2.imshow('FRAME', frame)
