import cv2
import numpy as np
from picamera import PiCamera
import time
from picamera.array import PiRGBArray



def steering(averaged_lines):
    a1 = 0
    a2 = 0
    a = 0
    if len(averaged_lines) == 2 and averaged_lines != 0:
        if (averaged_lines[1][0][0] - averaged_lines[1][0][2]) and (averaged_lines[0][0][0] - averaged_lines[0][0][2]) != 0:
            a1 = np.arctan((averaged_lines[1][0][3] - averaged_lines[1][0][1]) / (averaged_lines[1][0][2] - averaged_lines[1][0][0]))
            a2 = np.arctan((averaged_lines[0][0][1] - averaged_lines[0][0][3]) / (averaged_lines[0][0][0] - averaged_lines[0][0][2]))
            a = (a1 + a2) / 2
            a = int((360 / (2*np.pi)) * a)
            return 90 + a


def make_points(image, line):
    slope, intercept = line
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    if slope != 0:
        y1 = int(image.shape[0])  # bottom of the image
        y2 = int(y1*0.7)  # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        if x1 >= 320:
            x1 = 320
        if x1 <= -320:
            x1 = -320
        if x2 >= 320:
            x2 = 320
        if x2 <= -320:
            x2 = -320
    return [[x1, y1, x2, y2]]


def average_slope_intercept(cadru, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines
    height, width, _ = cadru.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(cadru, left_fit_average))
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append((make_points(cadru, right_fit_average)))
    if lane_lines is not None:
        return lane_lines


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny_met = cv2.Canny(blur, 150, 450)
    return canny_met


def display_lines(img, linie):
    line_frame = np.zeros_like(img)
    if linie is not None:
        for line in linie:
            for x1, y1, x2, y2 in line:
                cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_frame


def region_of_interest(canny_met):
    height = canny_met.shape[0]
    width = canny_met.shape[1]
    mask = np.zeros_like(canny_met)

    triangle = np.array([[
        (0, height),
        (width, height),
        (width/2, height/2), ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny_met, mask)
    return masked_image


camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    dim = (320, 240)
    frame = cv2.resize(frame, dim)
    canny_image = canny(frame)
    #cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=2)
    averaged_lines = average_slope_intercept(frame, lines)
    line_angle = steering(averaged_lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    print(len(averaged_lines), line_angle)
    rawCapture.truncate(0)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
