#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from go_board import GoBoard
from stone_detection import StoneDetection

import cv2 as cv
import numpy as np


channel = 2
radius = 8
delay = 3
bounds = (480, 640)
intensity_threshold_dark = 35
intensity_threshold_light = -50

go_board = GoBoard(channel)
fields = go_board.fields
coordinates = go_board.coordinates
board_size = go_board.go_board_params.get('board_size')

circles = StoneDetection.all_circle_environments(
    board_size,
    coordinates,
    bounds,
    radius=radius)

avg_env_intensity = StoneDetection.measure_avg_intensitiy(
    circles,
    coordinates,
    radius,
    channel,
    delay)

cap = cv.VideoCapture(channel)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        avg_env_intensity_check = StoneDetection.avg_env_intensity(
            gray,
            circles,
            coordinates,
            radius)
        errors = avg_env_intensity_check - avg_env_intensity
        for i in range(len(errors)):
            for j in range(len(errors)):
                print(errors)
                if errors[i][j] <= intensity_threshold_light:
                    print(coordinates[i][j][1])
                    cv.circle(
                        frame,
                        [coordinates[i][j][1][1], coordinates[i][j][1][0]],
                        5,
                        (125, 125, 0), -1)

                if errors[i][j] >= intensity_threshold_dark:
                    print(coordinates[i][j][1])
                    cv.circle(
                        frame,
                        [coordinates[i][j][1][1], coordinates[i][j][1][0]],
                        5,
                        (0, 125, 125),
                        -1)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    except Exception as e:
        print(e)
        break

cap.release()
cv.destroyAllWindows()
