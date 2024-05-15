#!/usr/bin/environment python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class StoneDetection():

    @staticmethod
    def draw_environment(
            environment: np.ndarray,
            image: np.ndarray,
            color: (np.uint8, np.uint8, np.uint8)):
        for pt in environment:
            image[pt[0]][pt[1]] = color

    @staticmethod
    def circle_points(
            point: list,
            bounds: tuple,
            radius: int):
        circle = []
        for x in range(point[1] - radius, point[1] + radius):
            for y in range(point[0] - radius, point[0] + radius):
                if (x - point[1]) ** 2 + (y - point[0]) ** 2 <= radius**2:
                    if 0 <= y < bounds[0] and 0 <= x < bounds[1]:
                        circle.append([y, x])
        return np.array(circle)

    @staticmethod
    def all_circle_environments(
            board_size: int,
            coordinates: list,
            bounds: tuple,
            radius: int) -> np.ndarray:

        circles = []
        for i in range(len(coordinates)):
            row = []
            for j in range(len(coordinates[i])):
                row.append(
                    StoneDetection.circle_points(
                        coordinates[i][j][1],
                        bounds,
                        radius))
            circles.append(row)
        return circles

    @staticmethod
    def avg_env_intensity(
            gray: np.ndarray,
            circles: list,
            coordinates: list,
            radius: int) -> list:

        avg_env_intensities = []
        for i in range(len(coordinates)):
            row = []
            for j in range(len(coordinates)):
                intensities = np.array(
                        [gray[circles[i][j][k][0]][circles[i][j][k][1]]
                         for k in range(len(circles[i][j]))])
                avg = np.int32(np.round(np.sum(intensities) / len(intensities)))
                row.append(avg)
            avg_env_intensities.append(row)
        return np.array(avg_env_intensities)

    @staticmethod
    def measure_avg_intensitiy(
            circles: list,
            coordinates: list,
            radius: int,
            channel: int,
            delay: int) -> list:

        start_time = time.time()
        total_elapsed_time = 0

        cap = cv.VideoCapture(channel)
        if not cap.isOpened():
            print("Cannot open camera")

        while total_elapsed_time < delay:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            try:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                avg_env_intensities = StoneDetection.avg_env_intensity(
                    gray,
                    circles,
                    coordinates,
                    radius)
            except Exception as e:
                print(e)
                break

            total_elapsed_time = time.time() - start_time

        cap.release()
        cv.destroyAllWindows()

        return avg_env_intensities