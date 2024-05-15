#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from line import Line
from utility import Frame

from dataclasses import dataclass
import cv2 as cv
import numpy as np


@dataclass
class GoBoard:

    go_board_params = {
        'board_size': 19,
        'min_line_length': 200,
        'horizontal_slope': 0.5,
        'vertical_slope': 2,
        'epsilon_unify': 5,
        }

    def __init__(self, channel: int, go_board_params: dict = None):
        while True:
            if go_board_params is None:
                go_board_params = GoBoard.go_board_params
                self.go_board_params = go_board_params
            else:
                self.go_board_params = go_board_params

            Line.outer_points(channel)
            self.fields = GoBoard.go_board_intersections(
                channel,
                min_line_length=go_board_params.get('min_line_length'),
                horizontal_slope=go_board_params.get('horizontal_slope'),
                vertical_slope=go_board_params.get('vertical_slope'),
                epsilon_unify=go_board_params.get('epsilon_unify'),
                board_size=go_board_params.get('board_size'))
            try:
                self.coordinates = self.get_field_coordinates(
                    self.go_board_params.get('epsilon_unify'))
                if len(self.coordinates) != self.go_board_params.get(
                        'board_size'):
                    raise ValueError('coordinates not set properly')
                break
            except Exception as e:
                print(e)

    def go_board_intersections(
            channel: int,
            min_line_length: int,
            horizontal_slope: float,
            vertical_slope: float,
            epsilon_unify: int,
            board_size: int) -> list:
        """
        generates the fields of a go board from image with the help of the
        Line class

        Parameters
        ----------
        channel : int
            Channel of the webacam.
        min_line_length : int
            minimal length of the found lines that pass filtering.
        horizontal_slope : float
            The slope that is supposed resemble horizontal lines
            (Approximation). Think about errors that are made when the lines
            are not actually horizontal due to camera angle and camera
            direction.
        vertical_slope : float
            The slope that is supposed resemble vertical lines (Approximation).
            Think about errors that are made when the lines are not actually
            vertical due to camera angle and camera direction.
        epsilon_unify : int
            epsilon to determine if two lines represent the same line on the
            board.
        board_size : int
            Board size.

        Returns
        -------
        list
            intersection of the found lines, resembles fields.
        """
        cap = cv.VideoCapture(channel)
        if not cap.isOpened():
            print("Cannot open camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            try:
                adaptive_thresh = Frame(frame).adaptive_thresh()
                lines = Line.hough_lines_p(adaptive_thresh)
                for line in lines:
                    line.filter_line(
                        min_line_length=min_line_length,
                        horizontal_slope=horizontal_slope,
                        vertical_slope=vertical_slope)

                extended_horizontal_lines = [line.extend_line(frame.shape)
                                             for line in Line.horizontal_lines]
                extended_vertical_lines = [line.extend_line(frame.shape)
                                           for line in Line.vertical_lines]

                sorted_horizontal_lines = Line.sort_horizontal_lines(
                    extended_horizontal_lines)
                sorted_vertical_lines = Line.sort_vertical_lines(
                    extended_vertical_lines)

                unique_horizontal_lines = Line.unique_horizontal_lines(
                    sorted_horizontal_lines,
                    epsilon_unify)
                unique_vertical_lines = Line.unique_vertical_lines(
                    sorted_vertical_lines,
                    epsilon_unify)

                unique_horizontal_lines = [line
                                           for line in unique_horizontal_lines
                                           if line.is_outer_line()]
                unique_vertical_lines = [line
                                         for line in unique_vertical_lines
                                         if line.is_outer_line()]

                intersections, _ = Line.line_intersections(
                    [*unique_horizontal_lines, *unique_vertical_lines])

                Line.horizontal_lines.clear()
                Line.vertical_lines.clear()

                if len(intersections) == board_size ** 2:
                    break

            except Exception as e:
                print(e)
                break
        cap.release()
        cv.destroyAllWindows()

        if intersections:
            return intersections
        else:
            return []

    def get_field_coordinates(self, epsilon: int) -> list:
        """
        Assigns coordinate names to the found points

        Parameters
        ----------
        epsilon : int
            Threshold value in order to measure distances between points and
            coordinate components.

        Returns
        -------
        list
            List of tuples: [(str, [x,y])].

        """

        fields = self.fields.copy()
        bound = self.go_board_params.get('board_size')

        distances = [np.linalg.norm([
            self.fields[i][0] - self.fields[i-1][0],
            self.fields[i][1] - self.fields[i-1][1]]) for i in range(
                1, len(self.fields))]
        distances.sort()

        for index in range(len(distances)-1):
            if distances[index+1] - distances[index] > epsilon:
                max_dist = distances[index]
                break

        coordinates = []
        col = []

        for i in range(bound):
            if not col and not coordinates:
                col.append((chr(97) + chr(97), fields[0]))
                fields.pop(0)
            if not col:
                for pt in fields:
                    if np.linalg.norm(
                            np.array(coordinates[i-1][0][1]) - np.array(pt)) <= max_dist and np.abs(
                                coordinates[i-1][0][1][1] - pt[1]) <= 2*epsilon:
                        col.append((chr(97 + i) + chr(97), pt))
                        index = fields.index(pt)
                        fields.pop(index)
                        break
            if col:
                for j in range(bound-1):
                    for pt in fields:
                        if np.linalg.norm(
                                np.array(col[-1][1]) - np.array(pt)) <= max_dist and np.abs(
                                    col[-1][1][0] - pt[0]) <= epsilon:
                            col.append((chr(97 + i) + chr(97 + j+1), pt))
                            index = fields.index(pt)
                            fields.pop(index)
                            break
                coordinates.append(col)
                col = []
        if coordinates:
            return coordinates
        else:
            return []
