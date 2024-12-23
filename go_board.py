#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from line import Line
from utility import Frame

from dataclasses import dataclass
import numpy as np
import cv2 as cv
#from bigtree import Node, list_to_tree


@dataclass
class GoBoard:

    go_board_params = {
        'board_size': 19,
        'min_line_length': 150,
        'horizontal_slope': 0.5,
        'vertical_slope': 2,
        'epsilon_unify': 8,
        }

    def __init__(self, channel: int, board_size: int, handicap: int):
        self.board_size = board_size
        self.board_dim = (
            GoBoard.go_board_params.get('board_size'),
            GoBoard.go_board_params.get('board_size'))
        Line.outer_points(channel)
        self.handicap = handicap
        self.fields = self.go_board_grid(channel)
        self.coordinates = self.get_grid_coordinates()
        self.tracker = np.zeros(self.board_dim, dtype=np.uint8)
        self.number_moves = 0
        self.game_state = []

    def go_board_grid(self, channel: int) -> np.ndarray:
        """
        Parameters
        ----------
        channel : int
            Channel of the webacam.

        Returns
        -------
        np.ndarray
            grid of the board.

        """
        
        vertical_lines, hor_grid = Line.create_vertical_lines(self.board_size)
                
        cap = cv.VideoCapture(channel)
        if not cap.isOpened():
            print("Cannot open camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            key = cv.waitKey(10) & 0xFF
            try:       
                adaptive_thresh = Frame(frame).adaptive_thresh()

                lines = Line.hough_lines_p(adaptive_thresh)
                
                line_params = self.go_board_params
                
                unique_horizontal_lines = Line.process_horizontal_lines(
                    lines,
                    line_params,
                    frame.shape
                    )

                if len(unique_horizontal_lines) == self.board_size:
                    unique_horizontal_lines.pop(0)
                    unique_horizontal_lines.pop(-1)                

                intersections = []
                for line in unique_horizontal_lines:
                    inter, _ = Line.line_intersections(
                        [line, *vertical_lines])
                    intersections.append(inter)
                
                frame_copy = frame.copy()
                Line.draw_lines(vertical_lines, frame_copy)
                    
                Line.draw_intersections(intersections, frame)
                cv.imshow('intersections', frame)
                cv.imshow('lines', frame_copy)

                if len(intersections) == self.board_size-2 and all(
                        [len(_i) == self.board_size for _i in intersections]):
                    intersections.insert(0, hor_grid[0])
                    intersections.append(hor_grid[1])
                    break

                Line.horizontal_lines.clear()
                Line.vertical_lines.clear()

                if key == ord('q'):
                    break
            except Exception as e:
                print(e)

        cap.release()
        cv.destroyAllWindows()

        return np.array(intersections)

    def get_grid_coordinates(self):
        return [[(chr(97 + i) + chr(97 + j), self.fields[i][j])
                 for j in range(len(self.fields[i]))]
                for i in range(len(self.fields))]

    def update_tracker(
            self,
            errors: np.ndarray,
            intensity_threshold_light: int,
            intensity_threshold_dark: int,
            frame: np.ndarray):

        for i in range(self.board_size):
            for j in range(self.board_size):
                if errors[i][j] <= intensity_threshold_light:
                    # visualisation
                    cv.circle(
                        frame,
                        [self.fields[i][j][1], self.fields[i][j][0]],
                        5,
                        (125, 125, 0), -1)

                    self.tracker[i][j] = 2

                if errors[i][j] >= intensity_threshold_dark:
                    # visualisation
                    cv.circle(
                        frame,
                        [self.fields[i][j][1], self.fields[i][j][0]],
                        5,
                        (125, 0, 125),
                        -1)

                    self.tracker[i][j] = 1

                if errors[i][j] > intensity_threshold_light \
                        and errors[i][j] < intensity_threshold_dark:
                    self.tracker[i][j] = 0

    @staticmethod
    def is_neighbor(field_a: list, field_b: list) -> bool:
        return (abs(field_a[0]-field_b[0]) == 1 and field_a[1] == field_b[1]) \
            or (abs(field_a[1]-field_b[1]) == 1 and field_a[0] == field_b[0])

    def define_groups(self):
        white_groups = []
        black_groups = []
        visited = set()

        def add_to_group(groups, i, j):
            for group in groups:
                for stone in group:
                    if self.is_neighbor(stone, (i, j)):
                        group.add((i, j))
                        return
            groups.append(set([(i, j)]))

        def delete_from_group(groups, i, j):
            for group in groups:
                if (i, j) in group:
                    group.remove((i, j))
                    return

        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) in visited:
                    continue

                if self.tracker[i][j] == 0:
                    visited.add((i, j))
                    delete_from_group(white_groups, i, j)
                    delete_from_group(black_groups, i, j)
                elif self.tracker[i][j] == 1:
                    add_to_group(black_groups, i, j)
                    visited.add((i, j))
                elif self.tracker[i][j] == 2:
                    add_to_group(white_groups, i, j)
                    visited.add((i, j))

        return list(visited), black_groups, white_groups

    def verify_tracker(self):
        visited, black_groups, white_groups = self.define_groups()

        def count_liberties(stone, visited):
            liberties = 0
            neighbors = [(stone[0] + 1, stone[1]), (stone[0] - 1, stone[1]),
                         (stone[0], stone[1] + 1), (stone[0], stone[1] - 1)]
            for neighbor in neighbors:
                if neighbor in visited:
                    liberties += 1
            return liberties

        def count_captured_stones(groups):
            stones_captured = 0
            for group in groups[:]:
                liberties = sum(count_liberties(stone, visited)
                                for stone in group)
                if liberties == 0:
                    stones_captured += len(group)
                    groups.remove(group)
            return stones_captured

        def valid_tracker(visited, black_groups, white_groups):
            black_stones_captured = count_captured_stones(black_groups)
            white_stones_captured = count_captured_stones(white_groups)
            black_stones = sum(len(group) for group in black_groups)
            white_stones = sum(len(group) for group in white_groups)

            if black_stones + black_stones_captured \
                    == white_stones + white_stones_captured:
                return True
            if black_stones + black_stones_captured \
                    == white_stones + white_stones_captured + self.handicap:
                return True
            return False

        return valid_tracker(visited, black_groups, white_groups)

"""    def set_game_state(self):
        if self.track_game():
            if self.game_state == []:
                root = Node()"""