#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import containing class where you run the board detection."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Transformation:
    """Contains all linear transformation to detect a goboard."""

    def transform_lowest_line(
            self,
            x_start: np.int32,
            y_start: np.int32,
            x_end: np.int32,
            y_end: np.int32) -> np.ndarray:
        """
        Transform the lowest line of the bounding lines of the goboard.

        Parameters
        ----------
        x_end : np.int32
            Ending x-value of the lowest line.
        y_end : np.int32
            Ending y-value of the lowest line.
        x_start : np.int32
            Starting x-value of the lowest line.
        y_start : np.int32
            Starting y-value of the lowest line.

        Returns
        -------
        np.ndarray
            Chooses the point of the line, which is lower. If they are
            horizontal the input is returned, else transformed by the
            translated_rotation method.

        """
        if y_end < y_start:
            x_proc, y_proc, x_trans, y_trans = x_end, y_end, x_start, y_start
        elif y_end > y_start:
            x_proc, y_proc, x_trans, y_trans = x_start, y_start, x_end, y_end
        else:
            return np.array([x_start, y_start, x_end, y_end])

        return self.translated_rotation(
            x_proc,
            y_proc,
            x_trans,
            y_trans,
            adjust_negative=True)

    @staticmethod
    def translated_rotation(
            x_trans: np.int32,
            y_trans: np.int32,
            x_proc: np.int32,
            y_proc: np.int32,
            adjust_negative: bool = False) -> np.ndarray:
        """
        Translate and rotate a point by its relation to another point.

        Parameters
        ----------
        x_trans : np.int32
            X-value that will be used to translate the other point. X-value
            of the rotational axis
        y_trans : np.int32
            Y-value that will be used to translate the other point. Y-value
            of the rotational axis
        x_proc: np.int32
            Processed x-value.
        y_proc : np.int32
            Processed y-value.

        Returns
        -------
        np.ndarray
            Array containing the translated and rotated point [new_x, new_y]
            and the fixed point [x_fixed, y_fixed].
        """
        # Translate to origin
        translation = np.array([x_proc - x_trans, y_proc - y_trans])

        # Compute the rotation angle
        alpha = np.pi + np.arctan2(translation[1], translation[0])

        # Rotation matrix
        R = np.array([
            [np.cos(alpha), np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])

        # Apply rotation
        rotated = np.matmul(R, translation)

        # Translate back
        translated_back = rotated + np.array([x_trans, y_trans])

        if translated_back[0] < 0 and adjust_negative:
            translated_back[0] += 2*(x_trans - translated_back[0])

        # Return rotated and fixed points
        return np.round([
                translated_back[0],
                translated_back[1],
                x_trans,
                y_trans
                ]).astype(np.int32)
