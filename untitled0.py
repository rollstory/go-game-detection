#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:42:17 2024

@author: igor
"""

extended_horizontal_lines = [line.extend_line(frame.shape)
                             for line in Line.horizontal_lines]

sorted_horizontal_lines = Line.sort_horizontal_lines(
    extended_horizontal_lines)

unique_horizontal_lines = Line.unique_horizontal_lines(
    sorted_horizontal_lines, 5)

unique_horizontal_lines = [
    line
    for line in unique_horizontal_lines
    if line.is_outer_line()]

print(len(unique_horizontal_lines))

if len(unique_horizontal_lines) == self.board_size:
    unique_horizontal_lines.pop(0)
    unique_horizontal_lines.pop(-1)
    
Line.draw_lines(
    unique_horizontal_lines,
    frame
    )
cv.imshow('horizontal lines', frame)
if key == ord('h'):
    cv.destroyWindow('horizontal lines')

intersections = []
for line in unique_horizontal_lines:
    inter, _ = Line.line_intersections(
        [line, *vertical_lines])
    intersections.append(inter)
    
Line.draw_intersections(intersections, frame)
cv.imshow('intersections', frame)
if key == ord('i'):
    cv.destroyWindow('intersections')

Line.horizontal_lines.clear()
Line.vertical_lines.clear()

print(len(intersections))

if len(intersections) == self.board_size-2 and all(
        [len(_i) == self.board_size for _i in intersections]):
    intersections.insert(0, hor_grid[0])
    intersections.append(hor_grid[1])
    break

fields = go_board.fields
coordinates = go_board.coordinates
board_size = go_board.go_board_params.get('board_size')
tracker = go_board.tracker

circles = StoneDetection.all_circle_environments(
    board_size,
    fields,
    bounds,
    radius=radius)

avg_env_intensity = StoneDetection.measure_avg_intensitiy(
    circles,
    fields,
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
        black = np.zeros((480, 640, 3))
        for line in circles:
            for env in line:
                StoneDetection.draw_environment(
                    env,
                    black,
                    (255, 255, 255))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        avg_env_intensity_check = StoneDetection.avg_env_intensity(
            gray,
            circles,
            coordinates,
            radius)
        errors = avg_env_intensity_check - avg_env_intensity
        print(errors)

        GoBoard.update_tracker(
            errors,
            intensity_threshold_light,
            intensity_threshold_dark,
            frame)

        # if GoBoard.track_game():
        #    pass

        cv.imshow('frame', frame)
        cv.imshow('black', black)
        if cv.waitKey(1) == ord('q'):
            break
    except Exception as e:
        print(e)
        break

cap.release()
cv.destroyAllWindows()


                for line in lines:
                    line.filter_line(
                        min_line_length=GoBoard.go_board_params.get(
                            'min_line_length'),
                        horizontal_slope=GoBoard.go_board_params.get(
                            'horizontal_slope'),
                        vertical_slope=GoBoard.go_board_params.get(
                            'vertical_slope'))
        
                extended_horizontal_lines = [line.extend_line(frame.shape)
                                             for line in Line.horizontal_lines]

                sorted_horizontal_lines = Line.sort_horizontal_lines(
                    extended_horizontal_lines)

                unique_horizontal_lines = Line.unique_horizontal_lines(
                    sorted_horizontal_lines, 5)
                
                print(unique_horizontal_lines)

                unique_horizontal_lines = [
                    line
                    for line in unique_horizontal_lines
                    if line.is_outer_line()]

                print(len(unique_horizontal_lines))

                
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
