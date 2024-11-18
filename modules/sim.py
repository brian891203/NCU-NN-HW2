# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:01 2023

@author: user
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Point


class Simulation:
    """
    模擬車輛在地圖上的移動，計算與牆壁的距離以及更新車輛的位置和方向。
    """
    
    def __init__(self):
        # 定義地圖牆壁為線段列表
        self.map_walls: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
            ((-6, -3), (-6, 22)),
            ((-6, 22), (18, 22)),
            ((18, 22), (18, 50)),
            ((18, 50), (30, 50)),
            ((30, 50), (30, 10)),
            ((30, 10), (6, 10)),
            ((6, 10), (6, -3)),
            ((6, -3), (-6, -3))
        ]
    
    def distance_to_nearest_wall(self, position: Point, angle: float) -> float:
        """
        計算給定位置和角度下，車頭前方和左右45度方向上與牆壁的最短距離。

        參數：
        - position (Point): 車輛當前的位置
        - angle (float): 車輛當前的朝向角度（度）

        回傳：
        - float: 與最近牆壁的距離，若無交點則回傳無窮大
        """
        # 定義射線的終點，這裡設置為一個足夠遠的點（例如100單位長）
        ray_length = 100.0
        direction_x = position.x + np.cos(np.radians(angle)) * ray_length
        direction_y = position.y + np.sin(np.radians(angle)) * ray_length
        ray = LineString([position.coords[0], (direction_x, direction_y)])
        
        # 儲存所有交點的距離
        intersection_distances: List[float] = []
        
        for wall_start, wall_end in self.map_walls:
            wall = LineString([wall_start, wall_end])
            intersection = ray.intersection(wall)
            
            if intersection.is_empty:
                continue
            elif intersection.geom_type == 'Point':
                distance = Point(intersection).distance(position)
                intersection_distances.append(distance)
            elif intersection.geom_type == 'MultiPoint':
                for pt in intersection:
                    distance = Point(pt).distance(position)
                    intersection_distances.append(distance)
        
        if intersection_distances:
            return min(intersection_distances)
        else:
            return np.inf  # 如果沒有交點，回傳無窮大

    def calculate_next_position(
        self,
        current_x: float,
        current_y: float,
        steering_angle: float,
        orientation: float,
        model: Optional[object],
        input_shape: int,
        answer_orientations: List[float],
        current_iteration: int
    ) -> Tuple[float, float, float, float, Tuple[float, float, float]]:
        """
        根據當前位置、方向和感測器距離，計算下一個位置和方向。

        參數：
        - current_x (float): 當前的 x 座標
        - current_y (float): 當前的 y 座標
        - steering_angle (float): 當前的轉向角度（度）
        - orientation (float): 當前的車輛朝向角度（度）
        - model (Optional[object]): 用於預測下一個方向的模型
        - input_shape (int): 輸入特徵的形狀（3 或 5）
        - answer_orientations (List[float]): 預定義的方向列表
        - current_iteration (int): 當前的迭代次數

        回傳：
        - Tuple[float, float, float, float, Tuple[float, float, float]]:
          下一個 x 座標、y 座標、方向角度、角度 phi，以及感測器距離 (前方、右側、左側)
        """
        position = Point(current_x, current_y)
        
        # 計算感測器距離
        front_distance = self.distance_to_nearest_wall(position, orientation)
        left_distance = self.distance_to_nearest_wall(position, orientation + 45)
        right_distance = self.distance_to_nearest_wall(position, orientation - 45)
        
        # 更新位置
        new_x = current_x + np.cos(np.radians(orientation + steering_angle)) + \
                np.sin(np.radians(steering_angle)) * np.sin(np.radians(orientation))
        new_y = current_y + np.sin(np.radians(orientation + steering_angle)) - \
                np.sin(np.radians(steering_angle)) * np.cos(np.radians(orientation))
        
        # 更新 phi 角度
        # 假設某種轉向規則，這裡需要根據具體需求調整
        phi_change = np.rad2deg(np.arcsin((2 * np.sin(np.radians(steering_angle))) / 6))
        new_phi = orientation - phi_change
        
        # 準備模型輸入
        if input_shape == 3:
            model_input = np.array([front_distance, right_distance, left_distance])
        elif input_shape == 5:
            model_input = np.array([new_x, new_y, front_distance, right_distance, left_distance])
        else:
            raise ValueError("Unsupported input shape. Must be 3 or 5.")
        
        # 預測新的方向角度
        if model is not None:
            predicted_steering = model.forward(model_input) * 80 - 40  # 根據具體模型調整
        else:
            predicted_steering = answer_orientations[current_iteration]
        
        return new_x, new_y, predicted_steering, new_phi, (front_distance, right_distance, left_distance)
