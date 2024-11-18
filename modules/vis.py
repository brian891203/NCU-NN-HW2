import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrow, Rectangle

from .simple_geometry import Line2D, Point2D


class Visualizer:
    def __init__(self):
        # Initialize the figure and axis for training visualization
        self.train_figure = plt.figure(figsize=(10, 9.5))
        self.train_axis = self.train_figure.add_subplot(111)
        self.train_axis.set_title("Training Results")

        self.front_text = self.train_axis.text(
            0.05, 0.95, '', transform=self.train_axis.transAxes, 
            verticalalignment='top', fontsize=10, color='black'
        )
        self.right_text = self.train_axis.text(
            0.05, 0.90, '', transform=self.train_axis.transAxes, 
            verticalalignment='top', fontsize=10, color='black'
        )
        self.left_text = self.train_axis.text(
            0.05, 0.85, '', transform=self.train_axis.transAxes, 
            verticalalignment='top', fontsize=10, color='black'
        )

    def plot_route_map(self, initial_position, finish_zone, route_data, 
                       car_image_path='car.png', initial_heading=90):
        """
        Plots the environment map including the route, start and finish zones, 
        initial car position, direction arrow, and car image.

        Parameters:
        - initial_position: numpy array with initial x and y coordinates
        - finish_zone: numpy array defining the finish area
        - route_data: numpy array containing the route points
        - car_image_path: path to the car image file
        - initial_heading: initial direction angle in degrees
        """
        # Plot the route lines
        for i in range(len(route_data) - 1):
            start = Point2D(route_data[i, 0], route_data[i, 1])
            end = Point2D(route_data[i + 1, 0], route_data[i + 1, 1])
            route_line = Line2D(start, end)
            self.train_axis.plot(
                [route_line.p1.x, route_line.p2.x],
                [route_line.p1.y, route_line.p2.y],
                color='blue'
            )

        # Plot the starting line
        start_line = Line2D(Point2D(-6, 0), Point2D(6, 0))
        self.train_axis.plot(
            [start_line.p1.x, start_line.p2.x],
            [start_line.p1.y, start_line.p2.y],
            color='yellow'
        )

        # Plot the finish zone as a rectangle
        finish_rectangle = Rectangle(
            (finish_zone[0, 0], finish_zone[1, 1]),
            width=finish_zone[1, 0] - finish_zone[0, 0],
            height=finish_zone[0, 1] - finish_zone[1, 1],
            edgecolor='red',
            facecolor='red'
        )
        self.train_axis.add_patch(finish_rectangle)

        # Plot the initial car position
        car_initial = Point2D(initial_position[0], initial_position[1])
        self.car_marker = self.train_axis.scatter(
            car_initial.x, car_initial.y, color='blue', zorder=5
        )
        self.car_outline = Circle(
            (car_initial.x, car_initial.y),
            radius=3,
            linewidth=0.5,
            edgecolor='blue',
            facecolor='none'
        )
        self.train_axis.add_patch(self.car_outline)
        self.train_axis.set_xlim(-7, 50)

        # Plot the initial direction arrow

        self.arrow_length = 5
        arrow_dx = self.arrow_length * np.cos(np.radians(initial_heading))
        arrow_dy = self.arrow_length * np.sin(np.radians(initial_heading))
        self.direction_arrow = FancyArrow(
            car_initial.x, car_initial.y,  # Starting point
            arrow_dx, arrow_dy,           # Length and direction
            color='orange',
            width=0.3,                    # Width of arrow shaft
            head_width=1,                 # Width of arrow head
            head_length=1.5,              # Length of arrow head
            length_includes_head=True     # Include head in total length
        )
        self.train_axis.add_patch(self.direction_arrow)

        # # Add the car image to the plot
        # car_image = mpimg.imread(car_image_path)
        # image_zoom = 0.04
        # image_box = OffsetImage(car_image, zoom=image_zoom)
        # annotation_box = AnnotationBbox(
        #     image_box,
        #     (0.12, 0.1),
        #     frameon=False,
        #     xycoords='axes fraction',
        #     boxcoords='axes fraction'
        # )
        # self.train_axis.add_artist(annotation_box)

        # self.train_figure.set_size_inches(12, 8)

        return self.train_figure

    def animate_movement(self, position_data, front_distances, 
                        right_distances, left_distances, headings):
        """
        Animates the car movement along the plotted route.

        Parameters:
        - position_data: numpy array with x and y positions over time
        - front_distances: list or array of front sensor distances
        - right_distances: list or array of right sensor distances
        - left_distances: list or array of left sensor distances
        - headings: list or array of heading angles in degrees
        """
        self.front_text.set_text('')
        self.right_text.set_text('')
        self.left_text.set_text('')

        # Initialize the trajectory line
        trajectory_color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        trajectory_line, = self.train_axis.plot(
            [], [], 
            marker='o', 
            markersize=4, 
            linestyle='-', 
            linewidth=2, 
            color=trajectory_color
        )
        trajectory_history_x, trajectory_history_y = [], []

        def update_frame(frame_index):
            """
            Updates the plot for each frame of the animation.

            Parameters:
            - frame_index: current frame number
            """
            # Update car position
            current_x, current_y = position_data[0][frame_index], position_data[1][frame_index]
            self.car_outline.set_center((current_x, current_y))
            self.car_marker.set_offsets((current_x, current_y))

            # Remove the old direction arrow
            if hasattr(self, 'direction_arrow') and self.direction_arrow in self.train_axis.patches:
                self.direction_arrow.remove()

            # Update direction arrow based on current heading
            current_heading = headings[frame_index]
            arrow_dx = self.arrow_length * np.cos(np.radians(current_heading))
            arrow_dy = self.arrow_length * np.sin(np.radians(current_heading))
            
            # Add new direction arrow
            self.direction_arrow = FancyArrow(
                current_x, current_y,  # Starting point
                arrow_dx, arrow_dy,    # Length and direction
                color='orange',
                width=0.3,             # Width of arrow shaft
                head_width=1,          # Width of arrow head
                head_length=1.5,       # Length of arrow head
                length_includes_head=True
            )
            self.train_axis.add_patch(self.direction_arrow)

            # Update sensor distance texts
            self.front_text.set_text(f'Front Distance: {front_distances[frame_index]:.2f}')
            self.right_text.set_text(f'Right Distance: {right_distances[frame_index]:.2f}')
            self.left_text.set_text(f'Left Distance: {left_distances[frame_index]:.2f}')

            # Update trajectory history
            trajectory_history_x.append(current_x)
            trajectory_history_y.append(current_y)
            trajectory_line.set_data(trajectory_history_x, trajectory_history_y)

            return (
                self.car_outline, 
                trajectory_line, 
                self.car_marker, 
                self.front_text, 
                self.right_text, 
                self.left_text, 
                self.direction_arrow  # Include the new arrow in the returned objects
            )

        # Create the animation
        animation = FuncAnimation(
            self.train_figure, 
            update_frame, 
            frames=len(position_data[0]), 
            interval=50, 
            blit=True
        )
        animation.repeat = False

        return animation

    def clear_plot(self):
        """
        Clears the training axis for fresh plotting.
        """
        self.train_axis.clear()
        self.train_axis.set_title("Training Results")
