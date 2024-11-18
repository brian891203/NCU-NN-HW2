import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import modules.K_means as K_means
import modules.RBF_model as RBF_model
import modules.sim as sim
import modules.vis as vis
from modules.data_processor import DataProcessor


class main_window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg='white')
        self.initUI()
        self.init_modules()

        self.canvas_widget_map: FigureCanvasTkAgg = None  # To hold training figure
        # self.canvas_widget_test: FigureCanvasTkAgg = None   # To hold testing figure
        # self.canvas_widget_loss: FigureCanvasTkAgg = None   # To hold loss curve figure
    
    def initUI(self):
        self.title('HW1 112526011')
        self.geometry('1200x700+10+0')
        self.minsize(1200, 700)  # Set minimum window size
        self.maxsize(1200, 700)  # Set maximum window size
        self.resizable(False, False)  # Disable window resizing by the user
        self.update_idletasks()  # Ensure widgets are initialized

        self.Content()
        self.Bottom_Bar()

    def init_modules(self):
        self.data_processor = DataProcessor()
        self.simulation = sim.Simulation()
        self.vis: vis.Visualizer = None
        self.route_fig: vis.Visualizer.train_figure = None
        self.model = None
    
    def Content(self):
        # Content_mainframe contains other Content_subframes
        self.Content_mainframe = tk.Frame(self, padx=0, pady=0, bg='white')
        self.Content_mainframe.grid(row=0, column=0)

        # sub1_component00 contains canvas1 for Training data
        # label_training = tk.Label(self.Content_mainframe, text="Training Result", bg='white', anchor=tk.W)
        # label_training.grid(row=0, column=0, padx=5, pady=0)
        self.sub1_component00 = tk.Frame(self.Content_mainframe, padx=10, pady=0, bg='white')
        self.sub1_component00.grid(row=0, column=0, padx=0, pady=0)
        self.canvas1 = tk.Canvas(self.sub1_component00, width=650, height=650, bg='white', highlightthickness=0, relief='solid')
        self.canvas1.grid(row=0, column=0)
        
    def Bottom_Bar(self):
        self.Bottom_Bar_mainframe = tk.Frame(self, padx=30, pady=50, bg='white')
        self.Bottom_Bar_mainframe.place(x=700, y=0, width=550, height=680)

        self.Bottom_left_frame = tk.Frame(self.Bottom_Bar_mainframe, padx=10, pady=0, bg='white', width=550, height=680)
        self.Bottom_left_frame.place(x=20, y=0)

        # Button to open file
        self.open_file_button = tk.Button(self.Bottom_left_frame, text='Browse', command=self.open_train_data_file_event, width=15, height=2, bg='white')
        self.open_file_button.place(x=10, y=10)  # Use place to position the button within the left frame
        self.open_map_button = tk.Button(self.Bottom_left_frame, text='Browse', command=self.open_map_data_file_event, width=15, height=2, bg='white')
        self.open_map_button.place(x=10, y=80)  # Use place to position the button within the left frame

        # Label to display the selected file path
        self.file_label = tk.Label(self.Bottom_left_frame, text="No file selected", bg='white', anchor='w')
        self.file_label.place(x=150, y=20)  # Position label using place

        # Label to display the selected file path
        self.map_file_label = tk.Label(self.Bottom_left_frame, text="No file selected", bg='white', anchor='w')
        self.map_file_label.place(x=150, y=90)  # Position label using place

        label_learning_rate = tk.Label(self.Bottom_left_frame, text="Learning rate :", bg='white', anchor=tk.W)
        label_learning_rate.place(x=20, y=145)
        self.learning_rate_entry = tk.Entry(self.Bottom_left_frame, width=7, highlightthickness=1)
        self.learning_rate_entry.place(x=150, y=145)
        self.learning_rate_entry.insert(0, "0.005")  # Default value for learning rate

        label_epoch = tk.Label(self.Bottom_left_frame, text="Epoch :", bg='white', anchor=tk.W)
        label_epoch.place(x=20, y=200)
        self.epoch_entry = tk.Entry(self.Bottom_left_frame, width=7, highlightthickness=1)
        self.epoch_entry.place(x=150, y=200)
        self.epoch_entry.insert(0, "30")  # Default value for epoch counts

        self.traing_button = tk.Button(self.Bottom_left_frame, text='Training', command=self.train_model_event, width=15, height=2, bg='white')
        self.traing_button.place(x=10, y=250)

        # Training Process Display
        label_process = tk.Label(self.Bottom_left_frame, text="Epoch & Loss :", bg='white', anchor=tk.W)
        label_process.place(x=10, y=320)

        self.weights_text = tk.Text(self.Bottom_left_frame, width=50, height=12, highlightthickness=2)
        self.weights_text.place(x=15, y=350)

        self.go_button = tk.Button(self.Bottom_left_frame, text='Go!!!', command=self.go_event, width=15, height=2, bg='white')
        self.go_button.place(x=10, y=550)
        
    def open_train_data_file_event(self):
        # Open file and read data
        file_path = filedialog.askopenfilename(initialdir="./data", filetypes=[("Text Files", "*.txt *.TXT")])
        if file_path:
            self.file_label.config(text=os.path.basename(file_path))
            self.data = self.data_processor.load_data(file_path)
            self.init_model(self.data)

            # print(data)
            # print(self.data[0].shape[1])

        else:
            tk.messagebox.showerror("Error", "No file selected..")
        
    def open_map_data_file_event(self):
        # Open file and read data
        self.map_file_path = filedialog.askopenfilename(initialdir="./data", filetypes=[("Text Files", "*.txt *.TXT")])
        if self.map_file_path:
            self.map_file_label.config(text=os.path.basename(self.map_file_path))

            self.visualizer = vis.Visualizer()
            self.figure_route = self.visualizer.train_figure

            self.init_loc, self.finish_loc, self.data_map = self.data_processor.load_map(self.map_file_path)
            self.figure_route = self.visualizer.plot_route_map(self.init_loc, self.finish_loc, self.data_map)
            
            # 計算對應的 Tkinter Canvas 實際尺寸 (轉換英吋為像素)
            figure_size = self.figure_route.get_size_inches()
            dpi = self.figure_route.get_dpi()
            canvas_width = int(figure_size[0] * dpi)
            canvas_height = int(figure_size[1] * dpi)
            self.canvas1.config(width=canvas_width, height=canvas_height)

            self.canvas_widget_map = FigureCanvasTkAgg(self.figure_route, master=self.canvas1)
            self.canvas_widget_map.get_tk_widget().place(relx=0, rely=0, relwidth=0.7, relheight=0.7)
            self.canvas_widget_map.draw()

        else:
            tk.messagebox.showerror("Error", "No file selected..")

    def init_model(self, data):
        kmean = K_means.KMeans(n_clusters=data[0].shape[1], data=data[0])
        center, sigma = kmean.fit()
        self.model = RBF_model.RBF(data[0].shape[1])
        
        self.model.set_data(data, center, sigma)

    def train_model_event(self):
        """Train the model and update the UI in real-time."""
        if self.model is not None:
            # Clear previous content
            self.weights_text.delete(1.0, tk.END)
            # if self.map_file_path:
            #     self.visualizer = vis_refactror_with_gem.Visualizer()
            #     self.figure_route = self.visualizer.train_figure

            #     # 重新載入地圖
            #     self.init_loc, self.finish_loc, self.data_map = self.data_processor.load_map(self.map_file_path)

            #     # 繪製地圖
            #     self.figure_route = self.visualizer.plot_route_map(self.init_loc, self.finish_loc, self.data_map)

            #     # 更新 Tkinter Canvas 中的地圖
            #     self.canvas_widget_map.get_tk_widget().destroy()  # 刪除舊的地圖
            #     self.canvas_widget_map = FigureCanvasTkAgg(self.figure_route, master=self.canvas1)
            #     self.canvas_widget_map.get_tk_widget().place(relx=0, rely=0, relwidth=0.7, relheight=0.7)
            #     self.canvas_widget_map.draw()
            
            # Get epoch and learning rate
            epoch = int(self.epoch_entry.get())
            learning_rate = float(self.learning_rate_entry.get())

            # Display initial info
            weights_info = f"Epoch {epoch}, Learning rate: {learning_rate}\n"
            self.weights_text.insert(tk.END, weights_info)

            # Initialize training flag
            training_end_flag = False

            while not training_end_flag:
                # Get the generator for training
                training_generator = self.model.train(num_epochs=epoch, learning_rate=learning_rate)

                # Iterate through generator and update UI
                for training_info in training_generator:
                    # Update the text box with training info
                    weights_info = f"Epoch {training_info['epoch']}, Loss: {training_info['loss']:.10f}\n"
                    self.weights_text.insert(tk.END, weights_info)
                    
                    # Scroll the text box to the end
                    self.weights_text.see(tk.END)
                    
                    # Force update of the UI
                    self.update_idletasks()

                    # Check if training is complete
                    if int(training_info['epoch']) == epoch:
                        training_end_flag = True
            
        else:
            tk.messagebox.showerror("Error", "No data loaded. Please open a file first.")

    def go_event(self):
        """
        Simulates the movement event, updates positions, distances, orientations,
        visualizes the movement, and saves tracking data to a file.
        """
        
        iteration = 0
        answer_orientation = self._initialize_answer_orientation()

        # Initialize the first position
        self._initialize_position(answer_orientation, iteration)

        # Lists to store simulation data
        trajectory = {
            'x': [],
            'y': [],
            'front_distance': [],
            'right_distance': [],
            'left_distance': [],
            'orientation': [],
            'phi': []
        }

        # Run the simulation loop
        while True:
            self._update_position(answer_orientation, iteration)
            self._record_trajectory(trajectory)

            if self._has_reached_target_area() or self._has_collision():
                break

            iteration += 1

        # Convert recorded data to NumPy arrays
        trajectory_arrays = self._convert_trajectory_to_arrays(trajectory)

        # Visualize the movement
        self._visualize_movement(trajectory_arrays)

        # Save tracking data if a model is present
        if self.model:
            self._save_tracking_data(trajectory_arrays)

    def _initialize_answer_orientation(self) -> Optional[np.ndarray]:
        """
        Initializes the answer orientation based on the presence of a model.

        Returns:
            Optional[np.ndarray]: The initialized answer orientations or None.
        """
        if self.model is None:
            return self.data[1] * 80 - 40
        return None

    def _initialize_position(self, answer_orientation: Optional[np.ndarray], iteration: int):
        """
        Initializes the starting position of the simulation.

        Args:
            answer_orientation (Optional[np.ndarray]): The answer orientations.
            iteration (int): The current iteration count.
        """
        self.x, self.y, self.sita, self.phi, distances = self.simulation.calculate_next_position(
            current_x=0.0,
            current_y=0.0,
            steering_angle=0.0,
            orientation=90.0,
            model=self.model,
            input_shape=self.data[0].shape[1],
            answer_orientations=answer_orientation,
            current_iteration=iteration
        )

    def _update_position(self, answer_orientation: Optional[np.ndarray], iteration: int):
        """
        Updates the current position by calculating the next position.

        Args:
            answer_orientation (Optional[np.ndarray]): The answer orientations.
            iteration (int): The current iteration count.
        """
        self.x, self.y, self.sita, self.phi, distances = self.simulation.calculate_next_position(
            current_x=self.x,
            current_y=self.y,
            steering_angle=self.sita,
            orientation=self.phi,
            model=self.model,
            input_shape=self.data[0].shape[1],
            answer_orientations=answer_orientation,
            current_iteration=iteration
        )

    def _record_trajectory(self, trajectory: dict):
        """
        Records the current state into the trajectory dictionary.

        Args:
            trajectory (dict): The dictionary storing trajectory data.
        """
        self.x, self.y, self.sita, self.phi, self.distances = self.simulation.calculate_next_position(
            self.x, self.y, self.sita, self.phi, self.model, self.data[0].shape[1],
            trajectory.get('answer_orientation'), trajectory.get('iteration')
        )
        trajectory['x'].append(self.x)
        trajectory['y'].append(self.y)
        trajectory['front_distance'].append(self.distances[0])
        trajectory['right_distance'].append(self.distances[1])
        trajectory['left_distance'].append(self.distances[2])
        trajectory['orientation'].append(self.sita)
        trajectory['phi'].append(self.phi)

    def _has_reached_target_area(self) -> bool:
        """
        Checks if the current position is within the target area.

        Returns:
            bool: True if within target area, else False.
        """
        TARGET_X_MIN, TARGET_X_MAX = 18, 30
        TARGET_Y_MIN, TARGET_Y_MAX = 40, 50
        return TARGET_X_MIN <= self.x <= TARGET_X_MAX and TARGET_Y_MIN <= self.y <= TARGET_Y_MAX

    def _has_collision(self) -> bool:
        """
        Checks if any of the distances indicate a collision or obstacle.

        Returns:
            bool: True if collision is detected, else False.
        """
        MIN_DISTANCE_THRESHOLD = 3
        return any(d - MIN_DISTANCE_THRESHOLD <= 0 for d in [self.distances[0], self.distances[1], self.distances[2]])

    def _convert_trajectory_to_arrays(self, trajectory: dict) -> dict:
        """
        Converts trajectory lists to NumPy arrays.

        Args:
            trajectory (dict): The dictionary storing trajectory data.

        Returns:
            dict: A dictionary with NumPy arrays.
        """
        return {key: np.array(values) for key, values in trajectory.items()}

    def _visualize_movement(self, trajectory_arrays: dict):
        """
        Visualizes the movement using the visualizer.

        Args:
            trajectory_arrays (dict): The dictionary containing trajectory NumPy arrays.
        """
        self.figure_route = self.visualizer.animate_movement(
            position_data=(trajectory_arrays['x'], trajectory_arrays['y']),
            front_distances=trajectory_arrays['front_distance'],
            right_distances=trajectory_arrays['right_distance'],
            left_distances=trajectory_arrays['left_distance'],
            headings=trajectory_arrays['phi']
        )

    def _save_tracking_data(self, trajectory_arrays: dict):
        """
        Saves the tracking data to a file based on input shape.

        Args:
            trajectory_arrays (dict): The dictionary containing trajectory NumPy arrays.
        """
        input_shape = self.data[0].shape[1]
        if input_shape == 3:
            file_path = 'track4D.txt'
            save_data = np.column_stack((
                trajectory_arrays['front_distance'],
                trajectory_arrays['right_distance'],
                trajectory_arrays['left_distance'],
                trajectory_arrays['orientation']
            ))
        elif input_shape == 5:
            file_path = 'track6D.txt'
            save_data = np.column_stack((
                trajectory_arrays['x'],
                trajectory_arrays['y'],
                trajectory_arrays['front_distance'],
                trajectory_arrays['right_distance'],
                trajectory_arrays['left_distance'],
                trajectory_arrays['orientation']
            ))
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        np.savetxt(file_path, save_data, fmt='%.6f', delimiter=' ')

    def save_img_event(self):
        if self.img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])
            if file_path:
                save_image = self.img
                save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
                save_image = cv2.resize(save_image, (self.original_width, self.original_height))
                cv2.imwrite(file_path, save_image)
                tk.messagebox.showinfo("Success", f"Image saved successfully at {file_path}")
            else:
                tk.messagebox.showerror("Error", "No file save path selected.")
        else:
            tk.messagebox.showerror("Error", "Please open an image first.")

    def show_histogram_event(self):
        pass

if __name__ == '__main__':
    app = main_window()
    app.mainloop()