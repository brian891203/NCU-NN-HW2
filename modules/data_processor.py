import numpy as np


class DataProcessor:
    def load_data(self, file_path):
        """
        Load and preprocess data from a specified file.

        Parameters:
        - file_path: str, path to the data file.

        Returns:
        - train_x: np.ndarray, feature matrix.
        - train_y: np.ndarray, normalized target values.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = np.loadtxt(file)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None, None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, None

        # Separate features and target
        train_x = data[:, :-1]
        train_y = data[:, -1]

        # Normalize target values
        self.train_y_min = -40
        self.train_y_max = 40
        train_y = (train_y - self.train_y_min) / (self.train_y_max - self.train_y_min)

        return train_x, train_y

    def load_map(self, file_path):
        """
        Load map data from a specified file, including initial and finish locations.

        Parameters:
        - file_path: str, path to the map file.

        Returns:
        - init_loc: np.ndarray, initial location coordinates.
        - finish_loc: np.ndarray, list of finish location coordinates.
        - data_map: np.ndarray, array of map points.
        """
        data_map = []
        finish_loc = []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Read and parse the initial location
                first_line = file.readline().strip().split(',')
                init_loc = np.array([float(coord) for coord in first_line])

                # Read and parse the first finish location
                sec_line = file.readline().strip().split(',')
                finish_loc.append(np.array([float(sec_line[0]), float(sec_line[1])]))

                # Read and parse the second finish location
                thr_line = file.readline().strip().split(',')
                finish_loc.append(np.array([float(thr_line[0]), float(thr_line[1])]))

                # Read and parse the remaining map data points
                for line in file:
                    s = line.strip().split(',')
                    if len(s) >= 2:
                        point = np.array([float(s[0]), float(s[1])])
                        data_map.append(point)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None, None, None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, None, None

        # Convert lists to NumPy arrays
        finish_loc = np.array(finish_loc)
        data_map = np.array(data_map)

        return init_loc, finish_loc, data_map


# Example usage
if __name__ == "__main__":
    processor = DataProcessor()

    # Load data
    data_file = "path/to/data_file.txt"  # Replace with your data file path
    train_x, train_y = processor.load_data(data_file)
    if train_x is not None and train_y is not None:
        print(f"Features shape: {train_x.shape}, Targets shape: {train_y.shape}")

    # Load map
    map_file = "path/to/map_file.txt"  # Replace with your map file path
    init_loc, finish_loc, data_map = processor.load_map(map_file)
    if init_loc is not None and finish_loc is not None and data_map is not None:
        print(f"Initial Location: {init_loc}")
        print(f"Finish Locations: {finish_loc}")
        print(f"Map Data Points: {data_map.shape}")
