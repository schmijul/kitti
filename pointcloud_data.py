import numpy as np
import os

class PointCloudData:
    """
    Handles on-disk storage (using memmap) of points.
    Each point: (x, y, z, intensity).
    """
    def __init__(self, filename: str, mode: str = 'r+', initial_capacity: int = 1024):
        self.filename = filename
        self.mode = mode
        self.initial_capacity = initial_capacity
        self.current_size = 0

        if mode.startswith('w'):
            if os.path.exists(filename):
                os.remove(filename)
            self._data = np.memmap(filename, dtype='float32', mode='w+', shape=(self.initial_capacity, 4))
            self.current_size = 0
        else:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File {filename} does not exist.")
            size = os.path.getsize(filename)
            if size % (4 * 4) != 0:
                raise ValueError("File size is not compatible with (N,4) float32 data.")
            num_points = size // (4 * 4)
            self._data = np.memmap(filename, dtype='float32', mode=mode, shape=(num_points, 4))
            self.current_size = num_points

    def append_points(self, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 4, "Points must have shape (N,4)"
        num_new = points.shape[0]
        new_total = self.current_size + num_new

        if new_total > self._data.shape[0]:
            new_capacity = max(self._data.shape[0] * 2, new_total, self.initial_capacity)
            self._data.flush()
            # Resize file
            with open(self.filename, 'r+b') as f:
                # Each point is 16 bytes (4 floats * 4 bytes)
                f.seek(new_capacity * 16 - 1)
                f.write(b'\x00')
            self._data = np.memmap(self.filename, dtype='float32', mode='r+', shape=(new_capacity, 4))

        self._data[self.current_size:new_total, :] = points
        self.current_size = new_total

    def get_points(self, indices: np.ndarray = None) -> np.ndarray:
        if indices is None:
            return self._data[:self.current_size, :]
        else:
            return self._data[indices, :]

    def flush(self):
        self._data.flush()

    def close(self):
        self.flush()
        del self._data

    def __len__(self):
        return self.current_size

