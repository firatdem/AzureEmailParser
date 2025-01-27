import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class DetectionResult:
    def __init__(self, result_number=None, coordinate_location=None, flag=None):
        self.result_number = result_number
        self.coordinate_location = coordinate_location
        self.flag = flag

    def __str__(self):
        return f"Result {self.result_number}: {self.coordinate_location} (Flag: {self.flag})"

