import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omniart_eye_generator import generate_eye

# Eye will be a PIL Image if eye_count is 1, otherwise a list of Image is returned
eye = generate_eye('hazel', eye_count=1)
eye.show()
