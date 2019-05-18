import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omniart_eye_generator import *

figure = plt.figure()

fixed_noise = generate_noise()

for index, eye_class in enumerate(classes):
    # Eye will be a PIL Image if eye_count is 1, otherwise a list of Image is returned
    eye = generate_eye(eye_class, eye_count=1, noise=fixed_noise)
    subplot = figure.add_subplot(2, 4, index + 1)
    subplot.title.set_text(eye_class)
    subplot.axis('off')
    plt.imshow(eye)
plt.show()
