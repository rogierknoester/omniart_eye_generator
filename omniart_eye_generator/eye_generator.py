import os
from typing import List, Union

import PIL
import numpy as np
import torch
from PIL.Image import Image

from omniart_eye_generator.generator import Generator, latent_space_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator()
state_path = os.path.join(os.path.dirname(__file__), 'generator_model_state_e150_d25x25.pth')
generator.load_state_dict(torch.load(state_path))

generator.eval()
generator.to(device)

classes = ('amber', 'blue', 'brown', 'gray', 'grayscale', 'green', 'hazel', 'red')


def __make_label(class_name: str, eye_count=1) -> torch.Tensor:
    if class_name not in classes:
        raise ValueError('The class name %s is not available'.format(class_name))

    onehot_labels = torch.zeros(eye_count, len(classes))
    # Create a tensor of {eye_count} items with the index of the class as value
    tensor_of_class_indexes = torch.full((eye_count, 1), classes.index(class_name)).long()
    # Mark the class index as 1 to get a one hot tensor
    onehot_labels.scatter_(1, tensor_of_class_indexes.view(eye_count, 1), 1)
    return onehot_labels.to(device)


def generate_noise(eye_count=1) -> torch.Tensor:
    return torch.randn(eye_count, latent_space_size, 1, 1).to(device)


def __to_image(eyes_tensors) -> List[Image]:
    eyes = eyes_tensors.numpy()

    eyes = eyes.transpose((0, 2, 3, 1))

    eyes = np.clip(((eyes + 1) / 2.0) * 256, 0, 255)
    eye_images = []
    for eye in eyes:
        eye_as_array = np.asarray(np.uint8(eye), dtype=np.uint8)
        eye_images.append(PIL.Image.fromarray(eye_as_array))

    return eye_images


def generate_eye(class_name: str, eye_count=1, noise=None) -> Union[List[Image], Image]:
    if noise is None:
        noise = generate_noise(eye_count)
    labels = __make_label(class_name, eye_count)

    eyes_tensors = generator(noise, labels).detach().cpu()
    eyes = __to_image(eyes_tensors)
    return eyes if eye_count > 1 else eyes[0]
