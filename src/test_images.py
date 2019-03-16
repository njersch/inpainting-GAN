
import numpy as np
from PIL import Image

idx = sorted(np.load('places/places_128.npz')['idx_test'])

for i in idx:
    name = f'Places365_val_{i:08d}.jpg'
    path = f'raw/{name}'
    Image.open(path).convert('RGB').show(title=name)



