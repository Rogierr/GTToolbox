import imageio
import os
from natsort import natsorted

path = 'C:\\Users\\HarmelinkRLA\\PycharmProjects\\GTToolbox\\figures'

files = []
images = []

for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

files = natsorted(files)

for filename in files:
    images.append(imageio.imread(filename))

imageio.mimsave(
    'C:\\Users\\HarmelinkRLA\\Desktop\\tanh_pd_mb.gif',
    images, duration=0.2)