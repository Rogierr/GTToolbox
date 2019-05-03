import imageio
import os
from natsort import natsorted

path = 'C:\\Users\\HarmelinkRLA\\Dropbox\\Universiteit Twente\\Papers\\Rarity Value\\figures\\Verzoek Reinoud\\gifs'

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
    'C:\\Users\\HarmelinkRLA\\Dropbox\\Universiteit Twente\\Papers\\Rarity Value\\figures\\Verzoek Reinoud\\gifs\\movie m loop, phi = 1.5.gif',
    images, duration=0.2)