import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.animation import FuncAnimation
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

# Hoe lager deze waarde is, hoe strenger hij wordt.
grootsteverschil = 50

# vervang 'bestand.txt' met het bestand wat je wil bijknippen.
with open('bestand.txt', 'r') as input_bestand:
    oude_lijst = []

    for regel in input_bestand:
       oude_lijst.append(int(regel))


#Lijst met alle verschillen uit de lijst
delta_values = []
for i in range (len(oude_lijst) - 1):
    delta_values.append(oude_lijst[i+1] - oude_lijst[i])



# Deze functie checkt of er te grote jumps in de lijst ziten, als dat zo is pakt hij het gemiddelde van zijn buren (mits deze geen grote jumps hebben)
index = 0
new_values = oude_lijst.copy()
for value in delta_values:

    if abs(value) > grootsteverschil:
        plussearch = 1

        while abs(delta_values[(index + plussearch)]) > 100:
            plussearch += 1
        new_values[index] = (oude_lijst[index - 1] + oude_lijst[index + plussearch]) / 2
    index += 1


plt.figure('figure 1')
plt.plot(oude_lijst)

plt.plot(new_values)
plt.show()
