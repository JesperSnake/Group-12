import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from lmfit import Model

equi_list =[]
x_list = []
y_list = []
pixel_scale = 5
with open('x_values.txt', 'r') as input_bestand:

    for regel in input_bestand:
       y_list.append(int(regel))
def Noramalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
def prep_data(values):

    sigma = 2
    values = gaussian_filter(values, sigma=sigma)

    values = Noramalize(values)
    values = values[0:200]
    equi = np.mean(values)
    for i in range(0 , len(values)):
        equi_list.append(equi)
        x_list.append(i / 30)

    #values = np.clip(values, 0, 1)
    return values
def omega_guess(values, equi):
    x_collision = []
    collision_count = 0
    x = 1
    y_line = equi[0]
    dx = 1
    while collision_count < 3:
        dy = values[x] - values[x-1]
        a = dy / dx
        b = values[x] - a * x
        x_intersect = (y_line - b) / a
        if x_intersect > (x - 1) and x_intersect < x:
            print("hello")
            collision_count += 1
            x_collision.append(x / 30)
        x += 1
    return x_collision
def function(t, A, w, phi, c):
    return A * np.sin(w * t+ phi) + c
plt.plot(equi_list)
y_list = prep_data(y_list)
x_collision = omega_guess(y_list, equi_list)
w_guess = 2 * np.pi / (x_collision[2] - x_collision[0])
model = Model(function)
params = model.make_params(A=1,  w=w_guess, phi=0, c=0)
result = model.fit(y_list, params, t=x_list)
print(result.fit_report())
print(f"Eigenfrequency = {result.params['w'].value} +- {result.params['w'].stderr}")
plt.plot(x_list, pixel_scale * np.array(result.best_fit))
plt.plot(x_list, pixel_scale * np.array(y_list))
plt.xlabel('time(s)')
plt.ylabel('Amplitude(m)')
plt.show()