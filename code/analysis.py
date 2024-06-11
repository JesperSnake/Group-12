import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from lmfit import Model

equi_list =[]
x_list = []
y_list = []


####################################################################################################################################

#TWEAKABLES:

# Hoe lager deze waarde is, hoe strenger hij maskeert.
grootsteverschil = 50

#.
pixel_scale = 5

#naam van bestand wat je opent:
bestandnaam = 'x_values.txt'

#####################################################################################################################################

with open(bestandnaam, 'r') as input_bestand:

    for regel in input_bestand:
       y_list.append(int(regel))


def maskeren(data):

    delta_values = []
    for i in range (len(data) - 1):
        delta_values.append(data[i+1] - data[i])


    # Deze functie checkt of er te grote jumps in de lijst ziten, als dat zo is pakt hij het gemiddelde van zijn buren (mits deze geen grote jumps hebben)
    index = 0
    new_values = data.copy()
    for value in delta_values:

        if abs(value) > grootsteverschil:
            plussearch = 1

            while abs(delta_values[(index + plussearch)]) > 100:
                plussearch += 1
            new_values[index] = (data[index - 1] + data[index + plussearch]) / 2
        index += 1


    #Check of alles goed is gegaan? Uncomment volgende lines:

    #plt.figure('figure 1')
    #plt.plot(oude_lijst)
    #plt.plot(new_values)
    #plt.show()

    return new_values

def Noramalize(data):

    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def prep_data(values):
    
    maskeddata = maskeren(values)

    sigma = 2
    prepped_values = gaussian_filter(maskeddata, sigma=sigma)

    prepped_values = Noramalize(prepped_values)
    prepped_values = prepped_values[0:200]
    equi = np.mean(prepped_values)
    for i in range(0 , len(prepped_values)):
        equi_list.append(equi)
        x_list.append(i / 30)

    #values = np.clip(values, 0, 1)
    return prepped_values

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
params = model.make_params(A=1,  w=w_guess, phi=0, c=0.5)
result = model.fit(y_list, params, t=x_list)
print(result.fit_report())
print(f"Eigenfrequency = {result.params['w'].value} +- {result.params['w'].stderr}")
plt.plot(x_list, pixel_scale * np.array(result.best_fit))
plt.plot(x_list, pixel_scale * np.array(y_list))
plt.xlabel('time(s)')
plt.ylabel('Amplitude(m)')
plt.show()