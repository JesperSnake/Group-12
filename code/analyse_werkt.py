# %%
from openpiv   import tools, pyprocess, validation, filters, scaling
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import pathlib

# %%
# 6000 frames in 10 minuten.

videopath = r"C:\Users\Gebruiker\Downloads\No_worm.avi"
cap = cv2.VideoCapture(videopath)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# %%

def vectorfield(i, dt=0.2, tresh=25, winsize = 32, searchsize = 38, overlap = 17):
    cap = cv2.VideoCapture(videopath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #1
    cap.set(1, i)
    ret, frame1 = cap.read()
    cap.set(1, i + int(10*dt))
    ret, frame2 = cap.read()

    cap.release()

    # Convert the frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Display the frames
    # fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    # ax[0].imshow(frame1_gray, cmap=plt.cm.gray)
    # ax[1].imshow(frame2_gray, cmap=plt.cm.gray)
  
    #2
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame1_gray.astype(np.int32),
        frame2_gray.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=5*dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
    )

    # Delete noise.
    invalid_mask = validation.sig2noise_val(
        sig2noise,
        threshold = 1.05,
    )
    u2, v2 = filters.replace_outliers(
        u0, v0,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame1_gray.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )

    x, y, u3, v3 = tools.transform_coordinates(x, y, u2, v2)

    L2 = u3**2 + v3**2

    u4 = np.where(L2 < tresh, u3, np.nan)
    v4 = np.where(L2 < tresh, v3, np.nan)

    string = f'project12_py{i}.txt'
    print(string)
    tools.save(string, x, y, u4, v4, invalid_mask)
    print('Saved!')

    #4
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_title(i)
    tools.display_vector_field(
        pathlib.Path(string),
        ax=ax, scaling_factor=1,
        scale=50, # scale defines here the arrow length
        width=0.0035, # width is the thickness of the arrow
        on_img=False, # overlay on the image
        )
    plt.show()
    plt.savefig(f'figure{i}.png')
    # saved niet op de gewenste plek.


# %%
for i in range(2):
    vectorfield(i)
