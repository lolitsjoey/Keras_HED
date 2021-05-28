import os
import cv2
import matplotlib.pyplot as plt

def crop_stripe(stripe):
    #2200:2600      STRIPE ZOOM
    #1234:1234+465  STRIPE BRIDGE NO ZOOM
    cut_zoom_stripe = cv2.imread(stripe_folder + stripe)[300:830, 300:800, :]
    #zoom_fac = 0.25
    #cut_zoom_stripe = cut_zoom_stripe[:,
    #                  int(cut_zoom_stripe.shape[1] * zoom_fac):int(cut_zoom_stripe.shape[1] * (1 - zoom_fac))]
    return cut_zoom_stripe

stripe_folder = 'D:/scoring_and_profiling/TrsSeal/'
for stripe in os.listdir(stripe_folder):
    cropped_stripe = crop_stripe(stripe)
    cv2.imwrite('D:/scoring_and_profiling/TrsSealCrest/' + stripe, cropped_stripe)

#
# edge_stripe_crop = 'D:/edges_here/'
# for edge in os.listdir(edge_stripe_crop):
#     cropped_stripe = cv2.imread(edge_stripe_crop + edge)
#     # -----Converting image to LAB Color model-----------------------------------
#     lab = cv2.cvtColor(cropped_stripe, cv2.COLOR_BGR2LAB)
#     # -----Splitting the LAB image to different channels-------------------------
#     l, a, b = cv2.split(lab)
#
#     # -----Applying CLAHE to L-channel-------------------------------------------
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#
#     # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
#     limg = cv2.merge((cl, a, b))
#
#     # -----Converting image from LAB Color model to RGB model--------------------
#     final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     plt.imshow(final)
#     plt.show()