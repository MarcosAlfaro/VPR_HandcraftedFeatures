from PIL import Image
import numpy as np


img_magnitude = np.array(Image.open("FIGURES/EXAMPLES_FEATURES/test_image_MAGNITUDE.png"))/255.0
img_angle = np.array(Image.open("FIGURES/EXAMPLES_FEATURES/test_image_ANGLE.png"))


image = np.dstack((img_angle*img_magnitude, img_angle*img_magnitude, img_angle*img_magnitude))
image = Image.fromarray((image).astype(np.uint8))
image.save("FIGURES/EXAMPLES_FEATURES/test_image_combined.png")