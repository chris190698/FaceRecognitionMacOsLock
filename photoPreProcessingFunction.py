from re import I, X
from PIL import Image
from math import sqrt
import io
from skimage import exposure
from skimage import io
from io import StringIO


std_R = 64
std_G = 64
std_B = 64
stdRGBMean = (
    std_B + std_G + std_R
) / 3  # calculate standard mean value from parameters

def brightnessCalculator(image):
    imag = image
    # Convert the image te RGB if it is a .gif for example
    imag = imag.convert("RGB")
    # coordinates of the pixel
    X, Y = 0, 0
    # Get RGB
    pixelRGB = imag.getpixel((X, Y))
    R, G, B = pixelRGB
    brightness = float(
        sum([R, G, B]) / 3
    )  # 0 is dark (black) and 255 is bright (white)
    return brightness



def photoBrightnessEvaluate(image, folder, frame):
    tmpPhotoBrightness = brightnessCalculator(
        image
    )  # calculate photo brightness
    if tmpPhotoBrightness < stdRGBMean:  # compare with standard medium value
        photoProcessor(frame, folder)


def photoProcessor(image, folder):

    print("Equalization started.")
    # img = io.imread(image)
    # Contrast stretching
    # p2, p98 = np.percentile(img, (2, 98))
    # img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    io.imsave(
        folder + "s.jpg",
        img_adapteq,
    )



# Example
# processingFlag = photoBrightnessEvaluate(photo_toCheck)
# photoProcessor(processingFlag)
