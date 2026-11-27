import numpy as np


def heatmap_text_color(image, value):
    rgb = np.asarray(image.to_rgba(value, bytes=True)[:3]) / 255.0
    # convert sRGB to linear light before calculating luminance
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear_rgb @ [0.2126, 0.7152, 0.0722]
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"
