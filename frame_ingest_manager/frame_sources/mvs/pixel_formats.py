"""GenICam PFNC pixel-format codes and conversion to mono images.

These numeric codes are defined by the GenICam Pixel Format Naming Convention
(identical to the PixelType_Gvsp_* values in HIKROBOT's PixelType_header.py),
so they are protocol constants rather than configuration.
"""

import cv2


MONO8 = 0x01080001
BAYER_GR8 = 0x01080008
BAYER_RG8 = 0x01080009
BAYER_GB8 = 0x0108000A
BAYER_BG8 = 0x0108000B
RGB8_PACKED = 0x02180014
BGR8_PACKED = 0x02180015

NAMES = {
    MONO8: "Mono8",
    BAYER_GR8: "BayerGR8",
    BAYER_RG8: "BayerRG8",
    BAYER_GB8: "BayerGB8",
    BAYER_BG8: "BayerBG8",
    RGB8_PACKED: "RGB8Packed",
    BGR8_PACKED: "BGR8Packed",
}
BY_NAME = {name: code for code, name in NAMES.items()}

TO_GRAY = {
    BAYER_GR8: cv2.COLOR_BAYER_GR2GRAY,
    BAYER_RG8: cv2.COLOR_BAYER_RG2GRAY,
    BAYER_GB8: cv2.COLOR_BAYER_GB2GRAY,
    BAYER_BG8: cv2.COLOR_BAYER_BG2GRAY,
    RGB8_PACKED: cv2.COLOR_RGB2GRAY,
    BGR8_PACKED: cv2.COLOR_BGR2GRAY,
}


def name_of(pixel_type):
    return NAMES.get(pixel_type, f"0x{pixel_type:08x}")


def raw_shape(pixel_type, height, width):
    if pixel_type == MONO8 or pixel_type in (BAYER_GR8, BAYER_RG8, BAYER_GB8, BAYER_BG8):
        return (height, width)
    if pixel_type in (RGB8_PACKED, BGR8_PACKED):
        return (height, width, 3)
    raise RuntimeError(
        f"Unsupported pixel format {name_of(pixel_type)}; use Mono8, Bayer*8, RGB8 or BGR8"
    )


def needs_conversion(pixel_type):
    return pixel_type in TO_GRAY


def to_gray(raw_image, pixel_type, destination):
    """Convert into the preallocated `destination` buffer; returns it."""
    cv2.cvtColor(raw_image, TO_GRAY[pixel_type], dst=destination)
    return destination
