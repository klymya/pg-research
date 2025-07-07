def rgb_to_xyz_pixel(r, g, b):
    def srgb_to_linear(c):
        c = c / 255.0
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    r_lin *= 100
    g_lin *= 100
    b_lin *= 100

    # sRGB D65 conversion
    x = r_lin * 0.4124 + g_lin * 0.3576 + b_lin * 0.1805
    y = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722
    z = r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505
    return x, y, z

def xyz_to_lab_pixel(x, y, z):
    # D65 reference white
    x /= 95.047
    y /= 100.000
    z /= 108.883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    # OpenCV scale
    L = int(round(L * 255 / 100))
    a = int(round(a + 128))
    b = int(round(b + 128))

    return L, a, b

def rgb_to_lab_pixel(r, g, b):
    x, y, z = rgb_to_xyz_pixel(r, g, b)
    return xyz_to_lab_pixel(x, y, z)


def lab_to_xyz_pixel(L, a, b):
    # OpenCV scale to standard
    L = L * 100 / 255
    a -= 128
    b -= 128

    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    def f_inv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16/116) / 7.787

    x = 95.047 * f_inv(fx)
    y = 100.000 * f_inv(fy)
    z = 108.883 * f_inv(fz)
    return x, y, z

def xyz_to_rgb_pixel(x, y, z):
    x /= 100
    y /= 100
    z /= 100

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def linear_to_srgb(c):
        return 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c

    r = linear_to_srgb(r)
    g = linear_to_srgb(g)
    b = linear_to_srgb(b)

    r = int(round(max(0, min(1, r)) * 255))
    g = int(round(max(0, min(1, g)) * 255))
    b = int(round(max(0, min(1, b)) * 255))
    return r, g, b

def lab_to_rgb_pixel(L, a, b):
    x, y, z = lab_to_xyz_pixel(L, a, b)
    return xyz_to_rgb_pixel(x, y, z)


def rgb_image_to_lab(image_rgb):
    height = len(image_rgb)
    width = len(image_rgb[0])
    image_lab = []
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = image_rgb[y][x]
            L, A, B = rgb_to_lab_pixel(r, g, b)
            row.append([L, A, B])
        image_lab.append(row)
    return image_lab


def lab_image_to_rgb(image_lab):
    height = len(image_lab)
    width = len(image_lab[0])
    image_rgb = []
    for y in range(height):
        row = []
        for x in range(width):
            L, A, B = image_lab[y][x]
            r, g, b = lab_to_rgb_pixel(L, A, B)
            row.append([r, g, b])
        image_rgb.append(row)
    return image_rgb
