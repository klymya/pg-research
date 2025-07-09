import re
import matplotlib.pyplot as plt
import numpy as np

IMG_PTH = "/home/klymy/projects/rtos_sensor/projects/alif_ml-embedded-evaluation-kit_pg/build/alif-gcc-he-revA/generated/pg_object_detection/src/img_320.cc"

with open(IMG_PTH, "r") as f:
    content = f.read()

def extract_c_array(filepath, varname="im0"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Match: const uint8_t im0[] ... = { ... };
    pattern = rf'{re.escape(varname)}\[\][^\=]*=\s*{{(.*?)}};'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Array {varname} not found in file.")

    # Extract all hex numbers like 0x6c
    hex_values = re.findall(r'0x[0-9a-fA-F]+', match.group(1))
    byte_list = [int(h, 16) for h in hex_values]

    return bytes(byte_list)


im0_bytes = extract_c_array(IMG_PTH, varname="im0")
print(im0_bytes[:20])      # Preview first 20 bytes
print(f"Total bytes: {len(im0_bytes)}")


img = list(im0_bytes)

plt.imshow(np.array(img).reshape((240, 320, 3), order="C"))
plt.axis('off')
plt.show()