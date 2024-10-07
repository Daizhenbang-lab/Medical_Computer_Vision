import os
import numpy as np
from PIL import Image


def get_patches_from_folder(folder_path):
    patches = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                coords = file[:-4].split('_')
                y = int(coords[0][1:])
                x = int(coords[1][1:])
                patches[(x, y)] = os.path.join(root, file)
    return patches


def restore_image(patches, patch_size=224, overlap=0.5):
    patch_half_size = patch_size // 2
    step = int(patch_size * (1 - overlap))

    coords = patches.keys()
    max_x = max(coord[0] for coord in coords)
    max_y = max(coord[1] for coord in coords)

    width = max_x + patch_half_size
    height = max_y + patch_half_size
    restored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for (x, y), patch_path in patches.items():
        patch_img = np.array(Image.open(patch_path))

        start_x = x - patch_half_size
        start_y = y - patch_half_size

        restored_image[start_y:start_y + patch_size, start_x:start_x + patch_size] = patch_img

    return Image.fromarray(restored_image)


# 示例使用
folder_path = "scan121"
patches = get_patches_from_folder(folder_path)
restored_image = restore_image(patches)
restored_image.show()
