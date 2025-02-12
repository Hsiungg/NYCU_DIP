import cv2
import numpy as np
import os
from union_find import union_find
"""
TODO Binary transfer
"""
def to_binary(img, idx):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    threshold = [150, 10]
    diff_from_white = np.abs(img - [255, 255, 255])
    non_white_pixels = np.any(diff_from_white > threshold[idx], axis=-1)
    binary_img = np.where(non_white_pixels, 255, 0).astype(np.uint8)
    return binary_img
"""
TODO Two-pass algorithm
"""
def two_pass(binary_img, connectivity):
    assert connectivity in [4, 8], 'Connectivity must be 4 or 8'
    rows, cols = binary_img.shape
    label_count = 0
    uf = union_find()
    padded_img = cv2.copyMakeBorder(binary_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    label_mask = np.zeros_like(padded_img, dtype = np.uint16)
    #first pass
    for i in range(rows):
        for j in range(cols):
            if padded_img[i+1, j+1] == 255:
                if connectivity == 4:
                    neighbors = [
                        label_mask[i, j+1],     # Top
                        label_mask[i+1, j]      # Left
                    ]
                else:
                    neighbors = [
                        label_mask[i, j],       # Top-left
                        label_mask[i, j+1],     # Top
                        label_mask[i, j+2],     # Top-right
                        label_mask[i+1, j]      # Left
                    ]
                non_zero_neighbors = [n for n in neighbors if n > 0]
                if not non_zero_neighbors:
                    # Empty, so assign new label to it
                    label_count += 1
                    label_mask[i + 1, j + 1] = label_count
                    node = uf.make_set(label_count)
                else:
                    min_label = min(non_zero_neighbors)
                    label_mask[i + 1, j + 1] = min_label
                    for neighbor_label in non_zero_neighbors:
                        neighbor_node = uf.node_values[neighbor_label]
                        min_node = uf.node_values[min_label]
                        uf.union(neighbor_node, min_node)
    # second pass
    label_mask = label_mask[1:-1, 1:-1]
    for i in range(rows):
        for j in range(cols):
            if label_mask[i, j] > 0:
                node = uf.node_values[label_mask[i, j]]
                label_mask[i, j] = uf.find(node).value
    return label_mask


"""
TODO Seed filling algorithm
"""
def seed_filling(binary_img, connectivity):
    assert connectivity in [4, 8], 'Connectivity must be 4 or 8'
    rows, cols = binary_img.shape
    label_count = 0
    stack = []
    visited = np.zeros_like(binary_img, dtype = np.bool_)
    label_mask = np.zeros_like(binary_img)
    if connectivity == 4:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    elif connectivity == 8:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 255 and not visited[i, j]:
                label_count += 1
                label_mask[i, j] = label_count
                stack.append((i, j))
                visited[i, j] = True
                while stack:
                    row, col = stack.pop()
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < rows and 0 <= new_col < cols \
                            and binary_img[new_row, new_col] == 255 \
                                and not visited[new_row, new_col]:
                            visited[new_row, new_col] = True
                            label_mask[new_row, new_col] = label_count
                            stack.append((new_row, new_col))
    return label_mask

"""
Bonus
"""
def other_cca_algorithm():
    raise NotImplementedError


"""
TODO Color mapping
"""
def color_mapping(label_img):
    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels > 0]
    color_map = {
        label: np.random.randint(0, 255, size=3, dtype=np.uint8) for label in unique_labels
    }
    colorized_image = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        area_size = np.sum(label_img == label)
        if area_size >= 500:
            colorized_image[label_img == label] = color
        else:
            colorized_image[label_img == label] = [0, 0, 0]
    return colorized_image



"""
Main function
"""
def main():

    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img, i)
            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
        
            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite("result/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)


if __name__ == "__main__":
    main()