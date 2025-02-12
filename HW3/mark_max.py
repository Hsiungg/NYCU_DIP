import cv2
def mark_max_points(image):
    channels = cv2.split(image)
    marked_image = image.copy()
    for i, channel in enumerate(channels):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(channel)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]  # Colors for B, G, R
        cv2.circle(marked_image, max_loc, radius=10, color=color, thickness=2)
        print(f"Channel {['B', 'G', 'R'][i]}: Max Value = {max_val}, Location = {max_loc}")
    return marked_image