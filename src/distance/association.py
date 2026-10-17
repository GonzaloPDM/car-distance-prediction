def is_in_lane(lane_mask, box):
    """ Verify wether the vehicle is over the lane """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int(y2)

    if 0 <= center_x < lane_mask.shape[1] and 0 <= center_y < lane_mask.shape[0]:
        # True if the vechile point in the lane_mask is colored
        return lane_mask[center_y, center_x, 1] > 0
    return False
