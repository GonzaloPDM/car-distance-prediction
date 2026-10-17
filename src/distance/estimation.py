# Real width (meters) of each vehicle class predicted by the model
VEHICLE_WIDTHS = {
    1:  1.8,    # car
    2:  2.6,    # truck
    3:  2.6,    # bus
    4:  2.4,    # caravan
    5:  2.5,    # trailer
    6:  0.8,    # motorcycle
    7:  0.6     # bicycle
}
FOCAL_LENGTH = 2000


def estimate_distance(box, label_id, focal_lenght=FOCAL_LENGTH):
    """ Pinhole camera model: D = (W_real * f) / W_pixel

        Args:
            box: Bounding box (x1, y1, x2, y2) in pixels
            label_id: The predicted class, used to know the real width of the vehicle
            focal_lenght: The focal lenght of the camera, in pixels
        Returns:
            The distance to the vehicle in meters
    """
    pixel_w = max(1, box[2] - box[0])
    real_w = VEHICLE_WIDTHS[label_id]
    return (real_w * focal_lenght) / pixel_w


def get_distance_color(dist):
    """ Generate RGB color gradient: Red (near) -> Green (far) """
    if dist < 10: return (0, 0, 255) # Red
    if dist > 50: return (0, 255, 0) # Green

    # Gradient
    ratio = (dist - 10) / 40
    g = int(255 * ratio)
    r = int(255 * (1 - ratio))
    return (0, g, r)
