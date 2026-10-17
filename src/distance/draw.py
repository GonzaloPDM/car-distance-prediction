import torch

from src.detection.draw import draw_predictions
from src.distance.association import is_in_lane
from src.distance.estimation import estimate_distance, get_distance_color, FOCAL_LENGTH


def draw_predictions_in_lanes(image_tensor, pred_masks, pred_boxes, pred_labels, lane_mask, focal_lenght=FOCAL_LENGTH):
    """
        This method draw over a given image the segmentation masks and bounding boxes of given predictions,
        and for those vehicles on the lane mask, calculate and represents their distance
    Args:
        image_tensor: the image in Tensor format
        pred_masks: The segmentation masks
        pred_boxes: The bounding boxes
        pred_labels: The labels for each predicted class
        lane_mask: The mask for the own lane


    Returns:
        processed_image: the image with masks and bounding boxes drawn on it
    """

    # if there are no elements detected, returns the original image
    if len(pred_boxes) == 0:
        return (image_tensor * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    # Lists with colors and labels for each detected vehicle
    dynamic_colors = []
    dynamic_labels = []

    # Asign especific color and distance for vehicles in same lane, and default for the others
    for i in range(len(pred_boxes)):
        box = pred_boxes[i].cpu().numpy().astype(int)
        label_id = pred_labels[i].item()

        # If the vehicle is in the same lane
        if is_in_lane(lane_mask, box):
            dist = estimate_distance(box, label_id, focal_lenght)

            # Get the RGB color
            color = get_distance_color(dist)
            label = f"Class: {label_id}\nDistance: {dist:.1f}m"
        else:
            # If it is in other lane
            color = (255, 255, 0)           # Cyan
            label = f"Class: {label_id}"

        dynamic_colors.append(color)
        dynamic_labels.append(label)

    # Draw the predictions over the image
    processed_image = draw_predictions(image_tensor, pred_masks, pred_boxes, dynamic_labels,
                                       masks_colors=dynamic_colors, custom_labels=True)

    return processed_image
