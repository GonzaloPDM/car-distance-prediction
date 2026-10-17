import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def draw_predictions(image_tensor, pred_masks, pred_boxes, pred_labels, masks_colors=None, custom_labels=False):
    """
        This method draw over a given image the segmentation masks and bounding boxes of given predictions
    Args:
        image_tensor: the image in Tensor format
        pred_masks: The segmentation masks
        pred_boxes: The bounding boxes
        pred_labels: The labels for each predicted class
        masks_colors: The array with the color of each mask
        custom_lables: If True, then the label within the box will be exactly as in pred_labels.
                       If False, then will be in format "Class: label".


    Returns:
        processed_image: the image with masks and bounding boxes drawn on it
    """

    # DRAW MASKS
    image_int = (image_tensor * 255).to(torch.uint8)

    # if there are no elements detected, returns the original image
    if len(pred_boxes) == 0:
        return image_int.permute(1, 2, 0).cpu().numpy()

    # Get only the vehicles in the masks
    masks_bool = pred_masks > 0.5
    masks_bool = masks_bool.squeeze(1)

    # Draw the masks
    if len(masks_bool) > 0:
        if masks_colors is not None:
            processed_image = draw_segmentation_masks(image_int, masks_bool, alpha=0.6, colors=masks_colors)
        else:
            processed_image = draw_segmentation_masks(image_int, masks_bool, alpha=0.6, colors="green")
    else:
        processed_image = image_int

    # Draw the bounding boxes
    if len(pred_boxes) > 0:
        if not custom_labels:
            pred_labels = [f"Class: {label}" for label in pred_labels]

        processed_image = draw_bounding_boxes(
            processed_image,
            pred_boxes,
            labels=pred_labels,
            colors="yellow",
            width=3
        )

    processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
    return processed_image
