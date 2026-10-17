import torch


def get_predictions(image_tensor, model, img_h, img_w, score_threshold=0.5, max_height=0):
    """
        This method calculates the predictions of the car segmentation model of a given image,
        and returns the predictions with a confidence level above a given threshold
    Args:
        image: The image to be processed
        model: The model that computes the predictions
        img_h, img_w: The dimensions of the image.
        score_threshold: The confidence threshold
        max_height: The row from which the model evaluates the image. 0 means no cropping (slower),
                    1 means cropping whole image (no image processed). Default 0 to no cropping.

    Returns:
        pred_masks: The segmentation masks
        pred_boxes: The bounding boxes
        pred_labels: The labels for each predicted class
    """

    # Check that max_height is not to low (minimum computes 40% of the image)
    if 0 < max_height < 0.6*img_h:
        input_tensor = image_tensor[:, max_height:, :]
    else:
        input_tensor = image_tensor

    # Turn off gradient mode during the prediction execution
    with torch.no_grad():
        predictions = model([input_tensor])

    prediction = predictions[0]

    # Only keep predictions if the model has a certain level of confidence about them (over 50% score)
    keep_idx = prediction['scores'] > score_threshold

    # If nothing detected, we return the original image
    if not keep_idx.any():
        empty_box = torch.empty((0, 4), device=image_tensor.device)
        empty_mask = torch.empty((0, 1, image_tensor.shape[1], image_tensor.shape[2]), device=image_tensor.device)
        empty_label = torch.empty((0), dtype=torch.int64, device=image_tensor.device)
        return empty_mask, empty_box, empty_label

    pred_boxes = prediction['boxes'][keep_idx]
    pred_masks = prediction['masks'][keep_idx]
    pred_labels = prediction['labels'][keep_idx]

    # Adjust coordinates if image cropped
    if 0 < max_height < 0.6*img_h:
        pred_boxes[:, [1, 3]] += max_height
        full_masks = torch.zeros((len(pred_masks), 1, img_h, img_w), device=image_tensor.device)
        full_masks[:, :, max_height:, :] = pred_masks
        return full_masks, pred_boxes, pred_labels

    return pred_masks, pred_boxes, pred_labels
