import os
import cv2
import numpy as np
import argparse
import sys

import torch
from torchvision.transforms import functional as F_torch

from train import get_model_instance_segmentation
from predictions import get_predictions, draw_predictions
from road_lines_sobel import road_line_filter_image, calculate_road_lines, get_road_mask
from road_lines_canny import image_edges

VEHICLE_CONFIG = {
    1:  1.8,
    2:  2.6,
    3:  2.6,
    4:  2.4,
    5:  2.5,
    6:  0.8,
    7:  0.6
}
FOCAL_LENGTH = 2000


def is_in_lane(lane_mask, box):
    """ Verify wether the vehicle is over the lane """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int(y2) 

    if 0 <= center_x < lane_mask.shape[1] and 0 <= center_y < lane_mask.shape[0]:
        # True if the vechile point in the lane_mask is colored
        return lane_mask[center_y, center_x, 1] > 0 
    return False


def get_distance_color(dist):
    """ Generate RGB color gradient: Red (near) -> Green (far) """
    if dist < 10: return (0, 0, 255) # Red
    if dist > 50: return (0, 255, 0) # Green
    
    # Gradient
    ratio = (dist - 10) / 40
    g = int(255 * ratio)
    r = int(255 * (1 - ratio))
    return (0, g, r)



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
            # Calculate distance
            pixel_w = max(1, box[2] - box[0])
            real_w = VEHICLE_CONFIG[label_id]
            dist = (real_w * focal_lenght) / pixel_w
            
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



def predict_distance(image, M, Minv, img_h, img_w, model, 
                     device='cpu', score_threshold=0.5, draw_lane=False):

    filter_image, _ = road_line_filter_image(image)
    no_persp = cv2.warpPerspective(filter_image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
    left_fit, right_fit, _ = calculate_road_lines(no_persp)

    # Create lane mask
    lane_mask = np.zeros_like(image)

    if left_fit is not None and right_fit is not None:
        # Get the road mask
        lane_mask = get_road_mask(no_persp, left_fit, right_fit, Minv, img_h, img_w)


    # ------------ VEHICLE SEGMENTATION MASKS ------

    # Convert the image to Tensor format
    image_tensor = F_torch.to_tensor(image).to(device)
    masks, boxes, labels = get_predictions(image_tensor, model, img_h, img_w, score_threshold)


    # ------------ GENERATE RESULTING FRAME --------

    # Draw the lane
    if draw_lane:
        image_processed = cv2.addWeighted(image, 1, lane_mask, 0.3, 0)
        image_tensor = F_torch.to_tensor(image_processed).to(device)

    image_processed = draw_predictions_in_lanes(image_tensor,
                                                masks, boxes, labels, lane_mask)
    
    return image_processed



def process_distance_video(video_path, output_path, model, device='cpu', score_threshold=0.5, 
                           frames_per_prediction=1, src_pts=None, dst_pts=None, focal_lenght=FOCAL_LENGTH, draw_lane=False, mode='sobel'):
    """
        This function procces a video by predicting the distance of the vechiles in the same lane that the vehicle from which the secuence is recorded.

    Args:
        video_path: The path to the video file
        output_path: The path where the resulting video is written
        model: The segmentation model that predicts masks and bouding boxes
        device: The device where the computations will be done. CPU by default
        score_threshold: The asurance of the model for shwoing the instance predictions 
        frames_per_prediction: The amount of frames that passes beetwen each prediction. The smaller, the faster the process, but results will be worst. Default is 1 (every frame)
        src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
        dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective.
        focal_lenght: The focal lenght of the camera that recorded the video

    Returns:
        true if video is correctly processed, False if an error ocurred.
    """
    # Control frames per second
    if frames_per_prediction <= 0:
        print("Frames per predictions must be > 0\n")
        return False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False
    else:
        # Create Writer with same properties that the original video
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) 
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_w, img_h))
        
        print(f"Processing video... Resolution: {img_w}x{img_h}, FPS: {fps}")

        if src_pts is None:
            p1_top_left = (int(img_w * 0.45), int(img_h * 0.63)) 
            p2_top_right = (int(img_w * 0.52), int(img_h * 0.63)) 
            p3_bot_right = (int(img_w * 0.84), int(img_h * 0.99)) 
            p4_bot_left  = (int(img_w * 0.28), int(img_h * 0.99)) 

            src_pts = np.float32([p1_top_left, p2_top_right, p3_bot_right, p4_bot_left])

        if dst_pts is None:
            dst_pts = np.float32([(int(img_w*0.4), 0), 
                                  (int(img_w*0.6), 0), 
                                  (int(img_w*0.6), img_h), 
                                  (int(img_w*0.4), img_h)
                    ])
            
        # Calculate homography
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = np.linalg.inv(M)
        
        frame_count = 0
        prev_max_height = 0     # First process the whole image
        RESET_MAX_HEIGHT = 25   # Each 25 frames, model predicts the whole image

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video ended

            # ------------ LANE MASK ------------------------

            if mode=='sobel':
                filter_frame, _ = road_line_filter_image(frame)
            elif mode== 'canny':
                filter_frame = image_edges(frame, lower_threshold=5, upper_threshold=250)
                
            no_perspective = cv2.warpPerspective(filter_frame, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
            left_fit, right_fit, _ = calculate_road_lines(no_perspective)

            # Create lane mask
            lane_mask = np.zeros_like(frame)

            if left_fit is not None and right_fit is not None:
                # Get the road mask
                lane_mask = get_road_mask(no_perspective, left_fit, right_fit, Minv, img_h, img_w)


            # ------------ VEHICLE SEGMENTATION MASKS ------

            # Convert the image to Tensor format
            frame_tensor = F_torch.to_tensor(frame).to(device)
            
            # Let between model predictions as many frames as the parameter frames_per_prediction
            if frame_count % frames_per_prediction == 0:

                # Each certain amount of frames, processes the whole image
                if frame_count % RESET_MAX_HEIGHT == 0:
                    prev_max_height = 0

                # Process the frame and write it in the ouput route
                masks, boxes, labels = get_predictions(frame_tensor, 
                                                       model, 
                                                       img_h, img_w, 
                                                       score_threshold,
                                                       prev_max_height
                                                       )
                
                # Update the maximum height reached by any bounding box
                if len(boxes) > 0:
                    prev_max_height = int(torch.min(boxes[:, 1]).item()) - int(img_h*0.1)   # margin of 10% above the maximum bounding box height

                

            # ------------ GENERATE RESULTING FRAME --------

            # Draw the lane
            if draw_lane:
                frame_processed = cv2.addWeighted(frame, 1, lane_mask, 0.3, 0)
                frame_tensor = F_torch.to_tensor(frame_processed).to(device)

            frame_processed = draw_predictions_in_lanes(frame_tensor,
                                                        masks, 
                                                        boxes, 
                                                        labels, 
                                                        lane_mask,
                                                        focal_lenght
                                                        )

            out.write(frame_processed)
            
            # Show the frame processing in real time
            cv2.imshow("Processed video", frame_processed)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"{frame_count} frames processed...")

            # Exit if press 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count == 0:
            print("Error, no frames have been processed")
            return False

        # Free resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return True




if __name__ == '__main__':

    # Define default base paths
    CWD = os.getcwd()
    DEFAULT_MODEL_PATH = os.path.join(CWD, "final_model/maskrcnn_cityscapes.pth")
    
    # 1. ARGUMENT PARSER CONFIGURATION
    parser = argparse.ArgumentParser(description="ADAS Prototype: Distance Estimation & Lane Tracking")
    
    # Mandatory argument: Input video
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help="Path to the input video file (e.g., videos/video1.mp4)")
    
    # Optional arguments
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help="Path to save the result. If not provided, saves automatically to 'results/'")
    
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_PATH, 
                        help="Path to the model weights .pth file")
    
    parser.add_argument('--conf', '-c', type=float, default=0.7, 
                        help="Model confidence threshold (0.0 to 1.0). Default: 0.7")
    
    parser.add_argument('--hide-lane', action='store_true', 
                        help="If set, the green lane mask will NOT be drawn")
    
    parser.add_argument('--mode', type=str, default='sobel', choices=['sobel', 'canny'],
                        help="Lane detection method: 'sobel' (default) or 'canny'")

    parser.add_argument('--frames-prediction', type=int, default=1, 
                        help="Frames between predictions")

    parser.add_argument('--focal-lenght', type=float, default=1000, 
                        help="Camera focal length")
    
    parser.add_argument('--srcpts', type=float, nargs=8, 
                        help="The four points of a trapezoid drawn over a straight lane, in a picture taken by the same camera than the video: x1 y1 x2 y2 x3 y3 x4 y4")
    
    # Parse arguments
    args = parser.parse_args()

    # 2. VALIDATIONS
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at: {args.model}")
        print("Please specify the correct path using --model")
        sys.exit(1)

    # 3. PREPARE DIRECTORIES
    # If no output path is provided, generate one automatically based on input filename
    if args.output is None:
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(CWD, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        args.output = os.path.join(results_dir, f"{name}_result{ext}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 4. LOAD MODEL
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading model from: {args.model}")
    print(f"Using device: {device}")

    # Instantiate architecture
    num_classes = 8 
    model = get_model_instance_segmentation(num_classes)

    # Load weights
    try:
        data_model = torch.load(args.model, map_location=device, weights_only=False) # weights_only=False might be needed depending on torch version
        model.load_state_dict(data_model['model_state_dict'])
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # 5. EXECUTE PROCESSING
    draw_lane_flag = not args.hide_lane # If hide-lane is True, draw is False
    src_pts = np.float32(np.array(args.srcpts).reshape(4, 2)) if args.srcpts else None
    
    print(f"Processing: {args.input}")
    print(f"Saving to: {args.output}")
    
    result = process_distance_video(
        video_path=args.input, 
        output_path=args.output, 
        model=model, 
        device=device, 
        score_threshold=args.conf,
        frames_per_prediction=args.frames_prediction,
        src_pts=src_pts,
        focal_lenght=args.focal_lenght,
        draw_lane=draw_lane_flag, 
        mode=args.mode
    )

    if result:
        print(f"Video saved in: {args.output}")
    else:
        print("An error ocurred while openning the video.")

