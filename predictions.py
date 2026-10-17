import os
import cv2
import torch
from torchvision.transforms import functional as F

from src.detection.model import load_model
from src.detection.inference import get_predictions
from src.detection.draw import draw_predictions

def segmentate_vehicles(image, model, device='cpu'):
    # Convert the image to Tensor format
    image_tensor = F.to_tensor(image).to(device)
    _, img_h, img_w = image_tensor.shape

    pred_masks, pred_boxes, pred_labels = get_predictions(image_tensor, model, img_h, img_w, 0.5)
    image_processed = draw_predictions(image_tensor, pred_masks, pred_boxes, pred_labels)

    return image_processed
    


def process_segmentation_video(video_path, output_path, model, device='cpu', score_threshold=0.5, frames_per_prediction=1, show=False):
    """
        This functions processes a video by calculating the model predictions and draw them on the video

    Args:
        video_path: The path to the video file
        output_path: The path where the resulting video is written
        model: The segmentation model that predicts masks and bouding boxes
        device: The device where the computations will be done. CPU by default
        score_threshold: The asurance of the model for shwoing the instance predictions 
        frames_per_prediction: The amount of frames that passes beetwen each prediction. The smaller, the faster the process, but results will be worst. Default is 1 (every frame)
        show: If True, then the video is displayed while computing.

    Returns:
        true if video is correctly processed, False if an error ocurred.
    """
    # Control frames per second
    if frames_per_prediction <= 0:
        print("Frmes per predictions must be > 0\n")
        return False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False
    else:
        # Create Writer with same properties that the original video
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))
        
        print(f"Processing video... Resolution: {img_w}x{img_h}, FPS: {fps}")
        
        frame_count = 0

        prev_max_height = 0     # First process the whole image
        RESET_MAX_HEIGHT = 25   # Each 25 frames, model predicts the whole image

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video ended

            # Convert the image to Tensor format
            frame_tensor = F.to_tensor(frame).to(device)
            
            # Let between model predictions as many frames as the parameter frames_per_prediction
            if frame_count % frames_per_prediction == 0:

                # Process the frame and write it in the ouput route
                pred_masks, pred_boxes, pred_labels = get_predictions(frame_tensor, model, 
                                                                      img_h, img_w, 
                                                                      score_threshold,
                                                                      prev_max_height
                                                                      )
                
                # Update the maximum height reached by any bounding box
                if len(pred_boxes) > 0:
                    prev_max_height = int(torch.min(pred_boxes[:, 1]).item()) - int(img_h*0.1)   # margin of 10% above the maximum bounding box height

                # Each certain amount of frames, processes the whole image
                if frame_count % RESET_MAX_HEIGHT == 0:
                    prev_max_height = 0

            frame_processed = draw_predictions(frame_tensor, pred_masks, pred_boxes, pred_labels)
            out.write(frame_processed)
            
            # Show the frame processing in real time
            if show:
                cv2.imshow("Processed video", frame_processed)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"{frame_count} frames processed...")

            # Exit if press 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Free resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return True


if __name__ == "__main__":
    CWD = os.getcwd()
    PATH_CITYSCAPE_DATA = os.path.join(CWD, 'datasets/cityscape_data')
    PATH_MODEL = os.path.join(CWD, "model/maskrcnn_cityscapes.pth")
    PATH_IMAGES = os.path.join(CWD, "display_elements/car_segmentation/images")
    PATH_VIDEOS = os.path.join(CWD, "display_elements/car_segmentation/videos")
    PATH_RESULT_IMAGES = os.path.join(CWD, "results/car_segmentation/images")
    PATH_RESULT_VIDEOS = os.path.join(CWD, "results/car_segmentation/videos")

    try:
        os.mkdir(PATH_RESULT_IMAGES)
        os.mkdir(PATH_RESULT_VIDEOS)
    except FileExistsError:
        pass

    # Set the acceleration device (or the CPU if cuda is not available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the fine-tuned model in evaluation mode
    model = load_model(PATH_MODEL, device)

    print("Loaded model and ready to inference.")

    # Get the video capture
    video_name = "video2"
    video_path = os.path.join(PATH_VIDEOS, f"{video_name}.mp4")
    output_path = os.path.join(PATH_RESULT_VIDEOS,f"{video_name}_result.mp4")

    frames_per_prediction = 1

    result = process_segmentation_video(video_path, output_path, model, device, score_threshold=0.7, frames_per_prediction=frames_per_prediction, show=False)

    if result:
        print(f"Video saved in: {output_path}")
    else:
        print("An error ocurred while openning the video.")