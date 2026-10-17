import os
import sys
import argparse

import torch

from src.detection.model import load_model
from src.pipeline import segmentation_processor
from src.video import process_video


if __name__ == "__main__":
    CWD = os.getcwd()
    DEFAULT_MODEL_PATH = os.path.join(CWD, "final_model/maskrcnn_cityscapes.pth")

    parser = argparse.ArgumentParser(description="Vehicle instance segmentation (Mask R-CNN)")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--output', '-o', type=str, required=True, help="Path to save the result")
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_PATH, help="Path to the model weights .pth file")
    parser.add_argument('--conf', '-c', type=float, default=0.7, help="Model confidence threshold. Default: 0.7")
    parser.add_argument('--frames-prediction', type=int, default=1, help="Frames between predictions")
    parser.add_argument('--show', action='store_true', help="If set, the video is displayed while processing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.frames_prediction <= 0:
        print("Error: --frames-prediction must be > 0")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Set the acceleration device (or the CPU if cuda is not available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the fine-tuned model in evaluation mode
    model = load_model(args.model, device)
    print("Loaded model and ready to inference.")

    processor = segmentation_processor(model, device, score_threshold=args.conf,
                                       frames_per_prediction=args.frames_prediction)
    result = process_video(args.input, args.output, processor, show=args.show)

    if result:
        print(f"Video saved in: {args.output}")
    else:
        print("An error ocurred while openning the video.")
