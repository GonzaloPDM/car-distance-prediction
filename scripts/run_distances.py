import os
import sys
import argparse

import numpy as np
import torch

from src.detection.model import load_model
from src.lanes.filters import LANE_FILTER_MODES
from src.pipeline import distance_processor
from src.video import process_video


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

    parser.add_argument('--mode', type=str, default='sobel', choices=LANE_FILTER_MODES,
                        help="Lane detection method: 'sobel' (default) or 'canny'")

    parser.add_argument('--frames-prediction', type=int, default=1,
                        help="Frames between predictions")

    parser.add_argument('--focal-lenght', type=float, default=1000,
                        help="Camera focal length")

    parser.add_argument('--srcpts', type=float, nargs=8,
                        help="The four points of a trapezoid drawn over a straight lane, in a picture taken by the same camera than the video: x1 y1 x2 y2 x3 y3 x4 y4")

    parser.add_argument('--show', action='store_true',
                        help="If set, the video is displayed while processing (press 'q' to stop)")

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

    if args.frames_prediction <= 0:
        print("Error: --frames-prediction must be > 0")
        sys.exit(1)

    # 3. PREPARE DIRECTORIES
    # If no output path is provided, generate one automatically based on input filename
    if args.output is None:
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        args.output = os.path.join(CWD, "results", f"{name}_result{ext}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 4. LOAD MODEL
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading model from: {args.model}")
    print(f"Using device: {device}")

    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    # 5. EXECUTE PROCESSING
    src_pts = np.float32(np.array(args.srcpts).reshape(4, 2)) if args.srcpts else None

    print(f"Processing: {args.input}")
    print(f"Saving to: {args.output}")

    processor = distance_processor(
        model=model,
        device=device,
        score_threshold=args.conf,
        frames_per_prediction=args.frames_prediction,
        src_pts=src_pts,
        focal_lenght=args.focal_lenght,
        show_lane=not args.hide_lane,
        mode=args.mode
    )
    result = process_video(args.input, args.output, processor, show=args.show)

    if result:
        print(f"Video saved in: {args.output}")
    else:
        print("An error ocurred while openning the video.")
