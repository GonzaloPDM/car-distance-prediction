import os
import argparse

from src.training.train import train


if __name__ == "__main__":
    CWD = os.getcwd()

    parser = argparse.ArgumentParser(description="Fine-tune Mask R-CNN on Cityscapes (resumes from the last checkpoint)")
    parser.add_argument('--data', type=str, default=os.path.join(CWD, 'datasets/cityscape_data'), help="Cityscapes root folder")
    parser.add_argument('--model-dir', type=str, default=os.path.join(CWD, 'model'), help="Folder for the best model")
    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join(CWD, 'checkpoints'), help="Folder for checkpoints")
    parser.add_argument('--epochs', type=int, default=15, help="Total number of epochs")
    parser.add_argument('--batch-size', type=int, default=3, help="Training batch size")
    args = parser.parse_args()

    train(args.data, args.model_dir, args.checkpoints_dir, num_epochs=args.epochs, batch_size=args.batch_size)
