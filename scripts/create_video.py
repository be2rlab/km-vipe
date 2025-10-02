import cv2
import os
import argparse
from tqdm import tqdm


def read_images_replica(args):
    input_dir = os.path.join(args.input_dir, args.scene_name, "results")
    images = [img for img in sorted(os.listdir(input_dir)) if img.startswith("frame")]
    return images, input_dir


def read_images_tum(args):
    input_dir = os.path.join(args.input_dir, args.scene_name, "rgb")
    images = sorted(os.listdir(input_dir))
    return images, input_dir


def read_images(args):
    if args.dataset == "replica" or args.dataset == 'aria':
        return read_images_replica(args)
    if args.dataset == "tum":
        return read_images_tum(args)
    else: 
        raise Exception(f"{args.dataset} is not supported!") 


def main():
    parser = argparse.ArgumentParser(description='Create video from images')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (Replica, TUM, etc.)")
    parser.add_argument('--scene_name', type=str, required=True, help='Name of the scene')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to folder with images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output video path')
    
    args = parser.parse_args()
    
    scene_name = args.scene_name    
    images, input_dir = read_images(args)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if not images:
        print(f"Error: No images found in {input_dir}")
        return
    
    frame = cv2.imread(os.path.join(input_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(output_dir, scene_name + ".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    
    for img_name in tqdm(images, desc='Creating a video'):
        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()

if __name__ == "__main__":
    main()
