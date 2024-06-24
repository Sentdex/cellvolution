import cv2
import os
import glob
from natsort import natsorted

def create_video(input_folder, output_file, fps=30):
    # Get all png files in the folder
    images = glob.glob(f"{input_folder}/*.png")
    
    # Sort the images naturally (1, 2, ..., 10, 11 instead of 1, 10, 11, 2, ...)
    images = natsorted(images)
    
    # Read the first image to get the frame size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the video
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    # Release the VideoWriter
    video.release()

    print(f"Video created successfully: {output_file}")

# Example usage

simulation = "1719178954"
simulation_dir = f"simulations/{simulation}"
output_video_dir = "simvids"
os.makedirs(output_video_dir, exist_ok=True)

output_video = f"{output_video_dir}/{simulation}.mp4"
create_video(simulation_dir, output_video, fps=30)