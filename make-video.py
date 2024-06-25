import os
import subprocess

def create_video_ffmpeg(input_folder, output_file, fps=30, crf=23):
    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Construct the FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', f'{input_folder}/*.png',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        output_file
    ]

    # Run the FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")


simulation = "1719264027"

simulation_dir = f"simulations/{simulation}"
output_video_dir = "simvids"
os.makedirs(output_video_dir, exist_ok=True)

output_video = f"{output_video_dir}/{simulation}.mp4"
create_video_ffmpeg(simulation_dir, output_video, fps=60, crf=23)
