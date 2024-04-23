import subprocess


def convert_mp4_to_gif(input_path, output_path):
    # Construct the command to convert MP4 to GIF
    command = [
        'ffmpeg',
        '-i', input_path,  # Input file
        # Frame rate and scale using lanczos filtering
        '-vf', 'fps=10,scale=500:-1:flags=lanczos',
        '-c:v', 'gif',  # Output format as GIF
        '-f', 'gif',  # Force output to be GIF even if more suitable format exists
        output_path  # Output file
    ]

    # Execute the command
    subprocess.run(command, check=True)


# Example usage
input_video = 'videos/crashing.mp4'
output_gif = 'videos/readMe_crash.gif'
convert_mp4_to_gif(input_video, output_gif)
