
## Setup
# Commented out IPython magic to ensure Python compatibility.
!pip install --quiet ipython-autotime
!pip install --quiet google-generativeai
!pip install --quiet g4f[all] --upgrade
!pip install --quiet elevenlabs

## 1. Generate image description and script"""

LLM = "G4F" # @param ["G4F", "Gemini", "Hercai"] {allow-input: true}

# Model Parameter is only useful if LLM is Hercai else it is Useless
MODEL = "turbo" # @param ["turbo", "gemma2-9b", "v3-32k"] {allow-input: true}


import re
import json
import pprint
from g4f.client import Client
import google.generativeai as genai

gemini_apikey = "IzaSyC6N1MVe9WmAFjWMNuXjlaLnYa8eO813tY"

def Genrate_Script_And_Prompts(prompt, LLM, model=None):
    if LLM == "G4F":
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()



    elif LLM == "Gemini":
        if not gemini_apikey:
            raise ValueError("Gemini API key is missing!")
        genai.configure(api_key=gemini_apikey)
        gemini_model = genai.GenerativeModel('gemini-pro')
        response = gemini_model.generate_content(prompt)
        response_text = response.text




    elif LLM == "Hercai":
        herc = Hercai("")  # Provide Hercai api key (optional)
        response = herc.question(model=MODEL, content=prompt)
        response_text = response["reply"]

    else:
        raise ValueError("Invalid LLM selected")




    # Extract JSON from the response
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        json_str = json_match.group(0)
        if not json_str.endswith(']'):
            json_str += ']'
        output = json.loads(json_str)
    else:
        raise ValueError("Invalid JSON")

    pprint.pprint(output)
    image_prompts = [item['image_description'] for item in output]
    sentences = [item['sentence'] for item in output]

    return image_prompts, sentences




# Daily motivation, personal growth and positivity
topic = "Success and Achievement"
goal = "inspire people to overcome challenges, achieve success, and celebrate their victories"

prompt_prefix = f"""You are tasked with creating a script for a {topic} video that is about 30 seconds.
Your goal is to {goal}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:"""

sample_output = """
   [
       { "image_description": "Description of the first image here.", "sentence": "Text accompanying the first scene cut." },
       { "image_description": "Description of the second image here.", "sentence": "Text accompanying the second scene cut." }
   ]
"""

prompt_postinstruction = f"""By following these instructions, you will create an impactful {topic} short-form video.
Output:"""

prompt = prompt_prefix + sample_output + prompt_postinstruction


image_prompts, sentences = Genrate_Script_And_Prompts(prompt, LLM)
print("image_prompts:", image_prompts)
print("sentences:", sentences)
print("Number of sentences:", len(sentences))

# 2. Create a new folder with a unique name.
import uuid

current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print(active_folder)

#############################################################################
# Generate high-quality images for those descriptions using Segmind API or Hercai
#############################################################################
import os
import io
import time
import uuid
import requests
from PIL import Image
import random

# User's choice of image generator
IMAGE_SOURCE = "Hercai" # @param ["Segmind",  "Hercai"] {allow-input: true}
IMAGE_MODEL = "prodia"  # @param ["animefy", "prodia",  "v3"] {allow-input: true}

# Segmind API
segmind_apikey = os.environ.get('SG_2d3504ba72dbeacc', 'SG_2d3504ba72dbeacc').split(',')
api_key_index = 0

# Generate images using Segmind API (supports waiting to comply with the rate limit of 5 image requests per minute)
def generate_images_segmind(prompts, fname):
    global api_key_index
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
    headers = {'x-api-key': segmind_apikey[api_key_index]}

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    requests_made = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        if requests_made >= 5 and time.time() - start_time <= 60:
            time_to_wait = 60 - (time.time() - start_time)
            print(f"Waiting for {time_to_wait:.2f} seconds to comply with rate limit...")
            time.sleep(time_to_wait)
            requests_made = 0
            start_time = time.time()

        final_prompt = "((High quality)), ((8K resolution)), ((Professional)), ((Best)), ((Cinematic)), ((Highly detailed)), ((Vibrant colors)), ((Sharp focus)), ((Dynamic lighting)), ((Photorealistic)), ((Masterpiece)), 4k, {}, no occlusion, Fujifilm XT3, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((Ugly)), ((Bad quality)), ((Unprofessional)), ((Distorted)), ((Pixelated)), ((Noisy)), ((Low resolution)), ((Unclear)), ((Blurry)), ((Overexposed)), ((Underexposed)), ((Artifacts)), ((Poorly lit)), ((Unnatural colors)), deformed, limbs cut off, quotes, extra fingers, deformed hands, extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": random.randint(1, 1000000),
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
        }

        while True:
            response = requests.post(url, json=data, headers=headers)
            requests_made += 1

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
                image_filename = os.path.join(fname, f"{i + 1}.jpg")
                image.save(image_filename)
                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                break
            else:
                print(response.status_code)
                print(response.text)
                if response.status_code == 429:
                    api_key_index = (api_key_index + 1) % len(segmind_apikey)
                    headers['x-api-key'] = segmind_apikey[api_key_index]
                    print(f"Switching to API key: {segmind_apikey[api_key_index]}")
                    time.sleep(1)
                    continue
                else:
                    print(f"Error: Failed to retrieve or save image {i + 1}")
                    break

# Generate images using Hercai
def generate_images_hercai(prompts, fname):
    os.makedirs(fname, exist_ok=True)

    for i, prompt in enumerate(prompts):
        final_prompt = "High quality, 8K resolution, Professional, Best, Cinematic, Highly detailed, Vibrant colors, Sharp focus, Dynamic lighting, Photorealistic, Masterpiece, 4k, {}, no occlusion, Fujifilm XT3, cinemascope".format(prompt.strip('.'))
        negative_prompt = "Ugly, Bad quality, Unprofessional, Distorted, Pixelated, Noisy, Low resolution, Unclear, Blurry, Overexposed, Underexposed, Artifacts, Poorly lit, Unnatural colors, deformed, limbs cut off, quotes, extra fingers, deformed hands, extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs, low quality, low resolution, dark, gloomy, overexposed, underexposed, noisy, grainy, distorted, pixelated, out of focus, poorly lit, unnatural colors, artifacts, text, logo, frame, border"
        url = f"https://hercai.onrender.com/{IMAGE_MODEL}/text2image?prompt={final_prompt}&negative_prompt={negative_prompt}"

        response = requests.get(url)
        parsed = response.json()
        image_url = parsed["url"]
        image_response = requests.get(image_url)
        image_data = image_response.content
        image = Image.open(io.BytesIO(image_data))
        image_filename = os.path.join(fname, f"{i + 1}.png")
        image.save(image_filename)
        print(f"Image {i + 1}/{len(prompts)} saved as '{image_filename}'")

# Call the appropriate function based on the user's choice
if IMAGE_SOURCE == "Segmind":
    generate_images_segmind(image_prompts, active_folder)
elif IMAGE_SOURCE == "Hercai":
    generate_images_hercai(image_prompts, active_folder)
else:
    print("Invalid image generator choice. Please choose either 'Segmind' or 'Hercai'.")

# 2. Generate images for that descriptions using [Segmind's](https://www.segmind.com/) API"""

import os
from elevenlabs import play, save
from elevenlabs.client import ElevenLabs

# Define the language and TTS settings
TTS = "ElevenLabs"  # Text-to-Speech service to use
VOICE = "Antoni"  # Voice to use for TTS

# Initialize the ElevenLabs client
client = ElevenLabs(
    api_key="c5ba23340a5ea55a2d2ab8004863fcb",  # Replace with your ElevenLabs API key
)


# Generate the audio using the specified voice and model
for i, sentence in enumerate(sentences):
    audio = client.generate(
        text=sentence,  # Use sentence to access the text for each iteration
        voice=VOICE,  # Specify the voice to use
        model="eleven_multilingual_v2"  # Specify the model to use
    )

    # Construct the full file path using os.path.join
    file_path = os.path.join(active_folder, f"{i+1}.mp3")

    # Save the generated audio
    save(audio, file_path)

    print(f"Audio {i+1} saved as '{file_path}'")

## 3. Convert text to speech using [Elevenlabs](https://elevenlabs.io/) API

## 4. Install Moviepy to stitch everything


!pip install --quiet git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
!pip install --quiet imageio==2.25.1

!apt install -qq imagemagick

!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml

from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
import os
import cv2
import numpy as np

def create_combined_video_audio(mp3_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    mp3_files = sorted([file for file in os.listdir(mp3_folder) if file.endswith(".mp3")])
    mp3_files = sorted(mp3_files, key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(os.path.join(mp3_folder, mp3_file))
        audio_clips.append(audio_clip)

        # Load the corresponding image for each mp3 and set its duration to match the mp3's duration
        img_path = os.path.join(mp3_folder, f"{mp3_file.split('.')[0]}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

        # Resize the original image to 1080x1080
        image_resized = cv2.resize(image, (1080, 1080))

        # Blur the image
        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - 1080) // 2
        blurred_img[y_offset:y_offset+1080, :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(final_audio)
    finalpath = mp3_folder+"/"+output_filename

    final_video.write_videofile(finalpath, fps=fps, codec='libx264',audio_codec="aac")

output_filename = "combined_video.mp4"
create_combined_video_audio(active_folder, output_filename)

# 5. Extract word level timestamps to overlap captions on top of the video"""

!pip install ffmpeg-python==0.2.0

import ffmpeg

def extract_audio_from_video(outvideo):
    """
    Extract audio from a video file and save it as an MP3 file.

    :param output_video_file: Path to the video file.
    :return: Path to the generated audio file.
    """

    audiofilename = outvideo.replace(".mp4",'.mp3')

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(outvideo)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename



audiofilename = extract_audio_from_video(output_video_file)
print(audiofilename)

from IPython.display import Audio

Audio(audiofilename)

!pip install --quiet faster-whisper==0.7.0

from faster_whisper import WhisperModel

model_size = "small"
model = WhisperModel(model_size)

segments, info = model.transcribe(audiofilename, word_timestamps=True)
segments = list(segments)  # The transcription will actually run here.
for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

wordlevel_info = []

for segment in segments:
    for word in segment.words:
      wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})

wordlevel_info

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
# Load the video file
video = VideoFileClip(output_video_file)

# Function to generate text clips
def generate_text_clip(word, start, end, video):
    txt_clip = (TextClip(word,font_size=80,color='white',font = "Courier" ,stroke_width=3, stroke_color='black').with_position('center')
               .with_duration(end - start))

    return txt_clip.with_start(start)

# Generate a list of text clips based on timestamps
clips = [generate_text_clip(item['word'], item['start'], item['end'], video) for item in wordlevel_info]

# Overlay the text clips on the video
final_video = CompositeVideoClip([video] + clips)

finalvideoname = active_folder+"/"+"final.mp4"
# Write the result to a file
final_video.write_videofile(finalvideoname, codec="libx264",audio_codec="aac")
