from requests import get, post
from random import randint
from requests.exceptions import RequestException
import time
from requests.adapters import HTTPAdapter
import requests



class Generation:
    def create(self, prompt):
        headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
        }
        try:
            changed_prompt = prompt
            print('changed_prompt: ', changed_prompt)
            s = requests.Session()
            s.mount("https://api.prodia.com/generate", HTTPAdapter(max_retries=5))
            resp = s.get(
                "https://api.prodia.com/generate",
                params={
                    "new": "true",
                    "prompt": changed_prompt,
                    "model": "blazing_drive_v10g.safetensors [ca1c1eab]", # EimisAnimeDiffusion_V1.ckpt [4f828a15] first
                    "negative_prompt": "verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.5),cross-eyed,sketches, (worst quality:2), (low quality:2.1), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, repeating hair",
                    "steps": "20",
                    "cfg": "7",
                    "seed": randint(1, 10000),
                    "sample": "DPM++ 2M Karras",
                    "aspect_ratio": "square"
                },
                headers=headers,
                timeout=30,
            )
            data = resp.json()
            print('data: ', data)
            time.sleep(5)
            while True:
                s = requests.Session()
                s.mount(f"https://api.prodia.com/job/{data['job']}", HTTPAdapter(max_retries=5))
                resp = s.get(f"https://api.prodia.com/job/{data['job']}", headers=headers)
                json = resp.json()
                print('json: ', json)
                time.sleep(5)
                if json["status"] == "succeeded":
                    return s.get(
                        f"https://images.prodia.xyz/{data['job']}.png?download=1",
                        headers=headers,
                    ).content
        except RequestException as exc:
            raise RequestException("Unable to fetch the response.") from exc
