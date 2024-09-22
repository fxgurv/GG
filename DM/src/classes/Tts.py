import os
import sys
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from config import ROOT_DIR, get_tts_type, get_elevenlabs_api_key, get_elevenlabs_voice, get_verbose
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from status import info, warning, error, success

try:
    from elevenlabs import generate, set_api_key, voices
    from elevenlabs.api import Models
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

class TTS:
    def __init__(self) -> None:
        self.verbose = get_verbose()
        self.tts_type = get_tts_type()
        
        if self.verbose:
            info(f"Initializing TTS with type: {self.tts_type}")
        
        if self.tts_type == "elevenlabs":
            if not ELEVENLABS_AVAILABLE:
                error("ElevenLabs is not installed. Please install it to use ElevenLabs TTS.")
                raise ImportError("ElevenLabs is not installed. Please install it to use ElevenLabs TTS.")
            
            self.elevenlabs_api_key = get_elevenlabs_api_key()
            self.elevenlabs_voice = get_elevenlabs_voice()
            
            if not self.elevenlabs_api_key:
                error("ElevenLabs API key is not set in the configuration.")
                raise ValueError("ElevenLabs API key is not set in the configuration.")
            
            set_api_key(self.elevenlabs_api_key)
            
            if self.verbose:
                info("ElevenLabs API key set successfully")
            
            try:
                available_voices = voices()
                if self.elevenlabs_voice not in [voice.name for voice in available_voices]:
                    warning(f"Selected voice '{self.elevenlabs_voice}' is not available. Using default voice.")
                    self.elevenlabs_voice = "Rachel"
            except Exception as e:
                error(f"Error checking ElevenLabs voices: {str(e)}")
                warning("Using default voice 'Rachel'")
                self.elevenlabs_voice = "Rachel"
        else:
            if self.verbose:
                info("Initializing Coqui TTS")
            self._init_coqui_tts()

    def _init_coqui_tts(self):
        try:
            site_packages = next((p for p in sys.path if 'site-packages' in p), None)

            if not site_packages:
                error("Could not find site-packages directory. Ensure that the TTS package is installed.")
                raise EnvironmentError("Could not find site-packages directory. Ensure that the TTS package is installed.")

            models_json_path = os.path.join(site_packages, "TTS", ".models.json")

            self._model_manager = ModelManager(models_json_path)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self._model_path, self._config_path, self._model_item = \
                        self._model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC_ph")

                    voc_path, voc_config_path, _ = self._model_manager. \
                        download_model("vocoder_models/en/ljspeech/univnet")
                    
                    self._synthesizer = Synthesizer(
                        tts_checkpoint=self._model_path,
                        tts_config_path=self._config_path,
                        vocoder_checkpoint=voc_path,
                        vocoder_config=voc_config_path
                    )
            
            if self.verbose:
                success("Coqui TTS initialized successfully")
        except Exception as e:
            error(f"Error initializing Coqui TTS: {str(e)}")
            raise

    @property
    def synthesizer(self) -> Optional[Synthesizer]:
        return self._synthesizer if self.tts_type == "coqui_tts" else None

    def synthesize(self, text: str, output_file: str = os.path.join(ROOT_DIR, ".mp", "audio.wav")) -> str:
        if self.verbose:
            info(f"Synthesizing text using {self.tts_type}")
        
        try:
            if self.tts_type == "elevenlabs":
                audio = generate(
                    text=text,
                    voice=self.elevenlabs_voice,
                    model=Models.ELEVEN_MULTILINGUAL_V2
                )
                with open(output_file, 'wb') as f:
                    f.write(audio)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        outputs = self.synthesizer.tts(text)
                        self.synthesizer.save_wav(outputs, output_file)
            
            if self.verbose:
                success(f"Audio synthesized successfully and saved to {output_file}")
            
            return output_file
        except Exception as e:
            error(f"Error synthesizing audio: {str(e)}")
            raise
