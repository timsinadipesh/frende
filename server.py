#!/usr/bin/env python3
"""
Local translation server using Helsinki-NLP Opus-MT models
Supports EN<->DE, DE<->FR, FR<->EN translation pairs with automatic language detection
Now includes Whisper audio transcription and Piper TTS

Speed improvements in this version:
- CPU INT8 dynamic quantization for Whisper-small + all translation models
- Piper voices preloaded at startup + caching of repeated TTS outputs
"""

import sys
import logging
import tempfile
import os
import platform
import subprocess
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask import send_from_directory
import librosa
import numpy as np
import shutil
import hashlib

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    logging.warning("lingua-language-detector not installed. Install with: pip install lingua-language-detector")

# Use CPU, but speed up Linear ops via threading (you can tune threads if desired)
torch.set_default_device('cpu')
# Leave thread count as detected; you can experiment by setting a fixed number:
# torch.set_num_threads(<num_physical_cores>)
torch.set_num_threads(torch.get_num_threads())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_CONFIGS = {
    'en-de': 'Helsinki-NLP/opus-mt-en-de',
    'de-en': 'Helsinki-NLP/opus-mt-de-en',
    'de-fr': 'Helsinki-NLP/opus-mt-de-fr',
    'fr-de': 'Helsinki-NLP/opus-mt-fr-de',
    'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr'
}

LANGUAGE_PAIRS = {
    'EN → DE': ['en', 'de'],
    'DE → FR': ['de', 'fr'], 
    'FR → EN': ['fr', 'en'],
    'EN ↔ DE': ['en', 'de'],
    'DE ↔ FR': ['de', 'fr'], 
    'FR ↔ EN': ['fr', 'en']
}

# Highest quality, most natural Piper voices
PIPER_VOICES = {
    'en': {
        'name': 'en_US-amy-medium',
        'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx',
        'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json'
    },
    'de': {
        'name': 'de_DE-thorsten-medium',
        'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx',
        'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json'
    },
    'fr': {
        'name': 'fr_FR-siwis-medium',
        'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx',
        'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json'
    }
}

# ---------- Utilities ----------

def sha1_of(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def quantize_linear_int8(model: torch.nn.Module) -> torch.nn.Module:
    """
    CPU INT8 dynamic quantization on Linear layers.
    Works with transformer-based models on CPU and typically yields 30–40% faster matmul-heavy paths.
    """
    try:
        qmodel = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return qmodel
    except Exception as e:
        logger.warning(f"Quantization failed (falling back to fp32): {e}")
        return model

# ---------- TTS Service (Piper) ----------

class TTSService:
    def __init__(self):
        self.piper_dir = Path.home() / '.cache' / 'piper'
        self.piper_dir.mkdir(parents=True, exist_ok=True)
        self.piper_executable = None
        self.voices_dir = self.piper_dir / 'voices'
        self.voices_dir.mkdir(exist_ok=True)
        # Simple cache: (text, lang) -> path
        self.audio_cache: Dict[Tuple[str, str], str] = {}
        # Track preloaded voices so we never re-download in hot path
        self.preloaded = set()
        
    def get_piper_download_info(self):
        """Get the appropriate Piper download URL for current OS"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'linux':
            if 'x86_64' in machine or 'amd64' in machine:
                return {
                    'url': 'https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz',
                    'executable': 'piper'
                }
            elif 'aarch64' in machine or 'arm64' in machine:
                return {
                    'url': 'https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz',
                    'executable': 'piper'
                }
        elif system == 'darwin':  # macOS
            return {
                'url': 'https://github.com/rhasspy/piper/releases/latest/download/piper_macos_x64.tar.gz',
                'executable': 'piper'
            }
        elif system == 'windows':
            return {
                'url': 'https://github.com/rhasspy/piper/releases/latest/download/piper_windows_amd64.zip',
                'executable': 'piper.exe'
            }
        
        raise RuntimeError(f"Unsupported platform: {system} {machine}")
    
    def download_and_extract(self, url: str, extract_to: Path):
        """Download and extract archive"""
        logger.info(f"Downloading from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            # Extract based on file extension
            if url.endswith('.tar.gz'):
                with tarfile.open(temp_file_path, 'r:gz') as tar:
                    tar.extractall(extract_to)
            elif url.endswith('.zip'):
                with zipfile.ZipFile(temp_file_path, 'r') as zip_file:
                    zip_file.extractall(extract_to)
            else:
                raise ValueError(f"Unsupported archive format: {url}")
        finally:
            os.unlink(temp_file_path)
    
    def download_file(self, url: str, filepath: Path):
        """Download a single file"""
        logger.info(f"Downloading {filepath.name}")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)

    def setup_piper(self):
        """Ensure a Piper executable is available."""
        # 1) Prefer system-installed piper
        system_piper = shutil.which("piper")
        if system_piper:
            self.piper_executable = system_piper
            logger.info(f"Using system Piper: {self.piper_executable}")
            return

        # 2) Look for previously downloaded binary
        for root, _, files in os.walk(self.piper_dir):
            for file in files:
                if file == "piper" or file == "piper.exe":
                    self.piper_executable = str(Path(root) / file)
                    break
            if self.piper_executable:
                break

        # 3) If still not found, download an appropriate build
        if not self.piper_executable:
            info = self.get_piper_download_info()
            self.download_and_extract(info['url'], self.piper_dir)
            # Search again
            for root, _, files in os.walk(self.piper_dir):
                for file in files:
                    if file == info['executable']:
                        self.piper_executable = str(Path(root) / file)
                        break
                if self.piper_executable:
                    break

        if not self.piper_executable:
            raise RuntimeError("Could not find Piper executable after extraction")

        try:
            os.chmod(self.piper_executable, 0o755)
        except Exception:
            pass

        logger.info(f"Piper ready: {self.piper_executable}")

    def download_voice(self, lang_code: str):
        """Download voice model and config for language"""
        if lang_code not in PIPER_VOICES:
            raise ValueError(f"Unsupported language: {lang_code}")
        
        voice_info = PIPER_VOICES[lang_code]
        voice_name = voice_info['name']
        
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"
        
        if not model_path.exists():
            self.download_file(voice_info['url'], model_path)
        if not config_path.exists():
            self.download_file(voice_info['config_url'], config_path)
        
        logger.info(f"Voice ready: {voice_name}")
        return model_path, config_path

    def preload_all_voices(self):
        """Preload EN/DE/FR voices at startup to avoid downloads on hot path."""
        for lang in ('en', 'de', 'fr'):
            try:
                self.download_voice(lang)
                self.preloaded.add(lang)
            except Exception as e:
                logger.error(f"Failed to preload voice {lang}: {e}")

    def synthesize_speech(self, text: str, lang_code: str) -> Optional[str]:
        """
        Generate speech audio for text in specified language.
        Optimizations:
        - Reuse downloaded voices
        - Cache repeated (text, lang) outputs within the process
        NOTE: We still invoke Piper per request because the CLI binds output file at start.
        """
        try:
            if not self.piper_executable or not os.path.exists(self.piper_executable):
                self.setup_piper()

            # Cache hit?
            cache_key = (text, lang_code)
            if cache_key in self.audio_cache and os.path.exists(self.audio_cache[cache_key]):
                return self.audio_cache[cache_key]

            # Ensure voice files exist (preloaded at startup)
            model_path, config_path = self.download_voice(lang_code)

            # Create temporary output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name

            cmd = [
                self.piper_executable,
                "--model", str(model_path),
                "--config", str(config_path),
                "--output_file", output_path
            ]

            # One-shot invocation (fast path now that binary + voices are already warm on disk)
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            try:
                process.communicate(input=text, timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                raise

            if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.audio_cache[cache_key] = output_path
                return output_path

            # Failure path
            stderr = process.stderr.read() if process.stderr else ""
            logger.error(f"Piper failed (code {process.returncode}) stderr={stderr}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None

# ---------- Translation + Transcription Service ----------

class TranslationService:
    def __init__(self):
        self.models: Dict[str, AutoModelForSeq2SeqLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.lang_detector = None
        self.whisper_model = None
        self.whisper_processor = None
        logger.info("Using CPU for inference")
        
        if LINGUA_AVAILABLE:
            try:
                languages = [Language.ENGLISH, Language.GERMAN, Language.FRENCH]
                self.lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()
                logger.info("Lingua language detector initialized for EN, DE, FR")
            except Exception as e:
                logger.error(f"Failed to initialize lingua detector: {e}")
                self.lang_detector = None
    
    def load_whisper_model(self):
        """Load Whisper model for audio transcription and apply INT8 dynamic quantization on Linear layers."""
        try:
            logger.info("Loading Whisper small model (will quantize to INT8 Linear)...")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            # Remove forced language to let detection/auto work
            base_model.config.forced_decoder_ids = None
            # INT8 dynamic quantization on Linear layers (CPU-friendly)
            self.whisper_model = quantize_linear_int8(base_model).eval()
            logger.info("Whisper-small loaded and quantized (INT8 Linear)")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
        
    def load_models(self):
        """
        Load all translation models, tokenize, and quantize (INT8 dynamic on Linear).
        Kept as preload (your original behavior) but now much lighter/faster at inference.
        """
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for direction, model_name in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {direction} model (will quantize to INT8 Linear)...")
                
                self.tokenizers[direction] = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32
                )
                # INT8 dynamic quantization on Linear layers
                q_model = quantize_linear_int8(base_model).eval()
                self.models[direction] = q_model
                
                logger.info(f"Loaded + quantized {direction}")
                
            except Exception as e:
                logger.error(f"Failed to load {direction} model: {e}")
                sys.exit(1)
        
        logger.info("All translation models loaded and quantized")
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using Whisper (quantized)."""
        if not self.whisper_model or not self.whisper_processor:
            logger.error("Whisper model not loaded")
            return None
            
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Load audio using librosa (short clips per your use case)
            audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
            logger.info(f"Loaded audio: {len(audio_array)} samples at {sampling_rate}Hz")
            
            # Process audio
            input_features = self.whisper_processor(
                audio_array, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_features
            
            # Generate transcription (no sampling; deterministic for quality)
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
            
            # Decode to text
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcribed_text = transcription[0].strip()
            
            logger.info(f"Transcription successful. Text: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of input text using lingua"""
        if not self.lang_detector:
            return None
            
        try:
            text = text.strip()
            if len(text) < 3:
                return None
            
            detected_language = self.lang_detector.detect_language_of(text)
            if detected_language is None:
                logger.info(f"Could not detect language for text: {text[:50]}...")
                return None
            
            language_code = None
            if detected_language == Language.ENGLISH:
                language_code = 'en'
            elif detected_language == Language.GERMAN:
                language_code = 'de'
            elif detected_language == Language.FRENCH:
                language_code = 'fr'
            
            confidence_values = self.lang_detector.compute_language_confidence_values(text)
            confidence = 0.0
            for lang_confidence in confidence_values:
                if lang_confidence.language == detected_language:
                    confidence = lang_confidence.value
                    break
            
            logger.info(f"Detected language: {language_code} (confidence: {confidence:.3f}) for text: {text[:50]}...")
            if confidence > 0.3:
                return language_code
            else:
                logger.info(f"Low confidence ({confidence:.3f}), using fallback")
                return None
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None
    
    def auto_translate(self, text: str, language_pair: str) -> Tuple[Optional[str], str]:
        """
        Automatically detect language and translate
        Returns: (translation, direction_used)
        """
        if language_pair not in LANGUAGE_PAIRS:
            return None, "unknown"
            
        lang1, lang2 = LANGUAGE_PAIRS[language_pair]
        detected_lang = self.detect_language(text)
        
        if detected_lang == lang1:
            direction = f"{lang1}-{lang2}"
            logger.info(f"Detected {lang1}, translating to {lang2}")
        elif detected_lang == lang2:
            direction = f"{lang2}-{lang1}"
            logger.info(f"Detected {lang2}, translating to {lang1}")
        else:
            logger.info(f"Language detection unclear or low confidence, trying both directions")
            direction1 = f"{lang1}-{lang2}"
            direction2 = f"{lang2}-{lang1}"
            translation1 = self.translate(text, direction1)
            translation2 = self.translate(text, direction2)
            if translation1 and translation2:
                input_len = len(text.strip())
                trans1_len = len(translation1.strip())
                trans2_len = len(translation2.strip())
                if (trans1_len < input_len * 0.5) or (translation1.lower().strip() == text.lower().strip()):
                    direction = direction2
                    logger.info(f"Chose {direction2} based on translation quality")
                elif (trans2_len < input_len * 0.5) or (translation2.lower().strip() == text.lower().strip()):
                    direction = direction1
                    logger.info(f"Chose {direction1} based on translation quality")
                else:
                    direction = direction1
                    logger.info(f"Using default direction: {direction}")
            else:
                direction = direction1
                logger.info(f"Using default direction: {direction}")
        
        translation = self.translate(text, direction)
        return translation, direction
    
    def translate(self, text: str, direction: str) -> Optional[str]:
        """Translate text using specified model direction"""
        if direction not in self.models:
            return None
            
        try:
            tokenizer = self.tokenizers[direction]
            model = self.models[direction]
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,          # keep your original quality setting
                    early_stopping=True,
                    do_sample=False
                )
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for {direction}: {e}")
            return None

translation_service = TranslationService()
tts_service = TTSService()

# ---------- Routes ----------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'models_loaded': len(translation_service.models),
        'lingua_available': LINGUA_AVAILABLE,
        'lang_detector_loaded': translation_service.lang_detector is not None,
        'whisper_loaded': translation_service.whisper_model is not None,
        'tts_available': tts_service.piper_executable is not None,
        'preloaded_tts_voices': list(tts_service.preloaded)
    })

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        language_pair = data.get('language_pair', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if language_pair not in LANGUAGE_PAIRS:
            return jsonify({'error': f'Invalid language pair: {language_pair}'}), 400
        
        result, direction_used = translation_service.auto_translate(text, language_pair)
        if result is None:
            return jsonify({'error': 'Translation failed'}), 500
        
        # Determine target language for TTS
        target_lang = direction_used.split('-')[1] if '-' in direction_used else 'en'
        
        return jsonify({
            'text': text,
            'language_pair': language_pair,
            'direction_used': direction_used,
            'translation': result,
            'target_language': target_lang
        })
        
    except Exception as e:
        logger.error(f"Request error: {e}", exc_info=True)
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language_pair = request.form.get('language_pair', '')
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        audio_file.seek(0)
        audio_content = audio_file.read()
        actual_size = len(audio_content)
        if actual_size == 0:
            return jsonify({'error': 'Audio file contains no data'}), 400
        if actual_size < 100:
            return jsonify({'error': 'Audio file is too small to be valid audio data'}), 400
        if language_pair not in LANGUAGE_PAIRS:
            return jsonify({'error': f'Invalid language pair: {language_pair}'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe the audio
            transcribed_text = translation_service.transcribe_audio(temp_file_path)
            if not transcribed_text or transcribed_text.strip() == "":
                return jsonify({'error': 'No speech detected in audio'}), 400
            
            # Translate the transcribed text
            translation, direction_used = translation_service.auto_translate(transcribed_text, language_pair)
            if translation is None:
                return jsonify({'error': 'Translation failed'}), 500
            
            # Determine target language for TTS
            target_lang = direction_used.split('-')[1] if '-' in direction_used else 'en'
            
            return jsonify({
                'transcribed_text': transcribed_text,
                'language_pair': language_pair,
                'direction_used': direction_used,
                'translation': translation,
                'target_language': target_lang
            })
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Transcription request error: {e}", exc_info=True)
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/fix-piper', methods=['POST'])
def fix_piper():
    """Manual endpoint to fix Piper permissions"""
    try:
        logger.info("Manual Piper fix requested")
        # Force re-setup
        tts_service.piper_executable = None
        tts_service.setup_piper()
        # Preload voices again
        tts_service.preload_all_voices()
        return jsonify({
            'status': 'success',
            'message': f'Piper fixed and ready at: {tts_service.piper_executable}',
            'executable': tts_service.piper_executable,
            'permissions': oct(os.stat(tts_service.piper_executable).st_mode)[-3:] if tts_service.piper_executable else None,
            'preloaded_voices': list(tts_service.preloaded)
        })
    except Exception as e:
        logger.error(f"Manual Piper fix failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        language = data.get('language', 'en').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if language not in PIPER_VOICES:
            return jsonify({'error': f'Unsupported language: {language}'}), 400
        
        audio_path = tts_service.synthesize_speech(text, language)
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'TTS generation failed'}), 500
        
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='speech.wav'
        )
    except Exception as e:
        logger.error(f"TTS request error: {e}", exc_info=True)
        return jsonify({'error': f'TTS failed: {str(e)}'}), 500

@app.route('/pairs', methods=['GET'])
def get_pairs():
    return jsonify({'pairs': list(LANGUAGE_PAIRS.keys())})

@app.route('/directions', methods=['GET'])
def get_directions():
    return jsonify({'directions': list(MODEL_CONFIGS.keys())})

def main():
    logger.info("Starting translation server with Whisper (quantized) and Piper TTS (preloaded voices)")
    
    if not LINGUA_AVAILABLE:
        logger.error("lingua-language-detector is required for auto-detection. Install with: pip install lingua-language-detector")
        sys.exit(1)
    
    # Setup Piper binary + preload all voices (kept “ready” from the start)
    logger.info("Setting up Piper TTS and preloading voices...")
    tts_service.setup_piper()
    tts_service.preload_all_voices()
    
    # Load translation models (+quantized) and Whisper (+quantized)
    translation_service.load_models()
    translation_service.load_whisper_model()
    
    logger.info("Server ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    main()
