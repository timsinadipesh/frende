#!/usr/bin/env python3
"""
Local translation server using Helsinki-NLP Opus-MT models
Supports EN<->DE, DE<->FR, FR<->EN translation pairs with automatic language detection
Now includes Whisper audio transcription
"""

import sys
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import librosa
import numpy as np

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    logging.warning("lingua-language-detector not installed. Install with: pip install lingua-language-detector")

torch.set_default_device('cpu')
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
        """Load Whisper model for audio transcription"""
        try:
            logger.info("Loading Whisper small model...")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.whisper_model.config.forced_decoder_ids = None
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
        
    def load_models(self):
        """Load all translation models into memory"""
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for direction, model_name in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {direction} model...")
                
                self.tokenizers[direction] = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                self.models[direction] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32
                )
                
                self.models[direction].eval()
                
                logger.info(f"Loaded {direction} model")
                
            except Exception as e:
                logger.error(f"Failed to load {direction} model: {e}")
                sys.exit(1)
        
        logger.info("All translation models loaded")
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using Whisper"""
        if not self.whisper_model or not self.whisper_processor:
            logger.error("Whisper model not loaded")
            return None
            
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Load audio using librosa
            audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
            logger.info(f"Loaded audio: {len(audio_array)} samples at {sampling_rate}Hz")
            
            # Process audio
            input_features = self.whisper_processor(
                audio_array, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_features
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
            
            # Decode to text
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcribed_text = transcription[0].strip()
            
            logger.info(f"Transcription successful. Text: {transcribed_text[:100]}...")
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for {direction}: {e}")
            return None

translation_service = TranslationService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'models_loaded': len(translation_service.models),
        'lingua_available': LINGUA_AVAILABLE,
        'lang_detector_loaded': translation_service.lang_detector is not None,
        'whisper_loaded': translation_service.whisper_model is not None
    })

@app.route('/translate', methods=['POST'])
def translate():
    try:
        logger.info(f"Received request: {request.data}")
        
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        if data is None:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        language_pair = data.get('language_pair', '')
        
        logger.info(f"Text: '{text}', Language pair: '{language_pair}'")
        logger.info(f"Available language pairs: {list(LANGUAGE_PAIRS.keys())}")
        
        if not text:
            logger.error("No text provided")
            return jsonify({'error': 'No text provided'}), 400
        
        if language_pair not in LANGUAGE_PAIRS:
            logger.error(f"Invalid language pair: '{language_pair}' not in {list(LANGUAGE_PAIRS.keys())}")
            return jsonify({'error': f'Invalid language pair: {language_pair}'}), 400
        
        result, direction_used = translation_service.auto_translate(text, language_pair)
        
        if result is None:
            logger.error("Translation failed - result is None")
            return jsonify({'error': 'Translation failed'}), 500
        
        logger.info(f"Translation successful: '{text}' -> '{result}' (direction: {direction_used})")
        
        return jsonify({
            'text': text,
            'language_pair': language_pair,
            'direction_used': direction_used,
            'translation': result
        })
        
    except Exception as e:
        logger.error(f"Request error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info(f"Transcribe endpoint called")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {dict(request.form)}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        if 'audio' not in request.files:
            logger.error("No 'audio' key in request.files")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language_pair = request.form.get('language_pair', '')
        
        logger.info(f"Audio file: {audio_file.filename}")
        logger.info(f"Audio file content type: {audio_file.content_type}")
        logger.info(f"Language pair: {language_pair}")
        
        if audio_file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Read the file content to get the actual data
        audio_file.seek(0)  # Make sure we're at the beginning
        audio_content = audio_file.read()
        actual_size = len(audio_content)
        
        logger.info(f"Audio file actual content size: {actual_size} bytes")
        
        if actual_size == 0:
            logger.error("Audio content is empty")
            return jsonify({'error': 'Audio file contains no data'}), 400
        
        if actual_size < 100:  # Very small files are likely invalid
            logger.error(f"Audio file is too small ({actual_size} bytes) - likely invalid")
            return jsonify({'error': 'Audio file is too small to be valid audio data'}), 400
        
        if language_pair not in LANGUAGE_PAIRS:
            logger.error(f"Invalid language pair: {language_pair}")
            return jsonify({'error': f'Invalid language pair: {language_pair}'}), 400
        
        # Save the audio content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved audio to temporary file: {temp_file_path}")
        logger.info(f"Temporary file size: {os.path.getsize(temp_file_path)} bytes")
        
        # Verify the file was written correctly
        if os.path.getsize(temp_file_path) == 0:
            logger.error("Temporary file is empty after writing")
            os.unlink(temp_file_path)
            return jsonify({'error': 'Failed to save audio data'}), 500
        
        try:
            # Transcribe the audio
            transcribed_text = translation_service.transcribe_audio(temp_file_path)
            
            if not transcribed_text or transcribed_text.strip() == "":
                logger.error("Transcription returned empty result")
                return jsonify({'error': 'No speech detected in audio'}), 400
            
            logger.info(f"Transcription successful: '{transcribed_text[:100]}...'")
            
            # Translate the transcribed text
            translation, direction_used = translation_service.auto_translate(transcribed_text, language_pair)
            
            if translation is None:
                logger.error("Translation failed")
                return jsonify({'error': 'Translation failed'}), 500
            
            logger.info(f"Translation successful: '{translation[:100]}...'")
            
            return jsonify({
                'transcribed_text': transcribed_text,
                'language_pair': language_pair,
                'direction_used': direction_used,
                'translation': translation
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
                
    except Exception as e:
        logger.error(f"Transcription request error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/pairs', methods=['GET'])
def get_pairs():
    return jsonify({'pairs': list(LANGUAGE_PAIRS.keys())})

@app.route('/directions', methods=['GET'])
def get_directions():
    return jsonify({'directions': list(MODEL_CONFIGS.keys())})

def main():
    logger.info("Starting translation server with Whisper support")
    
    if not LINGUA_AVAILABLE:
        logger.error("lingua-language-detector is required for auto-detection. Install with: pip install lingua-language-detector")
        sys.exit(1)
    
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