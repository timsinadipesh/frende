#!/usr/bin/env python3
"""
Local translation server using Helsinki-NLP Opus-MT models
Supports EN<->DE, DE<->FR, FR<->EN translation pairs with automatic language detection
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory

# Add language detection using lingua
try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    logging.warning("lingua-language-detector not installed. Install with: pip install lingua-language-detector")

# Force CPU-only usage
torch.set_default_device('cpu')
torch.set_num_threads(torch.get_num_threads())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model configurations
MODEL_CONFIGS = {
    'en-de': 'Helsinki-NLP/opus-mt-en-de',
    'de-en': 'Helsinki-NLP/opus-mt-de-en',
    'de-fr': 'Helsinki-NLP/opus-mt-de-fr',
    'fr-de': 'Helsinki-NLP/opus-mt-fr-de',
    'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
    'en-fr': 'Helsinki-NLP/opus-mt-en-fr'
}

# Language pair configurations
LANGUAGE_PAIRS = {
    'EN ↔ DE': ['en', 'de'],
    'DE ↔ FR': ['de', 'fr'], 
    'FR ↔ EN': ['fr', 'en']
}

class TranslationService:
    def __init__(self):
        self.models: Dict[str, AutoModelForSeq2SeqLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.lang_detector = None
        logger.info("Using CPU for inference")
        
        # Initialize lingua language detector
        if LINGUA_AVAILABLE:
            try:
                # Create detector with only the languages we support
                # English, German, French
                languages = [Language.ENGLISH, Language.GERMAN, Language.FRENCH]
                
                self.lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()
                logger.info("Lingua language detector initialized for EN, DE, FR")
                
            except Exception as e:
                logger.error(f"Failed to initialize lingua detector: {e}")
                self.lang_detector = None
        
    def load_models(self):
        """Load all translation models into memory"""
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for direction, model_name in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {direction} model...")
                
                # Load tokenizer
                self.tokenizers[direction] = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                # Load model on CPU
                self.models[direction] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32
                )
                
                # Enable eval mode for inference
                self.models[direction].eval()
                
                logger.info(f"Loaded {direction} model")
                
            except Exception as e:
                logger.error(f"Failed to load {direction} model: {e}")
                sys.exit(1)
        
        logger.info("All models loaded")
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of input text using lingua"""
        if not self.lang_detector:
            return None
            
        try:
            # Clean the text
            text = text.strip()
            if len(text) < 3:  # lingua needs at least a few characters
                return None
            
            # Detect language using lingua
            detected_language = self.lang_detector.detect_language_of(text)
            
            if detected_language is None:
                logger.info(f"Could not detect language for text: {text[:50]}...")
                return None
            
            # Convert lingua Language enum to string code
            language_code = None
            if detected_language == Language.ENGLISH:
                language_code = 'en'
            elif detected_language == Language.GERMAN:
                language_code = 'de'
            elif detected_language == Language.FRENCH:
                language_code = 'fr'
            
            # Get confidence score (lingua provides confidence values)
            confidence_values = self.lang_detector.compute_language_confidence_values(text)
            confidence = 0.0
            
            # Find confidence for detected language
            for lang_confidence in confidence_values:
                if lang_confidence.language == detected_language:
                    confidence = lang_confidence.value
                    break
            
            logger.info(f"Detected language: {language_code} (confidence: {confidence:.3f}) for text: {text[:50]}...")
            
            # Only return if confidence is reasonable
            if confidence > 0.3:  # Lowered from 0.6 to 0.3
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
        
        # Try to detect the language
        detected_lang = self.detect_language(text)
        
        # Determine translation direction
        if detected_lang == lang1:
            direction = f"{lang1}-{lang2}"
            logger.info(f"Detected {lang1}, translating to {lang2}")
        elif detected_lang == lang2:
            direction = f"{lang2}-{lang1}"
            logger.info(f"Detected {lang2}, translating to {lang1}")
        else:
            # Improved fallback: try both directions and pick the better one
            logger.info(f"Language detection unclear or low confidence, trying both directions")
            
            # Try translating in both directions
            direction1 = f"{lang1}-{lang2}"
            direction2 = f"{lang2}-{lang1}"
            
            translation1 = self.translate(text, direction1)
            translation2 = self.translate(text, direction2)
            
            # Simple heuristic: if one translation is much shorter or identical to input, 
            # the other is probably correct
            if translation1 and translation2:
                input_len = len(text.strip())
                trans1_len = len(translation1.strip())
                trans2_len = len(translation2.strip())
                
                # If translation1 is much shorter or same as input, use translation2
                if (trans1_len < input_len * 0.5) or (translation1.lower().strip() == text.lower().strip()):
                    direction = direction2
                    logger.info(f"Chose {direction2} based on translation quality")
                # If translation2 is much shorter or same as input, use translation1  
                elif (trans2_len < input_len * 0.5) or (translation2.lower().strip() == text.lower().strip()):
                    direction = direction1
                    logger.info(f"Chose {direction1} based on translation quality")
                else:
                    # Default to first direction
                    direction = direction1
                    logger.info(f"Using default direction: {direction}")
            else:
                # Fallback to first direction
                direction = direction1
                logger.info(f"Using default direction: {direction}")
        
        # Perform translation
        translation = self.translate(text, direction)
        return translation, direction
    
    def translate(self, text: str, direction: str) -> Optional[str]:
        """Translate text using specified model direction"""
        if direction not in self.models:
            return None
            
        try:
            tokenizer = self.tokenizers[direction]
            model = self.models[direction]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode output
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for {direction}: {e}")
            return None

# Global translation service
translation_service = TranslationService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'models_loaded': len(translation_service.models),
        'lingua_available': LINGUA_AVAILABLE,
        'lang_detector_loaded': translation_service.lang_detector is not None
    })

@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Debug: Log the raw request
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
        
        # Use automatic translation
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

@app.route('/pairs', methods=['GET'])
def get_pairs():
    return jsonify({'pairs': list(LANGUAGE_PAIRS.keys())})

@app.route('/directions', methods=['GET'])
def get_directions():
    return jsonify({'directions': list(MODEL_CONFIGS.keys())})

def main():
    logger.info("Starting translation server")
    
    if not LINGUA_AVAILABLE:
        logger.error("lingua-language-detector is required for auto-detection. Install with: pip install lingua-language-detector")
        sys.exit(1)
    
    # Load models at startup
    translation_service.load_models()
    
    # Start server
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