#!/usr/bin/env python3
"""
Local translation server using Helsinki-NLP Opus-MT models
Supports EN<->DE, DE<->FR, FR<->EN translation pairs
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory

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

class TranslationService:
    def __init__(self):
        self.models: Dict[str, AutoModelForSeq2SeqLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        logger.info("Using CPU for inference")
        
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
    return jsonify({'status': 'ok', 'models_loaded': len(translation_service.models)})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        direction = data.get('direction', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if direction not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid direction: {direction}'}), 400
        
        result = translation_service.translate(text, direction)
        
        if result is None:
            return jsonify({'error': 'Translation failed'}), 500
        
        return jsonify({
            'text': text,
            'direction': direction,
            'translation': result
        })
        
    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({'error': 'Invalid request'}), 400

@app.route('/directions', methods=['GET'])
def get_directions():
    return jsonify({'directions': list(MODEL_CONFIGS.keys())})

def main():
    logger.info("Starting translation server")
    
    # Load models at startup
    translation_service.load_models()
    
    # Start server
    logger.info("Server ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

@app.route('/')
def index():
    # Serve the HTML file from the same directory as server.py
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    # Allow serving CSS, JS, images, etc. from same folder
    return send_from_directory('.', filename)

if __name__ == '__main__':
    main()