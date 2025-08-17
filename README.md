# frende

Private on-device AI interpreter for breaking language barriers.

Real-time translation between English, German, and French with voice input/output. Everything runs locally - no data sent to external servers.

## Features

- Text and voice translation (EN ↔ DE, DE ↔ FR, FR ↔ EN)
- Auto language detection
- Text-to-speech output
- Voice mode for continuous conversation
- Dark/light theme
- Fully offline after setup

## Installation

### Prerequisites
- Python 3.12
- ~4GB free disk space (for models)

### Setup
```bash
git clone <https://github.com/timsinadipesh/frende.git>
cd frende
pip install -r requirements.txt
python server.py
```

First run will download models (~2GB) and voice files automatically.

Open http://localhost:5000 in your browser.

## Usage

1. **Text mode**: Type and click send
2. **Voice mode**: Click microphone icon, speak, click again to stop
3. **Continuous voice**: Click voice bars icon for hands-free conversation
4. **Language pairs**: Click language button to switch between EN↔DE, DE↔FR, FR↔EN

## Technical Details

- **Translation**: Helsinki-NLP Opus-MT models (quantized INT8)
- **Speech-to-text**: OpenAI Whisper-small (quantized)
- **Text-to-speech**: Piper voices
- **Language detection**: Lingua library
- **Backend**: Flask server
- **Frontend**: Vanilla HTML/JS

## System Requirements

- **RAM**: 16GB recommended (8GB minimum)
- **CPU**: x64 architecture
- **OS**: Linux, macOS, Windows
- **Browser**: Modern browser with WebRTC support

Models are cached locally after first download.