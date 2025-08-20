## dub-video
dub-video is a command line utility to translate and dub-over videos.

### Built With
- yt-dlp
- coqui-tts
- ffmpeg
- OpenAI Whisper

### Prerequisites
Install Python 3.11 and pip.
Locate (or record) a sample audio file.

### Installation
Clone the repository and install the required depencencies:

```bash
python3.11 -m pip install -r requirements.txt
```

## Usage

Dub a local video:
```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=y python3.11 dub-video.py --video /path/to/video.mp4 --out dubbed_video.mp4 --language en --speaker-wav sample.wav
```

Download and dub a video:
```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=y python3.11 dub-video.py --url https://tinyurl.com/yhr5na2b --out dubbed.mp4 --language en --speaker-wav sample.wav
```
