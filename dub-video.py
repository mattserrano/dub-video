#!/usr/bin/env python3
"""
Local Video Dubbing Script

Uses:
- yt-dlp
- ffmpeg (audio/video processing)
- OpenAI Whisper (speech-to-text)
- Coqui TTS XTTS (text-to-speech)
- SadTalker (face animation dubbing)
"""

import argparse
import os
import subprocess
import shutil
import tempfile
import torch
import whisper
from TTS.api import TTS
from tqdm import tqdm

# ---------------------------
# Utility functions
# ---------------------------

def run_cmd(cmd):
    """Run a shell command and raise on error."""
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def download_remote_video(url, output_path):
    """Download and convert input video to mp4 using yt-dlp."""
    run_cmd([
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", output_path,
        url
    ])

def extract_audio(video_path, audio_path):
    """Extract audio track from video."""
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ])

def combine_audio_video(video_path, audio_path, output_path):
    """Combine video and audio into final dubbed video."""
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ])

# ---------------------------
# Whisper transcription
# ---------------------------

def transcribe_audio(model_name, audio_path, language=None):
    """Transcribe audio to text and timestamps with Whisper."""
    model = whisper.load_model(model_name)
    if language:
        result = model.transcribe(audio_path, language=language)
    else:
        result = model.transcribe(audio_path)
    return result["segments"]

# ---------------------------
# Coqui TTS synthesis
# ---------------------------

def synthesize_segments_tts(model_name, segments, out_dir, voice=None, language=None, speaker_wav=None):
    """Synthesize each segment to a WAV file using Coqui TTS."""
    tts = TTS(model_name=model_name)

    # List available speakers if multi-speaker
    speaker = None
    if hasattr(tts, "speakers") and tts.speakers:
        print("\n[INFO] Available voices for this model:")
        for spk in tts.speakers:
            print(f"  - {spk}")
        print()

        if voice:
            if voice in tts.speakers:
                speaker = voice
            else:
                print(f"[WARN] Voice '{voice}' not found, defaulting to: {tts.speakers[0]}")
                speaker = tts.speakers[0]
        else:
            print(f"[INFO] No voice provided, defaulting to: {tts.speakers[0]}")
            speaker = tts.speakers[0]

    out_files = []
    for i, seg in enumerate(tqdm(segments, desc="Synthesizing TTS", unit="seg")):
        out_path = os.path.join(out_dir, f"segment_{i:04d}.wav")
        tts.tts_to_file(
            text=seg["text"],
            file_path=out_path,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav
        )
        out_files.append(out_path)

    return out_files


# ---------------------------
# Audio concatenation
# ---------------------------

def concatenate_audio(files, output_path):
    """Concatenate multiple audio segments into one file."""
    list_path = output_path + "_list.txt"
    with open(list_path, "w") as f:
        for file in files:
            f.write(f"file '{os.path.abspath(file)}'\n")

    run_cmd([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ])
    os.remove(list_path)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Local Video Dubbing Tool")
    parser.add_argument("--video", help="Path to local video file")
    parser.add_argument("--url", help="Video URL to download and dub")
    parser.add_argument("--output", default="dubbed_video.mp4", help="Output video path")
    parser.add_argument("--language", help="Language for Whisper transcription")
    parser.add_argument("--whisper-model", default="small", help="Whisper model name")
    parser.add_argument("--tts-model", default="tts_models/multilingual/multi-dataset/xtts_v2", help="Coqui TTS model name")
    parser.add_argument("--voice", help="Voice/speaker ID to use (if multi-speaker)")
    parser.add_argument("--speaker-wav", help="Speaker wav to clone")
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp()
    try:
        video_path = args.video
        if args.url:
            video_path = os.path.join(temp_dir, "input.mp4")
            print("[INFO] Downloading remote video...")
            download_remote_video(args.url, video_path)

        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError("No valid video file provided.")

        print("[INFO] Extracting audio...")
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(video_path, audio_path)

        print("[INFO] Transcribing with Whisper...")
        segments = transcribe_audio(args.whisper_model, audio_path, args.language)

        print("[INFO] Synthesizing TTS segments with Coqui...")
        tts_out_dir = os.path.join(temp_dir, "tts_segments")
        os.makedirs(tts_out_dir, exist_ok=True)
        tts_files = synthesize_segments_tts(args.tts_model, segments, tts_out_dir, voice=args.voice, speaker_wav=args.speaker_wav, language=args.language)

        print("[INFO] Concatenating TTS segments...")
        full_tts_path = os.path.join(temp_dir, "tts_full.wav")
        concatenate_audio(tts_files, full_tts_path)

        print("[INFO] Combining dubbed audio with original video...")
        combine_audio_video(video_path, full_tts_path, args.output)

        print(f"[DONE] Dubbed video saved to: {args.output}")

    finally:
        print("[INFO] Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()