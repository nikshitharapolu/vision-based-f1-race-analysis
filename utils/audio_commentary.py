"""
Future scope

Converts commentary text lines to speech and mixes them into the output video.

Three TTS backends (in order of quality):
  1. gTTS  — Google Text-to-Speech (best quality, needs internet)
  2. pyttsx3 — offline, uses system voices (good, no internet)
  3. say    — macOS built-in (fallback, British voice available)
"""

import os
import subprocess
import tempfile
from pathlib import Path


# Backend: Google TTS 

def _tts_gtts(text: str, out_path: str, lang: str = "en") -> bool:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(out_path)
        return True
    except Exception as e:
        print(f"  [WARN] gTTS failed: {e}")
        return False


#  Backend: pyttsx3 (offline) 

def _tts_pyttsx3(text: str, out_path: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            if "british" in voice.name.lower() or "daniel" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 165)   
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return Path(out_path).exists()
    except Exception as e:
        print(f"  [WARN] pyttsx3 failed: {e}")
        return False



def _tts_say(text: str, out_path: str, voice: str = "Daniel") -> bool:
    try:
        cmd = ["say", "-v", voice, "-o", out_path, "--data-format=LEF32@44100", text]
        subprocess.run(cmd, check=True, capture_output=True)
        # Convert to mp3 for ffmpeg compatibility
        mp3_path = out_path.replace(".aiff", ".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path, mp3_path],
            check=True, capture_output=True
        )
        os.replace(mp3_path, out_path.replace(".aiff", ".mp3"))
        return True
    except Exception as e:
        print(f"  [WARN] say failed: {e}")
        return False


#  Generate speech file for one line 

def text_to_speech(
    text:        str,
    out_path:    str,
    backend:     str = "gtts",
) -> bool:
    #Convert text to a speech audio file. Returns True on success
    if backend == "gtts":
        return _tts_gtts(text, out_path)
    elif backend == "pyttsx3":
        return _tts_pyttsx3(text, out_path)
    elif backend == "say":
        aiff = out_path.replace(".mp3", ".aiff")
        return _tts_say(text, aiff)
    else:
        print(f"  [WARN] Unknown TTS backend: {backend}")
        return False



def add_audio_commentary(
    video_path:       str,
    commentary_lines: list[tuple[int, int, str]],
    output_path:      str,
    fps:              float = 30.0,
    voice_backend:    str   = "gtts",
    keep_original_audio: bool = True,
) -> bool:
    """
    Generates TTS for each commentary line and
    mixes them into the video at the correct timestamps.

    Args:
        video_path:       input video (already annotated with boxes)
        commentary_lines: [(frame_start, frame_end, text), ...]
        output_path:      output video with audio
        fps:              video fps
        voice_backend:    "gtts" | "pyttsx3" | "say"
        keep_original_audio: mix with existing video audio if True

    """
    if not commentary_lines:
        print("  [Audio] No commentary lines — copying video as-is")
        import shutil
        shutil.copy2(video_path, output_path)
        return True

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  [ERROR] ffmpeg not found. Install with: brew install ffmpeg")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        speech_files = []

        # Generate TTS for each line
        print(f"  [Audio] Generating {len(commentary_lines)} speech clips ({voice_backend}) …")
        for i, (f_start, f_end, text) in enumerate(commentary_lines):
            if not text.strip():
                continue
            t_start = f_start / fps   
            speech_path = str(tmp / f"speech_{i:04d}.mp3")
            ok = text_to_speech(text, speech_path, backend=voice_backend)
            if ok and Path(speech_path).exists():
                speech_files.append((t_start, speech_path, text))
                print(f"    [{i+1}/{len(commentary_lines)}] {t_start:.1f}s: {text[:50]}")
            else:
                print(f"    [{i+1}/{len(commentary_lines)}] FAILED: {text[:40]}")

        if not speech_files:
            print("  [ERROR] No speech files generated")
            return False

        # Build ffmpeg command to mix all clips
        print(f"  [Audio] Mixing {len(speech_files)} clips into video …")

        cmd = ["ffmpeg", "-y"]

        cmd += ["-i", video_path]

        # Inputs 1..N: speech audio files
        for _, sp, _ in speech_files:
            cmd += ["-i", sp]

        n = len(speech_files)
        filter_parts = []

        for i, (t_start, _, _) in enumerate(speech_files):
            delay_ms = int(t_start * 1000)
            filter_parts.append(
                f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=2.0[s{i}]"
            )
        speech_inputs = "".join(f"[s{i}]" for i in range(n))
        filter_parts.append(f"{speech_inputs}amix=inputs={n}:duration=longest:dropout_transition=0[speech]")

        if keep_original_audio:
            filter_parts.append("[0:a][speech]amix=inputs=2:duration=first:weights=1 2[aout]")
            audio_out = "[aout]"
        else:
            audio_out = "[speech]"

        filter_complex = ";".join(filter_parts)
        cmd += ["-filter_complex", filter_complex]
        cmd += ["-map", "0:v", "-map", audio_out]
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        cmd += [output_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("  [Audio] Retrying without original audio track …")
                filter_parts_clean = []
                for i, (t_start, _, _) in enumerate(speech_files):
                    delay_ms = int(t_start * 1000)
                    filter_parts_clean.append(
                        f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=2.0[s{i}]"
                    )
                filter_parts_clean.append(
                    f"{speech_inputs}amix=inputs={n}:duration=longest:dropout_transition=0[aout]"
                )
                cmd2 = ["ffmpeg", "-y", "-i", video_path]
                for _, sp, _ in speech_files:
                    cmd2 += ["-i", sp]
                cmd2 += ["-filter_complex", ";".join(filter_parts_clean)]
                cmd2 += ["-map", "0:v", "-map", "[aout]"]
                cmd2 += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
                cmd2 += [output_path]
                result2 = subprocess.run(cmd2, capture_output=True, text=True)
                if result2.returncode != 0:
                    print(f"  [ERROR] ffmpeg failed:\n{result2.stderr[-500:]}")
                    return False

            print(f"  [Audio] Done → {output_path}")
            return True

        except Exception as e:
            print(f"  [ERROR] Audio mixing failed: {e}")
            return False