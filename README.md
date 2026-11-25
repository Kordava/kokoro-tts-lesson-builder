# Local Kokoro TTS

Turn an English practice script into a listen-and-answer audio lesson. Mark the prompt, give the listener time to respond, then play a model answer. Kokoro speaks the lines locally. The episode builder inserts the pauses and exports a mono WAV, an optional MP3 and a JSON timeline.

## From script to audio

The [train station example](examples/train-station.txt) gives the listener ten seconds to answer a question.

```text
[SECTION: AT_THE_STATION]

[VOICE: COACH]
Which platform does the train leave from?

[PAUSE: ANSWER]

[VOICE: MODEL]
It leaves from platform four.
```

### Install

Use Python 3.10–3.13. On Ubuntu, install the system tools, then run the Python commands from the directory containing this README.

```bash
sudo apt install espeak-ng libsndfile1 ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Build the example

Check the script first. Validation works before the voice model is loaded.

```bash
tts-episode examples/train-station.txt --validate-only
```

Render the same script. Allow network access on first use to download the model and voice if they are missing.

```bash
tts-episode examples/train-station.txt -o build/station.wav \
  --mp3 build/station.mp3 --manifest build/station.json
```

Listen to `build/station.mp3`. To measure the gap between audible speech on either side of the pause, run:

```bash
tts-qa-pauses build/station.wav build/station.json
```

For a single line, skip the script and write a WAV directly:

```bash
tts model --text "It leaves from platform four." -o platform.wav
```

## How the episode builder works

- **Validate before rendering.** Each spoken block has an explicit role. The parser catches errors such as an unanswered role tag or an answer pause followed by another coach prompt.
- **Keep time in samples.** The renderer trims excess silence around each line. The builder then inserts digital silence at 24 kHz. A ten-second answer window occupies exactly 240,000 samples in the WAV.
- **Reuse recorded lines.** The local cache keys speech by text, role, voice, speed and renderer settings. Edit one line to regenerate that line. Adjust a pause duration to change the timeline while reusing cached speech.
- **Inspect the result.** The JSON manifest records the script hash, renderer settings and frame counts. Pause defaults and speaking speeds live in [config.py](local_kokoro_tts/config.py).

## Run the tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The suite exercises parsing, cache keys, silence trimming and WAV assembly. Build tests use a small synthetic audio source so they can run without downloading Kokoro weights.
