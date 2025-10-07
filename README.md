# Holodeck Malfunction
A tv show episode generator pipeline held together by pure spite and GPT-powered hallucinations.  
It takes a one-line episode synopsis and outputs an AI-generated Star Trek fever dream with dialogue, voices, and lip-synced video.

[Example](https://www.youtube.com/watch?v=9KAWtlkWFpY)

---

## ⚙️ Setup

### 1. Install Wav2Lip
Clone and set up [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) according to their instructions.

You’ll need:
- `./Wav2Lip` folder in your project root  
- The pretrained checkpoint file at  
  `./Wav2Lip/checkpoints/wav2lip_gan.pth`

---

### 2. Create `keys.py`

In the project root, make a file called **`keys.py`** containing your API keys:

```python
FISHSPEECH_KEY = "your_fishspeech_key_here"
OPENROUTER_KEY = "your_openrouter_key_here"
```

---

## 🚀 Running the Pipeline

The process runs in stages.
Each stage writes its output into the same episode_YYYYMMDD_HHMMSS folder.

### 🧠 1. Script Generation

Generate beats and shots from a prompt:

```bash
python .\script_generator_worker.py --prompt "Episode synopsis"
```

### 🎙️ 2. Voice Generation

Use FishSpeech to synthesize dialogue audio:

```bash
python .\fishspeech_worker.py --episode_dir "episode_20251006_151759"
```

### 🎞️ 3. Video Generation

Generate scene visuals via ComfyUI:

```bash
python .\comfy_episode_worker.py --episode-dir "episode_20251006_151759" --frames-root "frames" --workflow ".\workflow_template.json" --concurrency 1 --comfy "http://127.0.0.1:8000"
```

### 👄 4. Lip Sync & Final Assembly

Apply Wav2Lip for lip movement:

```bash
python wav2lip_worker.py --episode-dir ./episode_20251006_151759 --wav2lip-path ./Wav2Lip --checkpoint ./Wav2Lip/checkpoints/wav2lip_gan.pth --concurrency 1 --breathing-room --lead-seconds 1 --tail-seconds 1
```
