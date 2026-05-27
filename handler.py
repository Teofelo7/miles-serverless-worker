import os
import torch
import base64
import runpod
import soundfile as sf
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder
from kokoro_onnx import Kokoro
from chunked_convert import chunked_voice_conversion
# --- Warm Optimization Stage (Loads ONCE when container boots) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ [RunPod Worker]: Initializing GPU Core Context ({DEVICE})...")

# Pre-load models into the cloud VRAM cache
kanade = KanadeModel.from_pretrained("frothywater/kanade-12.5hz").to(DEVICE).eval()
vocoder = load_vocoder(kanade.config.vocoder_name).to(DEVICE)
SAMPLE_RATE = kanade.config.sample_rate

kokoro = Kokoro("model/kokoro.onnx", "voice/voices-v1.0.bin")
print("🧠 [RunPod Worker]: Weights locked in VRAM. Ready for jobs.")

def handler(job):
    """Processes incoming zero-shot voice requests instantly via RunPod Serverless Queue."""
    try:
        job_input = job["input"]
        text = job_input.get("text")
        ref_audio_b64 = job_input.get("reference_audio")
        
        if not text or not ref_audio_b64:
            return {"error": "Missing required 'text' or 'reference_audio' payload fields."}

        # 1. Decode reference audio array from local machine
        temp_ref_path = f"/tmp/ref_{job['id']}.wav"
        with open(temp_ref_path, "wb") as f:
            f.write(base64.b64decode(ref_audio_b64))
            
        # 2. Run Base Synthesizer
        samples, sr = kokoro.create(text, voice="am_michael", speed=1.0, lang="en-us")
        temp_source_path = f"/tmp/src_{job['id']}.wav"
        sf.write(temp_source_path, samples, sr)
        
        # 3. Cast to GPU Tensors
        source_wav = load_audio(temp_source_path).to(DEVICE)
        ref_wav = load_audio(temp_ref_path).to(DEVICE)
        
        # 4. Process Voice Conversion Neural Graph
        with torch.inference_mode():
            converted_wav = chunked_voice_conversion(
                kanade=kanade,
                vocoder_model=vocoder,
                source_wav=source_wav,
                ref_wav=ref_wav,
                sample_rate=SAMPLE_RATE
            )
            
        # 5. Extract output array and encode back to text formatting
        temp_out_path = f"/tmp/out_{job['id']}.wav"
        audio_np = converted_wav.detach().cpu().numpy().flatten()
        sf.write(temp_out_path, audio_np, SAMPLE_RATE)
        
        with open(temp_out_path, "rb") as out_f:
            encoded_output = base64.b64encode(out_f.read()).decode('utf-8')
            
        # Clean up temporary disk files inside container to avoid memory bloating
        for path in [temp_ref_path, temp_source_path, temp_out_path]:
            if os.path.exists(path):
                os.remove(path)
                
        return {"audio_base64": encoded_output, "status": "success"}

    except Exception as e:
        return {"error": f"Internal Worker Inference Crash: {str(e)}", "status": "failed"}

# Start the RunPod Listener
runpod.serverless.start({"handler": handler})