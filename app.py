from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
import os
import shutil
import threading
import time
import requests

app = Flask(__name__)

GRADIO_SPACE = "openbmb/VoxCPM-Demo"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Render URL သတ်မှတ်ခြင်း (မိမိ Render Web App URL ထည့်ပါ)
RENDER_APP_URL = "https://voxcpm-myanmar.onrender.com"

# -------------------------------------------------------------
# App အမြဲ Live ဖြစ်နေစေရန် မိမိ Server ကို မိမိ ပြန် Ping သည့် Logic
# -------------------------------------------------------------
def keep_alive():
    # Server စတက်တက်ချင်း စက္ကန့် ၃၀ စောင့်ပြီးမှ စတင် Ping မည်
    time.sleep(30)
    while True:
        try:
            response = requests.get(RENDER_APP_URL, timeout=10)
            print(f"[Keep-Alive] Ping sent to {RENDER_APP_URL} | Status Code: {response.status_code}")
        except Exception as e:
            print(f"[Keep-Alive] Ping failed: {e}")
        
        # ၁၃ မိနစ် (၇၈၀ စက္ကန့်) တိုင်း တစ်ကြိမ် အလိုအလျောက် Ping ပြုလုပ်မည်
        # (Render Free Tier ၏ ၁၅ မိနစ် Timeout မတိုင်မီ နိုးပေးခြင်းဖြစ်သည်)
        time.sleep(780)

# Background Thread ကို App စတင်သည်နှင့် သီးသန့် Run ထားမည်
threading.Thread(target=keep_alive, daemon=True).start()

# -------------------------------------------------------------
# Flask Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-audio", methods=["POST"])
def generate_audio():
    ref_file_path = None
    
    try:
        text_input = request.form.get("text", "").strip()
        reference_audio = request.files.get("reference_audio")
        
        if not text_input:
            return jsonify({"error": "စာသား ရိုက်ထည့်ပေးပါ"}), 400

        # Character Length Validation (Max 500)
        if len(text_input) > 500:
            return jsonify({"error": "စာသား အရှည် ပမာဏသည် စာလုံးရေ ၅၀၀ ထက် မပိုရပါ။"}), 400

        if reference_audio:
            filename = reference_audio.filename or "input_voice.webm"
            ref_file_path = os.path.join(UPLOAD_FOLDER, filename)
            reference_audio.save(ref_file_path)

        client = Client(GRADIO_SPACE)

        # Gradio API Call (do_normalize=True ထည့်သွင်းထားသည်)
        result = client.predict(
            text_input=text_input,
            control_instruction="",
            reference_wav_path_input=handle_file(ref_file_path) if ref_file_path else None,
            use_prompt_text=False,
            prompt_text_input="",
            cfg_value_input=2,
            do_normalize=True,
            denoise=False,
            api_name="/generate"
        )

        audio_path = result if isinstance(result, str) else result.get('path') if isinstance(result, dict) else None

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"error": "Hugging Face Space မှ အသံဖိုင် တုံ့ပြန်မှု မရရှိပါ သို့မဟုတ် မော်ဒယ် နှေးနေပါသည်။"}), 500

        output_filename = "output.wav"
        destination_path = os.path.join(UPLOAD_FOLDER, output_filename)
        shutil.copy(audio_path, destination_path)

        return jsonify({"success": True, "audio_url": f"/get-audio/{output_filename}"})

    except Exception as e:
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

    finally:
        if ref_file_path and os.path.exists(ref_file_path):
            try:
                os.remove(ref_file_path)
            except Exception:
                pass

@app.route("/get-audio/<filename>")
def get_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/wav", as_attachment=False)
    return jsonify({"error": "Audio file not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
