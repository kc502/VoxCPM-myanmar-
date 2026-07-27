from flask import Flask, render_template, request, jsonify
from gradio_client import Client, handle_file
import os

app = Flask(__name__)

# Hugging Face Space Client
GRADIO_SPACE = "openbmb/VoxCPM-Demo"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-audio", methods=["POST"])
def generate_audio():
    try:
        data = request.json
        text_input = data.get("text", "")
        
        if not text_input:
            return jsonify({"error": "စာသား ရိုက်ထည့်ပေးပါ"}), 400

        # Gradio Client ချိတ်ဆက်ခြင်း
        client = Client(GRADIO_SPACE)

        # /generate API Endpoint ကို လှမ်းခေါ်ခြင်း
        result = client.predict(
            text_input=text_input,
            control_instruction="",
            reference_wav_path_input=None,
            use_prompt_text=False,
            prompt_text_input="",
            cfg_value_input=2,
            do_normalize=False,
            denoise=False,
            api_name="/generate"
        )

        # result ထဲတွင် generated audio ရဲ့ file path / URL ပါလာမည်ဖြစ်သည်
        return jsonify({"success": True, "audio_result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
