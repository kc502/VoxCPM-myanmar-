from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
import os
import shutil

app = Flask(__name__)

GRADIO_SPACE = "openbmb/VoxCPM-Demo"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-audio", methods=["POST"])
def generate_audio():
    try:
        text_input = request.form.get("text", "")
        reference_audio = request.files.get("reference_audio")
        
        if not text_input:
            return jsonify({"error": "စာသား ရိုက်ထည့်ပေးပါ"}), 400

        ref_file_path = None
        if reference_audio:
            # File နာမည်မရှိပါက default ပေးခြင်း (Direct Mic Record အတွက်)
            filename = reference_audio.filename or "recorded_voice.wav"
            ref_file_path = os.path.join(UPLOAD_FOLDER, filename)
            reference_audio.save(ref_file_path)

        client = Client(GRADIO_SPACE)

        # Gradio Predict
        result = client.predict(
            text_input=text_input,
            control_instruction="",
            reference_wav_path_input=handle_file(ref_file_path) if ref_file_path else None,
            use_prompt_text=False,
            prompt_text_input="",
            cfg_value_input=2,
            do_normalize=False,
            denoise=False,
            api_name="/generate"
        )

        audio_path = result if isinstance(result, str) else result.get('path') if isinstance(result, dict) else None

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"error": "Audio File မထုတ်လုပ်နိုင်ပါ။"}), 500

        output_filename = "output.wav"
        destination_path = os.path.join(UPLOAD_FOLDER, output_filename)
        shutil.copy(audio_path, destination_path)

        # Clean temp uploaded voice clone file
        if ref_file_path and os.path.exists(ref_file_path):
            os.remove(ref_file_path)

        return jsonify({"success": True, "audio_url": f"/get-audio/{output_filename}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-audio/<filename>")
def get_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, mimetype="audio/wav")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
