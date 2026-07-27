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
    ref_file_path = None
    try:
        text_input = request.form.get("text", "")
        reference_audio = request.files.get("reference_audio")
        
        if not text_input:
            return jsonify({"error": "စာသား ရိုက်ထည့်ပေးပါ"}), 400

        if reference_audio:
            # တင်လိုက်သော/ဖမ်းထားသော ဖိုင်ကို နာမည် ပေး၍ သိမ်းဆည်းခြင်း
            filename = reference_audio.filename or "input_voice.webm"
            ref_file_path = os.path.join(UPLOAD_FOLDER, filename)
            reference_audio.save(ref_file_path)

        client = Client(GRADIO_SPACE)

        # Gradio Predict ခေါ်ယူခြင်း
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
            return jsonify({"error": "VoxCPM မှ အသံဖိုင် တုံ့ပြန်မှု မရရှိပါ သို့မဟုတ် Temp မော်ဒယ် နှေးနေပါသည်။"}), 500

        output_filename = "output.wav"
        destination_path = os.path.join(UPLOAD_FOLDER, output_filename)
        shutil.copy(audio_path, destination_path)

        return jsonify({"success": True, "audio_url": f"/get-audio/{output_filename}"})

    except Exception as e:
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500
    finally:
        # Upload လုပ်ခဲ့သော Temp File ကို ဖျက်ပစ်ခြင်း
        if ref_file_path and os.path.exists(ref_file_path):
            try:
                os.remove(ref_file_path)
            except Exception:
                pass

@app.route("/get-audio/<filename>")
def get_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, mimetype="audio/wav")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
