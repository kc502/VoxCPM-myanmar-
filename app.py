from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
import os
import shutil
import re
import wave

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
    output_files = []
    
    try:
        text_input = request.form.get("text", "").strip()
        reference_audio = request.files.get("reference_audio")
        
        if not text_input:
            return jsonify({"error": "စာသား ရိုက်ထည့်ပေးပါ"}), 400

        if reference_audio:
            filename = reference_audio.filename or "input_voice.webm"
            ref_file_path = os.path.join(UPLOAD_FOLDER, filename)
            reference_audio.save(ref_file_path)

        client = Client(GRADIO_SPACE)

        # စာလုံးရေ ခွဲခြားခြင်း (မော်ဒယ် Timeout မဖြစ်စေရန် စာလုံး ၂၀၀ စီ ခွဲပါမည်)
        max_chunk_size = 200  
        chunks = []
        
        raw_sentences = re.split(r'(?<=[။\.\n])', text_input)
        current_chunk = ""

        for sentence in raw_sentences:
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks:
            chunks = [text_input]

        # Hugging Face API လှမ်းခေါ်ခြင်း
        for idx, chunk in enumerate(chunks):
            result = client.predict(
                text_input=chunk,
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
            
            if audio_path and os.path.exists(audio_path):
                temp_chunk_path = os.path.join(UPLOAD_FOLDER, f"part_{idx}.wav")
                shutil.copy(audio_path, temp_chunk_path)
                output_files.append(temp_chunk_path)

        if not output_files:
            return jsonify({"error": "Hugging Face API မှ အသံဖိုင် တုံ့ပြန်မှု မရရှိပါ။"}), 500

        # -------------------------------------------------------------
        # WAV Files များကို wave module ဖြင့် စနစ်တကျ ပေါင်းစပ်ခြင်း
        # -------------------------------------------------------------
        final_output_path = os.path.join(UPLOAD_FOLDER, "output.wav")
        
        if len(output_files) == 1:
            # အပိုင်း ၁ ပိုင်းတည်းဆိုလျှင် တိုက်ရိုက် copy ကူးမည်
            shutil.copy(output_files[0], final_output_path)
            os.remove(output_files[0])
        else:
            # အပိုင်းများစွာရှိပါက WAV header များကို ညှိ၍ ပေါင်းစပ်မည်
            data = []
            for fpath in output_files:
                with wave.open(fpath, 'rb') as w:
                    data.append((w.getparams(), w.readframes(w.getnframes())))
                os.remove(fpath)

            with wave.open(final_output_path, 'wb') as output:
                output.setparams(data[0][0])
                for params, frames in data:
                    output.writeframes(frames)

        return jsonify({"success": True, "audio_url": "/get-audio/output.wav"})

    except Exception as e:
        # Error တက်ပါက JSON format ဖြင့်သာ ပြန်ပို့ပေးရန်
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
    else:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
