from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client, handle_file
import os
import shutil
import re

app = Flask(__name__)

# Hugging Face Space Endpoint
GRADIO_SPACE = "openbmb/VoxCPM-Demo"

# Audio Files သိမ်းဆည်းမည့် Folder
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

        # Voice Clone အတွက် Upload/Record လုပ်ထားသော Audio ကို သိမ်းဆည်းခြင်း
        if reference_audio:
            filename = reference_audio.filename or "input_voice.webm"
            ref_file_path = os.path.join(UPLOAD_FOLDER, filename)
            reference_audio.save(ref_file_path)

        client = Client(GRADIO_SPACE)

        # -------------------------------------------------------------
        # စာလုံးရေ အလွန်များပါက VoxCPM API မဟန့်သွားစေရန် စာကြောင်းအလိုက် ခွဲခြင်း
        # -------------------------------------------------------------
        max_chunk_size = 250  # တစ်ကြိမ်လျှင် မော်ဒယ်အတွက် အဆင်ပြေဆုံး စာလုံးရေ
        chunks = []
        
        # စာကြောင်း ခွဲခြားနိုင်သည့် ပုဒ်ဖြတ်ပုဒ်ရပ်များဖြင့် ခွဲခြင်း (၊ ၊ ။ ၊ . ၊ \n)
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

        # စာသား လုံးဝမခွဲဘဲ ၁ ကြောင်းတည်း ရှိနေပါကလည်း အဆင်ပြေအောင်
        if not chunks:
            chunks = [text_input]

        # -------------------------------------------------------------
        # စာကြောင်း အပိုင်းများအလိုက် API လှမ်းခေါ်ပြီး အသံထုတ်ခြင်း
        # -------------------------------------------------------------
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

            # Gradio မှ ပြန်လာသည့် Audio File Path စစ်ဆေးခြင်း
            audio_path = result if isinstance(result, str) else result.get('path') if isinstance(result, dict) else None
            
            if audio_path and os.path.exists(audio_path):
                temp_chunk_path = os.path.join(UPLOAD_FOLDER, f"part_{idx}.wav")
                shutil.copy(audio_path, temp_chunk_path)
                output_files.append(temp_chunk_path)

        if not output_files:
            return jsonify({"error": "Hugging Face API မှ အသံဖိုင် တုံ့ပြန်မှု မရရှိပါ သို့မဟုတ် မော်ဒယ် နှေးနေပါသည်။"}), 500

        # -------------------------------------------------------------
        # ထွက်လာသော Audio File အပိုင်းများကို တစ်ဖိုင်တည်းဖြစ်အောင် ပေါင်းစပ်ခြင်း
        # -------------------------------------------------------------
        final_output_path = os.path.join(UPLOAD_FOLDER, "output.wav")
        
        with open(final_output_path, 'wb') as outfile:
            for fpath in output_files:
                with open(fpath, 'rb') as infile:
                    outfile.write(infile.read())
                # သုံးပြီးသား Temp Audio Chunk ကို ဖျက်ထုတ်ခြင်း
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        return jsonify({"success": True, "audio_url": "/get-audio/output.wav"})

    except Exception as e:
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

    finally:
        # Temporary Voice Clone Audio ကို ဖျက်ပစ်ခြင်း
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
    # Render.com Environment Port ကို ဖတ်ယူခြင်း
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
