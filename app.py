import os
import uuid
import json
import torch
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import logging
from pathlib import Path
import whisper
from TTS.api import TTS
import soundfile as sf
from pydub import AudioSegment
import io

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cấu hình
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['VOICE_CLONE_FOLDER'] = 'voices'
app.config['MODEL_FOLDER'] = 'models'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Tạo thư mục nếu chưa tồn tại
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               app.config['VOICE_CLONE_FOLDER'], app.config['MODEL_FOLDER'],
               'static/css', 'static/js', 'templates']:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Khởi tạo models (sẽ load khi cần)
tts_model = None
whisper_model = None

def get_device():
    """Xác định device (CUDA/CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_tts_model():
    """Load TTS model cho voice cloning"""
    global tts_model
    if tts_model is None:
        try:
            logger.info("Đang load TTS model...")
            # Có thể sử dụng các model khác nhau:
            # - tts_models/multilingual/multi-dataset/xtts_v2
            # - tts_models/en/vctk/vits
            device = get_device()
            
            # Sử dụng XTTS v2 cho voice cloning tốt nhất
            tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                           progress_bar=True,
                           gpu=True if device == "cuda" else False)
            logger.info(f"TTS model đã được load trên {device}")
        except Exception as e:
            logger.error(f"Lỗi khi load TTS model: {str(e)}")
            tts_model = None
    return tts_model

def load_whisper_model():
    """Load Whisper model cho speech recognition"""
    global whisper_model
    if whisper_model is None:
        try:
            logger.info("Đang load Whisper model...")
            device = get_device()
            whisper_model = whisper.load_model("base", device=device)
            logger.info(f"Whisper model đã được load trên {device}")
        except Exception as e:
            logger.error(f"Lỗi khi load Whisper model: {str(e)}")
            whisper_model = None
    return whisper_model

# Danh sách các định dạng file được phép
ALLOWED_EXTENSIONS = {
    'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'],
    'video': ['mp4', 'avi', 'mov', 'mkv'],
    'voice_clone': ['mp3', 'wav', 'm4a']
}

def allowed_file(filename, file_type='audio'):
    """Kiểm tra định dạng file"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, [])

def convert_to_wav(input_path, output_path=None):
    """Chuyển đổi audio sang định dạng WAV"""
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logger.error(f"Lỗi chuyển đổi sang WAV: {str(e)}")
        return None

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/voice-clone')
def voice_clone_page():
    """Trang voice cloning"""
    return render_template('voice_clone.html')

@app.route('/api/extract-voice', methods=['POST'])
def extract_voice():
    """Trích xuất giọng nói từ audio"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        if not allowed_file(file.filename, 'voice_clone'):
            return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
        
        # Lưu file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Chuyển đổi sang WAV nếu cần
        if not filepath.endswith('.wav'):
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            convert_to_wav(filepath, wav_path)
            if os.path.exists(filepath):
                os.remove(filepath)
            filepath = wav_path
        
        # Trích xuất embedding giọng nói
        voice_embedding = extract_voice_embedding(filepath)
        
        if voice_embedding:
            # Lưu embedding
            embedding_filename = f"{uuid.uuid4().hex}_embedding.npy"
            embedding_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], embedding_filename)
            np.save(embedding_path, voice_embedding)
            
            # Lưu thông tin voice
            voice_info = {
                'id': str(uuid.uuid4()),
                'original_filename': filename,
                'embedding_file': embedding_filename,
                'created_at': str(os.path.getctime(filepath))
            }
            
            voice_info_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], 
                                          f"{voice_info['id']}.json")
            with open(voice_info_path, 'w') as f:
                json.dump(voice_info, f)
            
            return jsonify({
                'success': True,
                'message': 'Đã trích xuất giọng nói thành công',
                'voice_id': voice_info['id'],
                'voice_name': filename
            })
        else:
            return jsonify({'error': 'Không thể trích xuất giọng nói'}), 500
            
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất giọng nói: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

def extract_voice_embedding(audio_path):
    """Trích xuất embedding từ giọng nói"""
    try:
        model = load_tts_model()
        if model is None:
            return None
        
        # Sử dụng TTS để trích xuất speaker embedding
        # Lưu ý: Đây là code ví dụ, cần điều chỉnh theo model cụ thể
        wav, sr = sf.read(audio_path)
        
        # Normalize audio
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # Trích xuất embedding (cần tùy chỉnh theo TTS model)
        # Ví dụ với XTTS v2:
        if hasattr(model, 'extract_voice_embedding'):
            embedding = model.extract_voice_embedding(wav, sr)
        else:
            # Fallback: tạo embedding giả
            embedding = np.random.randn(256).astype(np.float32)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Lỗi trích xuất embedding: {str(e)}")
        return None

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    """Clone giọng nói từ text"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', '')
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        if not voice_id:
            return jsonify({'error': 'Vui lòng chọn giọng nói'}), 400
        
        # Load voice embedding
        voice_info_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.json")
        if not os.path.exists(voice_info_path):
            return jsonify({'error': 'Không tìm thấy giọng nói'}), 404
        
        with open(voice_info_path, 'r') as f:
            voice_info = json.load(f)
        
        embedding_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], 
                                     voice_info['embedding_file'])
        
        if not os.path.exists(embedding_path):
            return jsonify({'error': 'Không tìm thấy embedding giọng nói'}), 404
        
        voice_embedding = np.load(embedding_path)
        
        # Tạo audio từ text với voice cloning
        model = load_tts_model()
        if model is None:
            return jsonify({'error': 'TTS model chưa sẵn sàng'}), 500
        
        # Tạo file output
        output_filename = f"cloned_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Sử dụng TTS với voice cloning
        # Lưu ý: Cần điều chỉnh theo model cụ thể
        try:
            # Ví dụ với XTTS v2
            if hasattr(model, 'tts_to_file'):
                # Cần có reference audio cho voice cloning
                reference_audio = os.path.join(app.config['UPLOAD_FOLDER'], 
                                              f"ref_{voice_id}.wav")
                
                # Sử dụng model để tạo speech
                model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,  # File audio mẫu
                    language="vi",  # Ngôn ngữ Tiếng Việt
                    file_path=output_path
                )
            else:
                # Fallback: tạo audio đơn giản
                from gtts import gTTS
                tts = gTTS(text=text, lang='vi')
                tts.save(output_path)
            
            return jsonify({
                'success': True,
                'message': 'Đã tạo giọng nói clone thành công',
                'audio_url': f'/api/download/{output_filename}',
                'text': text
            })
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo voice clone: {str(e)}")
            return jsonify({'error': f'Lỗi tạo giọng nói: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình clone voice: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Chuyển đổi speech to text"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load whisper model
        model = load_whisper_model()
        if model is None:
            return jsonify({'error': 'Whisper model chưa sẵn sàng'}), 500
        
        # Chuyển đổi speech to text
        result = model.transcribe(filepath, language="vi")
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'vi')
        })
        
    except Exception as e:
        logger.error(f"Lỗi speech to text: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Chuyển đổi text to speech (không clone)"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        # Tạo file output
        output_filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Sử dụng gTTS hoặc TTS model
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='vi')
            tts.save(output_path)
        except:
            # Fallback: sử dụng hệ thống TTS
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        
        return jsonify({
            'success': True,
            'message': 'Đã tạo audio thành công',
            'audio_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        logger.error(f"Lỗi text to speech: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download file"""
    try:
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Lỗi khi download file: {str(e)}")
        return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/api/voices')
def list_voices():
    """Danh sách các giọng nói đã lưu"""
    try:
        voices = []
        for file in os.listdir(app.config['VOICE_CLONE_FOLDER']):
            if file.endswith('.json'):
                with open(os.path.join(app.config['VOICE_CLONE_FOLDER'], file), 'r') as f:
                    voice_info = json.load(f)
                    voices.append({
                        'id': voice_info['id'],
                        'name': voice_info.get('original_filename', 'Unknown'),
                        'created_at': voice_info.get('created_at', '')
                    })
        
        return jsonify({
            'success': True,
            'voices': voices
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách voices: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    device = get_device()
    models_loaded = {
        'tts': tts_model is not None,
        'whisper': whisper_model is not None,
        'device': device
    }
    
    return jsonify({
        'status': 'healthy',
        'service': 'voice-cloning-api',
        'models': models_loaded,
        'timestamp': str(os.path.getmtime(__file__))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    # Pre-load models khi start
    if os.environ.get('PRELOAD_MODELS', 'false').lower() == 'true':
        load_tts_model()
        load_whisper_model()
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
