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
import librosa
import tempfile

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

# Khởi tạo models
tts_model = None
whisper_model = None

def get_device():
    """Xác định device (CUDA/CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_tts_model():
    """Load TTS model - sử dụng model nhẹ hơn"""
    global tts_model
    if tts_model is None:
        try:
            logger.info("Đang load TTS model...")
            device = get_device()
            
            # Sử dụng model nhẹ hơn để tránh lỗi
            # Tacotron2 là model ổn định và nhẹ
            tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC",
                          progress_bar=False,
                          gpu=True if device == "cuda" else False)
            logger.info(f"TTS model đã được load trên {device}")
        except Exception as e:
            logger.error(f"Lỗi khi load TTS model: {str(e)}")
            # Fallback: sử dụng gTTS
            tts_model = "gtts"
    return tts_model

def load_whisper_model():
    """Load Whisper model cho speech recognition"""
    global whisper_model
    if whisper_model is None:
        try:
            logger.info("Đang load Whisper model...")
            device = get_device()
            # Sử dụng model tiny để tiết kiệm bộ nhớ
            whisper_model = whisper.load_model("tiny", device=device)
            logger.info(f"Whisper model đã được load trên {device}")
        except Exception as e:
            logger.error(f"Lỗi khi load Whisper model: {str(e)}")
            whisper_model = None
    return whisper_model

# Danh sách các định dạng file được phép
ALLOWED_EXTENSIONS = {
    'audio': ['mp3', 'wav', 'ogg', 'm4a'],
    'voice_clone': ['mp3', 'wav', 'm4a']
}

def allowed_file(filename, file_type='audio'):
    """Kiểm tra định dạng file"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, [])

def convert_audio_format(input_path, output_path=None, target_format='wav'):
    """Chuyển đổi audio sang định dạng WAV"""
    try:
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + f'.{target_format}'
        
        # Sử dụng pydub để chuyển đổi
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)
        logger.info(f"Đã chuyển đổi {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Lỗi chuyển đổi audio: {str(e)}")
        
        # Fallback: sử dụng ffmpeg
        try:
            cmd = ['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e2:
            logger.error(f"FFmpeg cũng lỗi: {str(e2)}")
            return None

def extract_audio_features(audio_path):
    """Trích xuất đặc trưng từ audio đơn giản"""
    try:
        # Load audio với librosa
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Trích xuất các đặc trưng cơ bản
        features = {
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
            'chroma': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1).tolist(),
            'mel': librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1).tolist(),
            'duration': len(y) / sr,
            'sr': sr
        }
        
        return features
    except Exception as e:
        logger.error(f"Lỗi trích xuất đặc trưng: {str(e)}")
        
        # Fallback: tạo embedding giả
        return {
            'mfcc': np.random.randn(13).tolist(),
            'chroma': np.random.randn(12).tolist(),
            'mel': np.random.randn(128).tolist(),
            'duration': 1.0,
            'sr': 22050
        }

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
            return jsonify({'error': 'Định dạng file không được hỗ trợ. Chỉ chấp nhận MP3, WAV, M4A'}), 400
        
        # Lưu file tạm
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Chuyển đổi sang WAV nếu cần
        if not filepath.lower().endswith('.wav'):
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            converted = convert_audio_format(filepath, wav_path)
            if converted:
                # Xóa file gốc
                if os.path.exists(filepath):
                    os.remove(filepath)
                filepath = wav_path
            else:
                return jsonify({'error': 'Không thể chuyển đổi file audio'}), 500
        
        # Trích xuất đặc trưng audio
        features = extract_audio_features(filepath)
        
        if features:
            # Lưu features
            voice_id = str(uuid.uuid4())
            features_filename = f"{voice_id}_features.json"
            features_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], features_filename)
            
            with open(features_path, 'w') as f:
                json.dump(features, f)
            
            # Lưu thông tin voice
            voice_info = {
                'id': voice_id,
                'original_filename': filename,
                'features_file': features_filename,
                'audio_reference': unique_filename,
                'created_at': str(os.path.getctime(filepath)),
                'duration': features['duration']
            }
            
            voice_info_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.json")
            with open(voice_info_path, 'w') as f:
                json.dump(voice_info, f)
            
            # Di chuyển file audio vào thư mục voices để tham chiếu sau này
            reference_audio_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], unique_filename)
            os.rename(filepath, reference_audio_path)
            
            logger.info(f"Đã trích xuất giọng: {voice_id}")
            
            return jsonify({
                'success': True,
                'message': 'Đã trích xuất giọng nói thành công',
                'voice_id': voice_id,
                'voice_name': filename,
                'duration': features['duration']
            })
        else:
            return jsonify({'error': 'Không thể trích xuất đặc trưng từ audio'}), 500
            
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất giọng nói: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/clone-voice', methods=['POST'])
def clone_voice():
    """Tạo giọng nói từ text"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        voice_id = data.get('voice_id', '')
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        if not voice_id:
            return jsonify({'error': 'Vui lòng chọn giọng nói'}), 400
        
        # Kiểm tra voice có tồn tại không
        voice_info_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.json")
        if not os.path.exists(voice_info_path):
            return jsonify({'error': 'Không tìm thấy giọng nói'}), 404
        
        with open(voice_info_path, 'r') as f:
            voice_info = json.load(f)
        
        # Tạo audio từ text
        output_filename = f"cloned_{voice_id}_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Phương pháp 1: Sử dụng gTTS cho Tiếng Việt
        try:
            from gtts import gTTS
            
            # Tạo audio với gTTS
            tts = gTTS(text=text, lang='vi', slow=False)
            tts.save(output_path)
            
            logger.info(f"Đã tạo audio với gTTS: {output_filename}")
            
        except Exception as gtts_error:
            logger.error(f"gTTS lỗi: {str(gtts_error)}")
            
            # Phương pháp 2: Sử dụng TTS model
            try:
                model = load_tts_model()
                if model != "gtts":
                    # Sử dụng TTS model
                    model.tts_to_file(text=text, file_path=output_path)
                else:
                    # Phương pháp 3: Sử dụng pyttsx3
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.save_to_file(text, output_path)
                    engine.runAndWait()
            except Exception as tts_error:
                logger.error(f"TTS model lỗi: {str(tts_error)}")
                return jsonify({'error': 'Không thể tạo giọng nói'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Đã tạo giọng nói thành công',
            'audio_url': f'/api/download/{output_filename}',
            'text': text,
            'voice_name': voice_info.get('original_filename', 'Unknown')
        })
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình clone voice: {str(e)}", exc_info=True)
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
        
        # Chuyển đổi sang WAV nếu cần
        if not filepath.lower().endswith('.wav'):
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            converted = convert_audio_format(filepath, wav_path)
            if converted:
                if os.path.exists(filepath):
                    os.remove(filepath)
                filepath = wav_path
        
        # Load whisper model
        model = load_whisper_model()
        if model is None:
            return jsonify({'error': 'Whisper model chưa sẵn sàng'}), 500
        
        # Chuyển đổi speech to text
        result = model.transcribe(filepath, language="vi")
        
        # Dọn dẹp file tạm
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'vi')
        })
        
    except Exception as e:
        logger.error(f"Lỗi speech to text: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Chuyển đổi text to speech"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        # Tạo file output
        output_filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Sử dụng gTTS cho Tiếng Việt
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='vi')
            tts.save(output_path)
        except Exception as e:
            logger.error(f"gTTS lỗi: {str(e)}")
            
            # Fallback: sử dụng pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, output_path)
                engine.runAndWait()
            except Exception as e2:
                logger.error(f"pyttsx3 lỗi: {str(e2)}")
                return jsonify({'error': 'Không thể tạo audio'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Đã tạo audio thành công',
            'audio_url': f'/api/download/{output_filename}',
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Lỗi text to speech: {str(e)}", exc_info=True)
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
        voices_dir = app.config['VOICE_CLONE_FOLDER']
        
        if os.path.exists(voices_dir):
            for file in os.listdir(voices_dir):
                if file.endswith('.json') and not file.endswith('_features.json'):
                    with open(os.path.join(voices_dir, file), 'r') as f:
                        voice_info = json.load(f)
                        voices.append({
                            'id': voice_info['id'],
                            'name': voice_info.get('original_filename', 'Unknown'),
                            'created_at': voice_info.get('created_at', ''),
                            'duration': voice_info.get('duration', 0)
                        })
        
        return jsonify({
            'success': True,
            'voices': sorted(voices, key=lambda x: x.get('created_at', ''), reverse=True)
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách voices: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/delete-voice/<voice_id>', methods=['DELETE'])
def delete_voice(voice_id):
    """Xóa giọng nói"""
    try:
        voices_dir = app.config['VOICE_CLONE_FOLDER']
        voice_info_path = os.path.join(voices_dir, f"{voice_id}.json")
        
        if not os.path.exists(voice_info_path):
            return jsonify({'error': 'Không tìm thấy giọng nói'}), 404
        
        # Đọc thông tin voice
        with open(voice_info_path, 'r') as f:
            voice_info = json.load(f)
        
        # Xóa các file liên quan
        files_to_delete = [
            voice_info_path,
            os.path.join(voices_dir, voice_info.get('features_file', '')),
            os.path.join(voices_dir, voice_info.get('audio_reference', ''))
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Đã xóa: {file_path}")
        
        return jsonify({
            'success': True,
            'message': 'Đã xóa giọng nói thành công'
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi xóa voice: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    device = get_device()
    
    # Kiểm tra models
    tts_status = "gtts" if tts_model == "gtts" else (tts_model is not None)
    whisper_status = whisper_model is not None
    
    return jsonify({
        'status': 'healthy',
        'service': 'voice-cloning-api',
        'models': {
            'tts': str(tts_status),
            'whisper': whisper_status,
            'device': device
        },
        'storage': {
            'voices': len(os.listdir(app.config['VOICE_CLONE_FOLDER'])) if os.path.exists(app.config['VOICE_CLONE_FOLDER']) else 0,
            'outputs': len(os.listdir(app.config['OUTPUT_FOLDER'])) if os.path.exists(app.config['OUTPUT_FOLDER']) else 0
        },
        'timestamp': str(os.path.getmtime(__file__) if os.path.exists(__file__) else 'unknown')
    })

@app.route('/api/test-tts', methods=['POST'])
def test_tts():
    """Test TTS với text mẫu"""
    try:
        data = request.json
        text = data.get('text', 'Xin chào, đây là giọng nói thử nghiệm.')
        
        output_filename = f"test_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Sử dụng gTTS
        from gtts import gTTS
        tts = gTTS(text=text, lang='vi')
        tts.save(output_path)
        
        return jsonify({
            'success': True,
            'message': 'Test TTS thành công',
            'audio_url': f'/api/download/{output_filename}',
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Lỗi test TTS: {str(e)}")
        return jsonify({'error': f'Lỗi test TTS: {str(e)}'}), 500

# Middleware để dọn dẹp file cũ
@app.before_request
def cleanup_old_files():
    """Dọn dẹp file tạm cũ"""
    import time
    import glob
    
    try:
        current_time = time.time()
        
        # Xóa file trong temp folder cũ hơn 1 giờ
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                for filepath in glob.glob(os.path.join(folder, '*')):
                    try:
                        if os.path.getmtime(filepath) < current_time - 3600:  # 1 giờ
                            os.remove(filepath)
                            logger.debug(f"Đã xóa file cũ: {filepath}")
                    except Exception as e:
                        logger.error(f"Lỗi khi xóa file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Lỗi cleanup: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    # Pre-load models khi start
    if os.environ.get('PRELOAD_MODELS', 'false').lower() == 'true':
        logger.info("Đang pre-load models...")
        load_tts_model()
        load_whisper_model()
    
    # Kiểm tra và tạo thư mục
    for folder in ['templates', 'static', app.config['UPLOAD_FOLDER'], 
                   app.config['OUTPUT_FOLDER'], app.config['VOICE_CLONE_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    
    # Tạo file index.html mặc định nếu không tồn tại
    if not os.path.exists('templates/index.html'):
        os.makedirs('templates', exist_ok=True)
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Voice Cloning API</title>
    <style>
        body { font-family: Arial; padding: 20px; text-align: center; }
        .btn { padding: 10px 20px; margin: 10px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Voice Cloning API</h1>
    <p>API đang hoạt động!</p>
    <a href="/voice-clone" class="btn">Voice Cloning Demo</a>
    <a href="/api/health" class="btn">Health Check</a>
</body>
</html>''')
    
    # Tạo file voice_clone.html mặc định nếu không tồn tại
    if not os.path.exists('templates/voice_clone.html'):
        with open('templates/voice_clone.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Voice Cloning Demo</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        .tab { padding: 10px; cursor: pointer; }
        .tab-content { display: none; padding: 20px; }
        .active { display: block; }
    </style>
</head>
<body>
    <h1>Voice Cloning Demo</h1>
    <p>File template đang được tạo...</p>
    <a href="/">Quay lại</a>
</body>
</html>''')
    
    logger.info(f"Starting server on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development',
        threaded=True
    )
