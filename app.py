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
import requests
from typing import Optional, Dict, List

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cấu hình
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['VOICE_CLONE_FOLDER'] = 'voices'
app.config['MODEL_FOLDER'] = 'models'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Tạo thư mục
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               app.config['VOICE_CLONE_FOLDER'], app.config['MODEL_FOLDER'],
               'static/css', 'static/js', 'templates']:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Khởi tạo models
tts_models: Dict[str, TTS] = {}
whisper_model = None

def get_device():
    """Xác định device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_tts_model(model_name: str = "tts_models/vi/vivos/vits"):
    """Load TTS model cụ thể"""
    try:
        if model_name in tts_models:
            return tts_models[model_name]
        
        logger.info(f"Đang load TTS model: {model_name}")
        device = get_device()
        
        # Load model với các tham số tối ưu
        tts_model = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=True if device == "cuda" else False
        )
        
        tts_models[model_name] = tts_model
        logger.info(f"Model {model_name} đã được load trên {device}")
        return tts_model
        
    except Exception as e:
        logger.error(f"Lỗi khi load model {model_name}: {str(e)}")
        return None

def load_all_tts_models():
    """Load tất cả các model TTS có sẵn"""
    models_to_load = [
        # Model Tiếng Việt
        "tts_models/vi/vivos/vits",  # Giọng Tiếng Việt chất lượng cao
        "tts_models/vi/viettts/female",  # Giọng nữ Tiếng Việt
        "tts_models/vi/viettts/male",    # Giọng nam Tiếng Việt
        
        # Model đa ngôn ngữ với voice cloning
        "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS v2 - voice cloning tốt
        
        # Model Tiếng Anh (có thể đọc Tiếng Việt với accent)
        "tts_models/en/ljspeech/tacotron2-DDC",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/en/ljspeech/speedy-speech",
    ]
    
    loaded_models = {}
    for model_name in models_to_load:
        try:
            model = load_tts_model(model_name)
            if model:
                loaded_models[model_name] = model
        except Exception as e:
            logger.warning(f"Không thể load model {model_name}: {e}")
    
    return loaded_models

def load_whisper_model():
    """Load Whisper model"""
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

# Danh sách các định dạng file
ALLOWED_EXTENSIONS = {
    'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac'],
    'voice_clone': ['mp3', 'wav', 'm4a', 'ogg']
}

# Danh sách các giọng có sẵn
AVAILABLE_VOICES = {
    'vi_female_vivos': {
        'id': 'vi_female_vivos',
        'name': 'Giọng Nữ Việt Nam (VIVOS)',
        'model': 'tts_models/vi/vivos/vits',
        'language': 'vi',
        'gender': 'female',
        'description': 'Giọng nữ Tiếng Việt tự nhiên, chất lượng cao'
    },
    'vi_female_viettts': {
        'id': 'vi_female_viettts',
        'name': 'Giọng Nữ Việt Nam (VietTTS)',
        'model': 'tts_models/vi/viettts/female',
        'language': 'vi',
        'gender': 'female',
        'description': 'Giọng nữ Tiếng Việt từ VietTTS'
    },
    'vi_male_viettts': {
        'id': 'vi_male_viettts',
        'name': 'Giọng Nam Việt Nam (VietTTS)',
        'model': 'tts_models/vi/viettts/male',
        'language': 'vi',
        'gender': 'male',
        'description': 'Giọng nam Tiếng Việt từ VietTTS'
    },
    'xtts_v2_vietnamese': {
        'id': 'xtts_v2_vietnamese',
        'name': 'XTTS v2 - Tiếng Việt',
        'model': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'language': 'vi',
        'gender': 'neutral',
        'description': 'Model đa ngôn ngữ với hỗ trợ Tiếng Việt, có thể clone giọng'
    },
    'en_female_tacotron': {
        'id': 'en_female_tacotron',
        'name': 'Giọng Nữ Anh Mỹ',
        'model': 'tts_models/en/ljspeech/tacotron2-DDC',
        'language': 'en',
        'gender': 'female',
        'description': 'Giọng nữ Tiếng Anh Mỹ'
    },
    'en_male_glow': {
        'id': 'en_male_glow',
        'name': 'Giọng Nam Anh Mỹ',
        'model': 'tts_models/en/ljspeech/glow-tts',
        'language': 'en',
        'gender': 'male',
        'description': 'Giọng nam Tiếng Anh Mỹ'
    }
}

def allowed_file(filename, file_type='audio'):
    """Kiểm tra định dạng file"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, [])

def convert_audio_format(input_path, output_path=None, target_format='wav'):
    """Chuyển đổi audio format"""
    try:
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + f'.{target_format}'
        
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)
        return output_path
    except Exception as e:
        logger.error(f"Lỗi chuyển đổi audio: {str(e)}")
        return None

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/voice-clone')
def voice_clone_page():
    """Trang voice cloning"""
    return render_template('voice_clone.html')

@app.route('/api/available-voices')
def get_available_voices():
    """Lấy danh sách các giọng có sẵn"""
    return jsonify({
        'success': True,
        'voices': list(AVAILABLE_VOICES.values()),
        'total': len(AVAILABLE_VOICES)
    })

@app.route('/api/extract-voice', methods=['POST'])
def extract_voice():
    """Trích xuất giọng nói cho voice cloning (chỉ XTTS v2)"""
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
        
        # Chuyển đổi sang WAV
        wav_path = filepath.rsplit('.', 1)[0] + '.wav'
        if not convert_audio_format(filepath, wav_path):
            return jsonify({'error': 'Không thể chuyển đổi audio'}), 500
        
        # Load XTTS v2 model
        xtts_model = load_tts_model("tts_models/multilingual/multi-dataset/xtts_v2")
        if not xtts_model:
            return jsonify({'error': 'Không thể load XTTS v2 model'}), 500
        
        # Tạo voice ID
        voice_id = str(uuid.uuid4())
        
        # Lưu audio reference
        reference_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.wav")
        os.rename(wav_path, reference_path)
        
        # Lưu thông tin voice
        voice_info = {
            'id': voice_id,
            'name': filename.rsplit('.', 1)[0],
            'filename': f"{voice_id}.wav",
            'model': 'xtts_v2',
            'created_at': str(os.path.getctime(reference_path)),
            'language': 'vi'
        }
        
        with open(os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.json"), 'w') as f:
            json.dump(voice_info, f)
        
        return jsonify({
            'success': True,
            'message': 'Đã trích xuất giọng nói thành công',
            'voice_id': voice_id,
            'voice_name': voice_info['name']
        })
        
    except Exception as e:
        logger.error(f"Lỗi extract voice: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Tạo giọng nói từ văn bản với nhiều giọng khác nhau"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        voice_id = data.get('voice_id', 'vi_female_vivos')  # Mặc định giọng nữ VIVOS
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        # Kiểm tra voice_id
        if voice_id in AVAILABLE_VOICES:
            voice_config = AVAILABLE_VOICES[voice_id]
        else:
            # Kiểm tra xem có phải cloned voice không
            voice_json_path = os.path.join(app.config['VOICE_CLONE_FOLDER'], f"{voice_id}.json")
            if os.path.exists(voice_json_path):
                with open(voice_json_path, 'r') as f:
                    voice_config = json.load(f)
                voice_config['is_cloned'] = True
            else:
                voice_config = AVAILABLE_VOICES['vi_female_vivos']  # Fallback
        
        # Tạo file output
        output_filename = f"tts_{voice_id}_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        if voice_config.get('is_cloned'):
            # Sử dụng cloned voice với XTTS v2
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            model = load_tts_model(model_name)
            
            if model:
                # Sử dụng cloned voice
                reference_audio = os.path.join(app.config['VOICE_CLONE_FOLDER'], voice_config['filename'])
                model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,
                    language="vi",
                    file_path=output_path
                )
            else:
                return jsonify({'error': 'Không thể load model voice cloning'}), 500
        else:
            # Sử dụng pre-trained model
            model_name = voice_config['model']
            model = load_tts_model(model_name)
            
            if model:
                # Xác định language code
                lang_code = voice_config.get('language', 'vi')
                
                if 'xtts' in model_name:
                    # XTTS model cần speaker_wav
                    # Sử dụng speaker mặc định
                    model.tts_to_file(
                        text=text,
                        speaker_wav=None,  # Sử dụng giọng mặc định của model
                        language=lang_code,
                        file_path=output_path
                    )
                else:
                    # Các model khác
                    model.tts_to_file(text=text, file_path=output_path)
            else:
                # Fallback: sử dụng gTTS
                logger.warning(f"Model {model_name} không khả dụng, sử dụng gTTS")
                from gtts import gTTS
                tts = gTTS(text=text, lang='vi')
                tts.save(output_path)
        
        # Convert to MP3 nếu cần
        mp3_path = output_path.rsplit('.', 1)[0] + '.mp3'
        if convert_audio_format(output_path, mp3_path, 'mp3'):
            os.remove(output_path)  # Xóa file WAV gốc
            output_path = mp3_path
            output_filename = os.path.basename(mp3_path)
        
        return jsonify({
            'success': True,
            'message': 'Đã tạo giọng nói thành công',
            'audio_url': f'/api/download/{output_filename}',
            'voice_name': voice_config.get('name', 'Unknown'),
            'text': text,
            'voice_id': voice_id
        })
        
    except Exception as e:
        logger.error(f"Lỗi TTS: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi tạo giọng nói: {str(e)}'}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Chuyển speech to text"""
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
        
        # Chuyển đổi sang WAV
        wav_path = filepath.rsplit('.', 1)[0] + '.wav'
        if not convert_audio_format(filepath, wav_path):
            return jsonify({'error': 'Không thể chuyển đổi audio'}), 500
        
        # Load whisper model
        model = load_whisper_model()
        if not model:
            return jsonify({'error': 'Whisper model không khả dụng'}), 500
        
        # Transcribe
        result = model.transcribe(wav_path, language="vi")
        
        # Dọn dẹp
        for f in [filepath, wav_path]:
            if os.path.exists(f):
                os.remove(f)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'vi')
        })
        
    except Exception as e:
        logger.error(f"Lỗi STT: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi chuyển đổi: {str(e)}'}), 500

@app.route('/api/voices')
def list_voices():
    """Danh sách các giọng (bao gồm cloned)"""
    try:
        voices = []
        
        # Thêm các pre-trained voices
        for voice_id, voice_info in AVAILABLE_VOICES.items():
            voices.append({
                'id': voice_id,
                'name': voice_info['name'],
                'type': 'pre-trained',
                'language': voice_info.get('language', 'vi'),
                'gender': voice_info.get('gender', 'unknown'),
                'description': voice_info.get('description', '')
            })
        
        # Thêm cloned voices
        voices_dir = app.config['VOICE_CLONE_FOLDER']
        if os.path.exists(voices_dir):
            for file in os.listdir(voices_dir):
                if file.endswith('.json'):
                    with open(os.path.join(voices_dir, file), 'r') as f:
                        voice_info = json.load(f)
                        voices.append({
                            'id': voice_info['id'],
                            'name': f"Cloned: {voice_info['name']}",
                            'type': 'cloned',
                            'language': voice_info.get('language', 'vi'),
                            'gender': 'custom',
                            'description': 'Giọng đã clone từ audio mẫu'
                        })
        
        return jsonify({
            'success': True,
            'voices': voices,
            'total': len(voices)
        })
        
    except Exception as e:
        logger.error(f"Lỗi list voices: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/api/test-voice/<voice_id>', methods=['GET'])
def test_voice(voice_id):
    """Test giọng nói với văn bản mẫu"""
    test_text = "Xin chào! Tôi là giọng nói nhân tạo. Rất vui được gặp bạn."
    
    try:
        if voice_id in AVAILABLE_VOICES:
            voice_config = AVAILABLE_VOICES[voice_id]
        else:
            return jsonify({'error': 'Giọng không tồn tại'}), 404
        
        output_filename = f"test_{voice_id}_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        model_name = voice_config['model']
        model = load_tts_model(model_name)
        
        if model:
            if 'xtts' in model_name:
                model.tts_to_file(
                    text=test_text,
                    speaker_wav=None,
                    language=voice_config.get('language', 'vi'),
                    file_path=output_path
                )
            else:
                model.tts_to_file(text=test_text, file_path=output_path)
        else:
            from gtts import gTTS
            tts = gTTS(text=test_text, lang='vi')
            tts.save(output_path)
        
        return jsonify({
            'success': True,
            'audio_url': f'/api/download/{output_filename}',
            'voice_name': voice_config['name'],
            'text': test_text
        })
        
    except Exception as e:
        logger.error(f"Lỗi test voice: {str(e)}")
        return jsonify({'error': f'Lỗi test: {str(e)}'}), 500

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
        logger.error(f"Lỗi download: {str(e)}")
        return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/api/health')
def health_check():
    """Health check"""
    device = get_device()
    
    # Kiểm tra models
    models_loaded = {
        'whisper': whisper_model is not None,
        'tts_models_loaded': len(tts_models),
        'device': device
    }
    
    # Danh sách models đã load
    loaded_model_names = list(tts_models.keys())
    
    return jsonify({
        'status': 'healthy',
        'service': 'multi-voice-tts-api',
        'models': models_loaded,
        'loaded_models': loaded_model_names,
        'available_voices': len(AVAILABLE_VOICES),
        'timestamp': str(os.path.getmtime(__file__))
    })

@app.route('/api/preload-models', methods=['POST'])
def preload_models():
    """Preload tất cả models"""
    try:
        loaded = load_all_tts_models()
        load_whisper_model()
        
        return jsonify({
            'success': True,
            'message': f'Đã load {len(loaded)} models',
            'models_loaded': list(loaded.keys())
        })
    except Exception as e:
        logger.error(f"Lỗi preload: {str(e)}")
        return jsonify({'error': f'Lỗi preload: {str(e)}'}), 500

# Cleanup old files
@app.before_request
def cleanup():
    import time
    import glob
    
    try:
        current_time = time.time()
        max_age = 3600  # 1 giờ
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                for filepath in glob.glob(os.path.join(folder, '*')):
                    try:
                        if os.path.getmtime(filepath) < current_time - max_age:
                            os.remove(filepath)
                    except:
                        pass
    except:
        pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    # Preload models nếu được cấu hình
    if os.environ.get('PRELOAD_MODELS', 'false').lower() == 'true':
        logger.info("Preloading models...")
        load_all_tts_models()
        load_whisper_model()
    
    # Đảm bảo có templates
    if not os.path.exists('templates'):
        os.makedirs('templates', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development',
        threaded=True
    )
