import os
import threading
import queue
import json
import base64
import logging
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from openai import OpenAI
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedOK

# Load API Keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'interview_helper_secret'

# threading mode works locally AND under gunicorn's GeventWebSocketWorker
# (gunicorn monkey-patches threading → green threads automatically)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Global Variables
transcript_history = [] 
dg_service = None
user_context = ""
current_mode = "manual"

# Locking Logic
ai_response_locked = False

CODE_INSTRUCTION = (
    "\nWhen providing code, ALWAYS wrap it in triple backticks with the language name, "
    "e.g. ```python\\ncode\\n```. Keep explanations outside code blocks concise."
)

def stream_ai_answer(context, transcript_list, force=False, sid=None):
    """Stream GPT-4 response token-by-token via Socket.IO."""
    try:
        formatted_transcript = ""
        for entry in transcript_list[-15:]:
            formatted_transcript += f"[Speaker {entry['speaker']}]: {entry['text']}\n"

        if force:
            system_prompt = (
                "You are an expert interview assistant. The user has explicitly asked for an answer.\n"
                "1. Ignore who spoke last. Look for the last QUESTION asked.\n"
                "2. Provide a direct, high-impact answer (max 60 words for non-code, full solution for code questions)."
                + CODE_INSTRUCTION
                + f"\n\nCANDIDATE CONTEXT:\n{context}"
            )
        else:
            system_prompt = (
                "You are a stealth interview assistant. Analyze the LIVE TRANSCRIPT.\n"
                "1. Focus ONLY on what [Speaker 0] (The Interviewer) just said.\n"
                "2. Did [Speaker 0] ask a question?\n"
                "   - YES: Provide a concise answer (max 60 words for non-code, full solution for code questions).\n"
                "   - NO: Output exactly: [NO_ANSWER]\n"
                "3. Ignore [Speaker 1] (The Candidate)."
                + CODE_INSTRUCTION
                + f"\n\nCANDIDATE CONTEXT:\n{context}"
            )

        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"LIVE TRANSCRIPT:\n{formatted_transcript}"}
            ],
            temperature=0.6,
            stream=True,
        )
        
        full_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            token = delta.content if delta.content else ""
            if token:
                full_content += token
                socketio.emit('ai_token', {'token': token}, to=sid)
        
        # Signal stream end
        socketio.emit('ai_stream_end', {'full': full_content}, to=sid)
        
        if not force and "[NO_ANSWER]" in full_content:
            return None
            
        return full_content.replace("[ANSWER]", "").strip()

    except Exception as e:
        print(f"AI Error: {e}")
        socketio.emit('ai_stream_end', {'full': f'Error: {e}'}, to=sid)
        return None

def stream_image_answer(images_b64, context, sid=None):
    """Process one or more images via GPT-4 Vision and stream the response."""
    try:
        num_images = len(images_b64) if isinstance(images_b64, list) else 1
        if not isinstance(images_b64, list):
            images_b64 = [images_b64]

        system_prompt = (
            "You are an expert interview assistant. The user has shared screenshot(s) of a coding question.\n"
            f"There are {num_images} image(s) — they may be parts of the SAME question. Combine them.\n"
            "1. Extract the full question from ALL images.\n"
            "2. Provide a clear, complete solution.\n"
            "3. Use triple backtick code blocks with the language name for ALL code."
            + f"\n\nCANDIDATE CONTEXT:\n{context}"
        )

        # Build content array with all images
        content = [{"type": "text", "text": "Extract the coding question from these image(s) and provide a complete solution with code."}]
        for img_b64 in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.4,
            stream=True,
        )

        full_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            token = delta.content if delta.content else ""
            if token:
                full_content += token
                socketio.emit('ai_token', {'token': token}, to=sid)

        socketio.emit('ai_stream_end', {'full': full_content}, to=sid)
        return full_content

    except Exception as e:
        print(f"Vision AI Error: {e}")
        socketio.emit('ai_stream_end', {'full': f'Error: {e}'}, to=sid)
        return None

# --- DIRECT WEBSOCKET SERVICE ---
class DeepgramDirectService:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.send_thread = None
        self.recv_thread = None
        self.ws = None

    def start(self):
        if self.running: return
        self.running = True
        self.send_thread = threading.Thread(target=self._run_connection)
        self.send_thread.start()

    def _run_connection(self):
        print("Initializing Deepgram...")
        
        url = (
            "wss://api.deepgram.com/v1/listen?"
            "model=nova-2"
            "&smart_format=true"
            "&diarize=true"
            "&filler_words=false"
            "&punctuate=true"
            "&interim_results=true"
            "&endpointing=200"
            "&utterance_end_ms=1000"
        )
        
        headers = { "Authorization": f"Token {DEEPGRAM_API_KEY}" }

        try:
            with connect(url, additional_headers=headers) as websocket:
                self.ws = websocket
                print(">>> DEEPGRAM CONNECTED <<<")

                self.recv_thread = threading.Thread(target=self._recv_loop)
                self.recv_thread.start()

                last_keepalive = time.time()
                while self.running:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        websocket.send(data)
                    except queue.Empty:
                        # Send keepalive to prevent Deepgram timeout
                        if time.time() - last_keepalive > 8:
                            try:
                                websocket.send(json.dumps({"type": "KeepAlive"}))
                                last_keepalive = time.time()
                            except Exception:
                                break
                        continue
                    except ConnectionClosedOK:
                        break 
                    except Exception as e:
                        if "1000" in str(e): break 
                        print(f"Send Error: {e}")
                        break
                
                print("Closing connection...")
        
        except Exception as e:
            print(f"Connection Error: {e}")
        finally:
            self.running = False

    def _recv_loop(self):
        global current_mode, ai_response_locked
        try:
            while self.running:
                message = self.ws.recv()
                data = json.loads(message)
                
                if 'channel' in data and isinstance(data['channel'], dict):
                    alternatives = data['channel']['alternatives']
                    if alternatives:
                        alt = alternatives[0]
                        sentence = alt['transcript']
                        
                        if len(sentence) > 0:
                            # Extract Speaker (Safe extraction)
                            speaker = 0
                            if 'words' in alt and len(alt['words']) > 0:
                                speaker = alt['words'][0].get('speaker', 0)

                            is_final = data.get('is_final', False)

                            # 1. ALWAYS emit to UI (so you see it)
                            socketio.emit('transcript_update', {
                                'speaker': speaker, 
                                'text': sentence,
                                'is_final': is_final
                            })

                            # 2. Process Logic ONLY on Final Sentences
                            if is_final:
                                print(f"[{current_mode.upper()}] S{speaker}: {sentence}")
                                
                                # Store history
                                transcript_history.append({'speaker': speaker, 'text': sentence})

                                # Unlock if Interviewer (S0) speaks
                                if speaker == 0 and ai_response_locked:
                                    print(">> Interviewer speaking. Unlocking AI.")
                                    ai_response_locked = False
                                    socketio.emit('status_update', {'status': 'Listening...'})

                                # Trigger Auto-Pilot
                                if current_mode == 'auto':
                                    if speaker == 0 and not ai_response_locked:
                                        threading.Thread(target=self._trigger_auto_ai).start()

        except ConnectionClosedOK:
            pass  # Normal close, not an error
        except Exception as e:
            if self.running and "1000" not in str(e):
                print(f"Receive Error: {e}")

    def _trigger_auto_ai(self):
        global user_context, ai_response_locked
        
        if not user_context: return

        print("Auto-Pilot: Analyzing...")
        answer = stream_ai_answer(user_context, transcript_history, force=False)
        
        if answer:
            print("Auto-Pilot: Answer Generated! LOCKING AI.")
            ai_response_locked = True
            socketio.emit('status_update', {'status': 'Answer Locked'})

    def add_audio(self, data):
        self.audio_queue.put(data)

    def stop(self):
        self.running = False
        if self.ws:
            try: self.ws.close()
            except: pass
        if self.send_thread: self.send_thread.join()
        if self.recv_thread: self.recv_thread.join()

dg_service = DeepgramDirectService()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('mobile.html')

@socketio.on('connect')
def handle_connect():
    print(f"Device connected: {request.sid}")

@socketio.on('set_mode')
def handle_mode_set(data):
    global current_mode, ai_response_locked
    current_mode = data.get('mode', 'manual')
    ai_response_locked = False 
    print(f"Mode switched to: {current_mode}")

@socketio.on('update_context')
def handle_context_update(data):
    global user_context
    user_context = data.get('context', '')

@socketio.on('start_recording')
def start_recording():
    dg_service.start()
    emit('status_update', {'status': 'Listening...'})

@socketio.on('stop_recording')
def stop_recording():
    dg_service.stop()
    emit('status_update', {'status': 'Stopped'})

@socketio.on('binary_audio')
def handle_audio(data):
    dg_service.add_audio(data)

@socketio.on('manual_get_answer')
def handle_manual_answer(data):
    global user_context
    print(">>> MANUAL ANSWER REQUESTED <<<")
    emit('status_update', {'status': 'Generating Answer...'})
    sid = request.sid
    # Include interim (not-yet-finalized) transcript if user clicked fast
    interim = data.get('interim', '')
    transcript_with_interim = list(transcript_history)
    if interim:
        # Parse interim format "[Speaker N]: text" into a dict matching transcript_history format
        import re
        m = re.match(r'\[Speaker (\d+)\]: (.+)', interim)
        if m:
            transcript_with_interim.append({'speaker': int(m.group(1)), 'text': m.group(2)})
        else:
            # Fallback: treat as speaker 0 text
            transcript_with_interim.append({'speaker': 0, 'text': interim})
    answer = stream_ai_answer(user_context, transcript_with_interim, force=True, sid=sid)
    if answer:
        emit('status_update', {'status': 'Ready'})

@socketio.on('force_answer')
def handle_force_answer(data):
    global ai_response_locked
    handle_manual_answer(data)
    ai_response_locked = True 

@socketio.on('process_image')
def handle_image_process(data):
    global user_context
    sid = request.sid
    # Support both single image and multiple images
    images = data.get('images', [])
    if not images:
        single = data.get('image', '')
        if single:
            images = [single]
    if not images:
        emit('status_update', {'status': 'No image received'})
        return
    print(f">>> IMAGE PROCESSING REQUESTED ({len(images)} image(s)) <<<")
    emit('status_update', {'status': f'Analyzing {len(images)} image(s)...'})
    answer = stream_image_answer(images, user_context, sid=sid)
    if answer:
        emit('status_update', {'status': 'Ready'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Server starting on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
