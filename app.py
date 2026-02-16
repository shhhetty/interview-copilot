from gevent import monkey
monkey.patch_all()

import os
import threading
import queue
import json
import logging
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from openai import OpenAI
from OpenSSL import crypto 
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedOK

# Load API Keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'interview_helper_secret'

# Threading mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Global Variables
transcript_history = [] 
dg_service = None
user_context = ""
current_mode = "manual"

# Locking Logic
ai_response_locked = False

def get_ai_decision_and_answer(context, transcript_list, force=False):
    try:
        formatted_transcript = ""
        for entry in transcript_list[-15:]:
            formatted_transcript += f"[Speaker {entry['speaker']}]: {entry['text']}\n"

        if force:
            system_prompt = (
                "You are an expert interview assistant. The user has explicitly asked for an answer.\n"
                "1. Ignore who spoke last. Look for the last QUESTION asked.\n"
                "2. Provide a direct, high-impact answer (max 60 words)."
                f"\n\nCANDIDATE CONTEXT:\n{context}"
            )
        else:
            system_prompt = (
                "You are a stealth interview assistant. Analyze the LIVE TRANSCRIPT.\n"
                "1. Focus ONLY on what [Speaker 0] (The Interviewer) just said.\n"
                "2. Did [Speaker 0] ask a question?\n"
                "   - YES: Provide a concise answer (max 60 words).\n"
                "   - NO: Output exactly: [NO_ANSWER]\n"
                "3. Ignore [Speaker 1] (The Candidate)."
                f"\n\nCANDIDATE CONTEXT:\n{context}"
            )

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"LIVE TRANSCRIPT:\n{formatted_transcript}"}
            ],
            temperature=0.6,
        )
        
        content = response.choices[0].message.content.strip()
        
        if not force and "[NO_ANSWER]" in content:
            return None
            
        return content.replace("[ANSWER]", "").strip()

    except Exception as e:
        print(f"AI Error: {e}")
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
        
        # URL: interim_results=true (Fixes visibility issue)
        url = (
            "wss://api.deepgram.com/v1/listen?"
            "model=nova-2"
            "&smart_format=true"
            "&diarize=true"
            "&filler_words=false"
            "&punctuate=true"
            "&interim_results=true" 
        )
        
        headers = { "Authorization": f"Token {DEEPGRAM_API_KEY}" }

        try:
            with connect(url, additional_headers=headers) as websocket:
                self.ws = websocket
                print(">>> DEEPGRAM CONNECTED <<<")

                self.recv_thread = threading.Thread(target=self._recv_loop)
                self.recv_thread.start()

                while self.running:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        websocket.send(data)
                    except queue.Empty:
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
                
                if 'channel' in data:
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

        except Exception as e:
            if self.running:
                print(f"Receive Error: {e}") # Print error instead of silent fail

    def _trigger_auto_ai(self):
        global user_context, ai_response_locked
        
        if not user_context: return

        print("Auto-Pilot: Analyzing...")
        answer = get_ai_decision_and_answer(user_context, transcript_history, force=False)
        
        if answer:
            print("Auto-Pilot: Answer Generated! LOCKING AI.")
            socketio.emit('ai_response', {'answer': answer})
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
    answer = get_ai_decision_and_answer(user_context, transcript_history, force=True)
    if answer:
        emit('ai_response', {'answer': answer})
        emit('status_update', {'status': 'Ready'})

@socketio.on('force_answer')
def handle_force_answer(data):
    global ai_response_locked
    handle_manual_answer(data)
    ai_response_locked = True 

# --- SSL CERTIFICATE GENERATOR ---
def generate_self_signed_cert():
    if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
        print("Generating self-signed certificate...")
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "State"
        cert.get_subject().L = "City"
        cert.get_subject().O = "Organization"
        cert.get_subject().OU = "Organizational Unit"
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10*365*24*60*60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        with open("cert.pem", "wt") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
        with open("key.pem", "wt") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Server starting on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)