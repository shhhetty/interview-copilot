import os
import re
import threading
import queue
import json
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

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Per-Session State ─────────────────────────────────────────────────────────
# Each connected client gets its own isolated session dict.
sessions = {}
sessions_lock = threading.Lock()

def get_session(sid):
    """Get session for a given sid, or None if not found."""
    with sessions_lock:
        return sessions.get(sid)

def create_session(sid):
    """Create a fresh session for a new connection."""
    session = {
        'sid': sid,
        'transcript_history': [],
        'user_context': '',
        'current_mode': 'manual',
        'ai_response_locked': False,
        'dg_service': None,
        'auto_gen_id': 0,  # generation counter to prevent stale auto-answers
    }
    with sessions_lock:
        sessions[sid] = session
    return session

def destroy_session(sid):
    """Clean up session on disconnect."""
    with sessions_lock:
        session = sessions.pop(sid, None)
    if session and session.get('dg_service'):
        try:
            session['dg_service'].stop()
        except Exception:
            pass

CODE_INSTRUCTION = (
    "\nWhen providing code, ALWAYS wrap it in triple backticks with the language name, "
    "e.g. ```python\\ncode\\n```. Keep explanations outside code blocks concise."
)

# ─── AI Streaming ──────────────────────────────────────────────────────────────

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

        socketio.emit('ai_stream_end', {'full': full_content}, to=sid)

        if not force and "[NO_ANSWER]" in full_content:
            return None

        return full_content.replace("[ANSWER]", "").strip()

    except Exception as e:
        print(f"AI Error: {e}")
        socketio.emit('ai_stream_end', {'full': f'Error: {e}'}, to=sid)
        return None


def stream_image_answer(images_b64, context, transcript_text='', sid=None):
    """Process one or more images via GPT-4 Vision and stream the response."""
    try:
        num_images = len(images_b64) if isinstance(images_b64, list) else 1
        if not isinstance(images_b64, list):
            images_b64 = [images_b64]

        # Build transcript section if available
        transcript_section = ""
        if transcript_text:
            transcript_section = (
                "\n\nLIVE CONVERSATION TRANSCRIPT (from the interview):\n"
                f"{transcript_text}\n\n"
                "IMPORTANT: Check the transcript for any instructions about this question "
                "(e.g. specific programming language, constraints, approach preferences). "
                "If the interviewer or candidate mentioned how to solve it, follow those instructions."
            )

        system_prompt = (
            "You are an expert interview assistant. The user has shared screenshot(s) of a coding question.\n"
            f"There are {num_images} image(s) — they may be parts of the SAME question. Combine them.\n"
            "1. Extract the full question from ALL images.\n"
            "2. Check the conversation transcript (if provided) for any relevant context — "
            "such as preferred programming language, specific constraints, or verbal instructions from the interviewer.\n"
            "3. Provide a clear, complete solution following any instructions found in the transcript.\n"
            "4. Use triple backtick code blocks with the language name for ALL code."
            + f"\n\nCANDIDATE CONTEXT:\n{context}"
            + transcript_section
        )

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


# ─── Deepgram Service (per-session) ────────────────────────────────────────────

class DeepgramDirectService:
    def __init__(self, sid):
        self.sid = sid
        self.audio_queue = queue.Queue()
        self.running = False
        self.send_thread = None
        self.recv_thread = None
        self.ws = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.send_thread = threading.Thread(target=self._run_connection)
        self.send_thread.start()

    def _run_connection(self):
        print(f"[{self.sid[:8]}] Initializing Deepgram...")

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

        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        try:
            with connect(url, additional_headers=headers) as websocket:
                self.ws = websocket
                print(f"[{self.sid[:8]}] >>> DEEPGRAM CONNECTED <<<")

                self.recv_thread = threading.Thread(target=self._recv_loop)
                self.recv_thread.start()

                last_keepalive = time.time()
                while self.running:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        websocket.send(data)
                    except queue.Empty:
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
                        if "1000" in str(e):
                            break
                        print(f"[{self.sid[:8]}] Send Error: {e}")
                        break

                print(f"[{self.sid[:8]}] Closing Deepgram connection...")

        except Exception as e:
            print(f"[{self.sid[:8]}] Connection Error: {e}")
        finally:
            self.running = False

    def _recv_loop(self):
        try:
            while self.running:
                message = self.ws.recv()
                data = json.loads(message)

                session = get_session(self.sid)
                if not session:
                    break  # session was destroyed

                if 'channel' in data and isinstance(data['channel'], dict):
                    alternatives = data['channel']['alternatives']
                    if alternatives:
                        alt = alternatives[0]
                        sentence = alt['transcript']

                        if len(sentence) > 0:
                            speaker = 0
                            if 'words' in alt and len(alt['words']) > 0:
                                speaker = alt['words'][0].get('speaker', 0)

                            is_final = data.get('is_final', False)

                            # Emit to THIS client only
                            socketio.emit('transcript_update', {
                                'speaker': speaker,
                                'text': sentence,
                                'is_final': is_final
                            }, to=self.sid)

                            if is_final:
                                mode = session['current_mode']
                                print(f"[{self.sid[:8]}][{mode.upper()}] S{speaker}: {sentence}")

                                session['transcript_history'].append({
                                    'speaker': speaker,
                                    'text': sentence
                                })

                                # Unlock if Interviewer (S0) speaks
                                if speaker == 0 and session['ai_response_locked']:
                                    print(f"[{self.sid[:8]}] >> Interviewer speaking. Unlocking AI.")
                                    session['ai_response_locked'] = False
                                    socketio.emit('status_update', {'status': 'Listening...'}, to=self.sid)

                                # Trigger Auto-Pilot
                                if mode == 'auto':
                                    if speaker == 0 and not session['ai_response_locked']:
                                        # Increment generation counter
                                        session['auto_gen_id'] += 1
                                        gen_id = session['auto_gen_id']
                                        threading.Thread(
                                            target=self._trigger_auto_ai,
                                            args=(gen_id,)
                                        ).start()

        except ConnectionClosedOK:
            pass
        except Exception as e:
            if self.running and "1000" not in str(e):
                print(f"[{self.sid[:8]}] Receive Error: {e}")

    def _trigger_auto_ai(self, gen_id):
        session = get_session(self.sid)
        if not session:
            return

        context = session['user_context']
        if not context:
            return

        # Check if this is still the latest auto-pilot trigger
        if session['auto_gen_id'] != gen_id:
            print(f"[{self.sid[:8]}] Auto-Pilot: Superseded by newer question (gen {gen_id} < {session['auto_gen_id']})")
            return

        print(f"[{self.sid[:8]}] Auto-Pilot: Analyzing...")
        answer = stream_ai_answer(
            context,
            session['transcript_history'],
            force=False,
            sid=self.sid
        )

        if answer:
            # Double-check we're still the latest before locking
            if session['auto_gen_id'] == gen_id:
                print(f"[{self.sid[:8]}] Auto-Pilot: Answer Generated! LOCKING AI.")
                session['ai_response_locked'] = True
                socketio.emit('status_update', {'status': 'Answer Locked'}, to=self.sid)
            else:
                print(f"[{self.sid[:8]}] Auto-Pilot: Answer discarded (superseded).")

    def add_audio(self, data):
        self.audio_queue.put(data)

    def stop(self):
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.send_thread:
            self.send_thread.join(timeout=3)
        if self.recv_thread:
            self.recv_thread.join(timeout=3)


# ─── Flask Routes & Socket.IO Handlers ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('mobile.html')


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    session = create_session(sid)
    print(f"Device connected: {sid}")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Device disconnected: {sid}")
    destroy_session(sid)


@socketio.on('set_mode')
def handle_mode_set(data):
    session = get_session(request.sid)
    if not session:
        return
    session['current_mode'] = data.get('mode', 'manual')
    session['ai_response_locked'] = False
    print(f"[{request.sid[:8]}] Mode switched to: {session['current_mode']}")


@socketio.on('update_context')
def handle_context_update(data):
    session = get_session(request.sid)
    if not session:
        return
    session['user_context'] = data.get('context', '')


@socketio.on('start_recording')
def start_recording():
    sid = request.sid
    session = get_session(sid)
    if not session:
        return

    # Stop existing service if any
    if session['dg_service']:
        session['dg_service'].stop()

    # Create a new per-session Deepgram service
    dg = DeepgramDirectService(sid)
    session['dg_service'] = dg
    dg.start()
    emit('status_update', {'status': 'Listening...'})


@socketio.on('stop_recording')
def stop_recording():
    session = get_session(request.sid)
    if not session:
        return
    if session['dg_service']:
        session['dg_service'].stop()
        session['dg_service'] = None
    emit('status_update', {'status': 'Stopped'})


@socketio.on('binary_audio')
def handle_audio(data):
    session = get_session(request.sid)
    if not session:
        return
    if session['dg_service']:
        session['dg_service'].add_audio(data)


@socketio.on('manual_get_answer')
def handle_manual_answer(data):
    sid = request.sid
    session = get_session(sid)
    if not session:
        return

    print(f"[{sid[:8]}] >>> MANUAL ANSWER REQUESTED <<<")
    emit('status_update', {'status': 'Generating Answer...'})

    # Use the editable transcript from the frontend (includes user corrections)
    frontend_transcript = data.get('transcript', '')

    if frontend_transcript:
        # Frontend sent the user-editable transcript text — use it directly
        # Parse it into the list format stream_ai_answer expects
        transcript_list = []
        for line in frontend_transcript.split('\n'):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'S(\d+):\s*(.+)', line)
            if m:
                transcript_list.append({'speaker': int(m.group(1)), 'text': m.group(2)})
            else:
                # Non-speaker line (e.g. user typed something), treat as context
                transcript_list.append({'speaker': 0, 'text': line})

        # Append interim text if any
        interim = data.get('interim', '')
        if interim:
            m2 = re.match(r'\[Speaker (\d+)\]: (.+)', interim)
            if m2:
                transcript_list.append({'speaker': int(m2.group(1)), 'text': m2.group(2)})
            elif interim not in frontend_transcript:
                transcript_list.append({'speaker': 0, 'text': interim})
    else:
        # Fallback: use server-side transcript history
        transcript_list = list(session['transcript_history'])
        interim = data.get('interim', '')
        if interim:
            m = re.match(r'\[Speaker (\d+)\]: (.+)', interim)
            if m:
                transcript_list.append({'speaker': int(m.group(1)), 'text': m.group(2)})
            else:
                transcript_list.append({'speaker': 0, 'text': interim})

    answer = stream_ai_answer(session['user_context'], transcript_list, force=True, sid=sid)
    if answer:
        emit('status_update', {'status': 'Ready'})


@socketio.on('force_answer')
def handle_force_answer(data):
    session = get_session(request.sid)
    if not session:
        return
    handle_manual_answer(data)
    session['ai_response_locked'] = True


@socketio.on('process_image')
def handle_image_process(data):
    sid = request.sid
    session = get_session(sid)
    if not session:
        return

    images = data.get('images', [])
    if not images:
        single = data.get('image', '')
        if single:
            images = [single]
    if not images:
        emit('status_update', {'status': 'No image received'})
        return

    print(f"[{sid[:8]}] >>> IMAGE PROCESSING ({len(images)} image(s)) <<<")
    emit('status_update', {'status': f'Analyzing {len(images)} image(s)...'})

    # Get transcript text from frontend (user-editable) for context-aware solving
    transcript_text = data.get('transcript', '')

    answer = stream_image_answer(images, session['user_context'], transcript_text=transcript_text, sid=sid)
    if answer:
        emit('status_update', {'status': 'Ready'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Server starting on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
