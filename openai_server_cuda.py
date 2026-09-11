#!/usr/bin/env python3
"""
OpenAI-compatible-ish TTS server backed by Microsoft VibeVoice-Realtime-0.5B.

Endpoints
---------
1) POST /v1/audio/speech
   Compatible with the OpenAI Speech API request shape for the common fields:
     model, input, voice, response_format, speed, instructions, stream_format

   Notes:
   - response_format="pcm" streams raw PCM16LE @ 24 kHz mono as it is generated.
   - response_format="wav" is returned after generation completes.
   - mp3/opus/aac/flac are supported if ffmpeg is installed, but are buffered
     until generation completes.
   - speed != 1.0 is currently rejected because VibeVoice does not expose an
     equivalent parameter here.
   - "instructions" is accepted for compatibility but is not interpreted.

2) WS /v1/realtime
   Implements a small TTS-oriented subset of the OpenAI Realtime event shape:
     session.created
     session.update -> session.updated
     conversation.item.create
     response.create
     response.created
     response.output_audio.delta
     response.output_audio.done
     response.done
     error

   IMPORTANT:
   This is TTS-only. It does not run an LLM. The latest input_text sent with
   conversation.item.create is treated literally as the text to synthesize.

3) WS /v1/audio/speech/ws
   Simpler TTS WebSocket:
   client sends one JSON object shaped like /v1/audio/speech
   server returns:
     response.created
     response.output_audio.delta
     response.output_audio.done
     response.done

Authentication
--------------
Set a fixed API key in the environment:

    export LOCAL_TTS_API_KEY="change-me"

HTTP:
    Authorization: Bearer change-me

WebSocket:
    Authorization: Bearer change-me

For browser WebSocket clients (which cannot set arbitrary Authorization headers),
the server also accepts:

    ws://127.0.0.1:8000/v1/realtime?api_key=change-me

Configuration
-------------
    LOCAL_TTS_API_KEY          required
    VIBEVOICE_MODEL_DIR        default: ./VibeVoice-Realtime-0.5B
    VIBEVOICE_VOICES_DIR       default: ./vibevoice_voices
    VIBEVOICE_DEFAULT_VOICE    default: woman
    CUDA è obbligatoria. Il server termina all'avvio se CUDA non è disponibile.
    VIBEVOICE_DDPM_STEPS       default: 1
    VIBEVOICE_CFG_SCALE        default: 1.5
    VIBEVOICE_HOST             default: 0.0.0.0
    VIBEVOICE_PORT             default: 8000

Install
-------
From the Microsoft VibeVoice repo:

    pip install -e ".[streamingtts]"
    pip install fastapi "uvicorn[standard]" huggingface_hub numpy pydantic

Run
---
    export LOCAL_TTS_API_KEY="secret"
    python vibevoice_openai_server_cuda.py

OpenAI Python SDK example
-------------------------
    from openai import OpenAI

    client = OpenAI(
        api_key="secret",
        base_url="http://127.0.0.1:8000/v1",
    )

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input="Ciao, questa è una prova.",
        response_format="pcm",
    ) as response:
        with open("speech.pcm", "wb") as f:
            for chunk in response.iter_bytes(chunk_size=4096):
                f.write(chunk)

For the two native Italian presets you can also pass voice="woman" or voice="man".
All normal OpenAI voice names are accepted and mapped to VIBEVOICE_DEFAULT_VOICE.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import io
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Literal
from urllib.request import urlretrieve

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.modular.streamer import AudioStreamer
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
except ImportError as exc:
    raise SystemExit(
        "Non trovo il pacchetto 'vibevoice'.\n"
        "Installa il repository ufficiale Microsoft, poi:\n"
        '  pip install -e ".[streamingtts]"\n'
        "  pip install fastapi 'uvicorn[standard]' huggingface_hub numpy pydantic\n"
    ) from exc


MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
SAMPLE_RATE = 24_000

VOICE_URLS = {
    "woman": (
        "it-Spk0_woman.pt",
        "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
        "demo/voices/streaming_model/it-Spk0_woman.pt",
    ),
    "man": (
        "it-Spk1_man.pt",
        "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
        "demo/voices/streaming_model/it-Spk1_man.pt",
    ),
}

OPENAI_VOICE_NAMES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_dir: Path
    voices_dir: Path
    default_voice: str
    device: str
    ddpm_steps: int
    cfg_scale: float

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("LOCAL_TTS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "LOCAL_TTS_API_KEY non impostata. Esempio:\n"
                '  export LOCAL_TTS_API_KEY="secret"'
            )

        default_voice = os.getenv("VIBEVOICE_DEFAULT_VOICE", "woman").strip().lower()
        if default_voice not in VOICE_URLS:
            raise RuntimeError("VIBEVOICE_DEFAULT_VOICE deve essere 'woman' oppure 'man'.")

        return cls(
            api_key=api_key,
            model_dir=Path(
                os.getenv("VIBEVOICE_MODEL_DIR", "./VibeVoice-Realtime-0.5B")
            ).expanduser().resolve(),
            voices_dir=Path(
                os.getenv("VIBEVOICE_VOICES_DIR", "./vibevoice_voices")
            ).expanduser().resolve(),
            default_voice=default_voice,
            device="cuda",
            ddpm_steps=int(os.getenv("VIBEVOICE_DDPM_STEPS", "1")),
            cfg_scale=float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5")),
        )


class SpeechRequest(BaseModel):
    model: str = "gpt-4o-mini-tts"
    input: str = Field(min_length=1)
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = 1.0
    instructions: str | None = None
    stream_format: str | None = "audio"


def choose_device(requested: str) -> str:
    # Questo server è intenzionalmente CUDA-only.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. Questo server richiede una GPU NVIDIA "
            "con una build CUDA di PyTorch."
        )

    return "cuda"


def pcm16_from_float(chunk: np.ndarray) -> bytes:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if chunk.size == 0:
        return b""
    chunk = np.clip(chunk, -1.0, 1.0)
    pcm = (chunk * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes()


def wav_from_pcm16(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return out.getvalue()


def ffmpeg_encode(pcm: bytes, fmt: str) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"response_format={fmt!r} richiede ffmpeg. "
            "Installa ffmpeg oppure usa response_format='pcm'/'wav'."
        )

    output_args = {
        "mp3": ["-codec:a", "libmp3lame", "-f", "mp3"],
        "opus": ["-codec:a", "libopus", "-f", "opus"],
        "aac": ["-codec:a", "aac", "-f", "adts"],
        "flac": ["-codec:a", "flac", "-f", "flac"],
    }[fmt]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *output_args,
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg ha fallito ({proc.returncode}): "
            + proc.stderr.decode("utf-8", errors="replace")
        )
    return proc.stdout


class VibeVoiceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = choose_device(settings.device)
        self.processor = None
        self.model = None
        self.voice_prompts: dict[str, object] = {}
        # Il modello streaming supporta batch size 1: serializziamo l'inferenza.
        self._generation_lock = threading.Lock()

    def ensure_assets(self) -> None:
        self.settings.model_dir.parent.mkdir(parents=True, exist_ok=True)
        self.settings.voices_dir.mkdir(parents=True, exist_ok=True)

        print(f"[download] modello -> {self.settings.model_dir}")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(self.settings.model_dir),
        )

        for alias, (filename, url) in VOICE_URLS.items():
            path = self.settings.voices_dir / filename
            if not path.exists():
                print(f"[download] voce {alias} -> {path}")
                urlretrieve(url, path)

    def load(self) -> None:
        self.ensure_assets()

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(
            f"[load] device=cuda gpu={gpu_name} "
            f"compute_capability={capability[0]}.{capability[1]} "
            f"ddpm_steps={self.settings.ddpm_steps} cfg={self.settings.cfg_scale}"
        )

        self.processor = VibeVoiceStreamingProcessor.from_pretrained(
            str(self.settings.model_dir)
        )

        dtype = torch.bfloat16
        device_map = "cuda"
        attn = "flash_attention_2"

        try:
            self.model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(self.settings.model_dir),
                    torch_dtype=dtype,
                    device_map=device_map,
                    attn_implementation=attn,
                )
            )
        except Exception as exc:
            if self.device != "cuda":
                raise
            print(f"[load] flash_attention_2 non disponibile: {exc}")
            print("[load] fallback CUDA -> SDPA")
            self.model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(self.settings.model_dir),
                    torch_dtype=dtype,
                    device_map="cuda",
                    attn_implementation="sdpa",
                )
            )

        self.model.eval()
        self.model.set_ddpm_inference_steps(
            num_steps=self.settings.ddpm_steps
        )

        for alias, (filename, _) in VOICE_URLS.items():
            voice_path = self.settings.voices_dir / filename
            # I preset ufficiali contengono strutture Python complesse / KV cache.
            self.voice_prompts[alias] = torch.load(
                voice_path,
                map_location=torch.device(self.device),
                weights_only=False,
            )

        print("[ready] VibeVoice caricato.")

    def resolve_voice(self, requested: str | None) -> str:
        value = (requested or "").strip().lower()

        if value in {"woman", "female", "it-spk0_woman", "it-spk0_woman.pt"}:
            return "woman"

        if value in {"man", "male", "it-spk1_man", "it-spk1_man.pt"}:
            return "man"

        # Per compatibilità accettiamo i nomi standard OpenAI.
        # Non esiste una corrispondenza 1:1: li mappiamo alla voce locale di default.
        if value in OPENAI_VOICE_NAMES or not value:
            return self.settings.default_voice

        raise ValueError(
            f"Voce non supportata: {requested!r}. "
            "Usa woman/man oppure un nome voce OpenAI standard."
        )

    def stream_pcm(
        self,
        text: str,
        voice: str,
        cancel_event: threading.Event | None = None,
    ) -> Generator[bytes, None, None]:
        assert self.processor is not None
        assert self.model is not None

        text = text.strip().replace("’", "'")
        if not text:
            raise ValueError("input vuoto")

        voice_alias = self.resolve_voice(voice)
        cancel_event = cancel_event or threading.Event()

        cached_prompt = copy.deepcopy(self.voice_prompts[voice_alias])

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = torch.device(self.device)
        inputs = {
            key: value.to(target_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                with self._generation_lock:
                    if cancel_event.is_set():
                        return

                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=self.settings.cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": False},
                        audio_streamer=streamer,
                        verbose=False,
                        refresh_negative=True,
                        all_prefilled_outputs=copy.deepcopy(cached_prompt),
                        stop_check_fn=cancel_event.is_set,
                    )
            except BaseException as exc:
                errors.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        try:
            for chunk in streamer.get_stream(0):
                if cancel_event.is_set():
                    break

                if torch.is_tensor(chunk):
                    arr = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    arr = np.asarray(chunk, dtype=np.float32)

                pcm = pcm16_from_float(arr)
                if pcm:
                    yield pcm
        finally:
            cancel_event.set()
            thread.join()

        if errors:
            raise errors[0]

    def collect_pcm(self, text: str, voice: str) -> bytes:
        return b"".join(self.stream_pcm(text, voice))


settings = Settings.from_env()
tts = VibeVoiceService(settings)

app = FastAPI(
    title="VibeVoice OpenAI-compatible TTS",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    # Carica una sola volta all'avvio.
    tts.load()


def bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    prefix = "bearer "
    if value.lower().startswith(prefix):
        return value[len(prefix):].strip()
    return None


def require_http_auth(request: Request) -> None:
    token = bearer_token(request.headers.get("authorization"))
    if token != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


def websocket_is_authorized(ws: WebSocket) -> bool:
    token = bearer_token(ws.headers.get("authorization"))
    if token == settings.api_key:
        return True

    # Solo come comodità per browser WebSocket.
    query_key = ws.query_params.get("api_key")
    return query_key == settings.api_key


def mime_for(fmt: str) -> str:
    return {
        "pcm": "application/octet-stream",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }[fmt]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": MODEL_ID,
        "device": tts.device,
        "sample_rate": SAMPLE_RATE,
        "ddpm_steps": settings.ddpm_steps,
        "default_voice": settings.default_voice,
    }


@app.get("/v1/models")
def models(request: Request) -> dict:
    require_http_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4o-mini-tts",
                "object": "model",
                "owned_by": "local-vibevoice",
            },
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "microsoft",
            },
        ],
    }


@app.post("/v1/audio/speech")
def audio_speech(body: SpeechRequest, request: Request):
    require_http_auth(request)

    if body.speed != 1.0:
        raise HTTPException(
            status_code=400,
            detail="Questo backend locale supporta attualmente solo speed=1.0.",
        )

    if body.stream_format not in {None, "audio"}:
        raise HTTPException(
            status_code=400,
            detail="Questo server supporta stream_format='audio' soltanto.",
        )

    try:
        # PCM è il percorso realmente streaming e più vicino alla latenza minima.
        if body.response_format == "pcm":
            return StreamingResponse(
                tts.stream_pcm(body.input, body.voice),
                media_type=mime_for("pcm"),
                headers={
                    "X-Audio-Sample-Rate": str(SAMPLE_RATE),
                    "X-Audio-Channels": "1",
                    "X-Audio-Sample-Format": "s16le",
                    "X-TTS-Backend": MODEL_ID,
                },
            )

        # Gli altri container vengono bufferizzati per poter costruire un file valido.
        pcm = tts.collect_pcm(body.input, body.voice)

        if body.response_format == "wav":
            payload = wav_from_pcm16(pcm)
        else:
            payload = ffmpeg_encode(pcm, body.response_format)

        return Response(
            content=payload,
            media_type=mime_for(body.response_format),
            headers={"X-TTS-Backend": MODEL_ID},
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def realtime_session_payload(session_id: str, voice: str) -> dict:
    return {
        "id": session_id,
        "object": "realtime.session",
        "type": "realtime",
        "model": "gpt-realtime",
        "output_modalities": ["audio"],
        "audio": {
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": SAMPLE_RATE,
                },
                "voice": voice,
            }
        },
    }


def extract_text_from_item(item: dict) -> str | None:
    parts = item.get("content") or []
    out: list[str] = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        ptype = part.get("type")
        if ptype in {"input_text", "text"} and isinstance(part.get("text"), str):
            out.append(part["text"])

    text = "\n".join(x for x in out if x.strip()).strip()
    return text or None


def extract_text_from_response_create(event: dict) -> str | None:
    """
    Supporta anche response.create con response.input per facilitare client
    che vogliono inviare testo senza conversation.item.create.
    """
    response = event.get("response") or {}

    direct_text = response.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    inputs = response.get("input")
    if isinstance(inputs, list):
        texts: list[str] = []
        for item in inputs:
            if not isinstance(item, dict):
                continue
            text = extract_text_from_item(item)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts)

    return None


async def send_openai_audio_stream(
    ws: WebSocket,
    *,
    text: str,
    voice: str,
    response_id: str | None = None,
) -> None:
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    item_id = f"item_{uuid.uuid4().hex}"
    cancel_event = threading.Event()
    started = time.perf_counter()
    first_audio_at: float | None = None
    bytes_sent = 0

    await ws.send_json(
        {
            "type": "response.created",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "output": [],
            },
        }
    )

    await ws.send_json(
        {
            "type": "response.output_item.added",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
    )

    await ws.send_json(
        {
            "type": "response.content_part.added",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "audio",
                "transcript": text,
            },
        }
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    def producer() -> None:
        try:
            for pcm in tts.stream_pcm(text, voice, cancel_event=cancel_event):
                fut = asyncio.run_coroutine_threadsafe(queue.put(pcm), loop)
                fut.result()
        except BaseException as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    try:
        while True:
            item = await queue.get()

            if item is None:
                break

            if isinstance(item, BaseException):
                raise item

            if first_audio_at is None:
                first_audio_at = time.perf_counter()

            bytes_sent += len(item)

            await ws.send_json(
                {
                    "type": "response.output_audio.delta",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": base64.b64encode(item).decode("ascii"),
                }
            )

    except WebSocketDisconnect:
        cancel_event.set()
        raise
    except BaseException:
        cancel_event.set()
        raise
    finally:
        cancel_event.set()
        await asyncio.to_thread(producer_thread.join)

    elapsed = time.perf_counter() - started
    audio_seconds = bytes_sent / (SAMPLE_RATE * 2)
    ttfa_ms = (
        (first_audio_at - started) * 1000.0
        if first_audio_at is not None
        else None
    )

    await ws.send_json(
        {
            "type": "response.output_audio.done",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
    )

    await ws.send_json(
        {
            "type": "response.content_part.done",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "audio",
                "transcript": text,
            },
        }
    )

    await ws.send_json(
        {
            "type": "response.output_item.done",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "transcript": text,
                    }
                ],
            },
        }
    )

    await ws.send_json(
        {
            "type": "response.done",
            "event_id": f"event_{uuid.uuid4().hex}",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "output": [],
                "metadata": {
                    "backend": MODEL_ID,
                    "sample_rate": SAMPLE_RATE,
                    "audio_seconds": round(audio_seconds, 4),
                    "generation_seconds": round(elapsed, 4),
                    "rtf": round(elapsed / audio_seconds, 4)
                    if audio_seconds > 0
                    else None,
                    "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
                },
            },
        }
    )


async def websocket_auth_or_close(ws: WebSocket) -> bool:
    if not websocket_is_authorized(ws):
        await ws.accept()
        await ws.send_json(
            {
                "type": "error",
                "event_id": f"event_{uuid.uuid4().hex}",
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                    "message": "Invalid API key.",
                },
            }
        )
        await ws.close(code=1008)
        return False

    await ws.accept()
    return True


@app.websocket("/v1/audio/speech/ws")
async def audio_speech_ws(ws: WebSocket) -> None:
    if not await websocket_auth_or_close(ws):
        return

    try:
        raw = await ws.receive_text()
        data = json.loads(raw)
        body = SpeechRequest.model_validate(data)

        if body.speed != 1.0:
            raise ValueError("Questo backend locale supporta attualmente solo speed=1.0.")

        await send_openai_audio_stream(
            ws,
            text=body.input,
            voice=body.voice,
        )

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "error": {
                        "type": "server_error",
                        "code": "tts_error",
                        "message": str(exc),
                    },
                }
            )
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    if not await websocket_auth_or_close(ws):
        return

    session_id = f"sess_{uuid.uuid4().hex}"
    current_voice = settings.default_voice
    pending_text: str | None = None

    await ws.send_json(
        {
            "type": "session.created",
            "event_id": f"event_{uuid.uuid4().hex}",
            "session": realtime_session_payload(session_id, current_voice),
        }
    )

    try:
        while True:
            raw = await ws.receive_text()

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json(
                    {
                        "type": "error",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_json",
                            "message": "Il messaggio WebSocket deve essere JSON.",
                        },
                    }
                )
                continue

            event_type = event.get("type")

            if event_type == "session.update":
                session_cfg = event.get("session") or {}

                audio_cfg = session_cfg.get("audio") or {}
                output_cfg = audio_cfg.get("output") or {}
                requested_voice = output_cfg.get("voice")

                if requested_voice:
                    try:
                        current_voice = tts.resolve_voice(str(requested_voice))
                    except ValueError as exc:
                        await ws.send_json(
                            {
                                "type": "error",
                                "event_id": f"event_{uuid.uuid4().hex}",
                                "error": {
                                    "type": "invalid_request_error",
                                    "code": "invalid_voice",
                                    "message": str(exc),
                                },
                            }
                        )
                        continue

                # VibeVoice viene esposto come PCM16LE 24k mono.
                fmt = output_cfg.get("format") or {}
                requested_rate = fmt.get("rate", SAMPLE_RATE)
                requested_type = fmt.get("type", "audio/pcm")
                if requested_type != "audio/pcm" or requested_rate != SAMPLE_RATE:
                    await ws.send_json(
                        {
                            "type": "error",
                            "event_id": f"event_{uuid.uuid4().hex}",
                            "error": {
                                "type": "invalid_request_error",
                                "code": "unsupported_audio_format",
                                "message": (
                                    "Supportato solo audio/pcm 24000 Hz "
                                    "(PCM16LE mono)."
                                ),
                            },
                        }
                    )
                    continue

                await ws.send_json(
                    {
                        "type": "session.updated",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "session": realtime_session_payload(
                            session_id,
                            current_voice,
                        ),
                    }
                )
                continue

            if event_type == "conversation.item.create":
                item = event.get("item") or {}
                text = extract_text_from_item(item)
                if text:
                    pending_text = text

                item_id = item.get("id") or f"item_{uuid.uuid4().hex}"
                normalized_item = dict(item)
                normalized_item["id"] = item_id

                await ws.send_json(
                    {
                        "type": "conversation.item.added",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "previous_item_id": None,
                        "item": normalized_item,
                    }
                )
                await ws.send_json(
                    {
                        "type": "conversation.item.done",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "previous_item_id": None,
                        "item": normalized_item,
                    }
                )
                continue

            if event_type == "response.create":
                direct = extract_text_from_response_create(event)
                text = direct or pending_text

                if not text:
                    await ws.send_json(
                        {
                            "type": "error",
                            "event_id": f"event_{uuid.uuid4().hex}",
                            "error": {
                                "type": "invalid_request_error",
                                "code": "missing_input_text",
                                "message": (
                                    "Questo server è TTS-only. Invia prima "
                                    "conversation.item.create con content "
                                    "type=input_text, poi response.create."
                                ),
                            },
                        }
                    )
                    continue

                response_cfg = event.get("response") or {}
                requested_voice = (
                    ((response_cfg.get("audio") or {}).get("output") or {}).get("voice")
                )
                voice = current_voice
                if requested_voice:
                    voice = tts.resolve_voice(str(requested_voice))

                await send_openai_audio_stream(
                    ws,
                    text=text,
                    voice=voice,
                )
                continue

            if event_type == "response.cancel":
                # Questa implementazione processa una generazione per volta e non
                # riceve nuovi eventi mentre sta generando.
                await ws.send_json(
                    {
                        "type": "error",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "error": {
                            "type": "invalid_request_error",
                            "code": "cancel_not_supported",
                            "message": (
                                "response.cancel non è ancora supportato durante "
                                "una generazione attiva."
                            ),
                        },
                    }
                )
                continue

            await ws.send_json(
                {
                    "type": "error",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "unsupported_event",
                        "message": f"Evento non supportato: {event_type!r}",
                    },
                }
            )

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "error": {
                        "type": "server_error",
                        "code": "tts_error",
                        "message": str(exc),
                    },
                }
            )
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.getenv("VIBEVOICE_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VIBEVOICE_PORT", "8000")),
    )
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
