"""Parte 2 - Audio: audio -> transcripcion (Whisper) -> respuesta del modelo.

Uso:
  python parte2_audio.py              # procesa todos los audios de audios/metadata.json
  python parte2_audio.py --sin-vocab  # sin pista de vocabulario (para comparar errores en terminos tecnicos)
"""
import json
import sys
from pathlib import Path

import groq

from common import AUDIO_DIR, CHAT_MODEL, OUT_DIR, STT_MODEL, get_client

# Pista de vocabulario tecnico del dominio (Whisper la usa como contexto). Ajustala.
VOCAB = "RAG, embeddings, chunking, base vectorial, Ragas, fine-tuning, prompt, few-shot, token, LLM"

SYSTEM = ("Eres un tutor academico de Ingenieria de Sistemas. Responde en espanol, "
          "de forma clara y breve (maximo 5 lineas), con rigor tecnico.")


def transcribir(client, ruta: Path, usar_vocab: bool) -> dict:
    with ruta.open("rb") as f:
        kwargs = dict(file=(ruta.name, f.read()), model=STT_MODEL, language="es",
                      response_format="verbose_json", temperature=0.0)
    if usar_vocab:
        kwargs["prompt"] = VOCAB
    r = client.audio.transcriptions.create(**kwargs)
    return {"texto": r.text.strip(), "duracion": round(getattr(r, "duration", 0) or 0, 1)}


def responder(client, texto: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": texto}])
    return r.choices[0].message.content.strip()


def main():
    usar_vocab = "--sin-vocab" not in sys.argv
    meta = json.loads((AUDIO_DIR / "metadata.json").read_text(encoding="utf-8"))
    client = get_client()
    filas = []
    for m in meta:
        ruta = AUDIO_DIR / m["archivo"]
        print(f"\n=== {m['archivo']} ({m['hablante']}, {m['condicion']}) ===")
        try:
            t = transcribir(client, ruta, usar_vocab)
            print("Transcripcion:", t["texto"])
            resp = responder(client, t["texto"])
            print("Respuesta:", resp)
            filas.append({**m, **t, "respuesta": resp})
        except groq.APIStatusError as e:
            print(f"[ERROR API {e.status_code}] {e.message}")
            filas.append({**m, "error": f"{e.status_code}: {e.message}"})

    sufijo = "con_vocab" if usar_vocab else "sin_vocab"
    (OUT_DIR / f"audio_resultados_{sufijo}.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["| Audio | Hablante | Condicion | Duracion (s) | Lo que se dijo | Transcripcion |",
          "|---|---|---|---|---|---|"]
    md += [f"| {f['archivo']} | {f['hablante']} | {f['condicion']} | {f.get('duracion','')} | "
           f"{f['dijo']} | {f.get('texto','')} |" for f in filas]
    (OUT_DIR / f"audio_tabla_{sufijo}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nResultados guardados en resultados/ (*_{sufijo}.*)")


if __name__ == "__main__":
    main()
