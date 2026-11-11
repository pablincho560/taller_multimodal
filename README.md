# Taller: Entrada multimodal y audio

Dominio: tutor academico (figuras y tablas de articulos; preguntas orales sobre IA/ingenieria de sistemas).
Proveedor: Groq (cliente `groq`). Modelos configurables por variables de entorno.

## Preparacion
```bash
cd taller_multimodal
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # y completa GROQ_API_KEY
```

Variables de entorno: `GROQ_API_KEY` (obligatoria), `VISION_MODEL`, `STT_MODEL`, `CHAT_MODEL` (opcionales).
La clave va SOLO en `.env` (ignorado por git). Nunca la pegues en el codigo.

## Antes de ejecutar
- Pon al menos 5 imagenes en `imagenes/` (maximo ~4 MB cada una para base64).
- Edita `ground_truth.json` con los valores reales que lees en 3 de esas imagenes.
- Graba 4 audios (10-30 s) en `audios/` y completa `audios/metadata.json`.

## Parte 1 - Vision
```bash
python parte1_vision.py 1.1 imagenes/figura1.png "Pregunta del dominio"
python parte1_vision.py 1.2
python parte1_vision.py 1.3 imagenes/figura1.png imagenes/figura2.png "Compara estas figuras"
python parte1_vision.py 1.4 mas_imagenes      # o modelo_sin_vision | mime_incorrecto
```

## Parte 2 - Audio
```bash
python parte2_audio.py              # transcripcion + respuesta
python parte2_audio.py --sin-vocab  # opcional: comparar sin pista de vocabulario
```

Las salidas se guardan en `resultados/` para copiarlas al informe.
