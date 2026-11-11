"""Parte 1 - Vision.

Uso:
  python parte1_vision.py 1.1 imagenes/img1.png "Tu pregunta del dominio"
  python parte1_vision.py 1.2            # extraccion estructurada de las imagenes de ground_truth.json
  python parte1_vision.py 1.3 imagenes/img1.png imagenes/img2.png "Compara estas figuras"
  python parte1_vision.py 1.4 [mas_imagenes|modelo_sin_vision|mime_incorrecto]
"""
import json
import sys
from pathlib import Path

import groq

from common import (CHAT_MODEL, IMG_DIR, OUT_DIR, VISION_MODEL, extract_json,
                    get_client, image_part)

# ---------------------------------------------------------------- Esquema (1.2)
# Dominio: tutor academico -> figuras/tablas de articulos. Ajusta los campos a tu dominio.
SCHEMA = {
    "tipo_elemento": {"tipo": str, "valores": ["figura", "tabla", "ecuacion", "otro"],
                      "desc": "Tipo de elemento visual"},
    "numero": {"tipo": str, "nullable": True, "desc": "Numero del elemento (ej. 'Figura 3', 'Tabla 2')"},
    "titulo": {"tipo": str, "nullable": True, "desc": "Titulo o leyenda tal como aparece"},
    "variable_x": {"tipo": str, "nullable": True, "desc": "Variable del eje X o de las columnas"},
    "variable_y": {"tipo": str, "nullable": True, "desc": "Variable del eje Y o de las filas"},
    "unidades": {"tipo": str, "nullable": True, "desc": "Unidades de medida mostradas"},
    "valor_maximo": {"tipo": (int, float), "nullable": True, "desc": "Valor maximo visible (numero)"},
}


def validar_esquema(data: dict) -> list[str]:
    """Devuelve la lista de problemas; lista vacia = cumple el esquema."""
    errores = []
    for campo, regla in SCHEMA.items():
        if campo not in data:
            errores.append(f"Falta el campo '{campo}'")
            continue
        valor = data[campo]
        if valor is None:
            if not regla.get("nullable"):
                errores.append(f"'{campo}' no puede ser null")
            continue
        if isinstance(valor, bool) or not isinstance(valor, regla["tipo"]):
            errores.append(f"'{campo}' tiene tipo {type(valor).__name__}, se esperaba {regla['tipo']}")
        elif "valores" in regla and valor not in regla["valores"]:
            errores.append(f"'{campo}'='{valor}' no esta en {regla['valores']}")
    extra = set(data) - set(SCHEMA)
    if extra:
        errores.append(f"Campos no esperados: {sorted(extra)}")
    return errores


def prompt_extraccion() -> str:
    campos = "\n".join(f"- {k}: {v['desc']}" for k, v in SCHEMA.items())
    return ("Extrae de la imagen los siguientes campos y responde SOLO con un objeto JSON "
            "valido, sin texto adicional ni markdown. Usa null si un campo no aparece.\n"
            f"{campos}\n"
            f"'tipo_elemento' debe ser uno de {SCHEMA['tipo_elemento']['valores']}. "
            "'valor_maximo' debe ser un numero, no texto.")


def preguntar(client, imagenes: list[Path], pregunta: str, modelo: str = VISION_MODEL,
              mime_override: str | None = None) -> str:
    contenido = [{"type": "text", "text": pregunta}] + [image_part(p, mime_override) for p in imagenes]
    r = client.chat.completions.create(model=modelo, temperature=0,
                                       messages=[{"role": "user", "content": contenido}])
    return r.choices[0].message.content


# ---------------------------------------------------------------- 1.1 y 1.3
def parte_pregunta(args):
    *rutas, pregunta = args
    imagenes = [Path(r) for r in rutas]
    print(f"Pregunta: {pregunta}\nImagenes: {[str(i) for i in imagenes]}\n")
    respuesta = preguntar(get_client(), imagenes, pregunta)
    print("Respuesta del modelo:\n", respuesta)
    (OUT_DIR / f"vision_{len(imagenes)}img.txt").write_text(
        f"PREGUNTA: {pregunta}\nIMAGENES: {rutas}\n\nRESPUESTA:\n{respuesta}\n", encoding="utf-8")


# ---------------------------------------------------------------- 1.2
def parte_extraccion():
    gt_path = Path(__file__).parent / "ground_truth.json"
    verdad = json.loads(gt_path.read_text(encoding="utf-8"))
    client = get_client()
    resultados = {}
    for nombre in verdad:
        ruta = IMG_DIR / nombre
        print(f"\n=== {nombre} ===")
        crudo = preguntar(client, [ruta], prompt_extraccion())
        try:
            data = extract_json(crudo)
        except json.JSONDecodeError as e:  # JSON mal formado (subclase de ValueError: va primero)
            print(f"[JSON MAL FORMADO] {e}\nRespuesta cruda: {crudo}")
            resultados[nombre] = {"error": "json_mal_formado", "crudo": crudo}
            continue
        except ValueError as e:  # la respuesta no contiene ningun objeto JSON
            print(f"[JSON INVALIDO] {e}\nRespuesta cruda: {crudo}")
            resultados[nombre] = {"error": "json_invalido", "crudo": crudo}
            continue
        errores = validar_esquema(data)
        if errores:
            print("[NO CUMPLE EL ESQUEMA]", *errores, sep="\n  - ")
            resultados[nombre] = {"error": "esquema", "detalle": errores, "data": data}
        else:
            print("[OK] cumple el esquema:", json.dumps(data, ensure_ascii=False))
            resultados[nombre] = {"data": data}

    # Comparacion campo por campo contra ground_truth.json
    def norm(v):
        return str(v).strip().lower() if v is not None else None

    filas, aciertos, total = [], 0, 0
    for nombre, real in verdad.items():
        extraido = resultados[nombre].get("data", {})
        for campo in SCHEMA:
            ok = norm(real.get(campo)) == norm(extraido.get(campo))
            total += 1
            aciertos += ok
            filas.append((nombre, campo, real.get(campo), extraido.get(campo), "Si" if ok else "No"))
    md = ["| Imagen | Campo | Valor real | Valor extraido | Correcto? |", "|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in filas]
    pct = 100 * aciertos / total if total else 0
    md.append(f"\n**Porcentaje de aciertos: {aciertos} de {total} campos = {pct:.1f} %**")
    salida = "\n".join(md)
    print("\n" + salida)
    (OUT_DIR / "extraccion_tabla.md").write_text(salida, encoding="utf-8")
    (OUT_DIR / "extraccion_resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nNota: la comparacion es por igualdad exacta (sin mayusculas). Revisa a mano los "
          "casos 'No' que sean equivalentes (ej. 'Fig. 3' vs 'Figura 3') y ajusta la tabla del informe.")


# ---------------------------------------------------------------- 1.4
def parte_error(caso: str):
    client = get_client()
    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not imgs:
        raise SystemExit("Pon imagenes en imagenes/ primero.")
    print(f"Caso provocado: {caso}")
    try:
        if caso == "mas_imagenes":      # 4 imagenes > limite de 3 (repite la primera si hace falta)
            lote = (imgs * 4)[:4]
            preguntar(client, lote, "Describe brevemente estas imagenes.")
        elif caso == "modelo_sin_vision":  # modelo de solo texto recibiendo una imagen
            preguntar(client, imgs[:1], "Describe la imagen.", modelo=CHAT_MODEL)
        elif caso == "mime_incorrecto":   # declara un tipo que no corresponde al archivo
            mime = "application/pdf" if imgs[0].suffix.lower() != ".pdf" else "image/png"
            preguntar(client, imgs[:1], "Describe la imagen.", mime_override=mime)
        else:
            raise SystemExit("Casos: mas_imagenes | modelo_sin_vision | mime_incorrecto")
        print("La API NO devolvio error en este caso (anota que paso y prueba otro caso).")
    except groq.APIStatusError as e:
        print(f"\nHTTP {e.status_code}\nMensaje de la API: {e.message}\nCuerpo: {e.body}")
        (OUT_DIR / f"error_{caso}.txt").write_text(
            f"caso={caso}\nstatus={e.status_code}\nmensaje={e.message}\ncuerpo={e.body}\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    modo, resto = sys.argv[1], sys.argv[2:]
    if modo in ("1.1", "1.3"):
        parte_pregunta(resto)
    elif modo == "1.2":
        parte_extraccion()
    elif modo == "1.4":
        parte_error(resto[0] if resto else "mas_imagenes")
    else:
        raise SystemExit(__doc__)
