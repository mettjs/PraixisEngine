"""Register spike for hypothetical-question indexing.

Goal: confirm that Gemma generates questions in *civilian* register, and that
those generated questions sit closer (cosine) to how a real person asks than
the raw statute chunk does.

Run: uv run python scripts/spike_hyq_register.py
"""
import numpy as np
from openai import OpenAI

from src.config import AI_API_URL, AI_API_KEY, MODEL_NAME
from src.utils.vectordb.embeddings import embed


# Representative clauses of Ley 340-06 (Compras y Contrataciones) in statute register.
CHUNKS = [
    (
        "ley_340_06_art3",
        "Artículo 3.- Principios. Esta ley y su reglamento de aplicación tienen "
        "como objetivo establecer los principios y normas generales que rigen la "
        "contratación pública, relacionada con bienes, obras, servicios y "
        "concesiones del Estado, así como las modalidades que dentro de cada "
        "especialidad puedan considerarse. La contratación pública se regirá por "
        "los principios de eficiencia, igualdad y libre competencia, "
        "transparencia y publicidad, economía y flexibilidad, equidad, "
        "responsabilidad, moralidad y buena fe.",
    ),
    (
        "ley_340_06_art16",
        "Artículo 16.- De las excepciones. No obstante lo expresado en el "
        "artículo precedente, las siguientes actividades, no obstante tener "
        "carácter de contrataciones, quedan excluidas del procedimiento de "
        "selección de contratistas: 1) Las que por razones de seguridad o "
        "emergencia nacional se efectúen al amparo de una declaratoria; 2) Las "
        "que se realicen entre entidades del Estado.",
    ),
    (
        "ley_340_06_art63",
        "Artículo 63.- Sanciones. Los oferentes, proponentes o contratistas que "
        "incurran en las infracciones señaladas en la presente ley, serán "
        "pasibles de las siguientes sanciones, según la gravedad de la falta: "
        "amonestación escrita; ejecución de la garantía; e inhabilitación "
        "temporal o definitiva para contratar con el Estado.",
    ),
]

# How a real person (non-lawyer) would ask about each clause.
REAL_QUESTIONS = [
    "¿qué reglas tiene que seguir el gobierno cuando compra cosas o contrata obras?",
    "¿cuándo el Estado puede comprar algo sin hacer una licitación?",
    "¿qué me puede pasar si hago trampa en un contrato con el gobierno?",
]

QUESTION_GEN_PROMPT = (
    "Eres un asistente que ayuda a personas comunes a encontrar información en "
    "documentos legales. A continuación tienes un fragmento de una ley.\n\n"
    "Genera 5 preguntas que una persona SIN formación jurídica haría y que este "
    "fragmento responde. Usa lenguaje cotidiano y sencillo, como hablaría un "
    "ciudadano normal (no copies el lenguaje formal de la ley). Escribe en el "
    "mismo idioma del fragmento. Una pregunta por línea, sin numeración ni "
    "viñetas.\n\nFragmento:\n"
)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
    return a @ b.T


def main() -> None:
    client = OpenAI(base_url=AI_API_URL, api_key=AI_API_KEY)

    for (cid, chunk), real_q in zip(CHUNKS, REAL_QUESTIONS):
        print("=" * 80)
        print(f"CHUNK: {cid}")
        print(f"REAL USER QUESTION: {real_q}\n")

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": QUESTION_GEN_PROMPT + chunk}],
            temperature=0.4,
        )
        raw = resp.choices[0].message.content or ""
        gen_qs = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
        gen_qs = [q for q in gen_qs if len(q) > 8]

        print("GENERATED QUESTIONS:")
        for q in gen_qs:
            print(f"  - {q}")

        # Embed: [real_q], [chunk], gen_qs
        vecs = np.asarray(embed([real_q, chunk] + gen_qs), dtype=np.float32)
        real_vec, chunk_vec, gen_vecs = vecs[0:1], vecs[1:2], vecs[2:]

        sim_real_chunk = float(cosine_matrix(real_vec, chunk_vec)[0, 0])
        sims_real_gen = cosine_matrix(real_vec, gen_vecs)[0]
        best_gen = float(sims_real_gen.max())
        mean_gen = float(sims_real_gen.mean())

        print(f"\n  cosine(real_q, raw_chunk)      = {sim_real_chunk:.4f}")
        print(f"  cosine(real_q, best gen_q)     = {best_gen:.4f}")
        print(f"  cosine(real_q, mean gen_q)     = {mean_gen:.4f}")
        verdict = "IMPROVES" if best_gen > sim_real_chunk else "NO GAIN"
        print(f"  best gen vs raw chunk: {verdict} "
              f"(delta = {best_gen - sim_real_chunk:+.4f})\n")


if __name__ == "__main__":
    main()
