"""Prove the question index is load-bearing: a register-gap civilian query where
HYBRID alone ranks the target chunk LOW, and HQ-on fusion rescues it into top-n.

Run: PYTHONPATH=. uv run python scripts/spike_hyq_loadbearing.py
"""
import asyncio

from src.utils.vectordb.pool import init_db, get_pool, close_db
from src.utils.vectordb.ingestion import add_file_to_rag_db
from src.utils.vectordb.constants import DELETE_FILE, QUESTION_SEARCH, HYBRID_SEARCH
from src.utils.vectordb import questions as Q
from src.utils.vectordb import retrieval as R
from src.utils.vectordb.embeddings import embed

# A corpus of distinct articles so a target can genuinely rank outside top-n.
ARTICLES = [
    "Artículo 1.- Objeto. La presente ley establece el marco regulatorio del sistema nacional de compras y contrataciones públicas.",
    "Artículo 2.- Ámbito. Quedan sujetos a esta ley todos los organismos del gobierno central, las instituciones descentralizadas y los municipios.",
    "Artículo 3.- Definiciones. A los fines de esta ley se entiende por oferente toda persona física o jurídica que presenta una propuesta.",
    "Artículo 4.- Órgano rector. La Dirección General de Contrataciones Públicas es el órgano rector del sistema y dicta las políticas generales.",
    "Artículo 5.- Registro de proveedores. Todo proveedor del Estado debe inscribirse en el Registro de Proveedores del Estado antes de contratar.",
    "Artículo 6.- Plan anual. Cada institución elaborará un plan anual de compras que será publicado en el portal transaccional.",
    "Artículo 7.- Modalidades. Las modalidades de selección incluyen la licitación pública, la licitación restringida y la comparación de precios.",
    "Artículo 8.- Pliegos. Los pliegos de condiciones especificarán los requisitos técnicos y económicos exigidos a los participantes.",
    "Artículo 9.- Plazos. El plazo para la presentación de propuestas no será menor de treinta días contados desde la convocatoria.",
    "Artículo 10.- Garantías. Los participantes deberán constituir una garantía de seriedad de la oferta equivalente al uno por ciento.",
    "Artículo 11.- Adjudicación. El contrato se adjudicará a la propuesta que resulte más conveniente conforme a los criterios del pliego.",
    # TARGET — sanciones. Civilian asks about 'punishment for cheating'; this
    # article never uses those words (register + vocabulary gap).
    "Artículo 12.- Sanciones. Quienes incurran en las faltas previstas serán pasibles de amonestación escrita, ejecución de la garantía e inhabilitación temporal o definitiva para contratar con el Estado.",
]
DOC = "\n\n".join(ARTICLES)
# Resolved after chunking by locating the chunk that contains the sanciones text
# (character chunking merges articles, so index != article number).

# Register-gap civilian queries phrased to AVOID the sanciones chunk's vocabulary
# (no 'sanción', 'infracción', 'inhabilitación', 'falta', 'garantía', 'Estado',
# 'contratar') so the FTS arm can't trivially hand over the answer and the test
# actually stresses the register gap the question index is meant to close.
QUERIES = [
    "si engaño en un concurso de obras, ¿qué consecuencias enfrento yo?",
    "¿me pueden dejar fuera para siempre por portarme mal?",
    "si miento en mis papeles, ¿me castigan de alguna forma?",
    "¿pierdo mi dinero depositado si hago algo prohibido?",
]


def rank_of(target: int, ranked: list[tuple[str, int]]) -> str:
    for i, (_, idx) in enumerate(ranked, start=1):
        if idx == target:
            return str(i)
    return "NOT FOUND"


async def main() -> None:
    await init_db()
    p = get_pool()
    app = col = "__hyq_lb__"
    src = "ley.txt"
    await p.execute(DELETE_FILE, app, col, src)

    rows = await add_file_to_rag_db(
        text=DOC, collection_name=col, filename=src, app_name=app,
        chunk_overlap=0, chunk_size=200, chunking_strategy="character",
    )
    target_chunks = [r["chunk_index"] for r in rows if "Sanciones" in r["content"]]
    if len(target_chunks) != 1:
        raise SystemExit(f"Expected exactly one chunk with 'Sanciones', got {target_chunks}")
    TARGET_INDEX = target_chunks[0]
    print(f"chunks: {len(rows)} | target (sanciones) = chunk_index {TARGET_INDEX}")
    Q.schedule_question_generation(app_name=app, collection_name=col, source=src, chunks=rows)
    while Q._background_tasks:
        await asyncio.gather(*list(Q._background_tasks), return_exceptions=True)
    print("questions stored:", await p.fetchval("SELECT COUNT(*) FROM chunk_questions WHERE app=$1", app))

    pool_n = max(5 * 4, 40)
    print("\nrank of the sanciones chunk: lower is better. HQ off = hybrid only; "
          "HQ on = fused.\n'RESCUE@n' = HQ off misses it within top-n but HQ on lands it.\n")

    for query in QUERIES:
        emb = (await asyncio.to_thread(embed, [query]))[0]
        hy = await p.fetch(HYBRID_SEARCH, emb, app, col, pool_n, R._fts_query(query), None, pool_n)
        qs = await p.fetch(QUESTION_SEARCH, emb, app, col, pool_n, None, pool_n)
        hl = [(r["source"], r["chunk_index"]) for r in hy]
        ql = [(r["source"], r["chunk_index"]) for r in qs]
        fused = R._rrf_fuse([hl, ql], pool_n)

        h_rank = rank_of(TARGET_INDEX, hl)          # hybrid-only (HQ off)
        f_rank = rank_of(TARGET_INDEX, fused)        # fused (HQ on)

        def _r(s: str) -> int:
            return 9999 if s == "NOT FOUND" else int(s)

        flags = [f"RESCUE@{n}" for n in (1, 3, 5) if _r(h_rank) > n and _r(f_rank) <= n]
        flag = ("  <-- " + ", ".join(flags)) if flags else ""
        print(f"  query: {query}")
        print(f"     HQ off rank: {h_rank:>9} | HQ on rank: {f_rank:>9}{flag}\n")

    await p.execute(DELETE_FILE, app, col, src)
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
