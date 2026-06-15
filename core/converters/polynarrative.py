"""PolyNarrative -> KB converter (README §9).

Raw layout:  data/PolyNarrative/<split>/<lang>/
  train:      raw-documents/*.txt          + subtask-3-annotations.txt        (doc_id\tnarr;..\tsub;..[\texpl])
  dev / test: subtask-{1,2,3}-documents/*  + subtask-3-dominant-narratives.txt (doc_id\tnarr\tsub)

dev and test share the same on-disk layout (held-out documents + a
dominant-narratives label file), so both go through the non-train branch
below; only the recorded `split` field distinguishes them. The retrieval
benchmark builds its corpus from `train` and queries whichever held-out split
(`dev` or `test`) is configured.

Emits new-format Article records (name-based ids derived from the document
filename, source_language, synthetic outlet/date/author that feed the
coordination signals) plus retrieval ground truth. The hierarchy above the article is English-only; the benchmark filters
EN-CC via the ground-truth language/domain_topic fields.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from core.ids import article_id, IdRegistry, article_name_from_relpath
from core.structures import Article
from core.knowledge_base import KnowledgeBase

LANGUAGES = ["EN", "BG", "HI", "PT", "RU"]
SPLITS = ["train", "dev", "test"]
DATE_START = date(2023, 1, 1)
DATE_END = date(2023, 6, 30)
DATE_SPAN = (DATE_END - DATE_START).days

# --- synthetic outlet / author pools (verbatim from the original converter) ---
OUTLETS: dict[tuple, list] = {
    # ── English ──────────────────────────────────────────────────────────────
    ("EN", "URW"): [
        ("rt.com",           "RT — Russia Today"),
        ("sputniknews.com",  "Sputnik International"),
        ("tass.com",         "TASS Russian News Agency"),
        ("ria.ru",           "RIA Novosti"),
        ("southfront.org",   "SouthFront"),
        ("strategic-culture.org", "Strategic Culture Foundation"),
        ("thegrayzone.com",  "The Grayzone"),
        ("zerohedge.com",    "ZeroHedge"),
    ],
    ("EN", "CC"): [
        ("breitbart.com",    "Breitbart News"),
        ("dailymail.co.uk",  "Daily Mail"),
        ("wattsupwiththat.com", "Watts Up With That"),
        ("cfact.org",        "CFACT"),
        ("thefederalist.com","The Federalist"),
        ("spectator.co.uk",  "The Spectator"),
        ("zerohedge.com",    "ZeroHedge"),
        ("climatedepot.com", "Climate Depot"),
    ],
    ("EN", "Unknown"): [
        ("rt.com",           "RT — Russia Today"),
        ("sputniknews.com",  "Sputnik International"),
        ("breitbart.com",    "Breitbart News"),
        ("zerohedge.com",    "ZeroHedge"),
    ],
    # ── Bulgarian ────────────────────────────────────────────────────────────
    ("BG", "URW"): [
        ("blitz.bg",         "Blitz"),
        ("pogled.info",      "Поглед"),
        ("bgonair.bg",       "BG On Air"),
        ("trud.bg",          "Труд"),
        ("novinite.bg",      "Novinite"),
        ("dnes.bg",          "Dnes.bg"),
        ("econ.bg",          "Economy.bg"),
        ("fakti.bg",         "Fakti"),
    ],
    ("BG", "CC"): [
        ("blitz.bg",         "Blitz"),
        ("clubz.bg",         "Club Z"),
        ("dnevnik.bg",       "Дневник"),
        ("capital.bg",       "Капитал"),
        ("trud.bg",          "Труд"),
        ("bgonair.bg",       "BG On Air"),
    ],
    ("BG", "Unknown"): [
        ("blitz.bg",         "Blitz"),
        ("trud.bg",          "Труд"),
        ("novinite.bg",      "Novinite"),
    ],
    # ── Hindi ────────────────────────────────────────────────────────────────
    ("HI", "URW"): [
        ("navbharattimes.indiatimes.com", "Navbharat Times"),
        ("aajtak.in",        "Aaj Tak"),
        ("amarujala.com",    "Amar Ujala"),
        ("bhaskar.com",      "Dainik Bhaskar"),
        ("patrika.com",      "Patrika"),
        ("jagran.com",       "Dainik Jagran"),
        ("ndtv.in",          "NDTV India"),
        ("zeenews.india.com","Zee News"),
    ],
    ("HI", "CC"): [
        ("navbharattimes.indiatimes.com", "Navbharat Times"),
        ("aajtak.in",        "Aaj Tak"),
        ("bhaskar.com",      "Dainik Bhaskar"),
        ("patrika.com",      "Patrika"),
        ("jagran.com",       "Dainik Jagran"),
        ("hindi.news18.com", "News18 India"),
    ],
    ("HI", "Unknown"): [
        ("aajtak.in",        "Aaj Tak"),
        ("bhaskar.com",      "Dainik Bhaskar"),
        ("jagran.com",       "Dainik Jagran"),
    ],
    # ── Portuguese ───────────────────────────────────────────────────────────
    ("PT", "URW"): [
        ("revistaoeste.com.br",  "Revista Oeste"),
        ("jornaldabrasilia.com.br","Jornal de Brasília"),
        ("diariodopoder.com.br",  "Diário do Poder"),
        ("oantagonista.uol.com.br","O Antagonista"),
        ("gazetadopovo.com.br",   "Gazeta do Povo"),
        ("veja.abril.com.br",     "Veja"),
        ("dn.pt",                 "Diário de Notícias"),
        ("observador.pt",         "Observador"),
    ],
    ("PT", "CC"): [
        ("revistaoeste.com.br",  "Revista Oeste"),
        ("gazetadopovo.com.br",  "Gazeta do Povo"),
        ("jornaldabrasilia.com.br","Jornal de Brasília"),
        ("diariodopoder.com.br",  "Diário do Poder"),
        ("expresso.pt",           "Expresso"),
        ("observador.pt",         "Observador"),
    ],
    ("PT", "Unknown"): [
        ("gazetadopovo.com.br",  "Gazeta do Povo"),
        ("revistaoeste.com.br",  "Revista Oeste"),
        ("dn.pt",                "Diário de Notícias"),
    ],
    # ── Russian ──────────────────────────────────────────────────────────────
    ("RU", "URW"): [
        ("ria.ru",           "РИА Новости"),
        ("tass.ru",          "ТАСС"),
        ("rt.com",           "RT на русском"),
        ("rg.ru",            "Российская Газета"),
        ("life.ru",          "Life"),
        ("tsargrad.tv",      "Царьград"),
        ("vz.ru",            "Взгляд"),
        ("aif.ru",           "Аргументы и Факты"),
    ],
    ("RU", "CC"): [
        ("rg.ru",            "Российская Газета"),
        ("ria.ru",           "РИА Новости"),
        ("tass.ru",          "ТАСС"),
        ("life.ru",          "Life"),
        ("aif.ru",           "Аргументы и Факты"),
    ],
    ("RU", "Unknown"): [
        ("ria.ru",           "РИА Новости"),
        ("tass.ru",          "ТАСС"),
        ("rg.ru",            "Российская Газета"),
    ],
}

# Author name pools per language (first, last)
AUTHORS: dict[str, list] = {
    "EN": [
        "Alex Thompson", "Michael Carter", "Sarah Williams", "James Hobson",
        "Rachel Moore", "David Pearce", "Emma Sutton", "Mark Harrison",
        "Laura Bennett", "Paul Whitfield", "Anna Graham", "Robert Stone",
        "Claire Hudson", "Steven Walsh", "Diana Marsh", "Colin Fry",
        "Olivia Grant", "Nathan Cole", "Jessica Lang", "Andrew Burke",
    ],
    "BG": [
        "Иван Петров", "Мария Иванова", "Георги Стоянов", "Елена Тодорова",
        "Николай Димитров", "Светла Христова", "Борис Ангелов", "Радослава Кирова",
        "Петър Маринов", "Калина Стефанова", "Андрей Митев", "Тодор Василев",
        "Виктория Попова", "Красимир Нейков", "Деница Колева", "Огнян Тачев",
    ],
    "HI": [
        "राहुल शर्मा", "प्रिया वर्मा", "अमित कुमार", "सुनीता सिंह",
        "विकास गुप्ता", "पूजा मिश्रा", "संजय तिवारी", "अनिता यादव",
        "महेश पांडे", "कविता जोशी", "रवि चौधरी", "नेहा अग्रवाल",
        "दीपक सक्सेना", "ममता शुक्ला", "अजय त्रिपाठी", "रेखा भट्ट",
    ],
    "PT": [
        "Carlos Silva", "Ana Rodrigues", "Pedro Ferreira", "Sofia Martins",
        "João Costa", "Mariana Oliveira", "Rui Santos", "Catarina Sousa",
        "Miguel Carvalho", "Inês Pereira", "Bruno Alves", "Filipa Mendes",
        "Tiago Correia", "Beatriz Lopes", "Henrique Castro", "Luísa Nunes",
        "André Figueiredo", "Rita Cardoso", "Fernando Rocha", "Cláudia Pinto",
    ],
    "RU": [
        "Алексей Смирнов", "Ольга Новикова", "Дмитрий Козлов", "Наталья Морозова",
        "Сергей Волков", "Елена Зайцева", "Андрей Лебедев", "Татьяна Соколова",
        "Михаил Орлов", "Ирина Фёдорова", "Виктор Попов", "Людмила Кузнецова",
        "Павел Белов", "Светлана Тихонова", "Игорь Романов", "Валентина Крылова",
    ],
}


# --- deterministic synthetic metadata -------------------------------------
# A simple counter assigns each distinct doc_id a stable integer in encounter
# order; that integer seeds the per-document RNG. This replaces the previous
# md5-based seed — no hashing, just a counter.
_doc_counter: dict[str, int] = {}


def _doc_seed(doc_id: str) -> int:
    if doc_id not in _doc_counter:
        _doc_counter[doc_id] = len(_doc_counter) + 1
    return _doc_counter[doc_id]


def reset_doc_seeds() -> None:
    """Clear the counter so a fresh convert() run produces stable seeds."""
    _doc_counter.clear()


def generate_publish_date(doc_id: str) -> str:
    rng = random.Random(_doc_seed(doc_id))
    return (DATE_START + timedelta(days=rng.randint(0, DATE_SPAN))).isoformat()


def generate_outlet(doc_id: str, lang: str, topic: str) -> tuple[str, str]:
    pool = (OUTLETS.get((lang, topic)) or OUTLETS.get((lang, "Unknown"))
            or OUTLETS.get(("EN", "Unknown")))
    return random.Random(_doc_seed(doc_id + "_outlet")).choice(pool)


def generate_author(doc_id: str, lang: str) -> str:
    return random.Random(_doc_seed(doc_id + "_author")).choice(AUTHORS.get(lang, AUTHORS["EN"]))


def _domain_from_narratives(narrs) -> str | None:
    """Domain ('CC'/'URW') from the gold narrative-label prefix, which every
    language shares (e.g. 'CC: …', 'URW: …'). Returns None when the labels are
    absent or mix domains, so the caller can fall back to infer_topic."""
    doms = {n.split(":", 1)[0].strip().upper() for n in (narrs or []) if ":" in n}
    doms &= {"CC", "URW"}
    return next(iter(doms)) if len(doms) == 1 else None


def infer_topic(doc_id: str) -> str:
    u = doc_id.upper()
    if "_UA_" in u or u.startswith("UA"):
        return "URW"
    if "_CC_" in u or u.startswith("CC"):
        return "CC"
    return "Unknown"


def extract_title(content: str) -> str:
    for line in content.splitlines():
        if line.strip():
            return line.strip()[:200]
    return "Untitled"


def metadata_for(doc_id: str, content: str, lang: str,
                 narratives: list | None = None) -> dict:
    """Synthetic article metadata for one PolyNarrative document.

    Returns the title, author and a ``metadata`` sub-dict (outlet, language,
    publish date, domain topic) so the claim-detection generator can attach it
    to the per-article KB record without re-deriving any of it. Deterministic
    for a given ``doc_id`` via the counter-based seed.
    """
    topic = _domain_from_narratives(narratives) or infer_topic(doc_id)
    domain, outlet_name = generate_outlet(doc_id, lang, topic)
    return {
        "title":  extract_title(content),
        "author": generate_author(doc_id, lang),
        "metadata": {
            "source_domain":   domain,
            "outlet":          outlet_name,
            "source_language": lang,
            "published_at":    generate_publish_date(doc_id),
            "domain_topic":    topic,
        },
    }


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# --- annotation parsers ----------------------------------------------------
def parse_train_annotations(ann_path: Path) -> dict:
    result: dict = {}
    if not ann_path.exists():
        return result
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        doc_id, raw_narr, raw_sub = parts[0], parts[1], parts[2]
        narr = [n.strip() for n in raw_narr.split(";") if n.strip()]
        sub = [s.strip() for s in raw_sub.split(";") if s.strip()]
        m = max(len(narr), len(sub))
        narr += ["none"] * (m - len(narr))
        sub += ["none"] * (m - len(sub))
        result[doc_id] = {"narratives": narr, "sub_narratives": sub}
    return result


def parse_dev_annotations(ann_path: Path) -> dict:
    result: dict = {}
    if not ann_path.exists():
        return result
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        result[parts[0]] = {"narratives": [parts[1].strip()], "sub_narratives": [parts[2].strip()]}
    return result


def find_documents(split_lang_dir: Path, split: str) -> dict:
    docs: dict = {}
    subdirs = (["raw-documents"] if split == "train"
               else ["subtask-1-documents", "subtask-2-documents", "subtask-3-documents"])
    for sub in subdirs:
        d = split_lang_dir / sub
        if d.is_dir():
            for f in d.glob("*.txt"):
                docs.setdefault(f.name, f)
    return docs


# --- converter -------------------------------------------------------------
def convert(src: Path, out_root: Path,
            languages: list | None = None, splits: list | None = None) -> int:
    languages = languages or LANGUAGES
    splits = splits or SPLITS
    kb = KnowledgeBase(out_root)
    gt_dir = out_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    ground_truth: dict = {}
    total = 0
    registry = IdRegistry()          # per-run registry: IDs stay stable within one convert() call
    reset_doc_seeds()                # counter-based seeds stable within one convert() call
    for split in splits:
        for lang in languages:
            sld = src / split / lang
            if not sld.is_dir():
                continue
            ann = (parse_train_annotations(sld / "subtask-3-annotations.txt") if split == "train"
                   else parse_dev_annotations(sld / "subtask-3-dominant-narratives.txt"))
            for doc_id, path in sorted(find_documents(sld, split).items()):
                content = _read(path)
                if not content.strip():
                    continue
                url = f"polynarrative://{lang}/{split}/{doc_id}"
                # Canonical article_name from the path RELATIVE TO src — identical
                # to what the claim-detection generator derives from
                # data/PolyNarrative, so ground-truth keys join to the per-article
                # KB records. Includes the subdir (raw-documents / subtask-*).
                rel = path.relative_to(src)
                aid = article_id(article_name_from_relpath(rel), registry=registry)
                entry = ann.get(doc_id) or ann.get(Path(doc_id).stem)
                # Domain from the gold narrative-label prefix (CC:/URW:) — present in
                # every language — so non-EN articles are tagged correctly. The old
                # infer_topic(doc_id) read an EN-only id pattern and left every non-EN
                # article as 'Unknown'. Fall back to infer_topic only when no narrative
                # prefix is available.
                topic = _domain_from_narratives(entry.get("narratives") if entry else None) \
                    or infer_topic(doc_id)
                domain, _name = generate_outlet(doc_id, lang, topic)
                kb.save_article(Article(
                    id=aid, url=url, source_domain=domain, title=extract_title(content),
                    content=content, source_language=lang,
                    published_at=generate_publish_date(doc_id),
                    author=generate_author(doc_id, lang)))
                total += 1
                if entry:
                    ground_truth[aid] = {
                        "poly_doc_id": doc_id, "language": lang, "split": split,
                        "domain_topic": topic, **entry}
            print(f"  [{split}/{lang}] {total} articles so far")

    (gt_dir / "annotations.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    by_nar: dict = defaultdict(lambda: {"article_ids": [], "languages": set()})
    for aid, a in ground_truth.items():
        for narr in a["narratives"]:
            by_nar[narr]["article_ids"].append(aid)
            by_nar[narr]["languages"].add(a["language"])
    by_nar_clean = {n: {"article_ids": sorted(set(v["article_ids"])),
                        "languages": sorted(v["languages"])}
                    for n, v in sorted(by_nar.items())}
    (gt_dir / "annotations_by_narrative.json").write_text(
        json.dumps(by_nar_clean, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PolyNarrative: {total} articles, {len(ground_truth)} annotated, "
          f"{len(by_nar_clean)} narratives -> {out_root}")
    return total


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/PolyNarrative"))
    p.add_argument("--out", type=Path, default=Path("knowledge"))
    a = p.parse_args()
    convert(a.src, a.out)
