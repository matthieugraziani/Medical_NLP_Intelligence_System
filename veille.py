"""
veille.py — Veille IA Médicale hebdomadaire
=============================================
Améliorations v2 :
  - Gestion robuste des erreurs (LLM, SMTP, flux RSS)
  - Logging structuré (fichier + console)
  - Déduplication globale des articles cross-sources
  - SMTP avec context manager (connexion toujours fermée)
  - Compatibilité Unicode PDF (remplacement ★/☆ par ASCII)
  - URLs ClinicalTrials v2 (nouvelle API)
  - Arrêt propre du scheduler (CTRL+C)
  - LLM optionnel avec fallback extractif
  - Validation des variables d'environnement au démarrage
"""

import logging
import os
import smtplib
import time
import datetime
import schedule
import feedparser
import pandas as pd
from fpdf import FPDF
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

# ================================================================
#  LOGGING STRUCTURÉ
# ================================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"veille_{datetime.date.today():%Y-%m-%d}.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("veille")

# ================================================================
#  CONFIGURATION CENTRALE
# ================================================================
SOURCES = {
    # ── PubMed ─────────────────────────────────────────────────
    "PubMed": [
        {
            "theme": "Radiologie / Imagerie médicale",
            "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term=radiology+medical+imaging+AI&limit=20&sort=date",
        },
        {
            "theme": "Oncologie",
            "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term=oncology+deep+learning+diagnosis&limit=20&sort=date",
        },
        {
            "theme": "IA medicale generale",
            "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term=artificial+intelligence+clinical+medicine&limit=20&sort=date",
        },
        {
            "theme": "Neurologie",
            "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/?term=neurology+neuroimaging+machine+learning&limit=20&sort=date",
        },
        {
            "theme": "Tumeurs cerebrales & IRM",
            "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/1P5xyB4cY7nR?term=brain+tumor+MRI&limit=20&sort=date",
        },
    ],

    # ── ArXiv ──────────────────────────────────────────────────
    "ArXiv": [
        {"theme": "Traitement image medicale (eess.IV)", "url": "https://arxiv.org/rss/eess.IV"},
        {"theme": "Machine Learning medical (cs.LG)",   "url": "https://arxiv.org/rss/cs.LG"},
        {"theme": "Computer Vision medical (cs.CV)",    "url": "https://arxiv.org/rss/cs.CV"},
    ],

    # ── Google Scholar ─────────────────────────────────────────
    # ⚠️  Remplacez ces URLs par vos propres alertes Scholar RSS.
    # Format attendu : https://scholar.google.com/scholar_alerts?update_op=query_edit&...
    "Google Scholar": [
        {
            "theme": "Radiologie IA - Scholar",
            "url": "",   # TODO : coller ici l'URL de votre alerte RSS
        },
        {
            "theme": "Oncologie IA - Scholar",
            "url": "",   # TODO : coller ici l'URL de votre alerte RSS
        },
    ],

    # ── ClinicalTrials.gov (API v2) ────────────────────────────
    "ClinicalTrials": [
        {
            "theme": "Essais IA & Radiologie",
            "url": (
                "https://clinicaltrials.gov/api/v2/studies"
                "?format=rss&query.term=artificial+intelligence+radiology"
                "&filter.overallStatus=RECRUITING&pageSize=20"
            ),
        },
        {
            "theme": "Essais IA & Oncologie",
            "url": (
                "https://clinicaltrials.gov/api/v2/studies"
                "?format=rss&query.term=deep+learning+oncology"
                "&filter.overallStatus=RECRUITING&pageSize=20"
            ),
        },
        {
            "theme": "Essais IA & Neurologie",
            "url": (
                "https://clinicaltrials.gov/api/v2/studies"
                "?format=rss&query.term=AI+neurology+brain"
                "&filter.overallStatus=RECRUITING&pageSize=20"
            ),
        },
    ],
}

# Mots-cles de priorite (score +1 par mot, plafond 5)
HIGH_PRIORITY_KEYWORDS = [
    "AI", "deep learning", "neural network", "transformer",
    "segmentation", "detection", "glioblastoma", "FDA", "CE mark",
    "randomized", "multicenter", "prospective", "benchmark",
]

# ================================================================
#  VARIABLES D'ENVIRONNEMENT
# ================================================================
load_dotenv()

SMTP_EMAIL    = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SERVER   = os.getenv("SMTP_SERVER", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SLACK_TOKEN   = os.getenv("SLACK_TOKEN", "")
GPT4ALL_MODEL = os.getenv("GPT4ALL_MODEL", "")   # nom du modèle, ex: "mistral-7b-instruct-v0.1.Q4_0.gguf"
GPT4ALL_PATH  = os.getenv("GPT4ALL_PATH", "")    # dossier contenant le modèle


def _validate_env():
    """Avertit si des variables critiques sont manquantes."""
    missing = []
    for var in ("SMTP_EMAIL", "SMTP_PASSWORD", "SMTP_SERVER", "SLACK_TOKEN"):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        log.warning("Variables d'environnement manquantes : %s", ", ".join(missing))


_validate_env()

# ================================================================
#  LLM LOCAL (GPT4All) — optionnel
# ================================================================
_gpt_model = None

def _load_llm():
    """Charge le modèle LLM une seule fois, retourne None si indisponible."""
    global _gpt_model
    if _gpt_model is not None:
        return _gpt_model
    if not GPT4ALL_MODEL or not GPT4ALL_PATH:
        log.warning("GPT4ALL_MODEL ou GPT4ALL_PATH non defini — resumes extractifs utilises.")
        return None
    try:
        from gpt4all import GPT4All
        _gpt_model = GPT4All(model_name=GPT4ALL_MODEL, model_path=GPT4ALL_PATH)
        log.info("Modele LLM charge : %s", GPT4ALL_MODEL)
    except Exception as exc:
        log.error("Impossible de charger le modele LLM : %s", exc)
        _gpt_model = None
    return _gpt_model


def _extractive_summary(text: str, n_sentences: int = 2) -> str:
    """Fallback : retourne les n premières phrases du texte."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return ". ".join(sentences[:n_sentences]) + ("." if sentences else "")


def summarize_with_llm(text: str, max_tokens: int = 150) -> str:
    """Résume le texte via LLM ou par extraction si le LLM est indisponible."""
    model = _load_llm()
    if model is None:
        return _extractive_summary(text)
    prompt = (
        "Resume ce texte pour un medecin specialiste en 2-3 phrases "
        "claires et concises :\n" + text[:2000]
    )
    try:
        return model.generate(prompt, max_tokens=max_tokens)
    except Exception as exc:
        log.warning("Erreur LLM lors du resume : %s", exc)
        return _extractive_summary(text)


def compute_priority(title: str, summary: str) -> int:
    text = (title + " " + summary).lower()
    score = 1 + sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw.lower() in text)
    return min(score, 5)


# ================================================================
#  AGENT 1 — TechWatch
# ================================================================
def techwatch_agent() -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    global_seen_links: set[str] = set()   # déduplication cross-sources

    for source_name, feeds in SOURCES.items():
        articles = []

        for feed_info in feeds:
            theme = feed_info["theme"]
            url   = feed_info.get("url", "").strip()

            if not url:
                log.warning("[%s] URL vide pour le theme '%s' — ignoré.", source_name, theme)
                continue

            try:
                feed = feedparser.parse(url)
            except Exception as exc:
                log.error("[%s] Erreur parsing %s : %s", source_name, url, exc)
                continue

            if feed.bozo:
                log.warning("[%s] Flux malformé (%s) : %s", source_name, url, feed.bozo_exception)

            for entry in feed.entries[:20]:
                link = getattr(entry, "link", "").strip()
                if not link or link in global_seen_links:
                    continue
                global_seen_links.add(link)

                title = getattr(entry, "title", "Sans titre")
                raw   = getattr(entry, "summary", "") or title
                summary  = summarize_with_llm(raw)
                priority = compute_priority(title, raw)

                articles.append({
                    "source":   source_name,
                    "theme":    theme,
                    "title":    title,
                    "link":     link,
                    "date":     getattr(entry, "published", "N/A"),
                    "summary":  summary,
                    "priority": priority,
                })

        df = pd.DataFrame(articles)
        if not df.empty:
            df = df.sort_values(by="priority", ascending=False)

        safe_name = source_name.replace(" ", "_").replace("/", "-")
        df.to_csv(f"techwatch_{safe_name}.csv", index=False, encoding="utf-8-sig")
        results[source_name] = df
        log.info("[TechWatch] %s — %d articles collectes", source_name, len(df))

    return results


# ================================================================
#  AGENT 2 — MarketWatch
# ================================================================
def marketwatch_agent() -> pd.DataFrame:
    competitors = [
        {"name": "BrainScanAI",  "status": "market",  "funding": "5M€", "regulation": "FDA approved", "priority": 3},
        {"name": "NeuroVision",  "status": "preprod", "funding": "2M€", "regulation": "CE",           "priority": 2},
        {"name": "NeuroScanPro", "status": "R&D",     "funding": "1M€", "regulation": "pending",      "priority": 1},
    ]
    df = pd.DataFrame(competitors).sort_values(by="priority", ascending=False)
    df.to_csv("marketwatch_pro.csv", index=False, encoding="utf-8-sig")
    log.info("[MarketWatch] %d concurrents traites", len(df))
    return df


# ================================================================
#  AGENT 3 — PublicWatch
# ================================================================
def publicwatch_agent() -> pd.DataFrame:
    url = "https://www.boamp.fr/rss"
    ao_list = []

    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            log.warning("[PublicWatch] Flux BOAMP malformé : %s", feed.bozo_exception)

        for entry in feed.entries[:10]:
            title    = getattr(entry, "title", "Sans titre")
            priority = (
                3 if any(kw.lower() in title.lower()
                         for kw in ["IA", "imagerie", "intelligence artificielle"])
                else 1
            )
            ao_list.append({
                "title":    title,
                "link":     getattr(entry, "link", ""),
                "date":     getattr(entry, "published", "N/A"),
                "priority": priority,
            })
    except Exception as exc:
        log.error("[PublicWatch] Erreur flux BOAMP : %s", exc)

    df = pd.DataFrame(ao_list)
    if not df.empty:
        df = df.sort_values(by="priority", ascending=False)
    df.to_csv("publicwatch_pro.csv", index=False, encoding="utf-8-sig")
    log.info("[PublicWatch] %d appels d'offres collectes", len(df))
    return df


# ================================================================
#  REPORTGEN — PDF par source
# ================================================================
SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "PubMed":         (41,  128, 185),
    "ArXiv":          (192, 57,  43),
    "Google Scholar": (39,  174, 96),
    "ClinicalTrials": (142, 68,  173),
}
DEFAULT_COLOR = (52, 73, 94)

# Remplacement des caractères Unicode non supportés par FPDF Latin-1
_STAR_FILLED = "*"
_STAR_EMPTY  = "-"


def _priority_str(priority: int) -> str:
    """Représentation ASCII de la priorité (évite UnicodeEncodeError)."""
    p = int(priority)
    return _STAR_FILLED * p + _STAR_EMPTY * (5 - p)


def _set_source_color(pdf: FPDF, source_name: str):
    r, g, b = SOURCE_COLORS.get(source_name, DEFAULT_COLOR)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)


def _reset_color(pdf: FPDF):
    pdf.set_text_color(0, 0, 0)


def _safe_text(text: str) -> str:
    """Encode le texte en Latin-1 en remplaçant les caractères inconnus."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _source_header(pdf: FPDF, source_name: str, count: int):
    _set_source_color(pdf, source_name)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, _safe_text(f"  {source_name}  ({count} articles)"), ln=True, fill=True)
    _reset_color(pdf)
    pdf.ln(2)


def _theme_subheader(pdf: FPDF, theme: str):
    pdf.set_fill_color(235, 235, 235)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, _safe_text(f"   > {theme}"), ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    pdf.ln(1)


def _article_block(pdf: FPDF, row: pd.Series, idx: int):
    stars = _priority_str(row["priority"])
    pdf.set_font("Arial", "B", 10)
    pdf.multi_cell(0, 5, _safe_text(f"{idx}. {row['title']}"))
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 4, _safe_text(f"   Date : {row.get('date', 'N/A')}   Priorite : {stars}"))
    pdf.multi_cell(0, 4, _safe_text(f"   Resume : {row['summary']}"))
    pdf.set_text_color(0, 0, 200)
    pdf.multi_cell(0, 4, _safe_text(f"   {row['link']}"))
    _reset_color(pdf)
    pdf.ln(3)


def reportgen_agent(
    tech_results: dict[str, pd.DataFrame],
    df_market: pd.DataFrame,
    df_public: pd.DataFrame,
) -> str:
    today = datetime.date.today().strftime("%d-%m-%Y")
    total = sum(len(df) for df in tech_results.values())

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page de garde ────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.ln(40)
    pdf.cell(0, 14, "Rapport Veille IA Medicale", ln=True, align="C")
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, "Radiologie - Oncologie - IA medicale - Neurologie", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Date : {today}", ln=True, align="C")
    pdf.cell(0, 8, f"Articles analyses : {total}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Sources couvertes :", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    for src in tech_results:
        r, g, b = SOURCE_COLORS.get(src, DEFAULT_COLOR)
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(60, 7, _safe_text(f"  {src}"), fill=True)
        _reset_color(pdf)
        pdf.cell(5)
    pdf.ln(15)

    # ── Section 1 : Veille Technologique ─────────────────────
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "1. Veille Technologique - par source", ln=True)
    pdf.ln(2)

    for source_name, df in tech_results.items():
        if df.empty:
            continue
        _source_header(pdf, source_name, len(df))
        for theme, group in df.groupby("theme"):
            _theme_subheader(pdf, str(theme))
            for idx, (_, row) in enumerate(group.head(8).iterrows(), 1):
                _article_block(pdf, row, idx)
        pdf.ln(4)

    # ── Section 2 : Concurrence ──────────────────────────────
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "2. Veille Concurrentielle", ln=True)
    pdf.ln(3)
    for _, row in df_market.iterrows():
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, _safe_text(f"- {row['name']}"), ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, _safe_text(
            f"  Statut : {row['status']}  |  Financement : {row['funding']}"
            f"  |  Regulation : {row['regulation']}  |  Priorite : {row['priority']}/3"
        ))
        pdf.ln(2)

    # ── Section 3 : Marchés Publics ─────────────────────────
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "3. Veille Marches Publics", ln=True)
    pdf.ln(3)
    for _, row in df_public.iterrows():
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, _safe_text(f"- {row['title']}"))
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, _safe_text(f"  Date : {row['date']}  |  Priorite : {row['priority']}/3"))
        pdf.set_text_color(0, 0, 200)
        pdf.multi_cell(0, 5, _safe_text(f"  {row['link']}"))
        _reset_color(pdf)
        pdf.ln(2)

    # ── Export ───────────────────────────────────────────────
    reports_dir = "historique_reports"
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"weekly_report_{today}.pdf")
    pdf.output(pdf_path)
    log.info("PDF genere -> %s", pdf_path)
    return pdf_path


# ================================================================
#  ALERTES
# ================================================================
def send_email_alert(pdf_path: str, receiver: str):
    if not all([SMTP_EMAIL, SMTP_PASSWORD, SMTP_SERVER]):
        log.error("Configuration SMTP incomplete — email non envoye.")
        return

    msg = MIMEMultipart()
    msg["From"]    = SMTP_EMAIL
    msg["To"]      = receiver
    msg["Subject"] = f"Rapport veille IA medicale - {datetime.date.today():%d/%m/%Y}"
    msg.attach(MIMEText(
        "Bonjour,\n\n"
        "Veuillez trouver en piece jointe le rapport hebdomadaire de veille IA medicale.\n\n"
        "Sources couvertes : PubMed, ArXiv, Google Scholar, ClinicalTrials.gov\n"
        "Domaines : Radiologie, Oncologie, IA medicale generale, Neurologie\n\n"
        "Cordialement.",
        "plain",
    ))

    try:
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(pdf_path)}",
        )
        msg.attach(part)
    except OSError as exc:
        log.error("Impossible de lire le PDF pour l'email : %s", exc)
        return

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, receiver, msg.as_string())
        log.info("Email envoye a %s", receiver)
    except smtplib.SMTPException as exc:
        log.error("Erreur SMTP : %s", exc)


def send_slack_alert(pdf_path: str, channel: str = "#general"):
    if not SLACK_TOKEN:
        log.warning("SLACK_TOKEN non defini — alerte Slack ignoree.")
        return
    try:
        from slack_sdk import WebClient
        client = WebClient(token=SLACK_TOKEN)
        client.files_upload(channels=channel, file=pdf_path, title="Rapport veille IA medicale")
        log.info("Rapport envoye sur Slack (%s)", channel)
    except Exception as exc:
        log.error("Erreur Slack : %s", exc)


# ================================================================
#  ORCHESTRATEUR
# ================================================================
def run_weekly_veille():
    log.info("=" * 55)
    log.info("Veille hebdomadaire — %s", datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))
    log.info("=" * 55)

    log.info("[1/3] Collecte articles (PubMed · ArXiv · Scholar · ClinicalTrials)...")
    tech_results = techwatch_agent()

    log.info("[2/3] Veille concurrentielle & marches publics...")
    df_market = marketwatch_agent()
    df_public = publicwatch_agent()

    log.info("[3/3] Generation rapport PDF...")
    pdf_path = reportgen_agent(tech_results, df_market, df_public)

    send_email_alert(pdf_path, receiver="destinataire@example.com")  # ← modifiez ici
    send_slack_alert(pdf_path)

    log.info("Veille terminee avec succes.")


# ================================================================
#  POINT D'ENTRÉE
# ================================================================
if __name__ == "__main__":
    # Pour un test immediat, decommentez la ligne suivante :
    # run_weekly_veille()

    schedule.every().monday.at("12:00").do(run_weekly_veille)
    log.info("Scheduler actif — prochain lancement : Vendredi 12:00")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Arret demande par l'utilisateur (CTRL+C).")
