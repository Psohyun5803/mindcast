"""Utility functions for mindcast-online pipeline.

Sections:
  0  Common primitives  — channelish(), clean_counter(), label()
  1  Datasource         — load_table() (local mysqldump | HuggingFace)
  2  Prep               — clean_title(), hashtags(), prep_month()
  3  Online tracker     — OnlineTracker class, post_entities()
  4  Visualization      — set_korean_font(), viz_month()
  5  Export HTML        — build_month_data(), html_export(), TEMPLATE
  6  HF upload          — build_staging(), pii_guard(), upload_to_hf()
"""
# ── 경로 설정 (config/ 폴더를 sys.path에 추가) ─────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
# ────────────────────────────────────────────────────────────────────────────
import gzip, hashlib, json, math, pickle, re, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.cluster import HDBSCAN

from event_online_config import (
    ROOT, DATA, OUT, FIGS, STAGE,
    HF_REPO, DUMP, DEVICE, FONT,
    SCHEMA, NUMERIC, DATETIME, UNESCAPE,
    EMBED_MODEL, EMBED_BATCH, HASHTAG_BOILER,
    STOP, P_OFFLINE, P_ONLINE,
    FONT_CANDIDATES, FONT_NAMES,
    MONTHS, GAP_ACTIVE, GAP_DORMANT, CAP, COH_THR, FRAC,
    UPLOAD_TABLES, FORBIDDEN_COLS, HF_CARD,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Section 0 — Common primitives
# =============================================================================

def channelish(w):
    """True if a token is a channel/format tag rather than an event entity."""
    return (w in STOP or "TBC" in w or "MBC" in w or
            w.endswith(("MBC","방송","엠비씨","뉴스","TV","tv","생중계","라이브",
                        "직캠","브리핑","풀영상","다시보기")) or
            w.startswith(("MBC","KBS","SBS","TBC","채널A")))


def clean_counter(c):
    """Drop channel/format tokens from an entity/keyphrase Counter."""
    return Counter({w: n for w, n in c.items() if not channelish(w)})


def label(track):
    """Human-readable event label = top entities (fallback to keyphrases)."""
    ents = clean_counter(track["ents"])
    keys = clean_counter(track["keys"])
    top = [e for e, _ in ents.most_common(4)]
    if len(top) < 3:
        top += [k for k, _ in keys.most_common(4)]
    return " · ".join(list(dict.fromkeys(top))[:4])


# =============================================================================
# Section 1 — Datasource
# =============================================================================

_ESC = re.compile(r"\\[nrt0Z'\"\\]")


def _parse_values(s):
    """Yield tuples from a '(...),(...)' mysqldump VALUES body."""
    i, n = 0, len(s)
    while i < n:
        if s[i] != "(":
            i += 1; continue
        i += 1
        fields, cur, in_str = [], [], False
        while i < n:
            c = s[i]
            if in_str:
                if c == "\\":
                    cur.append(s[i:i+2]); i += 2; continue
                if c == "'":
                    in_str = False; i += 1; continue
                cur.append(c); i += 1; continue
            else:
                if c == "'":
                    in_str = True; cur.append("\x00STR\x00"); i += 1; continue
                if c == ",":
                    fields.append("".join(cur)); cur = []; i += 1; continue
                if c == ")":
                    fields.append("".join(cur)); i += 1
                    yield [_conv(f) for f in fields]
                    break
                cur.append(c); i += 1; continue


def _conv(tok):
    if tok.startswith("\x00STR\x00"):
        return _ESC.sub(lambda m: UNESCAPE[m.group(0)], tok[len("\x00STR\x00"):])
    tok = tok.strip()
    return None if tok == "NULL" else tok


def _extract_local(table):
    if not DUMP or not Path(DUMP).exists():
        raise FileNotFoundError(
            "source='local' needs the mysqldump path in $MINDCAST_DUMP "
            f"(current: {DUMP!r}). Set it, e.g. "
            "export MINDCAST_DUMP=/path/to/mindcast-prod-YYYYMMDD.sql.gz "
            "— or use source='hf' instead.")
    cols = SCHEMA[table]
    head = f"INSERT INTO `{table}` VALUES"
    rows, collecting, buf = [], False, []
    with gzip.open(DUMP, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not collecting:
                if line.startswith(head):
                    collecting = True; buf = []
                    rest = line[len(head):]
                    if rest.strip(): buf.append(rest)
                continue
            buf.append(line)
            if line.rstrip().endswith(";"):
                body = "".join(buf).rstrip().rstrip(";")
                for t in _parse_values(body):
                    if len(t) == len(cols): rows.append(t)
                collecting = False
    return pd.DataFrame(rows, columns=cols)


def _extract_hf(table):
    uri = f"hf://datasets/{HF_REPO}/{table}.parquet"
    try:
        return pd.read_parquet(uri)
    except Exception:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_REPO, filename=f"{table}.parquet",
                            repo_type="dataset")
        return pd.read_parquet(p)


def _hash_author(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


def _normalize(table, df, hash_author=True):
    for c in NUMERIC.get(table, []):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in DATETIME.get(table, []):
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    if table == "video_comment" and hash_author and "author" in df.columns:
        df["author"] = df["author"].map(_hash_author)
    return df


def load_table(table, source="hf", hash_author=True):
    """Return a typed DataFrame for `table` from `source` ('hf'|'local').
    Output schema/typing is identical across sources."""
    if table not in SCHEMA:
        raise ValueError(f"table not in allow-list (news tables only): {table}")
    df = _extract_local(table) if source == "local" else _extract_hf(table)
    return _normalize(table, df, hash_author=hash_author)


# =============================================================================
# Section 2 — Prep (text cleaning + embedding)
# =============================================================================

BRACKET   = re.compile(r"[\[\(][^\]\)]*[\]\)]")
TAGWORDS  = re.compile(
    r"(뉴스데스크|뉴스투데이|뉴스\.?zip|뉴스ZIP|뉴스꾹|오늘\s*이\s*뉴스|"
    r"자막뉴스|LIVE|Shorts|#Shorts|풀영상|다시보기|MBC\s*뉴스|MBCNEWS|"
    r"SBS\s*뉴스|KBS\s*News|앵커|현장)", re.I)
DATECH    = re.compile(r"\d{4}[.\-/]\s?\d{1,2}[.\-/]\s?\d{1,2}")
MULTISPACE = re.compile(r"\s+")
HASHTAG   = re.compile(r"#([^\s#·ㅤ,]+)")


def clean_title(t):
    if not t: return ""
    s = BRACKET.sub(" ", t)
    s = DATECH.sub(" ", s)
    s = TAGWORDS.sub(" ", s)
    s = re.sub(r"[\"'""''·|/]+", " ", s)
    s = re.sub(r"[^\w가-힣A-Za-z0-9 %]", " ", s)
    return MULTISPACE.sub(" ", s).strip()


def hashtags(desc):
    if not desc: return []
    out = []
    for h in HASHTAG.findall(desc):
        h = h.strip("ㅤ ").replace("_", "")
        if 1 < len(h) <= 20 and h not in HASHTAG_BOILER and not h.isdigit():
            out.append(h)
    seen, res = set(), []
    for h in out:
        if h not in seen: seen.add(h); res.append(h)
    return res[:12]


def prep_month(month, source="hf", device=None):
    """Clean titles, extract hashtags/keyphrases, embed with ko-sroberta.

    Writes:
      data/posts_<month>.parquet
      data/emb_<month>.npy
    """
    from kiwipiepy import Kiwi
    from sentence_transformers import SentenceTransformer
    import torch

    kiwi = Kiwi()
    v = load_table("video_video", source=source)
    v = v[v.created_at.dt.to_period("M").astype(str) == month].copy()
    v["day"] = v.created_at.dt.normalize()
    v["clean_title"] = v.title.map(clean_title)
    v["tags"] = v.description.map(hashtags)
    v = v[v.clean_title.str.len() >= 4].reset_index(drop=True)

    keys = []
    for toks in kiwi.tokenize(v.clean_title.tolist()):
        ks = [t.form for t in toks if t.tag in ("NNP","NNG") and len(t.form) >= 2]
        keys.append(ks)
    v["keyphrases"] = keys

    emb_text = (v.clean_title + " " + v.tags.map(lambda x: " ".join(x))).tolist()
    dev = device or DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prep] embedding on device={dev}")
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    emb = model.encode(emb_text, batch_size=EMBED_BATCH, normalize_embeddings=True,
                       show_progress_bar=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / f"emb_{month}.npy", emb.astype(np.float32))
    v[["id","video_id","day","created_at","clean_title","title","tags","keyphrases",
       "comment_count","view_count","like_count","channel_id"]].to_parquet(
        DATA / f"posts_{month}.parquet")
    print(f"posts: {len(v)}  | days: {v.day.nunique()}"
          f" | emb: {emb.shape}"
          f" | avg tags: {v.tags.map(len).mean():.2f}"
          f" | avg keys: {v.keyphrases.map(len).mean():.2f}")
    for _, r in v.head(5).iterrows():
        print(f"  {r.day.date()} | {r.clean_title[:55]} | tags={r.tags[:5]}")


# =============================================================================
# Section 3 — Online Tracker
# =============================================================================

def post_entities(row):
    return [w for w in list(row.tags) + list(row.keyphrases) if not channelish(w)]


class OnlineTracker:
    def __init__(self, params=None):
        self.P = {**P_ONLINE, **(params or {})}
        self.tracks   = {}
        self.next_tid = 0
        self.df       = Counter()
        self.Nseen    = 0
        self.residual = []
        self.assigned = set()

    def idf(self, w):
        return math.log((self.Nseen + 1) / (self.df.get(w, 0) + 1)) + 1e-6

    def prof(self, tk):
        ent = [w for w, _ in tk["ents"].most_common(self.P["topN"])]
        return set(ent), (sum(self.idf(w) for w in ent) or 1e-9)

    def wovlp(self, pset, tk):
        ent, sT = self.prof(tk)
        if not pset or not ent: return 0.0
        inter = sum(self.idf(w) for w in pset & ent)
        sP = sum(self.idf(w) for w in pset) or 1e-9
        return inter / min(sP, sT)

    def update_idf(self, rows):
        for r in rows:
            self.Nseen += 1
            for w in set(post_entities(r)):
                self.df[w] += 1

    def candidates(self, today):
        return [tid for tid, tk in self.tracks.items()
                if (today - tk["last_seen"]).days <= self.P["archive_gap"]]

    def assign_post(self, idx, day, e, ents, comments):
        cand = self.candidates(day)
        pset = set(ents); scored = []
        for tid in cand:
            tk = self.tracks[tid]
            s = self.P["w_cos"]*float(e @ tk["cen"]) + self.P["w_ent"]*self.wovlp(pset, tk)
            if s >= self.P["tau"]: scored.append((s, tid))
        scored.sort(reverse=True)
        chosen = scored[:self.P["K"]]
        if not chosen:
            self.residual.append(dict(idx=idx, day=day, emb=e, ents=ents, comments=comments))
            return False
        self.assigned.add(idx)
        for s, tid in chosen:
            self.update_track(tid, day, e, ents, comments, score=s, idx=idx)
        return True

    def update_track(self, tid, day, e, ents, comments, score, idx=None):
        tk = self.tracks[tid]
        a  = self.P["ema"]
        tk["cen"] = a*e + (1-a)*tk["cen"]
        tk["cen"] /= np.linalg.norm(tk["cen"]) + 1e-9
        tk["ents"].update(ents)
        cell = tk["days"].setdefault(day, {"n":0,"comments":0,"posts":[]})
        cell["n"] += 1; cell["comments"] += int(comments)
        if idx is not None: cell["posts"].append(int(idx))
        if day > tk["last_seen"]:
            if tk["status"] in ("dormant", "dead"):
                tk["status"] = "revived"; tk.setdefault("revivals", []).append(day)
            tk["last_seen"] = day
        tk["birth"] = min(tk["birth"], day)

    def ent_overlap(self, ea, tk, k=8):
        sa = {w for w, _ in ea.most_common(k)}
        ent, _ = self.prof(tk)
        if not sa or not ent: return 0.0
        inter = sum(self.idf(w) for w in sa & ent)
        denom = min(sum(self.idf(w) for w in sa), sum(self.idf(w) for w in ent)) or 1e-9
        return inter / denom

    def new_from_residual(self, today):
        lo = today - pd.Timedelta(days=self.P["residual_window"]-1)
        recent = [r for r in self.residual if r["day"] >= lo]
        self.residual = [r for r in self.residual if r["day"] >= lo]
        if len(recent) < self.P["min_cluster_size"]:
            return 0
        X = np.stack([r["emb"] for r in recent])
        lab = HDBSCAN(
            min_cluster_size=self.P["min_cluster_size"],
            min_samples=self.P["min_samples"],
            metric="euclidean",
            cluster_selection_method="leaf",
        ).fit_predict(X)
        created = 0; used = set()
        for c in sorted(set(lab)):
            if c == -1: continue
            members = [recent[i] for i in range(len(recent)) if lab[i] == c]
            cen = np.mean([m["emb"] for m in members], 0)
            cen /= np.linalg.norm(cen) + 1e-9
            ents = Counter()
            for m in members: ents.update(m["ents"])

            best, bs = None, 0.0
            for tid in self.candidates(today):
                tk = self.tracks[tid]
                sc = 0.5*float(cen @ tk["cen"]) + 0.5*self.ent_overlap(ents, tk)
                if sc > bs: bs, best = sc, tid
            if best is not None and bs >= self.P["merge_thr"]:
                for m in sorted(members, key=lambda x: x["day"]):
                    self.update_track(best, m["day"], m["emb"], m["ents"],
                                      m["comments"], score=bs, idx=m["idx"])
                    self.assigned.add(m["idx"])
                used.update(m["idx"] for m in members); continue

            days = {}
            for m in members:
                self.assigned.add(m["idx"])
                d = m["day"]; cell = days.setdefault(d, {"n":0,"comments":0,"posts":[]})
                cell["n"] += 1; cell["comments"] += int(m["comments"])
                cell["posts"].append(int(m["idx"]))
            tid = self.next_tid; self.next_tid += 1
            self.tracks[tid] = dict(cen=cen, ents=ents, keys=Counter(), days=days,
                birth=min(days), last_seen=max(days),
                status="new", confirmed=False, revivals=[])
            created += 1
            used.update(m["idx"] for m in members)
        self.residual = [r for r in self.residual if r["idx"] not in used]
        return created

    def consolidate(self, today):
        """Causal daily merge of co-active twin tracks (same event split at birth)."""
        tids = [t for t, tk in self.tracks.items()
                if (today - tk["last_seen"]).days <= self.P["dormant_gap"]]
        parent = {t: t for t in tids}
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                a, b = self.tracks[tids[i]], self.tracks[tids[j]]
                da, db = set(a["days"]), set(b["days"])
                if min((abs((x-y).days) for x in da for y in db), default=99) > self.P["cons_gap"]:
                    continue
                cos = float(a["cen"] @ b["cen"])
                ov  = self.ent_overlap(a["ents"], b)
                if cos >= self.P["cons_cos"] and ov >= self.P["cons_ov"]:
                    parent[find(tids[i])] = find(tids[j])
        groups = {}
        for t in tids: groups.setdefault(find(t), []).append(t)
        for root, members in groups.items():
            if len(members) == 1: continue
            keep = max(members, key=lambda t: sum(c["comments"] for c in self.tracks[t]["days"].values()))
            tk = self.tracks[keep]
            for m in members:
                if m == keep: continue
                o = self.tracks.pop(m)
                tk["ents"].update(o["ents"])
                for d, c in o["days"].items():
                    cell = tk["days"].setdefault(d, {"n":0,"comments":0,"posts":[]})
                    cell["n"] += c["n"]; cell["comments"] += c["comments"]
                    cell.setdefault("posts", []).extend(c.get("posts", []))
                tk["cen"] = (tk["cen"] + o["cen"])
                tk["cen"] /= np.linalg.norm(tk["cen"]) + 1e-9
                tk["birth"]     = min(tk["birth"],     o["birth"])
                tk["last_seen"] = max(tk["last_seen"], o["last_seen"])
                tk["confirmed"] = tk["confirmed"] or o["confirmed"]
                if o.get("revivals"):
                    tk.setdefault("revivals", []).extend(o["revivals"])

    def roll_lifecycle(self, today):
        for tk in self.tracks.values():
            gap = (today - tk["last_seen"]).days
            if not tk["confirmed"]:
                ad  = len(tk["days"])
                vol = sum(c["comments"] for c in tk["days"].values())
                if ad >= self.P["confirm_days"] or vol >= self.P["confirm_comments"]:
                    tk["confirmed"] = True
            if tk["status"] in ("new","revived") and gap == 0:
                continue
            if gap <= self.P["active_gap"]:   tk["status"] = "active"
            elif gap <= self.P["dormant_gap"]: tk["status"] = "dormant"
            else:                              tk["status"] = "dead"

    def step(self, day, rows, embs):
        self.update_idf(rows)
        for r, e in zip(rows, embs):
            self.assign_post(r.idx, day, e, post_entities(r), r.comment_count)
        self.new_from_residual(day)
        self.consolidate(day)
        self.roll_lifecycle(day)


def run_tracking(month):
    """Run online tracker for `month`.  Writes outputs/tracks_online[_<month>].pkl and .csv."""
    posts = pd.read_parquet(DATA / f"posts_{month}.parquet").reset_index(drop=True)
    emb   = np.load(DATA / f"emb_{month}.npy").astype(np.float32)
    posts["idx"] = np.arange(len(posts))
    days  = sorted(posts.day.unique())

    ot = OnlineTracker()
    for d in days:
        g = posts[posts.day == d]
        ot.step(d, [r for _, r in g.iterrows()], emb[g.idx.values])

    tracks  = {t: tk for t, tk in ot.tracks.items() if tk["confirmed"]}
    revivals = sum(1 for tk in tracks.values() if tk.get("revivals"))
    prov     = sum(1 for tk in ot.tracks.values() if not tk["confirmed"])
    cov      = len(ot.assigned) / len(posts)

    print(f"\n===== ONLINE / AUTOREGRESSIVE tracking [{month}] =====")
    print(f"days streamed    : {len(days)}  posts {len(posts)}")
    print(f"events confirmed : {len(tracks)}  (+{prov} still provisional)")
    print(f"revived events   : {revivals}")
    print(f"post coverage    : {cov:.1%}")

    # offline recall comparison (optional)
    sfx    = "" if month == "2025-09" else f"_{month}"
    offpkl = OUT / f"tracks_ml{sfx}.pkl"
    if offpkl.exists():
        O = pickle.load(open(offpkl, "rb"))["tracks"]
        def eset(tk, k=8): return {w for w, _ in tk["ents"].most_common(k)}
        offtop = sorted(O.items(), key=lambda kv: -sum(c["comments"] for c in kv[1]["days"].values()))[:15]
        lat = []; found = 0
        for _, otk in offtop:
            oe = eset(otk); best = None; bj = 0
            for _, tk in tracks.items():
                j = len(oe & eset(tk)) / max(1, len(oe | eset(tk)))
                if j > bj: bj = j; best = tk
            if best and bj >= 0.3:
                found += 1
                lat.append((min(best["days"]) - min(otk["days"])).days)
        print(f"offline top-15 recall: {found}/15  | median detect latency: "
              f"{int(np.median(lat)) if lat else '-'}d")

    births = Counter(min(tk["days"]) for tk in tracks.values())
    revs   = Counter(d for tk in tracks.values() for d in tk.get("revivals", []))
    summary = dict(n_events=len(tracks), provisional=prov, revived=revivals,
                   coverage=cov, month=month)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(
        tracks={t: {**{k: tk[k] for k in ("cen","ents","keys","days")},
                    "birth": tk["birth"], "revivals": tk.get("revivals",[])}
                for t, tk in tracks.items()},
        all_days=days, params={**P_OFFLINE, **P_ONLINE},
        births_per_day={d: births.get(d, 0) for d in days},
        revivals_per_day={d: revs.get(d, 0) for d in days},
        summary=summary,
    ), open(OUT / f"tracks_online{sfx}.pkl", "wb"))

    rows = []
    for t, tk in tracks.items():
        dd = tk["days"]
        rows.append(dict(
            tid=t, label=label(tk), start=min(dd).date(), end=max(dd).date(),
            active_days=len(dd), comments=sum(x["comments"] for x in dd.values()),
            revived=bool(tk.get("revivals")),
        ))
    df = pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / f"events_online{sfx}.csv", index=False)
    print("\ntop-12 online-tracked events:")
    with pd.option_context("display.max_colwidth", 42, "display.width", 200):
        print(df[df.active_days >= 2].head(12)
              [["label","start","end","active_days","comments","revived"]].to_string(index=False))
    return tracks, summary


# =============================================================================
# Section 4 — Visualization
# =============================================================================

def set_korean_font():
    for fp in FONT_CANDIDATES:
        if fp and Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    else:
        for name in FONT_NAMES:
            if any(f.name == name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = name; break
        else:
            print("[viz] warning: no Korean font found — set $MINDCAST_FONT to a .ttf/.ttc")
    plt.rcParams["axes.unicode_minus"] = False


def viz_month(month):
    """Plot streaming discovery timeline.  Saves figures/06_online_dynamics[_<month>].png."""
    set_korean_font()
    sfx = "" if month == "2025-09" else f"_{month}"
    on  = pickle.load(open(OUT / f"tracks_online{sfx}.pkl", "rb"))
    days = sorted(on["all_days"]); dstr = [d.strftime("%d") for d in days]
    bpd  = [on["births_per_day"][d] for d in days]
    rpd  = [on["revivals_per_day"][d] for d in days]
    s    = on["summary"]
    MLABEL = f"{days[0].year}년 {days[0].month}월"

    offp = OUT / f"tracks_ml{sfx}.pkl"
    off  = pickle.load(open(offp, "rb")) if offp.exists() else None

    fig, axs = plt.subplots(1, 2, figsize=(15,5.4), gridspec_kw={"width_ratios":[2,1]})

    ax = axs[0]
    x  = np.arange(len(days))
    ax.bar(x, bpd, color="#2ca02c", label="신규 이벤트 birth", zorder=3)
    ax.bar(x, rpd, bottom=bpd, color="#9467bd", label="이벤트 revival(부활)", zorder=3)
    cum = np.cumsum(bpd)
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#333", lw=2, marker="o", ms=3, label="누적 확정 이벤트")
    ax2.set_ylabel("누적 확정 이벤트 수", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(dstr, fontsize=7.5)
    ax.set_xlabel(f"{MLABEL} (일)", fontsize=10)
    ax.set_ylabel("일별 이벤트 수", fontsize=10)
    ax.set_title("온라인 스트리밍 이벤트 발견 타임라인 (인과적, 매일 갱신)", fontsize=13, weight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=9, loc="upper right", frameon=False)
    for sp in ["top"]: ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)

    ax = axs[1]
    ax.axis("off")
    txt = (f"■ 온라인(autoregressive)\n"
           f"   확정 이벤트: {s['n_events']}\n"
           f"   provisional(후보): {s['provisional']}\n"
           f"   부활(revival): {s['revived']}\n"
           f"   포스트 커버리지: {s['coverage']:.0%}\n\n")
    if off is not None:
        offv = len([1 for t in off["tracks"].values() if len(t["days"]) >= 2])
        txt += (f"■ 오프라인(multi-label)\n"
                f"   이벤트(다중일): {offv}\n"
                f"   포스트 커버리지: {off['metrics']['cov_multi']:.0%}\n"
                f"   부활: 미검출(전기간 일괄)\n\n"
                f"■ online 고유 이점\n"
                f"   · 미래 데이터 없이 매일 갱신\n"
                f"   · residual backfill로 지연 발견\n"
                f"   · dormant→revival 실시간 포착")
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=11, linespacing=1.5,
            transform=ax.transAxes)
    ax.set_title("오프라인 vs 온라인", fontsize=13, weight="bold", loc="left")

    fig.suptitle(f"Online / Autoregressive 이벤트 트래킹 — {MLABEL}", fontsize=14.5, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"06_online_dynamics{sfx}.png", dpi=140, bbox_inches="tight")
    print(f"saved figures/06_online_dynamics{sfx}.png")


# =============================================================================
# Section 5 — Export HTML
# =============================================================================

_RNG = np.random.RandomState(0)


def _statuses(days_dict, all_days, rev_days):
    """Per-day status list (mirrors online lifecycle)."""
    vol    = {d: c["comments"] for d, c in days_dict.items()}
    revset = set(rev_days)
    st     = [None]*len(all_days); prev = None
    for i, d in enumerate(all_days):
        if d in vol:
            if prev is None:
                s = "new"
            else:
                gap = (d - prev).days
                s = ("revived" if d in revset else
                     "active" if vol[d] >= 0.8*vol[prev] else "decaying")
            st[i] = s; prev = d
        elif prev is not None:
            gap = (d - prev).days
            st[i] = "dormant" if gap <= GAP_DORMANT else "dead"
    return st


def _pset(posts, i):
    return {w for w in list(posts.iloc[i].tags) + list(posts.iloc[i].keyphrases)
            if not channelish(w)}


def _cohesion(emb, ids, cap=120):
    if len(ids) < 2: return 1.0
    s = ids if len(ids) <= cap else list(_RNG.choice(ids, cap, replace=False))
    X = emb[s]; S = X @ X.T; np.fill_diagonal(S, -1.0)
    return float(np.median(S.max(1)))


def _and_label(posts, ids, n=4):
    df = Counter()
    for i in ids: df.update(_pset(posts, i))
    m = len(ids) or 1
    shared = [w for w, c in df.most_common() if c/m >= FRAC][:n]
    if not shared:
        shared = [w for w, _ in df.most_common(2)]
    return " · ".join(shared), df


def build_month_data(month):
    """Load tracks + embeddings for one month, build the JS data blob."""
    sfx    = "" if month == "2025-09" else f"_{month}"
    D      = pickle.load(open(OUT / f"tracks_online{sfx}.pkl", "rb"))
    tracks, all_days = D["tracks"], sorted(D["all_days"])
    di     = {d: i for i, d in enumerate(all_days)}
    posts  = pd.read_parquet(DATA / f"posts_{month}.parquet").reset_index(drop=True)
    emb    = np.load(DATA / f"emb_{month}.npy")
    ptitle = posts["title"].fillna("").tolist()
    used_titles = {}
    evs = []
    for t, tk in tracks.items():
        dd = tk["days"]
        if len(dd) < 2: continue
        allmem = list({i for d in dd for i in dd[d].get("posts", [])})
        if len(allmem) < 2: continue
        coh = _cohesion(emb, allmem)
        and_lab, df = _and_label(posts, allmem)
        misc  = coh < COH_THR
        lbl   = ("기타 · " + " · ".join(w for w, _ in df.most_common(2))) if misc else and_lab
        vols  = [0]*len(all_days); pbd = {}
        for d, c in dd.items():
            i = di[d]; vols[i] = int(c["comments"])
            pl = sorted(set(c.get("posts", [])),
                        key=lambda x: -int(posts.iloc[x].comment_count))[:CAP]
            if pl:
                pbd[i] = pl
                for x in pl: used_titles[x] = ptitle[x]
        rev = [di[d] for d in tk.get("revivals", []) if d in di]
        evs.append(dict(label=lbl, misc=misc, coh=round(coh,2),
                        birth=di[min(dd)], total=sum(vols), vols=vols, rev=rev,
                        st=_statuses(dd, all_days, tk.get("revivals",[])), pbd=pbd))
    evs.sort(key=lambda e: -e["total"])
    cmap   = {int(x): int(posts.iloc[x].comment_count) for x in used_titles}
    n_misc = sum(e["misc"] for e in evs)
    print(f"  {month}: {len(evs)} events ({n_misc} 기타/저응집)")
    return dict(
        days=[d.strftime("%m-%d") for d in all_days],
        dow=[["월","화","수","목","금","토","일"][d.weekday()] for d in all_days],
        events=evs,
        titles={str(k): v for k, v in used_titles.items()},
        cmt={str(k): v for k, v in cmap.items()},
    )


def html_export(months=None):
    """Build self-contained online_replay.html covering all specified months."""
    if months is None: months = MONTHS
    data = {m: build_month_data(m) for m in months}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out  = ROOT / "online_replay.html"
    out.write_text(html, encoding="utf-8")
    tot  = {m: len(data[m]["events"]) for m in months}
    print("wrote", out, "| events/month:", tot)


HTML_TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>온라인 뉴스 이벤트 트래킹 — 일별 리플레이</title>
<style>
:root{--bg:#0f1420;--panel:#171d2b;--ink:#e8ecf3;--mut:#8a94a8;--line:#26304a;
 --new:#2ca02c;--active:#2f7fe0;--decay:#ef8a2b;--dormant:#39435c;--revive:#9b6cff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Noto Sans KR",Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:800}
.sub{color:var(--mut);font-size:12px}
select,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:7px 12px;font-size:14px;cursor:pointer}
button:hover{border-color:#4a5a80}
button:disabled{opacity:.4;cursor:default}
.wrap{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;padding:14px 20px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--mut);font-weight:700;letter-spacing:.3px}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.daybadge{font-size:22px;font-weight:800;min-width:150px}
.stat{display:inline-flex;flex-direction:column;margin-right:18px}
.stat b{font-size:20px}.stat span{font-size:11px;color:var(--mut)}
input[type=range]{width:280px;accent-color:var(--revive)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--mut);margin-top:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#gantt{width:100%;overflow-x:auto}
.row{display:flex;align-items:center;height:22px;margin:2px 0}
.rlab{width:210px;flex:0 0 210px;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:8px;text-align:right;color:#cdd5e6}
.track{position:relative;flex:1;height:16px;background:#10162300;border-radius:3px}
.seg{position:absolute;top:50%;transform:translateY(-50%);border-radius:2px}
.today .rlab{color:#fff;font-weight:700}
.tbar{display:flex;align-items:center;gap:8px;margin:5px 0}
.tbar .bl{width:150px;flex:0 0 150px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}
.tbar .bar{height:16px;border-radius:3px;min-width:2px}
.tbar .bv{font-size:11px;color:var(--mut)}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;margin:2px 4px 2px 0}
.b-new{background:rgba(44,160,44,.18);color:#7bd67b;border:1px solid #2ca02c55}
.b-rev{background:rgba(155,108,255,.18);color:#c3a8ff;border:1px solid #9b6cff55}
.empty{color:var(--mut);font-size:12px}
#newslist{columns:2;column-gap:20px}
@media(max-width:900px){#newslist{columns:1}}
.ev-news{break-inside:avoid;margin:0 0 12px;padding:8px 10px;background:#12182633;border:1px solid var(--line);border-radius:8px}
.ev-h{font-size:12.5px;font-weight:700;margin-bottom:4px}
.ev-dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:0}
.ev-n{color:var(--mut);font-weight:400;font-size:11px}
.ev-news ul{margin:2px 0 0;padding-left:16px}
.ev-news li{font-size:12px;color:#d6ddec;margin:2px 0;line-height:1.35}
.ni-c{color:var(--mut);font-size:10.5px;white-space:nowrap}
.row.misc .rlab,.misc-lab{color:#79839a}
.divider{grid-column:1/-1;color:#79839a;font-size:11.5px;margin:6px 0 4px;border-top:1px dashed var(--line);padding-top:8px}
.ev-news.misc{opacity:.82}
</style></head><body>
<header>
  <h1>📰 온라인 뉴스 이벤트 트래킹 <span class="sub">— autoregressive replay (미래 데이터 없이 하루씩)</span></h1>
  <label class="sub">월 <select id="month"></select></label>
</header>
<div class="card" style="margin:14px 20px 0">
  <div class="controls">
    <button id="prev">◀ 이전날</button>
    <button id="play">▶ 자동재생</button>
    <button id="next">다음날 ▶</button>
    <input type="range" id="slider" min="0" value="0">
    <span class="daybadge" id="daybadge"></span>
    <span style="flex:1"></span>
    <span class="stat"><b id="s-ev">0</b><span>누적 이벤트</span></span>
    <span class="stat"><b id="s-new" style="color:var(--new)">0</b><span>오늘 신규</span></span>
    <span class="stat"><b id="s-rev" style="color:var(--revive)">0</b><span>오늘 부활</span></span>
    <span class="stat"><b id="s-cmt">0</b><span>오늘 댓글</span></span>
  </div>
  <div class="legend">
    <span><i class="dot" style="background:var(--new)"></i>생성</span>
    <span><i class="dot" style="background:var(--active)"></i>지속</span>
    <span><i class="dot" style="background:var(--decay)"></i>감쇠</span>
    <span><i class="dot" style="background:var(--dormant)"></i>휴면</span>
    <span><i class="dot" style="background:var(--revive)"></i>부활</span>
    <span>· 막대 높이 = 그날 댓글량 · 왼쪽 스토리라인은 오늘까지만 공개됩니다</span>
  </div>
</div>
<div class="wrap">
  <div class="card"><h2 id="g-title">스토리라인 (오늘까지 공개)</h2><div id="gantt"></div></div>
  <div class="card"><h2>오늘의 이슈 <span class="sub" id="t-sub"></span></h2>
    <div id="today"></div>
    <h2 style="margin-top:14px">오늘 일어난 일</h2>
    <div id="events"></div>
  </div>
</div>
<div class="card" style="margin:0 20px 20px">
  <h2>📋 오늘 분류된 실제 뉴스 제목 <span class="sub" id="n-sub"></span></h2>
  <div id="newslist"></div>
</div>
<script>
const DATA=__DATA__;
const COL={new:"#2ca02c",active:"#2f7fe0",decaying:"#ef8a2b",dormant:"#39435c",revived:"#9b6cff",dead:null};
let M="2025-09", cur=0, timer=null;

const $=id=>document.getElementById(id);
function initMonth(){
  const sel=$("month"); sel.innerHTML="";
  Object.keys(DATA).forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});
  sel.value=M; sel.onchange=()=>{M=sel.value;cur=0;setup();};
}
function setup(){
  const d=DATA[M]; $("slider").max=d.days.length-1; $("slider").value=cur;
  render();
}
function render(){
  const d=DATA[M], nD=d.days.length;
  $("slider").value=cur;
  $("daybadge").textContent=`Day ${cur+1}/${nD} · ${d.days[cur]}(${d.dow[cur]})`;
  const born=d.events.filter(e=>e.birth<=cur);
  $("s-ev").textContent=born.length;
  let todayNew=[],todayRev=[],todayCmt=0;
  born.forEach(e=>{ if(e.birth===cur)todayNew.push(e);
     if(e.rev.includes(cur))todayRev.push(e);
     todayCmt+=e.vols[cur]||0; });
  $("s-new").textContent=todayNew.length;
  $("s-rev").textContent=todayRev.length;
  $("s-cmt").textContent=todayCmt.toLocaleString();

  const TOPN=28;
  const pool=d.events.slice().sort((a,b)=>b.total-a.total).slice(0,TOPN);
  const rows=pool.filter(e=>e.birth<=cur).sort((a,b)=>a.birth-b.birth);
  const maxv=Math.max(1,...d.events.map(e=>Math.max(...e.vols)));
  const wpd=100/nD;
  let g="";
  rows.forEach(e=>{
    const isToday=(e.birth===cur||e.rev.includes(cur));
    let segs="";
    for(let i=0;i<=cur;i++){
      const s=e.st[i]; if(!s||s==="dead")continue;
      const c=COL[s]; if(!c)continue;
      const h=s==="dormant"?4:(5+11*(e.vols[i]/maxv));
      const glow=(i===cur&&isToday)?";box-shadow:0 0 6px "+c:"";
      segs+=`<div class="seg" style="left:${i*wpd}%;width:${wpd}%;height:${h}px;background:${c}${glow}"></div>`;
    }
    g+=`<div class="row ${isToday?'today':''} ${e.misc?'misc':''}"><div class="rlab">${isToday?'▶ ':''}${e.label}</div>`+
       `<div class="track">${segs}</div></div>`;
  });
  $("gantt").innerHTML=g||'<div class="empty">아직 이벤트 없음</div>';
  $("g-title").textContent=`스토리라인 · 월간 주요 ${TOPN}개 중 ${rows.length}개 등장 (${d.days[cur]}까지, 각자 birth일에 나타남)`;

  const act=born.filter(e=>e.vols[cur]>0).sort((a,b)=>b.vols[cur]-a.vols[cur]).slice(0,12);
  const tmax=Math.max(1,...act.map(e=>e.vols[cur]));
  $("t-sub").textContent=`(${d.days[cur]} 댓글량 기준 상위 ${act.length})`;
  $("today").innerHTML= act.length? act.map(e=>{
    const s=e.st[cur]||"active"; const c=COL[s]||"#2f7fe0";
    const w=100*e.vols[cur]/tmax;
    const tag=e.birth===cur?" 🆕":(e.rev.includes(cur)?" ♻️":"");
    return `<div class="tbar"><div class="bl ${e.misc?'misc-lab':''}">${e.label}${tag}</div>`+
      `<div class="bar" style="width:${w}%;background:${e.misc?'#5c6577':c}"></div>`+
      `<div class="bv">${e.vols[cur].toLocaleString()}</div></div>`;
  }).join(""): '<div class="empty">오늘 활동 이벤트 없음</div>';

  let ev="";
  if(todayNew.length) ev+='<div>'+todayNew.map(e=>`<span class="badge b-new">🆕 ${e.label}</span>`).join("")+'</div>';
  if(todayRev.length) ev+='<div style="margin-top:6px">'+todayRev.map(e=>`<span class="badge b-rev">♻️ 부활 · ${e.label}</span>`).join("")+'</div>';
  $("events").innerHTML=ev||'<div class="empty">신규/부활 없음 (기존 이슈 지속)</div>';

  const born2=d.events.filter(e=>e.birth<=cur);
  const withNews=born2.filter(e=>e.pbd&&e.pbd[cur]&&e.pbd[cur].length)
                      .sort((a,b)=>b.vols[cur]-a.vols[cur]);
  const coh=withNews.filter(e=>!e.misc), mis=withNews.filter(e=>e.misc);
  let cnt=0;
  const block=e=>{
    const ids=e.pbd[cur]; cnt+=ids.length;
    const s=e.st[cur]||"active"; const c=e.misc?"#5c6577":(COL[s]||"#2f7fe0");
    const tag=e.birth===cur?" 🆕":(e.rev.includes(cur)?" ♻️":"");
    let h=`<div class="ev-news ${e.misc?'misc':''}"><div class="ev-h"><span class="ev-dot" style="background:${c}"></span>`+
        `${e.label}${tag} <span class="ev-n">${ids.length}건 · 댓글 ${e.vols[cur].toLocaleString()}</span></div><ul>`;
    ids.forEach(i=>{ const ti=d.titles[i]||"(제목없음)"; const cm=d.cmt[i]||0;
      h+=`<li>${ti} <span class="ni-c">💬${cm.toLocaleString()}</span></li>`; });
    return h+"</ul></div>";
  };
  let nl=coh.map(block).join("");
  if(mis.length) nl+=`<div class="divider">── 기타 · 저응집(개별/잡다) 이슈 ${mis.length}개 — 하나의 사건으로 묶기 애매한 뉴스 ──</div>`+mis.map(block).join("");
  $("n-sub").textContent=`(${d.days[cur]} · ${cnt}건, 이벤트 ${coh.length}개 + 기타 ${mis.length}개)`;
  $("newslist").innerHTML=nl||'<div class="empty">오늘 분류된 뉴스 없음</div>';
}
function step(n){const nD=DATA[M].days.length;cur=Math.max(0,Math.min(nD-1,cur+n));render();}
$("prev").onclick=()=>step(-1);
$("next").onclick=()=>step(1);
$("slider").oninput=e=>{cur=+e.target.value;render();};
$("play").onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent="▶ 자동재생";return;}
  this.textContent="⏸ 정지";
  timer=setInterval(()=>{const nD=DATA[M].days.length; if(cur>=nD-1){cur=0;} else step(1);},700);
};
initMonth(); setup();
</script></body></html>"""


# =============================================================================
# Section 6 — HF Upload
# =============================================================================

def build_staging():
    """Extract tables from local dump to hf_staging/*.parquet."""
    STAGE.mkdir(exist_ok=True)
    for t in UPLOAD_TABLES:
        p = STAGE / f"{t}.parquet"
        if p.exists():
            print(f"  [skip] {t} already staged"); continue
        print(f"  [build] extracting {t} from local dump …")
        load_table(t, source="local").to_parquet(p)


def pii_guard():
    """Abort unless every staged file is free of raw PII."""
    for t in UPLOAD_TABLES:
        df  = pd.read_parquet(STAGE / f"{t}.parquet")
        bad = FORBIDDEN_COLS & set(df.columns)
        assert not bad, f"FORBIDDEN column(s) {bad} in {t} — refusing to upload"
        if t == "video_comment":
            a      = df["author"].dropna()
            nonhash = a.map(lambda x: not re.fullmatch(r"[0-9a-f]{16}", str(x))).sum()
            assert nonhash == 0, f"{nonhash} author values are not SHA-256 hashes — refusing"
            assert not a.str.startswith("@").any(), "raw @handle leaked in author — refusing"
        print(f"  [guard OK] {t}: {df.shape} cols={list(df.columns)}")
    print("  PII guard passed — safe to upload.")


def upload_to_hf():
    """Upload staged parquets + README to private HF dataset."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    (STAGE / "README.md").write_text(HF_CARD, encoding="utf-8")
    for t in UPLOAD_TABLES + ["README"]:
        fn = f"{t}.parquet" if t != "README" else "README.md"
        print(f"  [upload] {fn} …")
        api.upload_file(path_or_fileobj=str(STAGE / fn), path_in_repo=fn,
                        repo_id=HF_REPO, repo_type="dataset")
    print(f"  done -> https://huggingface.co/datasets/{HF_REPO} (private)")


def extract_db(with_comments=False):
    """Materialize local dump tables to data/*.parquet cache."""
    DATA.mkdir(parents=True, exist_ok=True)
    tables = ["video_video", "video_channel"] + (["video_comment"] if with_comments else [])
    for t in tables:
        df = load_table(t, source="local")
        df.to_parquet(DATA / f"{t}.parquet")
        print(f"{t}: {df.shape} -> {DATA/f'{t}.parquet'}")
