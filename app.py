"""Federasyon Sağlık Konsolu (Federation Health Console)
Streamlit Application for FedCost Prototype.
Developed by Bülent (Console & Exclusion Preview Track).
"""

import json
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from contract import (
    CLASS_NAMES, N_CLASSES, N_CLIENTS, RUNS_DIR, FEATURES_DIR
)
import contract
import preview
import audit_log

# ── Rare-class alert threshold ─────────────────────────────────────────────
RARE_ALERT_THRESHOLD = 0.20   # DF or VASC recall below this triggers a warning
RARE_CLASSES = {"DF", "VASC"}

st.set_page_config(
    page_title="Federasyon Sağlık Konsolu | FedCost",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar header */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* Status badges */
    .badge-ok   { background:#C8E6C9; color:#1B5E20; padding:3px 9px; border-radius:12px; font-weight:600; font-size:.82em; }
    .badge-exc  { background:#FFCDD2; color:#B71C1C; padding:3px 9px; border-radius:12px; font-weight:600; font-size:.82em; }
    .badge-syn  { background:#E3F2FD; color:#0D47A1; padding:3px 9px; border-radius:12px; font-weight:600; font-size:.82em; }
    .badge-real { background:#E8F5E9; color:#1B5E20; padding:3px 9px; border-radius:12px; font-weight:600; font-size:.82em; }

    /* Metric card */
    .metric-card {
        background:#F8FAFC; border-radius:10px; padding:14px 16px;
        border-left:5px solid #1E88E5;
        box-shadow:0 1px 4px rgba(0,0,0,.08);
        margin-bottom:8px;
    }

    /* Alert banner */
    .alert-rare {
        background:#FFF3E0; border-left:5px solid #E65100;
        padding:10px 14px; border-radius:6px; margin-bottom:8px;
        color:#BF360C; font-weight:600;
    }

    /* Run group headers in select */
    div[data-testid="stSelectbox"] { font-size: .93em; }

    /* Action buttons in sidebar columns 2 and 3 */
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button,
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button {
        padding: 0px !important;
        height: 40px !important;
        width: 100% !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Rename (edit) button */
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
    }
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
        background-color: #E2E8F0 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* Delete (trash) button: red theme */
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button {
        background-color: #FEE2E2 !important;
        border: 1px solid #FCA5A5 !important;
        color: #DC2626 !important;
    }
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button:hover {
        background-color: #EF4444 !important;
        border-color: #DC2626 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button span {
        color: #DC2626 !important;
    }
    div[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(3) button:hover span {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def is_synthetic_run(run_name: str) -> bool:
    """Return True if the run was produced by fake/synthetic simulation."""
    if run_name.startswith(("demo_", "sim_", "test_")):
        return True
    rd_path = RUNS_DIR / run_name / "round_0.json"
    if rd_path.exists():
        try:
            m = json.loads(rd_path.read_text(encoding="utf-8")).get("meta", {})
            if m.get("is_synthetic"):
                return True
        except Exception:
            pass
    return False


def run_group(run_name: str) -> str:
    """Return a human-readable group label for a run name."""
    low = run_name.lower()
    if "sweep_fedavg" in low:
        return "📊 Sweep — FedAvg"
    if "sweep_trimmed" in low:
        return "📊 Sweep — Trimmed Mean"
    if "sweep_krum" in low:
        return "📊 Sweep — Krum"
    if "engine_" in low:
        return "🟢 Gerçek Motor"
    if low.startswith("sim_"):
        return "🧪 Hızlı Simülasyon"
    if "trimmed" in low:
        return "🔵 Trimmed Mean (tekil)"
    if "krum" in low:
        return "🔵 Krum (tekil)"
    if "fedavg" in low:
        return "🔵 FedAvg (tekil)"
    return "📁 Diğer"


def format_run_label(run_name: str) -> str:
    icon = "🧪" if is_synthetic_run(run_name) else "🟢"
    return f"{icon} {run_name}"


def get_available_runs():
    if not RUNS_DIR.exists():
        return []
    runs = [d.name for d in RUNS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "audit_log.json"]
    return sorted(runs)


def load_run_rounds(run_id: str):
    run_path = RUNS_DIR / run_id
    round_files = sorted(
        run_path.glob("round_*.json"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    records = []
    for rf in round_files:
        try:
            records.append(json.loads(rf.read_text(encoding="utf-8")))
        except Exception:
            pass
    return records


def rare_class_alert(recalls: list[float]) -> list[str]:
    """Return list of rare classes whose recall is below the threshold."""
    return [
        name for name in RARE_CLASSES
        if recalls[CLASS_NAMES.index(name)] < RARE_ALERT_THRESHOLD
    ]


def trend_summary(rounds_data: list[dict]) -> str:
    """Return a short natural-language trend description for the last 5 rounds."""
    if len(rounds_data) < 2:
        return ""
    tail = rounds_data[-min(5, len(rounds_data)):]
    accs = [float(np.mean(r["global_class_recall"])) for r in tail]
    delta = accs[-1] - accs[0]
    n = len(tail)
    direction = "artıyor 📈" if delta > 0.005 else "düşüyor 📉" if delta < -0.005 else "stabil ➡️"
    rare_name_idx = [(n, CLASS_NAMES.index(n)) for n in RARE_CLASSES]
    rare_trend = []
    for cls_name, idx in rare_name_idx:
        vals = [r["global_class_recall"][idx] for r in tail]
        d = vals[-1] - vals[0]
        if abs(d) > 0.01:
            rare_trend.append(f"{cls_name} {d:+.1%}")
    rare_str = f" | Nadir sınıflar: {', '.join(rare_trend)}" if rare_trend else ""
    return f"Son {n} turda dengeli doğruluk {direction} ({delta:+.1%}){rare_str}."


@st.dialog("✏️ Koşumu Yeniden Adlandır")
def rename_run_dialog(old_name: str):
    st.markdown(f"Mevcut ad: `{old_name}`")
    new_name = st.text_input("Yeni isim:", value=old_name)
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("Kaydet", key="dlg_rename_save", use_container_width=True):
            clean = new_name.strip()
            if not clean:
                st.error("İsim boş bırakılamaz.")
            elif (RUNS_DIR / clean).exists() and clean != old_name:
                st.error("Bu isimde bir koşum zaten mevcut.")
            elif clean != old_name:
                (RUNS_DIR / old_name).rename(RUNS_DIR / clean)
                st.session_state["active_run"] = clean
                st.rerun()
    with col_cancel:
        if st.button("İptal", key="dlg_rename_cancel", use_container_width=True):
            st.rerun()


@st.dialog("🗑️ Koşumu Sil")
def delete_run_dialog(run_name: str):
    st.warning(f"**'{run_name}'** koşumunu silmek istediğinizden emin misiniz?")
    st.caption("Bu işlem diskteki tüm tur ve model dosyalarını kalıcı olarak silecektir.")
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("Silmeyi Onayla", key="dlg_del_confirm", type="primary", use_container_width=True):
            shutil.rmtree(RUNS_DIR / run_name, ignore_errors=True)
            avail = get_available_runs()
            st.session_state["active_run"] = avail[0] if avail else None
            st.rerun()
    with col_cancel:
        if st.button("Vazgeç", key="dlg_del_cancel", use_container_width=True):
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 FedCost Konsolu")
    st.caption("Federe Tıbbi Görüntülemede Sınıf Bazlı Koruma & Dışlama Maliyeti")
    st.divider()

    available_runs = get_available_runs()
    if not available_runs:
        st.warning("Henüz kayıtlı bir federasyon koşusu bulunamadı!")
        if st.button("Örnek Koşumları Oluştur"):
            import fake_run_generator
            fake_run_generator.main()
            st.rerun()
        st.stop()

    # ── Run type filter ────────────────────────────────────────────────────
    st.markdown("**Filtrele & Ara**")
    run_filter = st.radio(
        "Koşu türü:",
        ["Tümü", "🟢 Gerçek Motor", "🧪 Sentetik"],
        horizontal=True,
        label_visibility="collapsed"
    )
    search_query = st.text_input("🔍 Koşum ara…", placeholder="fedavg, krum, atk0…")

    if run_filter == "🟢 Gerçek Motor":
        filtered_runs = [r for r in available_runs if not is_synthetic_run(r)]
    elif run_filter == "🧪 Sentetik":
        filtered_runs = [r for r in available_runs if is_synthetic_run(r)]
    else:
        filtered_runs = available_runs

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_runs = [r for r in filtered_runs if q in r.lower()]

    if not filtered_runs:
        st.info("Bu filtreye uygun koşum bulunamadı.")
        filtered_runs = available_runs

    # ── Group runs for display ─────────────────────────────────────────────
    groups: dict[str, list[str]] = {}
    for r in filtered_runs:
        g = run_group(r)
        groups.setdefault(g, []).append(r)

    # Build a flat ordered list with group separators (──── GROUP) prepended
    # so users can see structure; actual selectbox uses flat list
    group_order = [
        "🟢 Gerçek Motor", "🔵 FedAvg (tekil)", "🔵 Trimmed Mean (tekil)",
        "🔵 Krum (tekil)", "📊 Sweep — FedAvg", "📊 Sweep — Trimmed Mean",
        "📊 Sweep — Krum", "🧪 Hızlı Simülasyon", "📁 Diğer"
    ]
    ordered_runs = []
    for g in group_order:
        ordered_runs.extend(groups.get(g, []))
    # Append any group not in the canonical order
    for g, items in groups.items():
        if g not in group_order:
            ordered_runs.extend(items)

    # ── Active run selection ───────────────────────────────────────────────
    st.markdown("**Aktif koşum**")
    default_idx = 0
    real_runs = [r for r in ordered_runs if not is_synthetic_run(r)]

    # Favorites stored in session state
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []

    fav_opts = [r for r in st.session_state["favorites"] if r in ordered_runs]
    if fav_opts:
        with st.expander(f"⭐ Favoriler ({len(fav_opts)})", expanded=False):
            fav_pick = st.selectbox("Favori koşumlar:", fav_opts,
                                    format_func=format_run_label,
                                    key="fav_select")
            if st.button("→ Bu koşumu aç", key="open_fav"):
                st.session_state["active_run"] = fav_pick
                st.rerun()

    if "active_run" in st.session_state and st.session_state["active_run"] in ordered_runs:
        default_idx = ordered_runs.index(st.session_state["active_run"])
    elif real_runs and real_runs[0] in ordered_runs:
        default_idx = ordered_runs.index(real_runs[0])

    col_sel, col_ren, col_del = st.columns([5.0, 1.2, 1.2], gap="small", vertical_alignment="center")
    with col_sel:
        selected_run = st.selectbox(
            "Koşum seç:",
            ordered_runs,
            index=default_idx,
            format_func=format_run_label,
            label_visibility="collapsed"
        )
        st.session_state["active_run"] = selected_run

    with col_ren:
        if st.button("", icon=":material/edit:", key="btn_rename_run", help=f"'{selected_run}' koşumunu yeniden adlandır", width="stretch"):
            rename_run_dialog(selected_run)

    with col_del:
        if st.button("", icon=":material/delete:", key="btn_delete_run", help=f"'{selected_run}' koşumunu sil", width="stretch"):
            delete_run_dialog(selected_run)

    # Favorite toggle
    is_fav = selected_run in st.session_state["favorites"]
    if st.button("⭐ Favorilere ekle" if not is_fav else "✖ Favoriden çıkar",
                 key="fav_toggle"):
        if is_fav:
            st.session_state["favorites"].remove(selected_run)
        else:
            st.session_state["favorites"].append(selected_run)
        st.rerun()

    rounds_data = load_run_rounds(selected_run)
    if not rounds_data:
        st.error("Seçilen koşumda geçerli tur verisi yok.")
        st.stop()

    total_rounds = len(rounds_data)
    selected_round = st.slider(
        "İncelenen Tur:", min_value=0, max_value=total_rounds - 1,
        value=total_rounds - 1
    )

    cur_meta      = rounds_data[selected_round].get("meta", {})
    cur_agg       = cur_meta.get("aggregation") or cur_meta.get("agg", "Bilinmiyor")
    cur_attack    = cur_meta.get("attack_client") if "attack_client" in cur_meta else cur_meta.get("attack")
    cur_sizes     = cur_meta.get("client_sizes", [9930, 3163, 2691, 1807, 655, 351])
    attack_label  = (f"Merkez {cur_attack} (Etiket Çevirme)"
                     if cur_attack is not None and cur_attack >= 0 else "Saldırı Yok (Temiz)")

    st.info(f"""
**Savunma:** `{str(cur_agg).upper()}`  
**Saldırı:** `{attack_label}`  
**Tür:** `{'🧪 Sentetik' if is_synthetic_run(selected_run) else '🟢 Gerçek Motor'}`  
**Toplam Tur:** `{total_rounds}`
""")

    # ── Rare-class alarm in sidebar ─────────────────────────────────────────
    cur_recalls = rounds_data[selected_round].get("global_class_recall", [0]*N_CLASSES)
    alerts = rare_class_alert(cur_recalls)
    if alerts:
        st.markdown(
            f'<div class="alert-rare">⚠️ Nadir sınıf uyarısı: '
            f'{", ".join(alerts)} &lt; %{RARE_ALERT_THRESHOLD*100:.0f}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── New run launcher ────────────────────────────────────────────────────
    with st.expander("🚀 Yeni Koşum Başlat", expanded=False):
        st.caption("Farklı savunma / saldırı senaryolarıyla yeni koşum üretin.")

        run_mode = st.radio(
            "Çalıştırma yöntemi:",
            ["🟢 Gerçek Motor (engine.py)", "🧪 Hızlı Sentetik Simülasyon"],
            index=0
        )

        inc_centers = st.multiselect(
            "Dahil edilecek hastaneler:",
            range(N_CLIENTS),
            default=list(range(N_CLIENTS)),
            format_func=lambda x: (
                f"Center {x} {'(Nadir DF/VASC)' if x == 2 else '(Cihaz Farkı)' if x == 4 else ''}"
            )
        )
        exc_centers = [c for c in range(N_CLIENTS) if c not in inc_centers]
        if exc_centers:
            st.warning(f"Dışlanan: {', '.join(f'Center {c}' for c in exc_centers)}")
        else:
            st.success("Tüm hastaneler dahil.")

        sim_agg    = st.selectbox("Savunma kuralı:", ["fedavg", "trimmed", "krum"], index=0)
        sim_attack = st.selectbox("Saldırı:", ["Yok", "Center 0 (Etiket Çevirme)"], index=0)
        atk_id     = 0 if "Center 0" in sim_attack else None
        sim_seed   = st.number_input("Seed:", min_value=0, max_value=999, value=0, step=1)

        exc_suffix   = f"_no_c{'_c'.join(map(str, exc_centers))}" if exc_centers else ""

        if run_mode == "🟢 Gerçek Motor (engine.py)":
            st.info("💡 EfficientNet öznitelikleri üzerinde saf NumPy federasyon motoru.")
            sim_rounds    = st.slider("Tur sayısı:", 5, 30, 15, 5)
            default_name  = f"engine_{sim_agg}{exc_suffix}_atk{atk_id if atk_id is not None else 'None'}_s{sim_seed}"
            custom_run_id = st.text_input("Koşum adı:", value=default_name)

            if st.button("🚀 Gerçek Motoru Çalıştır", use_container_width=True):
                feats_dir    = FEATURES_DIR
                missing_feats = [
                    f"center_{k}_{split}.npz"
                    for k in range(N_CLIENTS)
                    for split in ("train", "test")
                    if not (feats_dir / f"center_{k}_{split}.npz").exists()
                ]
                if missing_feats:
                    st.error(
                        f"❌ `features/` dizininde {len(missing_feats)} dosya eksik "
                        f"(ilk: `{missing_feats[0]}`). Simülasyon modunu deneyin."
                    )
                else:
                    with st.spinner(f"Gerçek motor {sim_rounds} tur eğitiliyor…"):
                        import importlib, engine
                        importlib.reload(engine)
                        engine.run(
                            agg=sim_agg, rounds=sim_rounds, attack=atk_id,
                            seed=int(sim_seed), run_id=custom_run_id,
                            excluded_clients=exc_centers
                        )
                        st.session_state["active_run"] = custom_run_id
                        st.success(f"'{custom_run_id}' tamamlandı!")
                        st.rerun()
        else:
            st.caption("Sentetik dağılımlar üzerinden anlık simülasyon.")
            default_name  = f"sim_{sim_agg}{exc_suffix}_atk{atk_id if atk_id is not None else 'None'}_s{sim_seed}"
            custom_run_id = st.text_input("Koşum adı:", value=default_name)

            if st.button("🚀 Simülasyonu Koş ve Kaydet", use_container_width=True):
                with st.spinner("15 turluk federasyon simüle ediliyor…"):
                    import importlib, fake_run_generator
                    importlib.reload(fake_run_generator)
                    fake_run_generator.generate_run(
                        run_id=custom_run_id, aggregation=sim_agg,
                        attack_client=atk_id, num_rounds=15,
                        seed=int(sim_seed), excluded_clients=exc_centers
                    )
                    st.session_state["active_run"] = custom_run_id
                    st.success(f"'{custom_run_id}' oluşturuldu!")
                    st.rerun()


    st.divider()
    if st.button("🔄 Sahte Koşumları Yenile"):
        import fake_run_generator
        fake_run_generator.main()
        st.success("Koşumlar yenilendi!")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# HEADER METRICS
# ═══════════════════════════════════════════════════════════════════════════
cur_round_info    = rounds_data[selected_round]
cur_global_recalls = cur_round_info.get("global_class_recall", [0]*N_CLASSES)
bal_acc           = float(np.mean(cur_global_recalls))
rare_acc          = float((cur_global_recalls[CLASS_NAMES.index("DF")] +
                            cur_global_recalls[CLASS_NAMES.index("VASC")]) / 2.0)
clients_round     = cur_round_info.get("clients", {})
excluded_count    = sum(1 for c in clients_round.values() if c.get("excluded", False))

if selected_round > 0:
    prev_round_info = rounds_data[selected_round - 1]
    prev_recalls    = prev_round_info.get("global_class_recall", [0]*N_CLASSES)
    prev_bal        = float(np.mean(prev_recalls))
    prev_rare       = float((prev_recalls[CLASS_NAMES.index("DF")] +
                             prev_recalls[CLASS_NAMES.index("VASC")]) / 2.0)
    delta_bal  = f"{(bal_acc - prev_bal):+.1%} (önceki tura göre)"
    delta_rare = f"{(rare_acc - prev_rare):+.1%} (önceki tura göre)"
else:
    delta_bal = delta_rare = None

# Top-level rare-class alarm banner
top_alerts = rare_class_alert(cur_global_recalls)
if top_alerts:
    st.markdown(
        f'<div class="alert-rare">⚠️ <strong>Nadir Sınıf Alarmı:</strong> '
        f'{", ".join(top_alerts)} recall &lt; %{RARE_ALERT_THRESHOLD*100:.0f} '
        f'— dışlama önizlemesi için Tab 3\'ü açın.</div>',
        unsafe_allow_html=True
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tur İlerlemesi", f"{selected_round + 1} / {total_rounds}")
with col2:
    st.metric("Dengeli Doğruluk", f"{bal_acc:.1%}", delta=delta_bal)
with col3:
    st.metric(
        "Nadir Sınıf Ort. (DF & VASC)", f"{rare_acc:.1%}",
        delta=delta_rare,
        delta_color="normal" if rare_acc >= RARE_ALERT_THRESHOLD else "inverse"
    )
with col4:
    st.metric("Dışlanan / Kırpılan Merkez", f"{excluded_count} / {N_CLIENTS}")

# Trend summary (last 5 rounds)
trend = trend_summary(rounds_data)
if trend:
    st.caption(f"📊 **Trend:** {trend}")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sınıf Bazlı İlerleme",
    "🏥 İstemci Telemetrisi & Savunma",
    "🔍 Dışlama Önizleme",
    "⚖️ Savunma Karşılaştırması",
    "📋 Denetim Raporu"
])


# ── TAB 1: Per-class progress ───────────────────────────────────────────────
with tab1:
    st.subheader("Tur Bazında Sınıf Başarımı (Per-Class Recall)")
    st.caption("Federe eğitim turları boyunca her bir cilt lezyonunun yakalanma oranındaki değişim.")

    # Toggle: show rare only or all classes
    show_rare_only = st.toggle("Yalnızca nadir sınıfları göster (DF & VASC)", value=False)

    history = []
    for r in rounds_data:
        t   = r["round"]
        rec = r["global_class_recall"]
        for c_idx, val in enumerate(rec):
            cls_name = CLASS_NAMES[c_idx]
            if show_rare_only and cls_name not in RARE_CLASSES:
                continue
            history.append({
                "Tur":   t,
                "Sınıf": cls_name,
                "Recall": val,
                "Tür":   "Nadir (Rare)" if cls_name in RARE_CLASSES else "Yaygın/Orta"
            })
    df_hist = pd.DataFrame(history)

    fig = px.line(
        df_hist, x="Tur", y="Recall", color="Sınıf",
        title="Tanı Sınıflarının Tur Bazlı Recall Eğrileri",
        markers=True,
        line_dash="Tür"
    )
    fig.add_vline(
        x=selected_round, line_width=2, line_dash="dash", line_color="red",
        annotation_text=f"Seçili Tur: {selected_round}"
    )
    # Add rare-class alert threshold line
    fig.add_hline(
        y=RARE_ALERT_THRESHOLD, line_dash="dot", line_color="orange",
        annotation_text=f"Nadir sınıf eşiği (%{RARE_ALERT_THRESHOLD*100:.0f})"
    )
    fig.update_layout(yaxis_range=[0, 1.0], height=480)
    st.plotly_chart(fig, width="stretch")

    # Per-class metric cards for selected round
    st.markdown("#### Seçili Turdaki Sınıf Başarım Detayları")
    c_cols = st.columns(8)
    for idx, name in enumerate(CLASS_NAMES):
        with c_cols[idx]:
            val     = cur_global_recalls[idx]
            is_rare = name in RARE_CLASSES
            is_low  = val < RARE_ALERT_THRESHOLD and is_rare
            st.metric(
                label=f"{name} {'🔴' if is_low else '⚠️' if is_rare else ''}",
                value=f"{val:.1%}"
            )

    # Automatic trend interpretation
    if trend:
        st.info(f"🤖 **Otomatik Yorum:** {trend}")


# ── TAB 2: Client telemetry ─────────────────────────────────────────────────
with tab2:
    st.subheader(f"Tur {selected_round} — İstemci Durumu ve Telemetri")
    st.caption("Hastanelerden görüntü alınmaz; yalnızca doğrulama özeti, norm ve koordinat kırpılma durumları.")

    client_rows = []
    for k in range(N_CLIENTS):
        c_info    = clients_round.get(str(k), {})
        c_recalls = c_info.get("class_recall", [0]*N_CLASSES)

        def fmt_rec(v):
            return "—" if (v is None or np.isnan(v)) else f"{v:.1%}"

        df_idx   = CLASS_NAMES.index("DF")
        vasc_idx = CLASS_NAMES.index("VASC")
        excluded = c_info.get("excluded", False)
        client_rows.append({
            "Merkez":          f"Center {k}",
            "Güncelleme Norm": round(c_info.get("update_norm", 0.0), 4),
            "Durum":           "DIŞLANDI / KIRPILDI" if excluded else "DAHİL",
            "Gerekçe":         c_info.get("reason") or "Normal varyans içinde",
            "MEL":             fmt_rec(c_recalls[0]),
            "NV":              fmt_rec(c_recalls[1]),
            "DF ⚠️":          fmt_rec(c_recalls[df_idx]),
            "VASC ⚠️":        fmt_rec(c_recalls[vasc_idx])
        })
    df_clients = pd.DataFrame(client_rows)

    def color_status(val):
        if "DIŞLANDI" in str(val):
            return "background-color:#FFCDD2; color:#B71C1C; font-weight:bold"
        return "background-color:#C8E6C9; color:#1B5E20; font-weight:bold"

    st.dataframe(df_clients.style.map(color_status, subset=["Durum"]),
                 width="stretch")

    # Heatmap
    st.markdown("#### İstemciler Arası Sınıf Başarım Karşılaştırması")
    matrix_recalls = [
        clients_round.get(str(k), {}).get("class_recall", [0]*N_CLASSES)
        for k in range(N_CLIENTS)
    ]
    fig_heat = px.imshow(
        matrix_recalls,
        x=CLASS_NAMES,
        y=[f"Center {k}" for k in range(N_CLIENTS)],
        labels=dict(x="Hastalık Sınıfı", y="Merkez", color="Recall"),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, width="stretch")


# --- TAB 3: Dışlama Önizleme ---
with tab3:
    st.subheader("🔍 Karar Öncesi Dışlama Önizlemesi (Exclusion Preview)")
    st.markdown("""
    Bir veya birden fazla hastaneyi dışlamadan önce, sistem **veriye dokunmadan ve tam eğitim yapmadan** alternatif modeli hesaplar.  
    Aşağıdan incelenecek / dışlanacak hastane(leri) seçerek dışlamanın hangi hastalığa ne kadara mal olacağını karar anında görün.
    """)

    st.info("""
    ℹ️ **Metodolojik Not (RQ3 — Tek Turluk Yeniden Toplama Tahmini):**  
    Bu önizleme, dışlanan merkez(ler) olmadan tüm federasyonu baştan 15 tur **yeniden eğitmek yerine**, mevcut turdaki istemci ağırlıklarını anlık olarak yeniden toplar (*One-Round Counterfactual Re-aggregation*).  
    Saldırısız FedAvg'de, son turda, 20 seed üzerinden yeniden eğitimle karşılaştırıldığında: tek bir koşuda korelasyon ortalama $r = 0.68$ (0.41–0.87 arası), 20 seed ortalamasında $r = 0.90$. Önizleme etkinin yönünü ve merkez sıralamasını büyük ölçüde doğru verir, ancak büyüklüğünü yaklaşık 1.5 kat küçük tahmin eder ve küçük merkezlerde (3, 4, 5) ayırt edici değildir. Tek bir koşudaki sayılar seed gürültüsü içerir; Trimmed Mean ve Krum için yeniden eğitim referansı yoktur.
    """)

    updates_dir = RUNS_DIR / selected_run / "updates"
    has_updates = updates_dir.exists() and any(updates_dir.glob("*.npy"))
    if not has_updates:
        st.warning(f"⚠️ **Model Güncellemeleri Bulunamadı:** `{selected_run}` koşusunda `updates/` klasörü yer almıyor. "
                   "Parametre tarama (sweep_*) koşularında disk alanı tasarrufu için ağırlık matrisleri kaydedilmemiş veya silinmiş olabilir. "
                   "Dışlama önizlemesi yalnızca ağırlıkları saklanan tekil koşularda çalışır.")
    else:
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            target_ks = st.multiselect(
                "İncelenecek / Dışlanacak Merkez(ler):",
                range(N_CLIENTS),
                default=[2],
                format_func=lambda x: f"Center {x} {'(Nadir Sınıf Taşıyıcısı)' if x == 2 else '(Cihaz Farkı)' if x == 4 else ''}"
            )
            rule_choice = cur_meta.get("aggregation") or cur_meta.get("agg", "trimmed")
            st.write(f"Kullanılan Toplama Kuralı: **{rule_choice.upper()}**")

        if not target_ks:
            st.info("Lütfen önizleme için en az bir merkez seçin.")
        else:
            # Run preview calculation with real client sizes
            with st.spinner("Ağırlıklar üzerinden dışlama önizlemesi hesaplanıyor..."):
                try:
                    import importlib
                    importlib.reload(preview)
                    prev_result = preview.preview_exclusion(
                        run_id=selected_run,
                        round_t=selected_round,
                        excluded_clients=target_ks,
                        rule=rule_choice,
                        client_sizes=cur_sizes
                    )
                except Exception as e:
                    st.error(f"Önizleme hesaplanırken hata oluştu: {e}")
                    prev_result = None

            if prev_result:
                expl = prev_result["explanation"]
                st.info(f"📊 **Dışlama Etki Özeti:** {expl}")
                if not prev_result["has_real_features"]:
                    st.warning("⚠️ **Yaklaşık Tahmin:** `features/` altında test öznitelikleri bulunamadı. Aşağıdaki sayılar "
                               "model değerlendirmesinden değil, tur telemetrisinden türetilen kaba bir tahmindir.")

                # Waterfall / Bar chart of Class Costs
                costs = prev_result["class_costs"]
                deltas = prev_result["delta_recalls"]

                df_cost = pd.DataFrame({
                    "Sınıf": CLASS_NAMES,
                    "Maliyet (Kayıp)": costs,
                    "Değişim": deltas,
                    "Renk": ["#D32F2F" if d < -0.0001 else "#388E3C" if d > 0.0001 else "#757575" for d in deltas]
                })

                target_str = ", ".join(f"Center {k}" for k in target_ks)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_cost["Sınıf"],
                    y=df_cost["Değişim"] * 100,
                    marker_color=df_cost["Renk"],
                    text=[f"{v:+.1f}%" for v in (df_cost["Değişim"] * 100)],
                    textposition="auto"
                ))
                fig_bar.update_layout(
                    title=f"{target_str} Dışlandığında Sınıf Başına Recall Değişimi (%)",
                    yaxis_title="Recall Değişimi (%) — Eksi Değer Kaybı Gösterir",
                    height=380
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Ground Truth LOO comparison (RQ3) - ONLY shown for clean FedAvg run
                ref_coalition_path = Path("reference/coalition_recall_seeds.npy")
                cur_agg_str = str(cur_meta.get("aggregation") or cur_meta.get("agg") or "").lower()
                cur_atk_val = cur_meta.get("attack_client") if "attack_client" in cur_meta else cur_meta.get("attack")
                is_clean_fedavg = (
                    ("fedavg" in cur_agg_str or ("fedavg" in selected_run.lower() and "trimmed" not in selected_run.lower() and "krum" not in selected_run.lower()))
                    and "trimmed" not in cur_agg_str
                    and "krum" not in cur_agg_str
                    and (cur_atk_val is None or str(cur_atk_val).lower() in ["none", "-1", "null", ""])
                    and ("atk0" not in selected_run and "atk_0" not in selected_run)
                )

                is_full_coalition = not cur_meta.get("excluded_clients")   # the reference compares against the full federation
                if ref_coalition_path.exists() and len(target_ks) == 1 and is_clean_fedavg and is_full_coalition:
                    tgt = target_ks[0]
                    try:
                        V_seeds = np.load(ref_coalition_path)  # (64, 20, 8)
                        full_v = V_seeds[63].mean(0)
                        loo_v = V_seeds[63 ^ (1 << tgt)].mean(0)
                        loo_costs = full_v - loo_v
                        
                        with st.expander(f"🔬 RQ3 Analizi: Merkez {tgt} İçin Önizleme vs Gerçek Yeniden Eğitim (Ground Truth LOO)", expanded=False):
                            st.caption(
                                "Şevval'in 20 tohum üzerinden baştan eğittiği tam koalisyon verisi (Ground Truth LOO) ile "
                                "konsolun tek turluk hızlı ağırlık önizlemesini yan yana kıyaslayın (Yalnızca saldırısız FedAvg referansı)."
                            )
                            comp_rows = []
                            for c_idx, c_name in enumerate(CLASS_NAMES):
                                comp_rows.append({
                                    "Sınıf": c_name,
                                    "Maliyet Türü": "Konsol Önizlemesi (Tek Tur)",
                                    "Recall Kaybı (%)": costs[c_idx] * 100
                                })
                                comp_rows.append({
                                    "Sınıf": c_name,
                                    "Maliyet Türü": "Gerçek Yeniden Eğitim (Ground Truth LOO)",
                                    "Recall Kaybı (%)": float(loo_costs[c_idx]) * 100
                                })
                            df_rq3 = pd.DataFrame(comp_rows)
                            fig_rq3 = px.bar(
                                df_rq3, x="Sınıf", y="Recall Kaybı (%)", color="Maliyet Türü", barmode="group",
                                title=f"Merkez {tgt}: Hızlı Önizleme vs 15 Turluk Tam Yeniden Eğitim Maliyeti",
                                color_discrete_map={"Konsol Önizlemesi (Tek Tur)": "#1E88E5", "Gerçek Yeniden Eğitim (Ground Truth LOO)": "#FF8F00"}
                            )
                            fig_rq3.update_layout(height=340)
                            st.plotly_chart(fig_rq3, use_container_width=True)
                            st.caption("📌 **Dikkat:** DF (recall ~0.03), SCC (~0.06) ve AK (~0.17) FedAvg tarafından zayıf öğrenildiği için bu sınıfların Shapley değerleri tohum varyansı içindedir.")
                    except Exception:
                        pass

                # Karar Alma Paneli
                st.markdown("### ⚖️ Operatör Kararı ve Denetim Kaydı")
                st.caption("Vereceğiniz karar, sınıf bazlı gerekçesiyle birlikte sistem denetim izine (Audit Trail) kalıcı olarak işlenecektir.")

                col_dec1, col_dec2 = st.columns([2, 1])
                default_retain_note = "Model katkısı ve azınlık sınıfları koruma amacıyla federasyonda tutuldu (İstisna uygulandı)."
                default_exclude_note = "Model kararlılığı ve sınıf başarım dengesi gözetilerek dışlama onaylandı."
                
                with col_dec1:
                    op_note = st.text_input("Karar Gerekçesi / Operatör Notu:", value=default_exclude_note)
                with col_dec2:
                    st.write("")
                    st.write("")
                    prev_summary = {
                        "explanation": expl,
                        "rare_cost_str": f"DF: {costs[5]:+.1%}, VASC: {costs[6]:+.1%}",
                        "bal_acc_delta": f"{prev_result['bal_acc_delta']:+.1%}"
                    }

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("🚫 DIŞLA", use_container_width=True):
                            for k_id in target_ks:
                                audit_log.log_decision(selected_run, selected_round, k_id, "EXCLUDED", op_note, prev_summary)
                            st.success(f"{target_str} dışlandı ve denetim kaydına yazıldı!")
                    with b_col2:
                        if st.button("✅ FEDERASYONDA TUT", use_container_width=True):
                            note_to_save = default_retain_note if op_note == default_exclude_note else op_note
                            for k_id in target_ks:
                                audit_log.log_decision(selected_run, selected_round, k_id, "RETAINED", note_to_save, prev_summary)
                            st.success(f"{target_str} için istisna tanımlandı ve federasyonda tutuldu!")

        # --- Shapley Ground Truth Referansı (Şevval'in 64 Koalisyonluk Referansı) ---
        ref_csv_path = Path("reference/shapley_class.csv")
        ref_img_path = Path("reference/shapley_heatmap.png")
        if ref_csv_path.exists() or ref_img_path.exists():
            st.divider()
            with st.expander("📊 64 Koalisyonluk Tam Shapley Değerleri Referansı (Şevval / Ground Truth)", expanded=False):
                st.caption("Konsolun hızlı dışlama tahminlerini doğrulamak için 64 koalisyonun tamamı eğitilerek elde edilen kesin matematiksel referans.")
                r_col1, r_col2 = st.columns([1, 1])
                with r_col1:
                    if ref_csv_path.exists():
                        st.markdown("**Sınıf Bazlı Shapley Tablosu:**")
                        df_shapley = pd.read_csv(ref_csv_path)
                        st.dataframe(df_shapley, use_container_width=True)
                with r_col2:
                    if ref_img_path.exists():
                        st.markdown("**Shapley Katkı Isı Haritası (Heatmap):**")
                        st.image(str(ref_img_path), width="stretch")


# ── TAB 4: Defense comparison ───────────────────────────────────────────────
with tab4:
    st.subheader("⚖️ Koşum & Savunma Karşılaştırması (RQ1)")
    st.markdown("Farklı savunma kurallarını veya dışlama senaryolarını yan yana karşılaştırın.")

    real_runs_list   = [r for r in available_runs if not is_synthetic_run(r)]
    candidate_pool   = real_runs_list if len(real_runs_list) >= 2 else available_runs

    # Auto-select one clean run per defense rule
    triad = []
    for def_rule in ["fedavg", "trimmed", "krum"]:
        match = next(
            (r for r in candidate_pool
             if def_rule in r.lower() and ("clean" in r.lower() or "atknone" in r.lower())),
            None
        )
        if not match:
            match = next((r for r in candidate_pool if def_rule in r.lower()), None)
        if match and match not in triad:
            triad.append(match)

    default_selected = (triad if len(triad) >= 2
                        else candidate_pool[:min(3, len(candidate_pool))])

    runs_to_compare = st.multiselect(
        "Karşılaştırılacak koşumlar:",
        available_runs,
        default=default_selected,
        format_func=format_run_label
    )

    if len(runs_to_compare) >= 2:
        comp_data = []
        for r_name in runs_to_compare:
            r_rounds = load_run_rounds(r_name)
            if r_rounds:
                last_r    = r_rounds[-1]
                rec       = last_r["global_class_recall"]
                agg_type  = last_r.get("meta", {}).get("aggregation", r_name).upper()
                exc_list  = last_r.get("meta", {}).get("excluded_clients", [])
                label_sfx = f" (Dışlanan: C{','.join(map(str, exc_list))})" if exc_list else ""

                for c_idx, val in enumerate(rec):
                    comp_data.append({
                        "Koşum":      f"{r_name}{label_sfx}",
                        "Savunma":    agg_type,
                        "Sınıf":      CLASS_NAMES[c_idx],
                        "Recall":     val,
                        "Sınıf Türü": "Nadir" if CLASS_NAMES[c_idx] in RARE_CLASSES else "Diğer"
                    })

        df_comp   = pd.DataFrame(comp_data)
        fig_comp  = px.bar(
            df_comp, x="Sınıf", y="Recall", color="Koşum", barmode="group",
            title="Seçili Koşumların Son Tur Sınıf Başarım Kıyaslaması",
            pattern_shape="Sınıf Türü",
            pattern_shape_map={"Nadir": "/", "Diğer": ""}
        )
        fig_comp.update_layout(yaxis_range=[0, 1.0], height=420)
        st.plotly_chart(fig_comp, width="stretch")

        st.info("""
💡 **Gözlem:**  
Nadir sınıfları (DF ve VASC) taşıyan Center 2 dışlandığında veya Trimmed Mean/Krum bu merkezi kırptığında,  
diğer sınıflar yüksek kalmaya devam etse de nadir sınıfların teşhis oranı dramatik biçimde düşmektedir.
""")
    else:
        st.warning("Karşılaştırma için lütfen en az iki koşum seçin.")


# ── TAB 5: Audit log ────────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Federasyon Denetim İzi (Audit Trail)")
    st.caption("AB Yapay Zekâ Yasası ve tıbbi tanı mevzuatına uygun, gerekçeli operatör dışlama kayıtları.")

    audit_df = audit_log.get_audit_df()
    if not audit_df.empty:
        st.dataframe(audit_df, width="stretch")
        csv_data = audit_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Denetim Raporunu İndir (CSV)",
            data=csv_data,
            file_name="fedcost_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz denetim kaydı yok. 'Dışlama Önizleme' sekmesinden bir karar vererek ilk kaydı oluşturun.")
