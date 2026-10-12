"""Federasyon Sağlık Konsolu (Federation Health Console)
Streamlit Application for FedCost Prototype.
Developed by Bülent (Console & Exclusion Preview Track).
"""

import json
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

st.set_page_config(
    page_title="Federasyon Sağlık Konsolu | FedCost",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-maverick {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-attack {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def is_synthetic_run(run_name):
    """Determine whether a run was produced by fake/synthetic simulation."""
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

def format_run_label(run_name):
    return f"🧪 [Sentetik] {run_name}" if is_synthetic_run(run_name) else f"🟢 [Gerçek Motor] {run_name}"

def get_available_runs():
    if not RUNS_DIR.exists():
        return []
    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(runs)

def load_run_rounds(run_id):
    run_path = RUNS_DIR / run_id
    round_files = sorted(run_path.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    records = []
    for rf in round_files:
        try:
            records.append(json.loads(rf.read_text(encoding="utf-8")))
        except Exception:
            pass
    return records

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏥 FedCost Konsolu")
    st.caption("Federe Tıbbi Görüntülemede Korumanın ve Dışlamanın Sınıf Bazlı Bedeli")
    st.divider()

    available_runs = get_available_runs()
    if not available_runs:
        st.warning("Henüz kayıtlı bir federasyon koşusu bulunamadı!")
        if st.button("Örnek Koşumları Oluştur"):
            import fake_run_generator
            fake_run_generator.main()
            st.rerun()
        st.stop()

    real_runs = [r for r in available_runs if not is_synthetic_run(r)]
    run_filter = st.radio(
        "Koşu Türü Filtresi:",
        ["Tümü", "🟢 Gerçek Motor", "🧪 Sentetik"],
        horizontal=True
    )
    if run_filter == "🟢 Gerçek Motor":
        filtered_runs = [r for r in available_runs if not is_synthetic_run(r)]
    elif run_filter == "🧪 Sentetik":
        filtered_runs = [r for r in available_runs if is_synthetic_run(r)]
    else:
        filtered_runs = available_runs

    if not filtered_runs:
        st.info("Bu filtreye uygun koşu bulunamadı.")
        filtered_runs = available_runs

    default_idx = 0
    if "active_run" in st.session_state and st.session_state["active_run"] in filtered_runs:
        default_idx = filtered_runs.index(st.session_state["active_run"])
    elif run_filter == "Tümü" and real_runs and real_runs[0] in filtered_runs:
        default_idx = filtered_runs.index(real_runs[0])

    selected_run = st.selectbox(
        "Aktif Federasyon Koşumu:",
        filtered_runs,
        index=default_idx,
        format_func=format_run_label
    )
    st.session_state["active_run"] = selected_run
    rounds_data = load_run_rounds(selected_run)
    
    if not rounds_data:
        st.error("Seçilen koşumda geçerli tur verisi yok.")
        st.stop()

    total_rounds = len(rounds_data)
    selected_round = st.slider("İncelenen Tur (Round):", min_value=0, max_value=total_rounds - 1, value=total_rounds - 1)

    cur_meta = rounds_data[selected_round].get("meta", {})
    cur_agg = cur_meta.get("aggregation") or cur_meta.get("agg", "Bilinmiyor")
    cur_attack = cur_meta.get("attack_client") if "attack_client" in cur_meta else cur_meta.get("attack")
    cur_sizes = cur_meta.get("client_sizes", [9930, 3163, 2691, 1807, 655, 351])
    attack_label = f"Merkez {cur_attack} (Etiket Çevirme)" if cur_attack is not None and cur_attack >= 0 else "Saldırı Yok (Temiz)"

    st.info(f"""
    **Savunma:** `{str(cur_agg).upper()}`  
    **Saldırı Durumu:** `{attack_label}`  
    **Tür:** `{'🧪 Sentetik Simülasyon' if is_synthetic_run(selected_run) else '🟢 Gerçek Motor Koşumu'}`  
    **Toplam Tur Sayısı:** `{total_rounds}`
    """)

    st.divider()
    with st.expander("🚀 Yeni Koşum Başlat (Gerçek Motor veya Simülasyon)", expanded=False):
        st.caption("Farklı savunma kuralları, saldırı senaryoları ve hastane koalisyonlarıyla yeni bir koşum üretin.")
        
        run_mode = st.radio(
            "Çalıştırma Yöntemi:",
            ["🟢 Gerçek Motor (engine.py)", "🧪 Hızlı Sentetik Simülasyon"],
            index=0
        )

        inc_centers = st.multiselect(
            "Dahil Edilecek Hastaneler:",
            range(N_CLIENTS),
            default=list(range(N_CLIENTS)),
            format_func=lambda x: f"Center {x} {'(Nadir DF/VASC)' if x == 2 else '(Cihaz Farkı)' if x == 4 else ''}"
        )
        exc_centers = [c for c in range(N_CLIENTS) if c not in inc_centers]
        if exc_centers:
            st.warning(f"Dışlanan Hastaneler: {', '.join(f'Center {c}' for c in exc_centers)}")
        else:
            st.success("Tüm hastaneler dahil (Tam Koalisyon).")
        
        sim_agg = st.selectbox("Savunma Kuralı:", ["fedavg", "trimmed", "krum"], index=0)
        sim_attack = st.selectbox("Saldırı Senaryosu:", ["Yok", "Center 0 (Etiket Çevirme)"], index=0)
        atk_id = 0 if "Center 0" in sim_attack else None
        sim_seed = st.number_input("Rastgele Tohum (Seed):", min_value=0, max_value=999, value=0, step=1,
                                   help="Referans deneyler 0–19 tohumları arasındadır.")

        exc_suffix = f"_no_c{'_c'.join(map(str, exc_centers))}" if exc_centers else ""

        if run_mode == "🟢 Gerçek Motor (engine.py)":
            st.info("💡 **Gerçek Motor:** Umut'un EfficientNet öznitelikleri üzerinde çalışan saf NumPy federasyon motorunu (SGD, yerel epochlar, gerçek gradyanlar) çalıştırır.")
            sim_rounds = st.slider("Tur Sayısı:", min_value=5, max_value=30, value=15, step=5)
            default_name = f"engine_{sim_agg}{exc_suffix}_atk{atk_id if atk_id is not None else 'None'}_s{sim_seed}"
            custom_run_id = st.text_input("Koşum Adı:", value=default_name)
            
            if st.button("🚀 Gerçek Motoru Çalıştır", use_container_width=True):
                feats_dir = FEATURES_DIR
                missing_feats = [
                    f"center_{k}_{split}.npz"
                    for k in range(N_CLIENTS)
                    for split in ("train", "test")
                    if not (feats_dir / f"center_{k}_{split}.npz").exists()
                ]
                if missing_feats:
                    st.error(
                        f"❌ **Gerçek Öznitelikler Bulunamadı:** `features/` dizininde {len(missing_feats)} adet dosya eksik "
                        f"(ilk eksik: `{missing_feats[0]}`). Gerçek federasyon motoru yalnızca Fed-ISIC2019 EfficientNet "
                        "öznitelikleriyle çalışabilir. Lütfen öznitelikleri `features/` klasörüne aktarın veya "
                        "simülasyon için '🧪 Hızlı Sentetik Simülasyon' modunu seçin."
                    )
                else:
                    with st.spinner(f"Gerçek federasyon motoru {sim_rounds} tur eğitiliyor..."):
                        import importlib
                        import engine
                        importlib.reload(engine)
                        engine.run(
                            agg=sim_agg,
                            rounds=sim_rounds,
                            attack=atk_id,
                            seed=int(sim_seed),
                            run_id=custom_run_id,
                            excluded_clients=exc_centers
                        )
                        st.session_state["active_run"] = custom_run_id
                        st.success(f"'{custom_run_id}' gerçek motor ile eğitildi ve kaydedildi!")
                        st.rerun()
        else:
            st.caption("Sentetik dağılımlar üzerinden anlık simülasyon üretir.")
            default_name = f"sim_{sim_agg}{exc_suffix}_atk{atk_id if atk_id is not None else 'None'}_s{sim_seed}"
            custom_run_id = st.text_input("Koşum Adı:", value=default_name)
            
            if st.button("🚀 Simülasyonu Koş ve Kaydet", use_container_width=True):
                with st.spinner("15 turluk federasyon simüle ediliyor..."):
                    import importlib
                    import fake_run_generator
                    importlib.reload(fake_run_generator)
                    fake_run_generator.generate_run(
                        run_id=custom_run_id,
                        aggregation=sim_agg,
                        attack_client=atk_id,
                        num_rounds=15,
                        seed=int(sim_seed),
                        excluded_clients=exc_centers
                    )
                    st.session_state["active_run"] = custom_run_id
                    st.success(f"'{custom_run_id}' başarıyla oluşturuldu!")
                    st.rerun()

    st.divider()
    if st.button("🔄 Sahte Koşumları Yenile"):
        import fake_run_generator
        fake_run_generator.main()
        st.success("Koşumlar yenilendi!")
        st.rerun()

# --- HEADER METRICS ---
cur_round_info = rounds_data[selected_round]
cur_global_recalls = cur_round_info.get("global_class_recall", [0]*N_CLASSES)
bal_acc = float(np.mean(cur_global_recalls))
rare_acc = float((cur_global_recalls[5] + cur_global_recalls[6]) / 2.0)  # DF & VASC

clients_round = cur_round_info.get("clients", {})
excluded_count = sum(1 for c in clients_round.values() if c.get("excluded", False))

if selected_round > 0:
    prev_round_info = rounds_data[selected_round - 1]
    prev_recalls = prev_round_info.get("global_class_recall", [0]*N_CLASSES)
    prev_bal = float(np.mean(prev_recalls))
    prev_rare = float((prev_recalls[5] + prev_recalls[6]) / 2.0)
    delta_bal = f"{(bal_acc - prev_bal):+.1%} (önceki tura göre)"
    delta_rare = f"{(rare_acc - prev_rare):+.1%} (önceki tura göre)"
else:
    delta_bal = None
    delta_rare = None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tur İlerlemesi", f"{selected_round + 1} / {total_rounds}")
with col2:
    st.metric("Dengeli Doğruluk (Bal. Acc)", f"{bal_acc:.1%}", delta=delta_bal)
with col3:
    st.metric("Nadir Sınıf Ortalaması (DF & VASC)", f"{rare_acc:.1%}", 
              delta=delta_rare, 
              delta_color="normal" if rare_acc >= 0.4 else "inverse")
with col4:
    st.metric("Dışlanan / Kırpılan Merkez", f"{excluded_count} / {N_CLIENTS}")

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sınıf Bazlı İlerleme",
    "🏥 İstemci Telemetrisi & Savunma",
    "🔍 Dışlama Önizleme (Karar Anı)",
    "⚖️ Savunma Karşılaştırması (RQ1)",
    "📋 Denetim Raporu"
])

# --- TAB 1: Sınıf Bazlı İlerleme ---
with tab1:
    st.subheader("Tur Bazında Sınıf Başarımı (Per-Class Recall)")
    st.caption("Federe eğitim turları boyunca her bir cilt lezyonunun yakalanma oranındaki değişim")

    history = []
    for r in rounds_data:
        t = r["round"]
        rec = r["global_class_recall"]
        for c_idx, val in enumerate(rec):
            history.append({
                "Tur": t,
                "Sınıf": CLASS_NAMES[c_idx],
                "Recall": val,
                "Tür": "Nadir (Rare)" if CLASS_NAMES[c_idx] in ["DF", "VASC"] else "Yaygın/Orta"
            })
    df_hist = pd.DataFrame(history)

    fig = px.line(
        df_hist, x="Tur", y="Recall", color="Sınıf",
        title="8 Tanı Sınıfının Tur Bazlı Recall Eğrileri",
        markers=True,
        line_dash="Tür"
    )
    fig.add_vline(x=selected_round, line_width=2, line_dash="dash", line_color="red", 
                  annotation_text=f"Seçili Tur: {selected_round}")
    fig.update_layout(yaxis_range=[0, 1.0], height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Güncel tur sınıf tablosu
    st.markdown("#### Seçili Turdaki Sınıf Başarım Detayları")
    c_cols = st.columns(8)
    for idx, name in enumerate(CLASS_NAMES):
        with c_cols[idx]:
            val = cur_global_recalls[idx]
            is_rare = name in ["DF", "VASC"]
            st.metric(
                label=f"{name} {'⚠️' if is_rare else ''}", 
                value=f"{val:.1%}"
            )

# --- TAB 2: İstemci Telemetrisi & Savunma ---
with tab2:
    st.subheader(f"Tur {selected_round} - İstemci Durumu ve Telemetri")
    st.caption("Hastanelerden görüntü alınmaz; yalnızca doğrulama özeti, norm ve koordinat kırpılma durumları toplanır.")

    client_rows = []
    for k in range(N_CLIENTS):
        c_info = clients_round.get(str(k), {})
        c_recalls = c_info.get("class_recall", [0]*N_CLASSES)
        def fmt_rec(v):
            return "—" if (v is None or np.isnan(v)) else f"{v:.1%}"

        client_rows.append({
            "Merkez": f"Center {k}",
            "Güncelleme Normu": c_info.get("update_norm", 0.0),
            "Durum": "DIŞLANDI / KIRPILDI" if c_info.get("excluded", False) else "DAHİL",
            "Gerekçe": c_info.get("reason") or "Normal varyans içinde",
            "MEL": fmt_rec(c_recalls[0]),
            "NV": fmt_rec(c_recalls[1]),
            "DF (Nadir)": fmt_rec(c_recalls[5]),
            "VASC (Nadir)": fmt_rec(c_recalls[6])
        })
    df_clients = pd.DataFrame(client_rows)

    def color_status(val):
        if "DIŞLANDI" in val:
            return "background-color: #FFCDD2; color: #B71C1C; font-weight: bold"
        return "background-color: #C8E6C9; color: #1B5E20; font-weight: bold"

    st.dataframe(df_clients.style.map(color_status, subset=["Durum"]), use_container_width=True)

    # İstemci bazlı sınıf recall ısı haritası
    st.markdown("#### İstemciler Arası Sınıf Başarım Karşılaştırması")
    matrix_recalls = []
    for k in range(N_CLIENTS):
        matrix_recalls.append(clients_round.get(str(k), {}).get("class_recall", [0]*N_CLASSES))
    
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
    st.plotly_chart(fig_heat, use_container_width=True)

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
    Bu hızlı operasyonel tahmin tek turluk ağırlık gradyanlarına dayanır (pilot ölçüm: seed 0, tur 14, FedAvg için $r \\approx 0.45$; 20-seed genelleme analizi hesaplanmaktadır). Önizleme anlık bir operasyonel karar desteğidir; mutlak koalisyon Shapley değeri değildir.
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

                # Waterfall / Bar chart of Class Costs
                costs = prev_result["class_costs"]
                deltas = prev_result["delta_recalls"]

                df_cost = pd.DataFrame({
                    "Sınıf": CLASS_NAMES,
                    "Maliyet (Kayıp)": costs,
                    "Değişim": deltas,
                    "Renk": ["#D32F2F" if c > 0.03 else "#388E3C" if c < -0.03 else "#757575" for c in costs]
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

                if ref_coalition_path.exists() and len(target_ks) == 1 and is_clean_fedavg:
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

# --- TAB 4: Savunma Karşılaştırması (RQ1) ---
with tab4:
    st.subheader("⚖️ Koşum ve Savunma Karşılaştırması (RQ1 & LOO Deneyleri)")
    st.markdown("""
    Farklı savunma kurallarını veya belirli hastanelerin dışlandığı koşumları yan yana karşılaştırarak sınıf bazlı etkisini inceleyin.
    """)

    real_runs = [r for r in available_runs if not is_synthetic_run(r)]
    candidate_pool = real_runs if len(real_runs) >= 2 else available_runs
    
    # Intelligently select FedAvg, Trimmed, Krum across defenses
    triad = []
    for def_rule in ["fedavg", "trimmed", "krum"]:
        match = next(
            (r for r in candidate_pool if def_rule in r.lower() and ("clean" in r.lower() or "atknone" in r.lower())),
            None
        )
        if not match:
            match = next((r for r in candidate_pool if def_rule in r.lower()), None)
        if match and match not in triad:
            triad.append(match)

    if len(triad) >= 2:
        default_selected = triad
    elif len(candidate_pool) >= 2:
        default_selected = candidate_pool[:min(3, len(candidate_pool))]
    else:
        default_selected = available_runs[:min(3, len(available_runs))]

    runs_to_compare = st.multiselect(
        "Karşılaştırılacak Koşumlar:",
        available_runs,
        default=default_selected,
        format_func=format_run_label
    )

    if len(runs_to_compare) >= 2:
        comp_data = []
        for r_name in runs_to_compare:
            r_rounds = load_run_rounds(r_name)
            if r_rounds:
                last_r = r_rounds[-1]
                rec = last_r["global_class_recall"]
                agg_type = last_r.get("meta", {}).get("aggregation", r_name).upper()
                exc_list = last_r.get("meta", {}).get("excluded_clients", [])
                label_suffix = f" (Dışlanan: C{','.join(map(str, exc_list))})" if exc_list else ""
                
                for c_idx, val in enumerate(rec):
                    comp_data.append({
                        "Koşum": f"{r_name}{label_suffix}",
                        "Savunma": agg_type,
                        "Sınıf": CLASS_NAMES[c_idx],
                        "Recall": val,
                        "Sınıf Türü": "Nadir (DF & VASC)" if CLASS_NAMES[c_idx] in ["DF", "VASC"] else "Diğerleri"
                    })
        df_comp = pd.DataFrame(comp_data)

        fig_comp = px.bar(
            df_comp, x="Sınıf", y="Recall", color="Koşum", barmode="group",
            title="Seçili Koşumların 8 Sınıf Üzerindeki Son Tur Başarım Kıyaslaması"
        )
        fig_comp.update_layout(yaxis_range=[0, 1.0], height=420)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.info("""
        💡 **Gözlem:**  
        Nadir sınıfları (DF ve VASC) taşıyan Center 2 dışlandığında veya Trimmed Mean/Krum bu merkezi kırptığında, 
        diğer sınıflar yüksek kalmaya devam etse de nadir sınıfların teşhis oranı dramatize bir şekilde düşmektedir.
        """)
    else:
        st.warning("Karşılaştırma grafiği için lütfen en az iki koşum seçin.")

# --- TAB 5: Denetim Raporu ---
with tab5:
    st.subheader("📋 Federasyon Denetim İzi (Audit Trail)")
    st.caption("AB Yapay Zekâ Yasası ve tıbbi tanı mevzuatına uygun, gerekçeli operatör dışlama kayıtları.")

    audit_df = audit_log.get_audit_df()
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True)
        csv_data = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Denetim Raporunu İndir (CSV)",
            data=csv_data,
            file_name="fedcost_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz denetim kaydı bulunmuyor. 'Dışlama Önizleme' sekmesinden bir karar vererek ilk kaydı oluşturabilirsiniz.")
