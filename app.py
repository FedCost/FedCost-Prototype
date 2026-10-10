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
    CLASS_NAMES, N_CLASSES, N_CLIENTS, RUNS_DIR
)
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

    default_idx = 0
    if "active_run" in st.session_state and st.session_state["active_run"] in available_runs:
        default_idx = available_runs.index(st.session_state["active_run"])
    elif len(available_runs) > 1:
        default_idx = 1

    selected_run = st.selectbox("Aktif Federasyon Koşumu:", available_runs, index=default_idx)
    st.session_state["active_run"] = selected_run
    rounds_data = load_run_rounds(selected_run)
    
    if not rounds_data:
        st.error("Seçilen koşumda geçerli tur verisi yok.")
        st.stop()

    total_rounds = len(rounds_data)
    selected_round = st.slider("İncelenen Tur (Round):", min_value=0, max_value=total_rounds - 1, value=total_rounds - 1)

    cur_meta = rounds_data[selected_round].get("meta", {})
    st.info(f"""
    **Savunma:** `{cur_meta.get('aggregation', 'Bilinmiyor').upper()}`  
    **Saldırı Durumu:** `{'Merkez ' + str(cur_meta.get('attack_client')) if cur_meta.get('attack_client') is not None else 'Saldırı Yok (Temiz)'}`  
    **Toplam Tur Sayısı:** `{total_rounds}`
    """)

    st.divider()
    with st.expander("🧪 Yeni Koşum Simüle Et (LOO / Özel Koalisyon)", expanded=False):
        st.caption("İstediğiniz hastaneleri dışarıda bırakıp 15 turluk yeni bir federasyon simülasyonu başlatın.")
        inc_centers = st.multiselect(
            "Dahil Edilecek Hastaneler:",
            range(N_CLIENTS),
            default=[0, 1, 3, 4, 5],  # Center 2 excluded by default as an example
            format_func=lambda x: f"Center {x} {'(Nadir DF/VASC)' if x == 2 else '(Cihaz Farkı)' if x == 4 else ''}"
        )
        exc_centers = [c for c in range(N_CLIENTS) if c not in inc_centers]
        if exc_centers:
            st.warning(f"Dışlanan Hastaneler: {', '.join(f'Center {c}' for c in exc_centers)}")
        else:
            st.success("Tüm hastaneler dahil (Tam Koalisyon).")
            
        sim_agg = st.selectbox("Savunma Kuralı:", ["fedavg", "trimmed", "krum"], index=0)
        sim_attack = st.selectbox("Saldırı Senaryosu:", ["Yok", "Center 0 (Etiket Çevirme)"], index=0)
        
        default_name = f"sim_no_c{'_c'.join(map(str, exc_centers))}_{sim_agg}" if exc_centers else f"sim_all_{sim_agg}"
        custom_run_id = st.text_input("Koşum Adı:", value=default_name)
        
        if st.button("🚀 Simülasyonu Koş ve Kaydet", use_container_width=True):
            with st.spinner("15 turluk federasyon simüle ediliyor..."):
                import importlib
                import fake_run_generator
                importlib.reload(fake_run_generator)
                fake_run_generator.generate_run(
                    run_id=custom_run_id,
                    aggregation=sim_agg,
                    attack_client=0 if "Center 0" in sim_attack else None,
                    num_rounds=15,
                    seed=42,
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

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tur İlerlemesi", f"{selected_round + 1} / {total_rounds}")
with col2:
    st.metric("Dengeli Doğruluk (Bal. Acc)", f"{bal_acc:.1%}", delta=f"{(bal_acc - 0.5):+.1%}")
with col3:
    st.metric("Nadir Sınıf Ortalaması (DF & VASC)", f"{rare_acc:.1%}", 
              delta=f"{(rare_acc - 0.5):+.1%}", 
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
        client_rows.append({
            "Merkez": f"Center {k}",
            "Güncelleme Normu": c_info.get("update_norm", 0.0),
            "Durum": "DIŞLANDI / KIRPILDI" if c_info.get("excluded", False) else "DAHİL",
            "Gerekçe": c_info.get("reason") or "Normal varyans içinde",
            "MEL": f"{c_recalls[0]:.1%}",
            "NV": f"{c_recalls[1]:.1%}",
            "DF (Nadir)": f"{c_recalls[5]:.1%}",
            "VASC (Nadir)": f"{c_recalls[6]:.1%}"
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

    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        target_ks = st.multiselect(
            "İncelenecek / Dışlanacak Merkez(ler):",
            range(N_CLIENTS),
            default=[2],
            format_func=lambda x: f"Center {x} {'(Nadir Sınıf Taşıyıcısı)' if x == 2 else '(Cihaz Farkı)' if x == 4 else ''}"
        )
        rule_choice = cur_meta.get("aggregation", "trimmed")
        st.write(f"Kullanılan Toplama Kuralı: **{rule_choice.upper()}**")

    if not target_ks:
        st.info("Lütfen önizleme için en az bir merkez seçin.")
    else:
        # Run preview calculation
        with st.spinner("Ağırlıklar üzerinden dışlama önizlemesi hesaplanıyor..."):
            try:
                prev_result = preview.preview_exclusion(
                    run_id=selected_run,
                    round_t=selected_round,
                    excluded_clients=target_ks,
                    rule=rule_choice
                )
            except Exception as e:
                st.error(f"Önizleme hesaplanırken hata oluştu: {e}")
                prev_result = None

        if prev_result:
            diag = prev_result["diagnosis"]
            expl = prev_result["explanation"]

            if diag == "VALUABLE_MAVERICK":
                st.error(f"🚨 **DİKKAT — DEĞERLİ AYKIRI MERKEZ (MAVERICK):** {expl}")
            elif diag == "HARMFUL_ATTACKER":
                st.warning(f"🛡️ **ZARARLI İSTEMCİ TESPİTİ:** {expl}")
            else:
                st.info(f"ℹ️ **STANDART MERKEZ:** {expl}")

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

            # Karar Alma Paneli
            st.markdown("### ⚖️ Operatör Kararı ve Denetim Kaydı")
            st.caption("Vereceğiniz karar, sınıf bazlı gerekçesiyle birlikte sistem denetim izine (Audit Trail) kalıcı olarak işlenecektir.")

            col_dec1, col_dec2 = st.columns([2, 1])
            with col_dec1:
                op_note = st.text_input("Karar Gerekçesi / Operatör Notu:", 
                                        value="Nadir hastalık sınıflarındaki (DF/VASC) kritik kaybı önlemek için istisna uygulandı." if diag == "VALUABLE_MAVERICK" else "Güvenlik gereği dışlama onaylandı.")
            with col_dec2:
                st.write("")
                st.write("")
                prev_summary = {
                    "diagnosis": diag,
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
                        for k_id in target_ks:
                            audit_log.log_decision(selected_run, selected_round, k_id, "RETAINED", op_note, prev_summary)
                        st.success(f"{target_str} için istisna tanımlandı ve tutuldu!")

# --- TAB 4: Savunma Karşılaştırması (RQ1) ---
with tab4:
    st.subheader("⚖️ Koşum ve Savunma Karşılaştırması (RQ1 & LOO Deneyleri)")
    st.markdown("""
    Farklı savunma kurallarını veya belirli hastanelerin dışlandığı koşumları yan yana karşılaştırarak sınıf bazlı etkisini inceleyin.
    """)

    default_selected = [r for r in available_runs if r in ['demo_fedavg_clean', 'demo_trimmed_clean', 'demo_krum_clean']]
    if not default_selected:
        default_selected = available_runs[:min(3, len(available_runs))]
        
    runs_to_compare = st.multiselect("Karşılaştırılacak Koşumlar:", available_runs, default=default_selected)

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
