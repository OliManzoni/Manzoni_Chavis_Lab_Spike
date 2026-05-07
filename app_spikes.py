import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Neural Excitability Pipeline", layout="wide")

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Branding") 
with col_r:
    st.markdown("# Pipeline Expert : Excitabilité & Propriétés Intrinsèques")
    st.markdown("### Manzoni Lab | Analyse de la Plasticité Synaptique")
    st.markdown("#### *Extraction automatisée Passif, Excitabilité & Morphologie des PA*")

st.divider()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header("📂 1. Chargement & Unités")
uploaded_file = st.sidebar.file_uploader("Charger un fichier ABF", type=["abf"])
current_unit = st.sidebar.radio("Unité du canal de courant (I_cmd)", ["pA", "nA"])

st.sidebar.header("⚙️ 2. Réglages de Détection")
spike_threshold = st.sidebar.number_input("Seuil de détection des PA (mV)", value=0.0)
dvdt_threshold = st.sidebar.number_input("Seuil dV/dt pour V_threshold (mV/ms)", value=15.0)

# --- LOGIQUE ANALYTIQUE COMPLÈTE ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        # 1. Listes pour l'extraction exhaustive
        courants, voltages_stat, voltages_peak, voltages_rest = [], [], [], []
        spike_counts_raw, v_thresholds = [], []
        
        # Nouvelles listes pour les propriétés du 1er PA de chaque sweep
        ap_amps, ap_widths, ap_rise_times, ap_decay_times, ap_ahps = [], [], [], [], []
        
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6)
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_rest = np.mean(abf.sweepY[0:idx_start])
            v_stat = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            
            v_peak = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            trace_window = abf.sweepY[idx_start:idx_end]
            peaks, _ = find_peaks(trace_window, height=spike_threshold)
            num_spikes = len(peaks)
            
            # Initialisation des valeurs par défaut (NaN) pour ce sweep
            v_thresh_sw = np.nan
            amp_sw, width_sw, rise_sw, decay_sw, ahp_sw = np.nan, np.nan, np.nan, np.nan, np.nan
            
            if num_spikes > 0:
                first_peak_idx = peaks[0]
                search_start = max(0, first_peak_idx - int(sr * 0.015))
                segment = trace_window[search_start:first_peak_idx]
                
                if len(segment) > 1:
                    # Lissage et dérivée pour trouver le seuil
                    smoothed_segment = gaussian_filter1d(segment, sigma=1)
                    dvdt = np.diff(smoothed_segment) / dt_ms
                    crossings = np.where(dvdt > dvdt_threshold)[0]
                    
                    if len(crossings) > 0:
                        idx_thresh_segment = crossings[0]
                        v_thresh_sw = segment[idx_thresh_segment]
                        idx_thresh_global = search_start + idx_thresh_segment
                        
                        # --- ANALYSE MORPHOLOGIQUE DU PA ---
                        v_peak_ap = trace_window[first_peak_idx]
                        amp_sw = v_peak_ap - v_thresh_sw
                        
                        if amp_sw > 0:
                            # Calculs des niveaux de tension (10%, 50%, 90%)
                            v_10 = v_thresh_sw + 0.10 * amp_sw
                            v_50 = v_thresh_sw + 0.50 * amp_sw
                            v_90 = v_thresh_sw + 0.90 * amp_sw
                            
                            # Délimitation temporelle pour la recherche (avant et après le pic)
                            ap_segment_up = trace_window[idx_thresh_global:first_peak_idx]
                            search_end_down = min(len(trace_window), first_peak_idx + int(sr * 0.010)) # 10ms max après pic
                            ap_segment_down = trace_window[first_peak_idx:search_end_down]
                            
                            # 1. Rise Time (10% -> 90%)
                            cross_10_up = np.where(ap_segment_up >= v_10)[0]
                            cross_90_up = np.where(ap_segment_up >= v_90)[0]
                            if len(cross_10_up) > 0 and len(cross_90_up) > 0:
                                rise_sw = (cross_90_up[0] - cross_10_up[0]) * dt_ms
                                
                            # 2. Decay Time (90% -> 10% repolarisation)
                            cross_90_down = np.where(ap_segment_down <= v_90)[0]
                            cross_10_down = np.where(ap_segment_down <= v_10)[0]
                            if len(cross_90_down) > 0 and len(cross_10_down) > 0:
                                decay_sw = (cross_10_down[0] - cross_90_down[0]) * dt_ms
                                
                            # 3. Half-Width (Durée à 50%)
                            cross_50_up = np.where(ap_segment_up >= v_50)[0]
                            cross_50_down = np.where(ap_segment_down <= v_50)[0]
                            if len(cross_50_up) > 0 and len(cross_50_down) > 0:
                                idx_50_up = idx_thresh_global + cross_50_up[0]
                                idx_50_down = first_peak_idx + cross_50_down[0]
                                width_sw = (idx_50_down - idx_50_up) * dt_ms
                                
                            # 4. AHP (Minimum dans les 10ms suivant le pic)
                            ahp_sw = np.min(ap_segment_down)

            # Ajout aux listes globales
            courants.append(i_cmd); voltages_stat.append(v_stat); voltages_peak.append(v_peak)
            voltages_rest.append(v_rest); spike_counts_raw.append(num_spikes); v_thresholds.append(v_thresh_sw)
            ap_amps.append(amp_sw); ap_widths.append(width_sw); ap_rise_times.append(rise_sw)
            ap_decay_times.append(decay_sw); ap_ahps.append(ahp_sw)

        # 2. Calculs Biophysiques Globaux
        v_rest_global = np.mean(voltages_rest)
        
        # Rhéobase et index
        rheobase_idx = next((i for i, count in enumerate(spike_counts_raw) if count > 0), None)
        rheobase_i = courants[rheobase_idx] if rheobase_idx is not None else None
        rheobase_v = v_thresholds[rheobase_idx] if rheobase_idx is not None else np.nan
        
        # Propriétés Passives
        neg_indices = [i for i, c in enumerate(courants) if c < 0]
        rin_mohm, tau_m_ms, cm_pf = np.nan, np.nan, np.nan
        if neg_indices:
            neg_indices_sorted = sorted(neg_indices, key=lambda i: abs(courants[i]))[:4]
            i_neg = [courants[i] for i in neg_indices_sorted] + [0]
            v_neg = [voltages_stat[i] for i in neg_indices_sorted] + [v_rest_global]
            rin_mohm = np.polyfit(i_neg, v_neg, 1)[0] * (1 if current_unit == "nA" else 1000)
            
            idx_t = sorted(neg_indices, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_baseline = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start])
            v_target = v_baseline + 0.632 * (voltages_stat[idx_t] - v_baseline)
            cross = np.where(abf.sweepY[idx_start:idx_end] <= v_target)[0]
            if len(cross) > 0:
                tau_m_ms = (cross[0] / sr) * 1000.0
                cm_pf = (tau_m_ms / rin_mohm) * 1000.0 if rin_mohm > 0 else np.nan

        # 3. TABLEAUX DE BORD (Metrics)
        st.subheader("📊 Propriétés Intrinsèques & Excitabilité")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vrest (Repos)", f"{v_rest_global:.1f} mV")
        c2.metric("Rin (Entrée)", f"{rin_mohm:.1f} MΩ" if not np.isnan(rin_mohm) else "N/A")
        c3.metric("Cm (Capacitance)", f"{cm_pf:.1f} pF" if not np.isnan(cm_pf) else "N/A")
        c4.metric("Tau_m (Constante)", f"{tau_m_ms:.1f} ms" if not np.isnan(tau_m_ms) else "N/A")
        
        c5, c6, c7 = st.columns(3)
        rheo_scientific = f"{rheobase_i * (1e-12 if current_unit == 'pA' else 1e-9):.2e} A" if rheobase_i else "N/A"
        c5.metric("Rhéobase (Intensité)", rheo_scientific)
        c6.metric("Rhéobase (Seuil mV)", f"{rheobase_v:.1f} mV" if not np.isnan(rheobase_v) else "N/A")
        c7.metric("Sag Amplitude (max)", f"{voltages_stat[np.argmin(courants)] - voltages_peak[np.argmin(courants)]:.1f} mV")

        # NOUVEAU BLOC : Morphologie du PA
        st.subheader("⚡ Propriétés du Premier PA (à la Rhéobase)")
        if rheobase_idx is not None:
            ap1, ap2, ap3, ap4, ap5 = st.columns(5)
            ap1.metric("Amplitude", f"{ap_amps[rheobase_idx]:.1f} mV")
            ap2.metric("Half-Width", f"{ap_widths[rheobase_idx]:.2f} ms")
            ap3.metric("Rise (10-90%)", f"{ap_rise_times[rheobase_idx]:.2f} ms")
            ap4.metric("Decay (90-10%)", f"{ap_decay_times[rheobase_idx]:.2f} ms")
            ap5.metric("AHP (Min Volt)", f"{ap_ahps[rheobase_idx]:.1f} mV")
        else:
            st.info("Aucun potentiel d'action détecté dans cet enregistrement.")

        st.divider()

        # 4. EXPORTATION DES DONNÉES (Mises à jour)
        st.subheader("📥 Exportation des Résultats")
        exp1, exp2 = st.columns(2)
        
        df_bio = pd.DataFrame({
            "Fichier": [uploaded_file.name], "Vrest_mV": [v_rest_global], "Rin_Mohm": [rin_mohm], 
            "Cm_pF": [cm_pf], "Tau_ms": [tau_m_ms], "Rheo_I_A": [rheo_scientific], 
            "Rheo_V_mV": [rheobase_v], "Sag_Max_mV": [voltages_stat[np.argmin(courants)] - voltages_peak[np.argmin(courants)]]
        })
        exp1.download_button("💾 Profil Biophysique Global (CSV)", df_bio.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_biophysique.csv", "text/csv", use_container_width=True)
        
        # Ajout des propriétés AP dans le DataFrame des courbes
        df_curv = pd.DataFrame({
            "Sweep": list(range(abf.sweepCount)), "I_inj": courants, "V_steady": voltages_stat, 
            "V_peak": voltages_peak, "V_threshold": v_thresholds, "Spikes_Raw": spike_counts_raw,
            "AP1_Amplitude_mV": ap_amps, "AP1_HalfWidth_ms": ap_widths, 
            "AP1_RiseTime_ms": ap_rise_times, "AP1_DecayTime_ms": ap_decay_times, "AP1_AHP_mV": ap_ahps
        })
        exp2.download_button("📊 Données Courbes IV & Morphologie PA (CSV)", df_curv.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_donnees_courbes.csv", "text/csv", use_container_width=True)

        st.divider()

        # 5. VISUALISATION AVANCÉE
        st.subheader("📈 Exploration des Traces & Courbes")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            sw_idx = st.slider("Sélectionner un Sweep individuel", 0, abf.sweepCount - 1, 0)
        with col_v2:
            stk_indices = st.multiselect("Superposer des sweeps (Overlay)", list(range(abf.sweepCount)), default=[0, abf.sweepCount//2, abf.sweepCount-1])

        plt.switch_backend('Agg') 
        plt.style.use('seaborn-v0_8-paper')
        fig = plt.figure(figsize=(18, 14), dpi=110)
        gs = fig.add_gridspec(3, 2)
        
        def clean_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10)

        # Trace Individuelle
        ax0 = fig.add_subplot(gs[0, 0])
        abf.setSweep(sw_idx)
        ax0.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
        ax0.set_title(f"Sweep {sw_idx} ({courants[sw_idx]:.1f} {current_unit})", fontweight='bold')
        if not np.isnan(v_thresholds[sw_idx]):
            ax0.axhline(v_thresholds[sw_idx], color='green', ls=':', label='Seuil')
        clean_ax(ax0)

        # Superposition (Overlay)
        ax1 = fig.add_subplot(gs[0, 1])
        cmap = plt.colormaps.get_cmap('viridis')
        for i, s in enumerate(stk_indices):
            abf.setSweep(s)
            ax1.plot(abf.sweepX, abf.sweepY, color=cmap(i/len(stk_indices)), lw=0.8, alpha=0.8)
        ax1.set_title(f"Overlay ({len(stk_indices)} sweeps)", fontweight='bold')
        clean_ax(ax1)
        
        # Courbe I-V
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(courants, voltages_stat, 'o-', label="Steady State")
        ax2.plot(courants, voltages_peak, '^--', alpha=0.4, label="Peak (Sag)")
        ax2.plot(courants[sw_idx], voltages_stat[sw_idx], 'ro', markersize=9, zorder=5) 
        ax2.axvline(0, color='gray', lw=0.5); ax2.axhline(v_rest_global, color='gray', lw=0.5, ls='--')
        ax2.set_title("Relation I-V (Passif & Sag)", fontweight='bold')
        ax2.set_xlabel(f"Injection ({current_unit})")
        ax2.set_ylabel("Vm (mV)")
        ax2.legend(frameon=False)
        clean_ax(ax2)
        
        # Courbe f-I
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(courants, spike_counts_raw, 's-', color='orange')
        ax3.plot(courants[sw_idx], spike_counts_raw[sw_idx], 'ro', markersize=9, zorder=5)
        if rheobase_i: ax3.axvline(rheobase_i, color='red', ls='--')
        ax3.set_title("Excitabilité : Courbe f-I", fontweight='bold')
        ax3.set_xlabel(f"Injection ({current_unit})")
        ax3.set_ylabel("Nombre de PA (Brut)")
        clean_ax(ax3)
        
        st.pyplot(fig)

        # 6. DOCUMENTATION INTÉGRÉE
        with st.expander("📖 Aide Mémoire : Formalisme & Biophysique"):
            st.markdown("""
            * **Rise Time (10-90%) :** Temps mis par le voltage pour passer de 10% à 90% de son amplitude (du seuil au pic).
            * **Half-Width :** Largeur du potentiel d'action mesurée exactement à la moitié de son amplitude maximale.
            * **Decay Time (90-10%) :** Vitesse de repolarisation, mesurée lors de la descente du pic.
            * **AHP (Post-hyperpolarisation) :** Voltage le plus négatif atteint dans la phase réfractaire (fenêtre de 10ms post-pic).
            """)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
else:
    st.info("Veuillez charger un fichier .abf pour activer le pipeline expert.")
