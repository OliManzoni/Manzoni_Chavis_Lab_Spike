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
st.set_page_config(page_title="Manzoni Lab - Neural Excitability", layout="wide")

# --- EN-TÊTE ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Branding") 
with col_r:
    st.markdown("# Pipeline Expert : Excitabilité & Propriétés Intrinsèques")
    st.markdown("### Manzoni Lab | Analyse de la Plasticité Synaptique")
    st.markdown("#### *Standard de publication : Modèle RC, Seuil dV/dt & Morphométrie*")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header("📂 1. Chargement")
uploaded_file = st.sidebar.file_uploader("Charger un fichier ABF", type=["abf"])

st.sidebar.header("⚙️ 2. Réglages de Détection")
spike_threshold = st.sidebar.number_input("Seuil de détection des PA (mV)", value=0.0)
dvdt_threshold = st.sidebar.number_input("Seuil dV/dt (mV/ms)", value=15.0)

# --- LOGIQUE ANALYTIQUE ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        # DÉTECTION AUTOMATIQUE DES UNITÉS (Telegraph via ABF Header)
        current_unit = abf.sweepUnitsC  
        voltage_unit = abf.sweepUnitsY
        st.sidebar.success(f"Unités détectées : {current_unit} / {voltage_unit}")
        
        # Paramètres temporels
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6) # Fenêtre de pulse type (100-600ms)
        
        # Listes d'extraction
        courants, voltages_stat, voltages_peak, voltages_rest = [], [], [], []
        spike_counts_raw, v_thresholds = [], []
        ap_amps, ap_widths, ap_rise_times, ap_decay_times, ap_ahps = [], [], [], [], []
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_rest = np.mean(abf.sweepY[0:idx_start])
            v_stat = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_peak = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            trace_window = abf.sweepY[idx_start:idx_end]
            peaks, _ = find_peaks(trace_window, height=spike_threshold)
            num_spikes = len(peaks)
            
            v_thresh_sw = np.nan
            amp_sw, width_sw, rise_sw, decay_sw, ahp_sw = np.nan, np.nan, np.nan, np.nan, np.nan
            
            if num_spikes > 0:
                first_peak_idx = peaks[0]
                search_start = max(0, first_peak_idx - int(sr * 0.015))
                segment = trace_window[search_start:first_peak_idx]
                
                if len(segment) > 1:
                    smoothed_segment = gaussian_filter1d(segment, sigma=1)
                    dvdt = np.diff(smoothed_segment) / dt_ms
                    crossings = np.where(dvdt > dvdt_threshold)[0]
                    
                    if len(crossings) > 0:
                        idx_thresh_segment = crossings[0]
                        v_thresh_sw = segment[idx_thresh_segment]
                        idx_thresh_global = search_start + idx_thresh_segment
                        
                        # Morphométrie
                        v_peak_ap = trace_window[first_peak_idx]
                        amp_sw = v_peak_ap - v_thresh_sw
                        if amp_sw > 0:
                            v_10, v_50, v_90 = v_thresh_sw + 0.10*amp_sw, v_thresh_sw + 0.50*amp_sw, v_thresh_sw + 0.90*amp_sw
                            up_seg = trace_window[idx_thresh_global:first_peak_idx]
                            dn_lim = min(len(trace_window), first_peak_idx + int(sr*0.010))
                            dn_seg = trace_window[first_peak_idx:dn_lim]
                            
                            # Rise/Decay/Width
                            r10 = np.where(up_seg >= v_10)[0]; r90 = np.where(up_seg >= v_90)[0]
                            if len(r10)>0 and len(r90)>0: rise_sw = (r90[0]-r10[0])*dt_ms
                            d90 = np.where(dn_seg <= v_90)[0]; d10 = np.where(dn_seg <= v_10)[0]
                            if len(d90)>0 and len(d10)>0: decay_sw = (d10[0]-d90[0])*dt_ms
                            w50u = np.where(up_seg >= v_50)[0]; w50d = np.where(dn_seg <= v_50)[0]
                            if len(w50u)>0 and len(w50d)>0: width_sw = ((first_peak_idx+w50d[0])-(idx_thresh_global+w50u[0]))*dt_ms
                            ahp_sw = np.min(dn_seg)

            courants.append(i_cmd); voltages_stat.append(v_stat); voltages_peak.append(v_peak)
            voltages_rest.append(v_rest); spike_counts_raw.append(num_spikes); v_thresholds.append(v_thresh_sw)
            ap_amps.append(amp_sw); ap_widths.append(width_sw); ap_rise_times.append(rise_sw); ap_decay_times.append(decay_sw); ap_ahps.append(ahp_sw)

        # Calculs passifs
        v_rest_global = np.mean(voltages_rest)
        neg_indices = [i for i, c in enumerate(courants) if c < 0]
        rin_mohm, tau_m_ms, cm_pf = np.nan, np.nan, np.nan
        if neg_indices:
            neg_indices_sorted = sorted(neg_indices, key=lambda i: abs(courants[i]))[:4]
            i_neg, v_neg = [courants[i] for i in neg_indices_sorted]+[0], [voltages_stat[i] for i in neg_indices_sorted]+[v_rest_global]
            rin_mohm = np.polyfit(i_neg, v_neg, 1)[0] * (1 if current_unit == "nA" else 1000)
            idx_t = sorted(neg_indices, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_base = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start])
            v_targ = v_base + 0.632 * (voltages_stat[idx_t] - v_base)
            cross = np.where(abf.sweepY[idx_start:idx_end] <= v_targ)[0]
            if len(cross)>0: 
                tau_m_ms = (cross[0]/sr)*1000.0
                cm_pf = (tau_m_ms/rin_mohm)*1000.0 if rin_mohm > 0 else np.nan

        # --- DASHBOARD ---
        st.subheader("📊 Résultats Biophysiques & Excitabilité")
        rheobase_idx = next((i for i, count in enumerate(spike_counts_raw) if count > 0), None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vrest", f"{v_rest_global:.1f} mV")
        c2.metric("Rin", f"{rin_mohm:.1f} MΩ")
        c3.metric("Capacitance (Cm)", f"{cm_pf:.1f} pF")
        c4.metric("Tau_m", f"{tau_m_ms:.1f} ms")

        st.divider()

        # --- EXPORTS (Modifié : Ajout Nb_AP_par_step) ---
        st.subheader("📥 Exportation des Données")
        exp1, exp2 = st.columns(2)
        
        df_bio = pd.DataFrame({"Fichier": [uploaded_file.name], "Vrest_mV": [v_rest_global], "Rin_MOh": [rin_mohm], "Cm_pF": [cm_pf], "Tau_ms": [tau_m_ms]})
        exp1.download_button("💾 Profil Passif (CSV)", df_bio.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_passif.csv", "text/csv")
        
        df_curv = pd.DataFrame({
            "Sweep": list(range(abf.sweepCount)), "I_inj_unit": courants, "Nb_AP_par_step": spike_counts_raw,
            "V_steady": voltages_stat, "V_threshold": v_thresholds, "AP_Amp": ap_amps, "AP_Width_ms": ap_widths, "AP_Rise_ms": ap_rise_times
        })
        exp2.download_button("📊 Courbes & Morphométrie (CSV)", df_curv.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_complet.csv", "text/csv")

        st.divider()

        # --- VISUALISATION ---
        sw_idx = st.slider("Visualiser un sweep", 0, abf.sweepCount-1, rheobase_idx if rheobase_idx else 0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        abf.setSweep(sw_idx)
        ax1.plot(abf.sweepX, abf.sweepY, color='black')
        ax1.set_title(f"Sweep {sw_idx} : {courants[sw_idx]:.1f} {current_unit}")
        ax2.plot(courants, spike_counts_raw, 'o-', color='orange')
        ax2.set_xlabel(current_unit); ax2.set_ylabel("Nombre de PA")
        st.pyplot(fig)

        # --- AIDE MÉMOIRE (Modifié : Formalisme détaillé) ---
        with st.expander("📖 Aide Mémoire : Formalisme, Biophysique & Limites"):
            st.markdown("### 1. Mesure de la Capacitance Membranaire ($C_m$)")
            st.write("Basée sur le modèle RC d'un neurone à un seul compartiment :")
            st.latex(r"C_m = \frac{\tau_m}{R_{in}}")
            st.markdown("""
            * **Méthode :** $R_{in}$ est extraite par régression linéaire sur les échelons passifs. $\\tau_m$ est le temps pour atteindre **63.2%** du régime stationnaire.
            * **Limites de mesure :** * *Space-Clamp :* Dans les neurones à morphologie complexe (ex: Pyramides CA1), les compartiments distaux ne sont pas parfaitement clampés, ce qui peut sous-estimer la capacitance totale.
                * *Courants actifs :* La présence de courants $I_h$ (Sag) peut distordre l'exponentielle de charge.
            """)
            st.markdown("---")
            st.markdown("### 2. Morphométrie des PA")
            st.markdown("""
            * **Seuil (dV/dt) :** Défini au point où l'accélération du voltage atteint **15 mV/ms** (paramétrable).
            * **Half-Width :** Mesurée à 50% de l'amplitude pic-seuil. C'est un indicateur clé de la cinétique des canaux $K^+$ et de la maturation neuronale (ex: modèles FXS).
            * **Rise/Decay :** Calculés sur les segments 10-90% pour minimiser l'impact du bruit de ligne.
            """)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
