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
st.set_page_config(page_title="Manzoni Lab - Excitability Pipeline", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Interface Language:", ["Français", "English"])

T = {
    "title": {"Français": "Pipeline Expert : Excitabilité & Morphométrie", "English": "Expert Pipeline: Excitability & Morphometry"},
    "subtitle": {"Français": "Analyse de la Plasticité Synaptique | Standard de Publication", "English": "Synaptic Plasticity Analysis | Publication Standard"},
    "load": {"Français": "📂 1. Chargement", "English": "📂 1. Upload File"},
    "upload_btn": {"Français": "Charger un fichier ABF", "English": "Upload an ABF file"},
    "settings": {"Français": "⚙️ 2. Réglages de Détection", "English": "⚙️ 2. Detection Settings"},
    "spike_th": {"Français": "Seuil de détection (mV)", "English": "Spike detection threshold (mV)"},
    "dvdt_th": {"Français": "Seuil dV/dt (mV/ms)", "English": "dV/dt threshold (mV/ms)"},
    "global_metrics": {"Français": "📊 Propriétés Intrinsèques Globales", "English": "📊 Global Intrinsic Properties"},
    "rheo_th": {"Français": "Rhéobase (Seuil)", "English": "Rheobase (Threshold)"},
    "visuals": {"Français": "📈 Visualisations des Traces & Courbes", "English": "📈 Trace & Curve Visualizations"},
    "select_sweep": {"Français": "Sélectionner un Sweep individuel", "English": "Select individual Sweep"},
    "select_overlay": {"Français": "Sélectionner les sweeps pour l'Overlay", "English": "Select sweeps for Overlay"},
    "morph_title": {"Français": "⚡ Morphologie du 1er PA pour le Sweep", "English": "⚡ 1st AP Morphometry for Sweep"},
    "no_ap": {"Français": "Trace Passive : Aucun Potentiel d'Action détecté.", "English": "Passive Trace: No Action Potential detected."},
    "export": {"Français": "📥 Exportation des Résultats", "English": "📥 Export Results"},
    "exp_global": {"Français": "💾 Exporter le Profil Global (CSV)", "English": "💾 Export Global Profile (CSV)"},
    "exp_sweeps": {"Français": "💾 Exporter les Données par Sweep (CSV)", "English": "💾 Export Sweep Data (CSV)"},
    "readme_title": {"Français": "📚 README, Formalisme & Citation", "English": "📚 README, Formalism & Citation"}
}

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=360) 
    except: st.info("Manzoni Lab") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- BLOC ACCÈS RAPIDE : MÉTHODES ET INFORMATION BIOPHYSIQUE ---
with st.expander("ℹ️ **Information Biophysique, Méthodologie & Raccourcis / Biophysical Methods & Shortcuts**", expanded=False):
    if lang == "Français":
        st.markdown("""
        ### 🔬 Résumé Méthodologique
        * **Seuil d'initiation (Threshold) :** Calculé sur le premier potentiel d'action via la méthode de la première dérivée ($dV/dt \ge 15$ mV/ms).
        * **AHP (Post-hyperpolarisation) :** Mesurée de manière robuste à l'aide d'une fenêtre glissante de 50 ms après le pic, protégée contre le bruit de crête par une période réfractaire de 3 ms. L'amplitude est calculée relativement au seuil du PA.
        * **Résistance d'entrée ($R_{in}$) :** Déterminée par régression linéaire sur la courbe Courant-Voltage des échelons hyperpolarisants.
        * **Bloc de dépolarisation (*Depolarization Block*) :** Détecté automatiquement si le potentiel stationnaire d'un échelon dépolarisation dépasse $-45$ mV et provoque un arrêt de la décharge. Ces traces anormales sont automatiquement écartées des analyses et des fichiers d'exports. **Ne s'applique qu'après avoir atteint la rhéobase.**
        
        👉 **[Aller directement au README complet au bas de la page](#readme-formalise-citation)**
        """)
    else:
        st.markdown("""
        ### 🔬 Methodological Summary
        * **Spike Initiation Threshold:** Calculated on the first action potential using the first derivative method ($dV/dt \ge 15$ mV/ms).
        * **AHP (After-Hyperpolarization):** Measured robustly using a 50 ms sliding window post-peak, protected against crest noise via a 3 ms refractory period. Amplitude is computed relative to the spike threshold.
        * **Input Resistance ($R_{in}$):** Determined by linear regression on the hyperpolarizing current-voltage relationship steps.
        * **Depolarization Block:** Automatically flagged if the steady-state potential during a depolarizing step exceeds $-45$ mV and causes cessation of firing. These traces are automatically excluded from curves and exports. **Only applies after rheobase is reached.**
        
        👉 **[Jump directly to the comprehensive README at the bottom](#readme-formalise-citation)**
        """)

# --- BARRE LATÉRALE ---
st.sidebar.header(T["load"][lang])
uploaded_file = st.sidebar.file_uploader(T["upload_btn"][lang], type=["abf"])

st.sidebar.header(T["settings"][lang])
spike_threshold = st.sidebar.number_input(T["spike_th"][lang], value=0.0)
dvdt_threshold = st.sidebar.number_input(T["dvdt_th"][lang], value=15.0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        unit_i, unit_v = abf.sweepUnitsC, abf.sweepUnitsY
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6) 
        
        courants, v_stat, v_peak, v_rest_list, n_spikes = [], [], [], [], []
        v_thresh_list, ap_amps, ap_widths, ap_rise, ap_decay, ap_ahp = [], [], [], [], [], []
        sweep_all_ahps_indices = [] 
        
        # Listes de suivi pour le bloc de dépolarisation
        excluded_sweeps = []
        is_excluded_list = []
        has_reached_rheobase = False
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_r = np.mean(abf.sweepY[0:idx_start])
            v_s = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_p = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            trace_win = abf.sweepY[idx_start:idx_end]
            
            min_dist_samples = int(sr * 0.003) 
            peaks, _ = find_peaks(trace_win, height=spike_threshold, distance=min_dist_samples)
            num_spikes = len(peaks)
            
            # --- DÉTECTION INTELLIGENTE DU DEPOLARIZATION BLOCK ---
            is_depol_block = False
            
            if num_spikes > 0:
                has_reached_rheobase = True # La cellule a prouvé qu'elle pouvait décharger
                if i_cmd > 0 and v_s > -45:
                    # Arrêt précoce : On demande num_spikes > 1 pour ne pas faussement exclure la rhéobase
                    if num_spikes > 1 and peaks[-1] < int((idx_end - idx_start) * 0.6):
                        is_depol_block = True
            else:
                # 0 Spike : Est-ce un Depol Block ou juste avant la rhéobase ?
                if i_cmd > 0 and v_s > -45:
                    # Ne déclenche le bloc que si on a déjà dépassé la rhéobase lors d'un sweep précédent
                    if has_reached_rheobase:
                        is_depol_block = True
            
            if is_depol_block:
                excluded_sweeps.append(sweep)
            is_excluded_list.append(is_depol_block)
            
            vt, amp, width, rise, decay, ahp_1st = [np.nan]*6
            ahp_indices_for_current_sweep = []
            
            if num_spikes > 0:
                pk_idx_1st = peaks[0]
                s_start = max(0, pk_idx_1st - int(sr * 0.015))
                seg = trace_win[s_start:pk_idx_1st]
                
                if len(seg) > 1:
                    smoothed = gaussian_filter1d(seg, sigma=1)
                    dvdt = np.diff(smoothed) / dt_ms
                    cross = np.where(dvdt > dvdt_threshold)[0]
                    
                    if len(cross) > 0:
                        idx_t_seg = cross[0]
                        vt = seg[idx_t_seg]
                        idx_t_glob = s_start + idx_t_seg
                        amp = trace_win[pk_idx_1st] - vt
                        
                        if amp > 0:
                            v50 = vt + 0.5*amp; v10 = vt + 0.1*amp; v90 = vt + 0.9*amp
                            up = trace_win[idx_t_glob:pk_idx_1st]
                            
                            dn_end_1st = peaks[1] if num_spikes > 1 else min(len(trace_win), pk_idx_1st + int(sr * 0.1))
                            dn_1st = trace_win[pk_idx_1st:dn_end_1st]
                            
                            r10 = np.where(up >= v10)[0]; r90 = np.where(up >= v90)[0]
                            if len(r10)>0 and len(r90)>0: rise = (r90[0]-r10[0])*dt_ms
                            
                            d90 = np.where(dn_1st <= v90)[0]; d10 = np.where(dn_1st <= v10)[0]
                            if len(d90)>0 and len(d10)>0: decay = (d10[0]-d90[0])*dt_ms
                            
                            wup = np.where(up >= v50)[0]; wdn = np.where(dn_1st <= v50)[0]
                            if len(wup)>0 and len(wdn)>0: width = ((pk_idx_1st+wdn[0])-(idx_t_glob+wup[0]))*dt_ms

                for i, pk_idx in enumerate(peaks):
                    max_search_window = pk_idx + int(sr * 0.05) 
                    
                    if i < num_spikes - 1:
                        ahp_end = min(max_search_window, peaks[i+1])
                    else:
                        ahp_end = min(max_search_window, len(trace_win))
                    
                    dn_segment = trace_win[pk_idx:ahp_end]
                    
                    if len(dn_segment) > 0:
                        local_min_idx = np.argmin(dn_segment)
                        global_ahp_idx = idx_start + pk_idx + local_min_idx
                        ahp_indices_for_current_sweep.append(global_ahp_idx)
                        
                        if i == 0:
                            ahp_1st = dn_segment[local_min_idx]

            courants.append(i_cmd); v_stat.append(v_s); v_peak.append(v_p)
            v_rest_list.append(v_r); n_spikes.append(num_spikes); v_thresh_list.append(vt)
            ap_amps.append(amp); ap_widths.append(width); ap_rise.append(rise); ap_decay.append(decay); ap_ahp.append(ahp_1st)
            sweep_all_ahps_indices.append(ahp_indices_for_current_sweep)

        # --- INDICATION DES TRACES EXCLUES (DEPOLARIZATION BLOCK) ---
        if excluded_sweeps:
            st.warning(f"⚠️ **Depolarization Block détecté !** Les sweeps suivants ont été exclus de l'analyse globale et des courbes : {', '.join([str(s) for s in excluded_sweeps])}" if lang == "Français" else f"⚠️ **Depolarization Block detected!** The following sweeps have been excluded from global analysis and curves: {', '.join([str(s) for s in excluded_sweeps])}")
        else:
            st.success("✅ Aucun Depolarization Block détecté." if lang == "Français" else "✅ No Depolarization Block detected.")

        # --- CALCULS GLOBAUX ÉCHANTILLONNÉS (SANS LES TRACES EXCLUES) ---
        valid_indices = [i for i, excl in enumerate(is_excluded_list) if not excl]
        
        v_rest_final = np.mean(v_rest_list)
        neg = [i for i, c in enumerate(courants) if c < 0]
        rin, tau = np.nan, np.nan
        if neg:
            neg_s = sorted(neg, key=lambda i: abs(courants[i]))[:4]
            rin = np.polyfit([courants[i] for i in neg_s]+[0], [v_stat[i] for i in neg_s]+[v_rest_final], 1)[0] * (1 if unit_i=="nA" else 1000)
            idx_t = sorted(neg, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_targ = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]) + 0.632*(v_stat[idx_t]-np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]))
            cross_t = np.where(abf.sweepY[idx_start:idx_end] <= v_targ)[0]
            if len(cross_t)>0:
