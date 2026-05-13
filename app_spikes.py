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
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_r = np.mean(abf.sweepY[0:idx_start])
            v_s = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_p = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            trace_win = abf.sweepY[idx_start:idx_end]
            
            # --- Période réfractaire (3ms) pour contrer le bruit de crête ---
            min_dist_samples = int(sr * 0.003) 
            peaks, _ = find_peaks(trace_win, height=spike_threshold, distance=min_dist_samples)
            num_spikes = len(peaks)
            
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

                # --- Détection des AHPs pour TOUS les spikes ---
                for i, pk_idx in enumerate(peaks):
                    max_search_window = pk_idx + int(sr * 0.05) # 50 ms max
                    
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

        # --- CALCULS GLOBAUX ---
        v_rest_final = np.mean(v_rest_list)
        neg = [i for i, c in enumerate(courants) if c < 0]
        rin, tau, cm = np.nan, np.nan, np.nan
        if neg:
            neg_s = sorted(neg, key=lambda i: abs(courants[i]))[:4]
            rin = np.polyfit([courants[i] for i in neg_s]+[0], [v_stat[i] for i in neg_s]+[v_rest_final], 1)[0] * (1 if unit_i=="nA" else 1000)
            idx_t = sorted(neg, key=lambda i: abs(courants[i]))[0]
            abf.setSweep(idx_t)
            v_targ = np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]) + 0.632*(v_stat[idx_t]-np.mean(abf.sweepY[idx_start-int(sr*0.01):idx_start]))
            cross_t = np.where(abf.sweepY[idx_start:idx_end] <= v_targ)[0]
            if len(cross_t)>0: 
                tau = (cross_t[0]/sr)*1000.0
                cm = (tau/rin)*1000.0 if rin>0 else np.nan
                
        rheo_idx_global = next((i for i, count in enumerate(n_spikes) if count > 0), None)
        rheo_v = v_thresh_list[rheo_idx_global] if rheo_idx_global is not None else np.nan
        rheo_i = courants[rheo_idx_global] if rheo_idx_global is not None else np.nan

        # --- DASHBOARD ---
        st.subheader(T["global_metrics"][lang])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Vrest", f"{v_rest_final:.1f} mV")
        c2.metric("Rin", f"{rin:.1f} MΩ")
        c3.metric("Cm (Capacitance)", f"{cm:.1f} pF")
        c4.metric("Tau_m", f"{tau:.1f} ms")
        c5.metric(T["rheo_th"][lang], f"{rheo_v:.1f} mV" if not np.isnan(rheo_v) else "N/A")

        st.divider()

        # --- VISUALISATION INTERACTIVE ---
        st.subheader(T["visuals"][lang])
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            # CORRECTION : Protège le curseur si un seul sweep
            if abf.sweepCount > 1:
                default_sw = int(rheo_idx_global) if rheo_idx_global is not None else 0
                default_sw = min(max(default_sw, 0), abf.sweepCount - 1) 
                sw_idx = st.slider(T["select_sweep"][lang], 0, abf.sweepCount-1, default_sw)
            else:
                sw_idx = 0
                st.info("Fichier à Sweep unique (Gap-free).")
                
        with col_v2:
            # CORRECTION : Protège le multiselect si moins de 3 sweeps
            if abf.sweepCount == 1:
                default_stk = [0]
            elif abf.sweepCount == 2:
                default_stk = [0, 1]
            else:
                default_stk = [0, abf.sweepCount//2, abf.sweepCount-1]
                
            stk_indices = st.multiselect(T["select_overlay"][lang], list(range(abf.sweepCount)), default=default_stk)
        
        st.markdown(f"**{T['morph_title'][lang]} {sw_idx} ({courants[sw_idx]:.1f} {unit_i})**")
        
        if n_spikes[sw_idx] > 0:
            c_ap1, c_ap2, c_ap3, c_ap4, c_ap5 = st.columns(5)
            c_ap1.metric("Amplitude", f"{ap_amps[sw_idx]:.1f} mV")
            c_ap2.metric("Half-Width", f"{ap_widths[sw_idx]:.2f} ms")
            c_ap3.metric("Rise (10-90%)", f"{ap_rise[sw_idx]:.2f} ms")
            c_ap4.metric("Decay (90-10%)", f"{ap_decay[sw_idx]:.2f} ms")
            ahp_relative = v_thresh_list[sw_idx] - ap_ahp[sw_idx] if not np.isnan(v_thresh_list[sw_idx]) else np.nan
            c_ap5.metric("AHP Amplitude", f"{ahp_relative:.1f} mV" if not np.isnan(ahp_relative) else "N/A")
        else:
            st.info(T["no_ap"][lang])
            
        st.write("") 
        
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        abf.setSweep(sw_idx)
        ax1.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
        
        if not np.isnan(v_thresh_list[sw_idx]):
            ax1.axhline(v_thresh_list[sw_idx], color='red', ls='--', alpha=0.6, label="Threshold dV/dt")
            
            indices_to_plot = sweep_all_ahps_indices[sw_idx]
            if indices_to_plot:
                ax1.plot(abf.sweepX[indices_to_plot], abf.sweepY[indices_to_plot], 'bx', markersize=8, markeredgewidth=2, label="AHP Min")
                
            ax1.legend(loc='upper right')
            
        ax1.set_title(f"Sweep {sw_idx}", fontweight='bold')
        ax1.set_ylabel("mV")

        ax2 = fig.add_subplot(gs[0, 1])
        cmap = plt.colormaps.get_cmap('viridis')
        for i, s in enumerate(stk_indices):
            abf.setSweep(s)
            ax2.plot(abf.sweepX, abf.sweepY, color=cmap(i/max(1, len(stk_indices))), alpha=0.8, lw=0.8)
        ax2.set_title(f"Overlay ({len(stk_indices)} traces)", fontweight='bold')
        ax2.set_ylabel("mV")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(courants, v_stat, 'o-', color='tab:blue', label="Steady-state")
        ax3.plot(courants, v_peak, 'x--', color='tab:blue', alpha=0.5, label="Peak (Sag)")
        ax3.plot(courants[sw_idx], v_stat[sw_idx], 'ro', markersize=9, zorder=5) 
        ax3.set_title("I-V Curve", fontweight='bold')
        ax3.set_xlabel(f"Current ({unit_i})"); ax3.set_ylabel("mV"); ax3.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(courants, n_spikes, 's-', color='tab:orange')
        ax4.plot(courants[sw_idx], n_spikes[sw_idx], 'ro', markersize=9, zorder=5) 
        ax4.set_title("f-I Curve", fontweight='bold')
        ax4.set_xlabel(f"Current ({unit_i})"); ax4.set_ylabel("Spike count")
        
        st.pyplot(fig)

        # --- EXPORT ---
        st.divider()
        st.subheader(T["export"][lang])
        col_exp1, col_exp2 = st.columns(2)
        
        df_global = pd.DataFrame({
            "File": [uploaded_file.name], "Vrest_mV": [v_rest_final], "Rin_MOhms": [rin],
            "Cm_pF": [cm], "Tau_ms": [tau], "Rheobase_I": [rheo_i], "Rheobase_mV": [rheo_v]
        })
        col_exp1.download_button(T["exp_global"][lang], df_global.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_Global.csv", use_container_width=True)

        ap_ahp_relative = [v_t - a if not np.isnan(v_t) and not np.isnan(a) else np.nan for v_t, a in zip(v_thresh_list, ap_ahp)]

        df_sweeps = pd.DataFrame({
            "Sweep": abf.sweepList, "I_inj": courants, "Nb_Spikes": n_spikes,
            "V_steady": v_stat, "V_threshold": v_thresh_list, "AP_Amp": ap_amps, 
            "AP_Width_ms": ap_widths, "AP_Rise_ms": ap_rise, "AP_Decay_ms": ap_decay, "AP_AHP": ap_ahp_relative
        })
        col_exp2.download_button(T["exp_sweeps"][lang], df_sweeps.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_Sweeps.csv", use_container_width=True)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
