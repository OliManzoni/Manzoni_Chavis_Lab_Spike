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

# Dictionnaire de traduction simplifié
T = {
    "title": {"Français": "Pipeline Expert : Excitabilité & Morphométrie", "English": "Expert Pipeline: Excitability & Morphometry"},
    "subtitle": {"Français": "Analyse des Potentiels d'Action", "English": "Action Potential Analysis | Publication Standard"},
    "load": {"Français": "📂 1. Chargement", "English": "📂 1. Upload File"},
    "upload_btn": {"Français": "Charger un fichier ABF", "English": "Upload an ABF file"},
    "settings": {"Français": "⚙️ 2. Réglages de Détection", "English": "⚙️ 2. Detection Settings"},
    "spike_th": {"Français": "Seuil de détection (mV)", "English": "Spike detection threshold (mV)"},
    "dvdt_th": {"Français": "Seuil dV/dt (mV/ms)", "English": "dV/dt threshold (mV/ms)"},
    "prominence_th": {"Français": "Proéminence min (mV)", "English": "Min Prominence (mV)"},
    "refractory_ms": {"Français": "Période Réfractaire (ms)", "English": "Refractory Period (ms)"},
    "artefact_ms": {"Français": "Ignorer l'artefact initial (ms)", "English": "Ignore initial artifact (ms)"},
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
    try: 
        st.image("logo_chavis_final.png", width=360) 
    except: 
        st.info("Manzoni Lab - Neurosciences") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header(T["load"][lang])
uploaded_file = st.sidebar.file_uploader(T["upload_btn"][lang], type=["abf"])

st.sidebar.header(T["settings"][lang])

# Valeurs permissives pour capter l'accommodation et éviter les coupures
spike_threshold = st.sidebar.number_input(T["spike_th"][lang], value=-25.0)
dvdt_threshold = st.sidebar.number_input(T["dvdt_th"][lang], value=10.0)
prominence_th = st.sidebar.number_input(T["prominence_th"][lang], value=8.0, help="Rejette les oscillations sous-liminaires tout en acceptant les spikes fatigués.")
refractory_ms = st.sidebar.number_input(T["refractory_ms"][lang], value=1.5, help="Empêche le comptage multiple sur un même événement élargi.")
artefact_ms = st.sidebar.number_input(T["artefact_ms"][lang], value=2.0, help="Délai après le début du pulse pour ignorer le transitoire capacitif de l'amplificateur.")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        unit_i = abf.sweepUnitsC  
        unit_v = abf.sweepUnitsY
        st.sidebar.success(f"Units : {unit_i} / {unit_v}")
        
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6) 
        
        # Marge de sécurité (repolarisation lente) et exclusion d'artefact capacitif
        padding_samples = int(sr * 0.05) 
        artefact_samples = int(sr * (artefact_ms / 1000.0))
        
        courants, v_stat, v_peak, v_rest_list, n_spikes = [], [], [], [], []
        v_thresh_list, ap_amps, ap_widths, ap_rise, ap_decay, ap_ahp = [], [], [], [], [], []
        depol_blocks = [] # Traçage des sweeps en bloc de dépolarisation
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_r = np.mean(abf.sweepY[0:idx_start])
            
            # Le calcul des variables passives reste strictement calé sur l'injection de courant
            v_s = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_p = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            # NOUVEAU FENÊTRAGE DE RECHERCHE (ignore l'artefact de début, englobe la queue de fin)
            idx_start_search = idx_start + artefact_samples
            idx_end_search = min(idx_end + padding_samples, len(abf.sweepY)) 
            trace_win = abf.sweepY[idx_start_search:idx_end_search]
            
            # Application des contraintes topologiques
            distance_samples = int(sr * (refractory_ms / 1000.0))
            peaks, _ = find_peaks(trace_win, height=spike_threshold, prominence=prominence_th, distance=max(1, distance_samples))
            num_spikes = len(peaks)
            
            vt, amp, width, rise, decay, ahp = [np.nan]*6
            
            if num_spikes > 0:
                pk_idx = peaks[0]
                s_start = max(0, pk_idx - int(sr * 0.015))
                seg = trace_win[s_start:pk_idx]
                
                if len(seg) > 1:
                    smoothed = gaussian_filter1d(seg, sigma=1)
                    dvdt = np.diff(smoothed) / dt_ms
                    cross = np.where(dvdt > dvdt_threshold)[0]
                    
                    if len(cross) > 0:
                        idx_t_seg = cross[0]
                        vt = seg[idx_t_seg]
                        idx_t_glob = s_start + idx_t_seg
                        amp = trace_win[pk_idx] - vt
                        
                        if amp > 0:
                            v50 = vt + 0.5*amp; v10 = vt + 0.1*amp; v90 = vt + 0.9*amp
                            up = trace_win[idx_t_glob:pk_idx]
                            
                            if num_spikes > 1:
                                dn_end = peaks[1] 
                            else:
                                dn_end = min(len(trace_win), pk_idx + int(sr * 0.1)) # 100 ms
                                
                            dn = trace_win[pk_idx:dn_end]
                            
                            r10 = np.where(up >= v10)[0]; r90 = np.where(up >= v90)[0]
                            if len(r10)>0 and len(r90)>0: rise = (r90[0]-r10[0])*dt_ms
                            
                            d90 = np.where(dn <= v90)[0]; d10 = np.where(dn <= v10)[0]
                            if len(d90)>0 and len(d10)>0: decay = (d10[0]-d90[0])*dt_ms
                            
                            wup = np.where(up >= v50)[0]; wdn = np.where(dn <= v50)[0]
                            if len(wup)>0 and len(wdn)>0: width = ((pk_idx+wdn[0])-(idx_t_glob+wup[0]))*dt_ms
                            
                            ahp = np.min(dn)

                # --- FILTRE HEURISTIQUE : BLOC DE DÉPOLARISATION ---
                if vt < -60 or np.isnan(decay):
                    num_spikes = 0
                    vt, amp, width, rise, decay, ahp = [np.nan]*6
                    depol_blocks.append(sweep)

            courants.append(i_cmd); v_stat.append(v_s); v_peak.append(v_p)
            v_rest_list.append(v_r); n_spikes.append(num_spikes); v_thresh_list.append(vt)
            ap_amps.append(amp); ap_widths.append(width); ap_rise.append(rise); ap_decay.append(decay); ap_ahp.append(ahp)

        # --- GESTION DE L'AFFICHAGE DU BLOC DE DÉPOLARISATION ---
        if depol_blocks:
            st.warning(f"⚠️ **Bloc de dépolarisation identifié.** Les événements ont été exclus du comptage pour les sweeps : {', '.join(map(str, depol_blocks))}")

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
            sw_idx = st.slider(T["select_sweep"][lang], 0, abf.sweepCount-1, int(rheo_idx_global) if rheo_idx_global else 0)
        with col_v2:
            stk_indices = st.multiselect(T["select_overlay"][lang], list(range(abf.sweepCount)), default=[0, abf.sweepCount//2, abf.sweepCount-1])
        
        st.markdown(f"**{T['morph_title'][lang]} {sw_idx} ({courants[sw_idx]:.1f} {unit_i})**")
        
        if n_spikes[sw_idx] > 0:
            c_ap1, c_ap2, c_ap3, c_ap4, c_ap5 = st.columns(5)
            c_ap1.metric("Amplitude", f"{ap_amps[sw_idx]:.1f} mV")
            c_ap2.metric("Half-Width", f"{ap_widths[sw_idx]:.2f} ms")
            c_ap3.metric("Rise (10-90%)", f"{ap_rise[sw_idx]:.2f} ms")
            c_ap4.metric("Decay (90-10%)", f"{ap_decay[sw_idx]:.2f} ms")
            c_ap5.metric("AHP (Min)", f"{ap_ahp[sw_idx]:.1f} mV")
        else:
            st.info(T["no_ap"][lang])
            
        st.write("") 
        
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        abf.setSweep(sw_idx)
        ax1.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
        
        # --- DIAGNOSTIC VISUEL : Ajout des points rouges sur les pics détectés ---
        trace_win_visu = abf.sweepY[idx_start_search:idx_end_search]
        distance_samples = int(sr * (refractory_ms / 1000.0))
        peaks_visu, _ = find_peaks(trace_win_visu, height=spike_threshold, prominence=prominence_th, distance=max(1, distance_samples))
        
        if len(peaks_visu) > 0:
            peak_times = abf.sweepX[idx_start_search + peaks_visu]
            peak_values = abf.sweepY[idx_start_search + peaks_visu]
            ax1.plot(peak_times, peak_values, 'ro', markersize=5, label=f"Spikes détectés ({len(peaks_visu)})")

        if not np.isnan(v_thresh_list[sw_idx]):
            ax1.axhline(v_thresh_list[sw_idx], color='red', ls='--', alpha=0.3, label="Threshold dV/dt (1er PA)")
            
        ax1.legend(loc='upper right')
        ax1.set_title(f"Sweep {sw_idx}", fontweight='bold')
        ax1.set_ylabel("mV")

        # Reste des graphiques
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

        df_sweeps = pd.DataFrame({
            "Sweep": abf.sweepList, "I_inj": courants, "Nb_AP_par_step": n_spikes,
            "V_steady": v_stat, "V_threshold": v_thresh_list, "AP_Amp": ap_amps, 
            "AP_Width_ms": ap_widths, "AP_Rise_ms": ap_rise, "AP_Decay_ms": ap_decay, "AP_AHP": ap_ahp
        })
        col_exp2.download_button(T["exp_sweeps"][lang], df_sweeps.to_csv(index=False).encode('utf-8'), f"{uploaded_file.name}_Sweeps.csv", use_container_width=True)

        # --- README, FORMALISME & CITATION ---
        st.divider()
        with st.expander(T["readme_title"][lang]):
            if lang == "Français":
                st.markdown("""
                ### 📄 README (Mode d'emploi)
                Cet outil est conçu pour le traitement par lots et l'extraction biophysique de traces *Current-Clamp* (.abf).
                1. Chargez votre fichier via le panneau latéral.
                2. Ajustez la **Proéminence** et la **Période Réfractaire** pour contrôler la rigueur de détection des événements actifs.
                3. Utilisez l'**Artefact Initial** si le pipeline manque le tout premier potentiel d'action.
                4. Exportez le profil biophysique global et les données métriques par échelons pour vos analyses statistiques.

                ### 🧠 Formalisme & Limites
                * **Capacitance ($C_m = \\tau_m / R_{in}$) :** Calculée au point de charge de 63.2%. *Limite (Space-Clamp)* : Dans des neurones à arborescence dendritique riche (ex: neurones pyramidaux CA1/PFC, ou modèles de pathologies comme l'X Fragile), $C_m$ peut être sous-estimée.
                * **Filtre Heuristique (Bloc de Dépolarisation) :** Si le neurone échoue à repolariser (Decay = NaN) ou si le seuil de dV/dt chute sous -60mV, l'algorithme annule le comptage pour exclure le bruit stochastique sous haute stimulation.

                ### 🎓 Citation
                Si vous utilisez ce code ou ce pipeline pour une publication scientifique, merci d'inclure le DOI et la citation suivante :
                > **Manzoni Lab (2026).** *Expert Pipeline: Neural Excitability & Morphometry.* > **DOI:** `10.5281/zenodo.XXXXXXX` *(Placeholder)*
                > **Github:** [github.com/ManzoniLab/ElectrophyPipeline](https://github.com)
                """)
            else:
                st.markdown("""
                ### 📄 README (Instructions)
                This tool is designed for batch processing and biophysical extraction of *Current-Clamp* traces (.abf).
                1. Upload your file via the sidebar.
                2. Adjust the **Prominence** and **Refractory Period** to control the stringency of active event detection.
                3. Use the **Initial Artifact** parameter if the pipeline misses the very first action potential.
                4. Export the global biophysical profile and sweep-by-sweep metric data for statistical analysis.

                ### 🧠 Formalism & Limitations
                * **Capacitance ($C_m = \\tau_m / R_{in}$) :** Calculated at the 63.2% charge point. *Limitation (Space-Clamp)*: In neurons with complex dendritic arborizations (e.g., CA1/PFC pyramidal neurons, or disease models like Fragile X), $C_m$ may be underestimated.
                * **Heuristic Filter (Depolarization Block) :** If the neuron fails to repolarize (Decay = NaN) or if the dV/dt threshold drops below -60mV, the algorithm resets the spike count to exclude stochastic noise under high stimulation.

                ### 🎓 Citation
                If you use this code or pipeline for a scientific publication, please include the DOI and following citation:
                > **Manzoni Lab (2026).** *Expert Pipeline: Neural Excitability & Morphometry.* > **DOI:** `10.5281/zenodo.XXXXXXX` *(Placeholder)*
                > **Github:** [github.com/ManzoniLab/ElectrophyPipeline](https://github.com)
                """)

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
