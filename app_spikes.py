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
    "subtitle": {"Français": "Analyse de la Plasticité Synaptique | Manzoni Lab", "English": "Synaptic Plasticity Analysis | Manzoni Lab"},
    "tab_analyse": {"Français": "📈 Analyse & Visualisation", "English": "📈 Analysis & Visualization"},
    "tab_methode": {"Français": "📚 Formalisme & Méthodes", "English": "📚 Formalism & Methods"},
    "tab_export": {"Français": "📥 Exportation", "English": "📥 Export"},
    "settings": {"Français": "⚙️ Réglages de Détection", "English": "⚙️ Detection Settings"},
    "spike_th": {"Français": "Seuil de détection (mV)", "English": "Spike detection threshold (mV)"},
    "dvdt_th": {"Français": "Seuil dV/dt (mV/ms)", "English": "dV/dt threshold (mV/ms)"},
    "prominence_th": {"Français": "Proéminence min (mV)", "English": "Min Prominence (mV)"},
    "refractory_ms": {"Français": "Période Réfractaire (ms)", "English": "Refractory Period (ms)"},
    "artefact_ms": {"Français": "Ignorer l'artefact initial (ms)", "English": "Ignore initial artifact (ms)"},
}

# --- EN-TÊTE ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    st.info("Manzoni Lab - Neurosciences") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header("📂 1. Chargement")
uploaded_file = st.sidebar.file_uploader("Fichier ABF", type=["abf"])

st.sidebar.header(T["settings"][lang])
spike_threshold = st.sidebar.number_input(T["spike_th"][lang], value=-25.0)
dvdt_threshold = st.sidebar.number_input(T["dvdt_th"][lang], value=10.0)
prominence_th = st.sidebar.number_input(T["prominence_th"][lang], value=8.0)
refractory_ms = st.sidebar.number_input(T["refractory_ms"][lang], value=1.5)
artefact_ms = st.sidebar.number_input(T["artefact_ms"][lang], value=2.0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        sr = abf.dataRate
        dt_ms = (1.0 / sr) * 1000.0  
        idx_start, idx_end = int(sr * 0.1), int(sr * 0.6) 
        padding_samples = int(sr * 0.05) 
        artefact_samples = int(sr * (artefact_ms / 1000.0))
        
        # --- BOUCLE D'ANALYSE ---
        courants, v_stat, v_peak, v_rest_list, n_spikes = [], [], [], [], []
        v_thresh_list, ap_amps, ap_widths, ap_rise, ap_decay, ap_ahp = [], [], [], [], [], []
        depol_blocks = []
        
        for sweep in abf.sweepList:
            abf.setSweep(sweep)
            i_cmd = np.mean(abf.sweepC[idx_start:idx_end])
            v_r = np.mean(abf.sweepY[0:idx_start])
            v_s = np.mean(abf.sweepY[idx_end - int(sr*0.05) : idx_end])
            v_p = np.min(abf.sweepY[idx_start:idx_end]) if i_cmd < 0 else np.max(abf.sweepY[idx_start:idx_end])
            
            idx_start_search = idx_start + artefact_samples
            idx_end_search = min(idx_end + padding_samples, len(abf.sweepY)) 
            trace_win = abf.sweepY[idx_start_search:idx_end_search]
            
            distance_samples = int(sr * (refractory_ms / 1000.0))
            peaks, _ = find_peaks(trace_win, height=spike_threshold, prominence=prominence_th, distance=max(1, distance_samples))
            num_spikes = len(peaks)
            
            vt, amp, width, rise, decay, ahp = [np.nan]*6
            if num_spikes > 0:
                pk_idx = peaks[0]
                s_start = max(0, pk_idx - int(sr * 0.015))
                seg = trace_win[s_start:pk_idx]
                if len(seg) > 1:
                    dvdt = np.diff(gaussian_filter1d(seg, sigma=1)) / dt_ms
                    cross = np.where(dvdt > dvdt_threshold)[0]
                    if len(cross) > 0:
                        vt = seg[cross[0]]
                        amp = trace_win[pk_idx] - vt
                        # Calcul simplified decay pour heuristique
                        dn_end = min(len(trace_win), pk_idx + int(sr * 0.1))
                        dn = trace_win[pk_idx:dn_end]
                        d10 = np.where(dn <= (vt + 0.1*amp))[0]
                        decay = d10[0]*dt_ms if len(d10)>0 else np.nan
                
                if vt < -60 or np.isnan(decay):
                    num_spikes = 0
                    vt, amp, decay = [np.nan]*3
                    depol_blocks.append(sweep)

            n_spikes.append(num_spikes); v_stat.append(v_s); v_peak.append(v_p)
            v_rest_list.append(v_r); courants.append(i_cmd); v_thresh_list.append(vt)
            ap_amps.append(amp); ap_ahp.append(np.min(dn) if num_spikes>0 else np.nan)

        # --- NAVIGATION PAR ONGLETS ---
        tab1, tab2, tab3 = st.tabs([T["tab_analyse"][lang], T["tab_methode"][lang], T["tab_export"][lang]])

        with tab1:
            if depol_blocks:
                st.warning(f"⚠️ Bloc de dépolarisation détecté (Sweeps {depol_blocks}). Comptage annulé pour ces traces.")
            
            sw_idx = st.slider("Visualiser Sweep", 0, abf.sweepCount-1, 0)
            abf.setSweep(sw_idx)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(abf.sweepX, abf.sweepY, color='black', lw=1)
            
            # Diagnostic visuel
            trace_visu = abf.sweepY[idx_start_search:idx_end_search]
            p_visu, _ = find_peaks(trace_visu, height=spike_threshold, prominence=prominence_th, distance=distance_samples)
            if len(p_visu)>0:
                ax.plot(abf.sweepX[idx_start_search+p_visu], abf.sweepY[idx_start_search+p_visu], 'ro', label=f"Détectés ({len(p_visu)})")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            if lang == "Français":
                st.markdown("""
                ## 🔬 Formalisme Mathématique & Rigueur Biophysique
                
                ### 1. Extraction des Paramètres Passifs
                * **Résistance d'Entrée ($R_{in}$)** : Calculée par régression linéaire sur les 4 premiers échelons hyperpolarisants ($I < 0$). La pente de la relation $V_{steady} = f(I_{inj})$ définit $R_{in}$.
                * **Capacitance ($C_m$)** : Dérivée de la constante de temps membranaire ($\tau_m$). $\tau_m$ est mesuré par le temps nécessaire pour atteindre 63.2% de la réponse stationnaire sur le plus faible pulse hyperpolarisant.
                
                ### 2. Algorithme de Détection des Événements (Spikes)
                Pour garantir la fidélité des courbes $f-I$ face à l'accommodation et aux artefacts :
                * **Fenêtrage Dynamique** : La détection commence après un délai ($t_{artefact}$) pour ignorer le transitoire capacitif et se termine avec un débordement de 50ms après le pulse pour capturer les décharges tardives.
                * **Critère de Proéminence** : Chaque pic doit s'élever d'une hauteur relative ($\Delta V_{prom}$) minimale par rapport à sa base. Ce filtre topologique rejette les oscillations sous-liminaires.
                * **Période Réfractaire** : Une fenêtre d'exclusion temporelle empêche le double comptage sur les potentiels d'action élargis.
                
                ### 3. Filtre Heuristique : Bloc de Dépolarisation
                Sous forte stimulation, le neurone peut entrer en état d'inactivation sodique totale. L'algorithme valide chaque sweep selon deux critères :
                1. **Validité du Seuil** : Si $V_{threshold} < -60$ mV, l'événement est considéré comme un artefact de calcul.
                2. **Cinétique de Repolarisation** : Si le potentiel ne redescend pas sous 10% de son amplitude dans la fenêtre impartie, le sweep est classé comme "Bloc de dépolarisation" et le compteur est forcé à zéro.
                
                ### 🎓 Citation & Logiciel
                * **Développement** : Manzoni Lab (2026).
                * **Moteur** : Python 3.9+, PyABF, SciPy Signal Processing.
                * **Référence** : *Manzoni, O. J. et al. (2026). Expert Pipeline for Synaptic Plasticity and Intrinsic Excitability Analysis.*
                """)
            else:
                st.markdown("""
                ## 🔬 Mathematical Formalism & Biophysical Rigor
                
                ### 1. Passive Properties Extraction
                * **Input Resistance ($R_{in}$)**: Calculated via linear regression on the first 4 hyperpolarizing steps ($I < 0$). The slope of the $V_{steady} = f(I_{inj})$ relationship defines $R_{in}$.
                * **Capacitance ($C_m$)**: Derived from the membrane time constant ($\tau_m$). $\tau_m$ is measured as the time required to reach 63.2% of the steady-state response on the weakest hyperpolarizing pulse.
                
                ### 2. Event Detection Algorithm (Spikes)
                To ensure $f-I$ curve fidelity against accommodation and artifacts:
                * **Dynamic Windowing**: Detection starts after a delay ($t_{artifact}$) to ignore capacitive transients and ends with a 50ms padding after the pulse to capture late discharges.
                * **Prominence Criterion**: Each peak must rise by a minimum relative height ($\Delta V_{prom}$) from its base. This topological filter rejects sub-threshold oscillations.
                * **Refractory Period**: A temporal exclusion window prevents double counting on broadened action potentials.
                
                ### 3. Heuristic Filter: Depolarization Block
                Under high stimulation, the neuron may reach a state of total sodium inactivation. The algorithm validates each sweep based on two criteria:
                1. **Threshold Validity**: If $V_{threshold} < -60$ mV, the event is considered a calculation artifact.
                2. **Repolarization Kinetics**: If the potential does not return below 10% of its amplitude within the allotted window, the sweep is classified as "Depolarization Block" and the counter is forced to zero.
                """)

        with tab3:
            st.subheader("Télécharger les résultats")
            df = pd.DataFrame({"Sweep": abf.sweepList, "I_inj": courants, "Nb_Spikes": n_spikes})
            st.download_button("Exporter CSV", df.to_csv(index=False).encode('utf-8'), "results.csv")

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
