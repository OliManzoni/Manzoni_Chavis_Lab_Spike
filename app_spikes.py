import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Spike Analysis Pipeline", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Interface Language:", ["Français", "English"])

# --- DICTIONNAIRE DE TRADUCTION & METHODOLOGIE ---
T = {
    "title": {"Français": "⚡ Analyse des Potentiels d'Action & Courbes I-V", "English": "⚡ Spike Analysis & I-V Curve Pipeline"},
    "subtitle": {"Français": "Pipeline Biophysique de Pointe | Manzoni Lab Standards", "English": "Advanced Biophysical Pipeline | Manzoni Lab Standards"},
    "load": {"Français": "📂 1. Chargement du fichier ABF", "English": "📂 1. Upload ABF File"},
    "settings": {"Français": "⚙️ 2. Paramètres de Détection", "English": "⚙️ 2. Detection Settings"},
    "dvdt_th": {"Français": "Seuil dV/dt (mV/ms) pour le seuil du PA :", "English": "dV/dt threshold (mV/ms) for AP threshold:"},
    "amplitude_th": {"Français": "Seuil minimal du pic de PA (mV) :", "English": "Minimum AP peak threshold (mV):"},
    "results_tab": {"Français": "📊 Profil Biophysique Global", "English": "📊 Global Biophysical Profile"},
    "sweeps_tab": {"Français": "🔢 Analyse par Sweep", "English": "🔢 Sweep-by-Sweep Analysis"},
    "methodo_tab": {"Français": "📚 Principes Biophysiques & Méthodologie", "English": "📚 Biophysical Principles & Method"},
    "warning_db": {"Français": "⚠️ AVERTISSEMENT : Bloc de Dépolarisation Détecté", "English": "⚠️ WARNING: Depolarization Block Detected"},
}

st.markdown(f"# {T['title'][lang]}")
st.markdown(f"### {T['subtitle'][lang]}")
st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header(T["load"][lang])
uploaded_file = st.sidebar.file_uploader(
    "Upload .abf file", 
    type=["abf"], 
    help="Fichiers bruts issus d'Axon pClamp (Axopatch/Multiclamp)."
)

st.sidebar.header(T["settings"][lang])
dvdt_threshold = st.sidebar.slider(T["dvdt_th"][lang], 5, 50, 15, step=1)
peak_voltage_threshold = st.sidebar.slider(T["amplitude_th"][lang], -40, 20, -10, step=5)

# --- CORPS DE L'APPLICATION ---
if not uploaded_file:
    st.info("👈 Veuillez charger un fichier d'enregistrement patch-clamp (`.abf`) dans le menu latéral pour débuter l'analyse.")
    
    # Affichage immédiat des liens académiques requis
    st.markdown("### 🎓 Documentation & Citation")
    st.markdown("[📄 Consulter le guide d'utilisation complet (README sur GitHub)](https://github.com/ManzoniLab/ElectrophyPipeline/blob/main/README.md)")
    st.markdown("Pour citer cette pipeline dans vos articles (*Science*, *Nature*, etc.), veuillez utiliser l'identifiant numérique d'objet permanent suivant :")
    st.code("DOI: 10.5281/zenodo.XXXXXXX (Lien direct : https://doi.org/10.5281/zenodo.XXXXXXX)")

else:
    # Sauvegarde temporaire du fichier binaire ABF pour permettre la lecture par pyabf
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_filepath = tmp.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        
        sweep_data = []
        depolarization_blocks = []
        excluded_sweeps = []
        
        # Initialisation des variables pour l'analyse de population
        max_spikes = 0
        rheobase_curr = None
        
        # --- BOUCLE DE TRAITEMENT DES SWEEPS ---
        for sweep_idx in abf.sweepList:
            abf.setSweep(sweep_idx)
            
            time = abf.sweepX # en secondes
            voltage = abf.sweepY # en mV
            
            # Échantillonnage de la commande de courant injecté (généralement au milieu de la trace)
            current_command = abf.sweepC
            i_inj = np.median(current_command[int(len(current_command)*0.4):int(len(current_command)*0.6)])
            
            # Calcul du dV/dt (dérivée temporelle du potentiel de membrane)
            dt = time[1] - time[0]
            dvdt = np.diff(voltage) / (dt * 1000) # V/s ou mV/ms
            
            # Détection des pics des potentiels d'action
            peaks, _ = find_peaks(voltage, height=peak_voltage_threshold, distance=int(0.002/dt))
            nb_spikes = len(peaks)
            
            # Mesure du voltage stationnaire (Steady-State Voltage) en fin d'échelon de courant
            v_steady = np.mean(voltage[int(len(voltage)*0.7):int(len(voltage)*0.75)])
            
            # --- ALGORITHME DE DÉTECTION DU BLOC DE DÉPOLARISATION ---
            # Si le neurone subit une injection dépolarisante massive mais que sa membrane sature 
            # à un plateau très positif (> -40 mV) sans pouvoir générer de vrais PA (inactivation des canaux NaV)
            is_db = False
            if i_inj > 0 and v_steady > -40.0:
                if nb_spikes == 0:
                    is_db = True
                elif sweep_idx > 0 and len(sweep_data) > 0:
                    # Chute brutale des spikes accompagnée d'un plateau de tension anormalement haut
                    prev_spikes = sweep_data[-1]['Nb_Spikes']
                    if nb_spikes < (prev_spikes / 2) and v_steady > -35.0:
                        is_db = True
            
            # Extraction des caractéristiques du premier potentiel d'action du sweep (si existant)
            v_thresh, ap_amp, ap_width = np.nan, np.nan, np.nan
            if nb_spikes > 0:
                first_peak_idx = peaks[0]
                
                # Remonter dans le temps pour trouver le franchissement du seuil dV/dt
                search_region = range(max(0, first_peak_idx - int(0.005/dt)), first_peak_idx)
                thresh_idx = next((idx for idx in search_region if dvdt[idx] >= dvdt_threshold), None)
                
                if thresh_idx is not None:
                    v_thresh = voltage[thresh_idx]
                    ap_amp = voltage[first_peak_idx] - v_thresh
                    
                    # Calcul de la demi-largeur (FWHM)
                    half_amplitude_voltage = v_thresh + (ap_amp / 2)
                    above_half = np.where(voltage[thresh_idx:first_peak_idx + int(0.01/dt)] >= half_amplitude_voltage)[0]
                    if len(above_half) > 0:
                        ap_width = len(above_half) * dt * 1000 # conversion en ms
            
            if is_db:
                depolarization_blocks.append(sweep_idx)
                excluded_sweeps.append(sweep_idx)
            
            # Enregistrement des données du sweep
            sweep_data.append({
                "Sweep": sweep_idx,
                "I_inj_nA": round(i_inj, 4),
                "Nb_Spikes": nb_spikes,
                "V_steady_mV": round(v_steady, 2),
                "V_threshold_mV": round(v_thresh, 2) if not np.isnan(v_thresh) else None,
                "AP_Amp_mV": round(ap_amp, 2) if not np.isnan(ap_amp) else None,
                "AP_Width_ms": round(ap_width, 3) if not np.isnan(ap_width) else None,
                "Status": "Depolarization Block" if is_db else "Normal"
            })
            
            # Suivi de la Rhéobase
            if nb_spikes > 0 and rheobase_curr is None and i_inj >= 0:
                rheobase_curr = i_inj

        df_sweeps = pd.DataFrame(sweep_data)
        
        # --- CALCULS BIOPHYSIQUES GLOBAUX ---
        # Potentiel de repos (Vrest) calculé sur les sweeps où l'injection est nulle (0 pA/nA)
        zero_current_sweeps = df_sweeps[df_sweeps['I_inj_nA'].abs() < 1e-3]
        v_rest = zero_current_sweeps['V_steady_mV'].mean() if not zero_current_sweeps.empty else df_sweeps['V_steady_mV'].iloc[0]
        
        # Résistance d'entrée (R_in) : calculée sur la plage hyperpolarisante linéaire saine
        hyper_df = df_sweeps[(df_sweeps['I_inj_nA'] < 0) & (df_sweeps['Status'] == "Normal")]
        if len(hyper_df) >= 2:
            slope, _ = np.polyfit(hyper_df['I_inj_nA'], hyper_df['V_steady_mV'], 1)
            r_in = slope # nA et mV s'annulent pour donner des MOhms
        else:
            r_in = np.nan

        # --- EXCLUSION DES TRACES DE L'IV ---
        # Filtrage des sweeps sains pour le tracé de la relation Courant-Voltage
        df_iv_clean = df_sweeps[~df_sweeps['Sweep'].isin(excluded_sweeps)]

        # --- AFFICHAGE DES ONGLETS ---
        tab1, tab2, tab3 = st.tabs([T["results_tab"][lang], T["sweeps_tab"][lang], T["methodo_tab"][lang]])
        
        with tab1:
            # Affichage des avertissements de bloc de dépolarisation de manière très visible
            if depolarization_blocks:
                st.warning(f"{T['warning_db'][lang]} : Les balayages (Sweeps) suivants ont été exclus de l'analyse stationnaire de la membrane car le neurone a saturé : {depolarization_blocks}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Potentiel de Repos (Vrest)", f"{v_rest:.1f} mV")
            col2.metric("Résistance d'Entrée (Rin)", f"{r_in:.1f} MΩ" if not np.isnan(r_in) else "N/A")
            col3.metric("Rhéobase (I Rheobase)", f"{rheobase_curr:.3f} nA" if rheobase_curr is not None else "Non atteinte")
            
            st.divider()
            
            # Graphiques de synthèse (f-I et I-V)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Courbe f-I (Courants positifs)
            df_fi = df_sweeps[df_sweeps['I_inj_nA'] >= -1e-3]
            ax1.plot(df_fi['I_inj_nA'], df_fi['Nb_Spikes'], '-o', color='firebrick', linewidth=2)
            ax1.set_title("Courbe Fréquence - Courant (f-I)")
            ax1.set_xlabel("Courant Injecté (nA)")
            ax1.set_ylabel("Nombre de Spikes (PA)")
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Courbe I-V (uniquement les sweeps non exclus pour éviter l'artefact du bloc de dépolarisation)
            ax2.plot(df_iv_clean['I_inj_nA'], df_iv_clean['V_steady_mV'], '-s', color='royalblue', linewidth=2, label="Traces saines")
            if excluded_sweeps:
                df_excl = df_sweeps[df_sweeps['Sweep'].isin(excluded_sweeps)]
                ax2.scatter(df_excl['I_inj_nA'], df_excl['V_steady_mV'], color='orange', marker='x', s=100, zorder=5, label="Bloc (Exclu)")
            ax2.set_title("Relation Courant - Voltage (I-V)")
            ax2.set_xlabel("Courant Injecté (nA)")
            ax2.set_ylabel("Voltage Stationnaire (mV)")
            ax2.legend(frameon=False)
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            sns_style = ["top", "right"]
            for ax in [ax1, ax2]:
                for edge in sns_style: ax.spines[edge].set_visible(False)
                
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.markdown("### Tableau des métriques sweep par sweep")
            
            # AJOUT REQUIS : Intégration explicite de la colonne I_Rheobase_nA dans le tableau exporté
            df_export = df_sweeps.copy()
            df_export['I_Rheobase_nA'] = round(rheobase_curr, 4) if rheobase_curr is not None else np.nan
            
            # Réorganisation pour mettre en valeur les unités requises
            st.dataframe(df_export, use_container_width=True)
            
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Exporter le profil biophysique (CSV)",
                data=csv_data,
                file_name=f"Spike_Analysis_Metrics_{uploaded_file.name}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab3:
            if lang == "Français":
                st.markdown("""
                ### 📚 Méthodologie Clinique & Formalisme Biophysique
                
                Cette pipeline d'analyse extrait de manière automatisée les constantes fondamentales de la membrane neuronale à partir d'enregistrements en *Whole-Cell Current-Clamp*.
                
                #### 1. Potentiel de Repos de la Membrane ($V_{rest}$)
                Le $V_{rest}$ représente le potentiel de membrane stable mesuré en l'absence de toute injection ou commande de courant extérieure ($I = 0$). Il reflète l'état d'équilibre électrochimique de la cellule au repos, principalement maintenu par les conductances potassiques de fuite et la pompe $Na^+/K^+$ ATPase.
                
                #### 2. Résistance d'Entrée ($R_{in}$)
                Calculée en appliquant la loi d'Ohm macroscopique ($\Delta V = R \cdot \Delta I$). L'algorithme calcule la pente de la relation linéaire sur les échelons de courants hyperpolarisants. Une résistance d'entrée élevée traduit une faible densité de canaux ioniques ouverts au repos, rendant le neurone plus sensible aux entrées synaptiques.
                
                #### 3. Courbes F-I & Rhéobase ($I_{rheobase}$)
                * **La Rhéobase** est définie comme l'intensité minimale de courant injecté (exprimée en **nA**) nécessaire pour atteindre le seuil de décharge et déclencher un potentiel d'action unique.
                * **La courbe f-I** traduit le gain d'excitabilité somato-dendritique du neurone. Elle quantifie la fréquence de décharge en fonction de l'intensité du stimulus.
                
                #### 4. Cinétique du Potentiel d'Action (PA)
                * **Seuil ($V_{threshold}$)** : Point d'inflexion cinétique où l'ouverture coopérative des canaux $Na_V$ dépendants du voltage outrepasse les courants potassiques de fuite. Il est isolé mathématiquement là où la dérivée de la membrane ($dV/dt$) franchit le seuil critique (ex: $15$ mV/ms).
                * **Amplitude ($AP_{amp}$)** : Différence de potentiel stricte entre le niveau du seuil ($V_{threshold}$) et le sommet (pic) du potentiel d'action.
                * **Demi-largeur ($AP_{width}$)** : Durée totale du potentiel d'action mesurée à 50% de son amplitude maximale. Une modification de ce paramètre traduit une altération de la cinétique d'inactivation du sodium ou d'activation des canaux potassiques retardés ($K_V$).
                
                #### 5. Bloc de Dépolarisation (*Depolarization Block*)
                Lors d'une stimulation dépolarisante continue et massive, le potentiel stationnaire de la membrane s'élève au-dessus d'une valeur critique (typiquement > -40 mV). À ce niveau, les canaux $Na_V$ n'ont plus la capacité physique de se désinactiver (fermeture de la porte de vannes $h$). Les potentiels d'action s'amortissent, s'effondrent en amplitude puis disparaissent complètement. Conserver ces traces fausserait les calculs de résistance ou de dynamique stationnaire, raison pour laquelle l'algorithme les isole et les exclut de la courbe I-V.
                """)
            else:
                st.markdown("""
                ### 🔬 Biophysical Core Principles & Methodologies
                
                This pipeline provides automated extraction of fundamental neuronal membrane properties from *Whole-Cell Current-Clamp* recordings.
                
                #### 1. Resting Membrane Potential ($V_{rest}$)
                $V_{rest}$ is the baseline membrane potential measured when no current is being injected ($I = 0$). It indicates the electrochemical equilibrium of the cell, primarily governed by background leak potassium conductances and the $Na^+/K^+$ ATPase pump.
                
                #### 2. Input Resistance ($R_{in}$)
                Determined by Ohm's Law ($\Delta V = R \cdot \Delta I$) through a linear fit across hyperpolarizing current steps. A higher input resistance means fewer open channels at rest, indicating that less synaptic current is required to alter the membrane potential.
                
                #### 3. F-I Curves & Rheobase ($I_{rheobase}$)
                * **Rheobase** is the minimal current intensity (expressed here in **nA**) required to depolarize the membrane up to the firing threshold, triggering at least one action potential.
                * **The f-I curve** describes the spiking frequency output as a function of current input, reflecting the active gain properties of the somatodendritic compartment.
                
                #### 4. Action Potential (AP) Kinetics
                * **Threshold ($V_{threshold}$)**: The voltage where voltage-gated $Na^+$ currents exceed outward potassium leak currents. It is defined mathematically where the first derivative ($dV/dt$) crosses a user-defined threshold (e.g., $15$ mV/ms).
                * **Amplitude ($AP_{amp}$)**: The potential difference between the calculated threshold and the absolute peak of the action potential.
                * **Half-Width ($AP_{width}$)**: The total duration of the spike measured at 50% of its maximum amplitude (FWHM). Alterations in this metric indicate changes in $Na^+$ channel inactivation or voltage-gated delayed rectifier $K^+$ channel kinetics.
                
                #### 5. Depolarization Block
                Under heavy, sustained depolarizing inputs, the membrane potential reaches a critical steady-state plateau (typically > -40 mV). At this sustained voltage, voltage-gated sodium channels are trapped in an inactivated state (the $h$-gate remains closed). Action potentials first attenuate in amplitude and then vanish. Retaining these traces would introduce massive artifacts into voltage calculations; the algorithm automatically isolates and excludes them from the proper I-V plot.
                """)

    finally:
        if os.path.exists(tmp_filepath): 
            os.remove(tmp_filepath)
