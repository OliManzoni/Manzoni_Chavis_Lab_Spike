import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Averaged Curves", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Interface Language:", ["Français", "English"])

# --- DICTIONNAIRE DE TRADUCTION ---
T = {
    "title": {"Français": "Courbes Moyennées f-I & I-V", "English": "Averaged f-I & I-V Curves"},
    "subtitle": {"Français": "Analyse de Population & Excitabilité", "English": "Population Analysis & Excitability"},
    "upload": {"Français": "📂 1. Charger les fichiers", "English": "📂 1. Upload Files"},
    "upload_help": {"Français": "Sélectionnez plusieurs fichiers '_Sweeps.csv'", "English": "Select multiple '_Sweeps.csv' files"},
    "settings": {"Français": "⚙️ 2. Paramètres d'Alignement", "English": "⚙️ 2. Alignment Settings"},
    "align_method": {"Français": "Méthode de groupement :", "English": "Grouping method:"},
    "align_sweep": {"Français": "Par numéro de Sweep (Balayage)", "English": "By Sweep number"},
    "align_round": {"Français": "Par Courant (Arrondi à 3 décimales)", "English": "By Current (Rounded to 3 decimals)"},
    "err_type": {"Français": "Type d'Erreur (Ombrage)", "English": "Error Type (Shading)"},
    "err_sem": {"Français": "SEM (Erreur Standard)", "English": "SEM (Standard Error)"},
    "err_ci": {"Français": "CI (95% Confiance)", "English": "CI (95% Confidence)"},
    "tab_graphs": {"Français": "📈 Visualisation des Courbes", "English": "📈 Curve Visualization"},
    "tab_data": {"Français": "🔢 Données Moyennées (Export)", "English": "🔢 Averaged Data (Export)"},
    "export_btn": {"Français": "💾 Exporter les Moyennes (CSV)", "English": "💾 Export Averages (CSV)"},
}

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([1, 6]) 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")
st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header(T["upload"][lang])
uploaded_files = st.sidebar.file_uploader(
    T["upload_help"][lang], 
    type=["csv"], 
    accept_multiple_files=True
)

st.sidebar.header(T["settings"][lang])
alignment_choice = st.sidebar.radio(
    T["align_method"][lang], 
    [T["align_sweep"][lang], T["align_round"][lang]]
)

error_choice = st.sidebar.radio(
    T["err_type"][lang], 
    [T["err_sem"][lang], T["err_ci"][lang]]
)

# --- CORPS DE L'APPLICATION ---
if not uploaded_files:
    st.info("👈 " + ("Veuillez charger vos fichiers '_Sweeps.csv' dans le menu latéral pour commencer." if lang == "Français" else "Please upload your '_Sweeps.csv' files in the sidebar to begin."))
else:
    df_list = []
    cell_count = 0
    
    for f in uploaded_files:
        # Réinitialisation du curseur fichier (Corrige le bug de re-run Streamlit)
        f.seek(0) 
        try:
            df = pd.read_csv(f)
            df['Cell_ID'] = f.name.replace("_Sweeps.csv", "").replace(".csv", "")
            df_list.append(df)
            cell_count += 1
        except Exception as e:
            st.warning(f"Impossible de lire {f.name} : {e}")
            
    if df_list:
        master_df = pd.concat(df_list, ignore_index=True)
        required_cols = ['Sweep', 'I_inj', 'Nb_Spikes', 'V_steady']
        
        if not all(col in master_df.columns for col in required_cols):
            st.error(f"Format incorrect. Les colonnes requises sont : {required_cols}")
        else:
            # Sécurisation des types numériques
            for col in required_cols:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
            
            # 🧮 LOGIQUE D'ALIGNEMENT
            if alignment_choice == T["align_sweep"][lang]:
                # On groupe par le numéro du balayage, l'axe X sera la moyenne exacte du courant à ce sweep
                stats_df = master_df.groupby('Sweep').agg(
                    I_inj_Mean=('I_inj', 'mean'),
                    N_Cells=('Cell_ID', 'nunique'),
                    Nb_Spikes_Mean=('Nb_Spikes', 'mean'),
                    Nb_Spikes_SEM=('Nb_Spikes', 'sem'),
                    V_steady_Mean=('V_steady', 'mean'),
                    V_steady_SEM=('V_steady', 'sem')
                ).reset_index()
            else:
                # On arrondit le courant (3 décimales = pA exact) pour fusionner les variations du digitaliseur
                master_df['I_inj_Rounded'] = master_df['I_inj'].round(3)
                stats_df = master_df.groupby('I_inj_Rounded').agg(
                    I_inj_Mean=('I_inj_Rounded', 'mean'), # Identique au groupe
                    N_Cells=('Cell_ID', 'nunique'),
                    Nb_Spikes_Mean=('Nb_Spikes', 'mean'),
                    Nb_Spikes_SEM=('Nb_Spikes', 'sem'),
                    V_steady_Mean=('V_steady', 'mean'),
                    V_steady_SEM=('V_steady', 'sem')
                ).reset_index()

            # Remplacement des NaN (si N=1) par 0 pour éviter les bugs graphiques
            stats_df = stats_df.fillna(0)
            
            # Calcul des intervalles de confiance (95%)
            stats_df['Nb_Spikes_CI'] = stats_df['Nb_Spikes_SEM'] * 1.96
            stats_df['V_steady_CI'] = stats_df['V_steady_SEM'] * 1.96

            # Sélection de la métrique d'erreur à afficher
            err_col_spikes = 'Nb_Spikes_SEM' if error_choice == T["err_sem"][lang] else 'Nb_Spikes_CI'
            err_col_vsteady = 'V_steady_SEM' if error_choice == T["err_sem"][lang] else 'V_steady_CI'

            st.success(f"✅ {cell_count} " + ("cellules consolidées." if lang == "Français" else "cells consolidated."))
            
            tab1, tab2 = st.tabs([T["tab_graphs"][lang], T["tab_data"][lang]])

            # --- ONGLET 1 : GRAPHIQUES (Format Publication) ---
            with tab1:
                # Construction explicite et robuste avec Matplotlib
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Sous-échantillonnage pour la courbe f-I (seulement courants dépolarisants)
                stats_depol = stats_df[stats_df['I_inj_Mean'] >= 0]
                
                # --- Graphe f-I ---
                ax1.plot(stats_depol['I_inj_Mean'], stats_depol['Nb_Spikes_Mean'], marker='o', color='firebrick', linewidth=2, label="Mean")
                ax1.fill_between(
                    stats_depol['I_inj_Mean'], 
                    stats_depol['Nb_Spikes_Mean'] - stats_depol[err_col_spikes], 
                    stats_depol['Nb_Spikes_Mean'] + stats_depol[err_col_spikes], 
                    color='firebrick', alpha=0.2, label=error_choice
                )
                ax1.set_title("Excitabilité (f-I)", fontweight='bold', fontsize=14)
                ax1.set_xlabel("Courant Injecté (nA)", fontsize=12)
                ax1.set_ylabel("Nombre de Potentiels d'Action", fontsize=12)
                ax1.legend(frameon=False)
                ax1.grid(True, linestyle='--', alpha=0.3)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                # --- Graphe I-V ---
                ax2.plot(stats_df['I_inj_Mean'], stats_df['V_steady_Mean'], marker='s', color='royalblue', linewidth=2, label="Mean")
                ax2.fill_between(
                    stats_df['I_inj_Mean'], 
                    stats_df['V_steady_Mean'] - stats_df[err_col_vsteady], 
                    stats_df['V_steady_Mean'] + stats_df[err_col_vsteady], 
                    color='royalblue', alpha=0.2, label=error_choice
                )
                ax2.set_title("Relation Courant-Voltage (I-V)", fontweight='bold', fontsize=14)
                ax2.set_xlabel("Courant Injecté (nA)", fontsize=12)
                ax2.set_ylabel("Voltage Stationnaire (mV)", fontsize=12)
                ax2.legend(frameon=False)
                ax2.grid(True, linestyle='--', alpha=0.3)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Libère la mémoire

            # --- ONGLET 2 : DONNÉES ET EXPORT ---
            with tab2:
                display_cols = ['I_inj_Mean', 'N_Cells', 'Nb_Spikes_Mean', 'Nb_Spikes_SEM', 'Nb_Spikes_CI', 'V_steady_Mean', 'V_steady_SEM', 'V_steady_CI']
                st.dataframe(stats_df[display_cols].style.format(precision=3), use_container_width=True)
                
                csv_export = stats_df[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=T["export_btn"][lang],
                    data=csv_export,
                    file_name="Consolidated_Population_Data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
