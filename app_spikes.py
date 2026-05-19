import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Averaged Curves", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Interface Language:", ["Français", "English"])

# --- DICTIONNAIRE DE TRADUCTION ---
T = {
    "title": {"Français": "Courbes Moyennées f-I & I-V", "English": "Averaged f-I & I-V Curves"},
    "subtitle": {"Français": "Analyse de Population & Excitabilité | Manzoni Lab", "English": "Population Analysis & Excitability | Manzoni Lab"},
    "upload": {"Français": "📂 1. Charger les fichiers", "English": "📂 1. Upload Files"},
    "upload_help": {"Français": "Sélectionnez plusieurs fichiers terminant par '_Sweeps.csv'", "English": "Select multiple '_Sweeps.csv' files"},
    "settings": {"Français": "⚙️ 2. Option d'Alignement", "English": "⚙️ 2. Alignment Options"},
    "align_method": {"Français": "Choisir la méthode d'alignement :", "English": "Choose alignment method:"},
    "align_sweep": {"Français": "Option 1 : Par numéro de Sweep", "English": "Option 1: By Sweep number"},
    "align_round": {"Français": "Option 2 : Par Courant arrondi à l'entier le plus proche (pA)", "English": "Option 2: By Current rounded to the nearest integer (pA)"},
    "step_size": {"Français": "Cible de l'arrondi (pA) :", "English": "Rounding target (pA):"},
    "err_type": {"Français": "Type d'Erreur (Ombrage)", "English": "Error Type (Shading)"},
    "err_sem": {"Français": "SEM (Erreur Standard)", "English": "Standard Error (SEM)"},
    "err_ci": {"Français": "CI (95% Confiance)", "English": "Confidence Interval (95% CI)"},
    "tab_graphs": {"Français": "📈 Visualisation des Courbes", "English": "📈 Curve Visualization"},
    "tab_data": {"Français": "🔢 Données Moyennées (Export)", "English": "🔢 Averaged Data (Export)"},
    "export_btn": {"Français": "💾 Exporter les Moyennes (CSV)", "English": "💾 Export Averages (CSV)"},
}

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

# Option permettant de choisir entre l'entier brut (ex: 49 pA) ou le pas théorique (ex: 50 pA)
round_target = 1
if alignment_choice == T["align_round"][lang]:
    round_target = st.sidebar.selectbox(
        T["step_size"][lang],
        [1, 10, 50],
        index=0,
        help="1 = Entier le plus proche (ex: 49 pA). 50 = Calage automatique sur la grille du protocole (ex: 50 pA)."
    )

error_choice = st.sidebar.radio(
    T["err_type"][lang], 
    [T["err_sem"][lang], T["err_ci"][lang]]
)

# --- ANALYSE DES DONNÉES ---
if not uploaded_files:
    st.info("👈 " + ("Veuillez charger vos fichiers '_Sweeps.csv' dans le menu latéral." if lang == "Français" else "Please upload your '_Sweeps.csv' files in the sidebar."))
else:
    df_list = []
    cell_count = 0
    
    for f in uploaded_files:
        f.seek(0) # Corrige le comportement de rafraîchissement Streamlit
        try:
            df = pd.read_csv(f)
            df['Cell_ID'] = f.name.replace("_Sweeps.csv", "").replace(".csv", "")
            df_list.append(df)
            cell_count += 1
        except Exception as e:
            st.warning(f"Erreur lors de la lecture de {f.name} : {e}")
            
    if df_list:
        master_df = pd.concat(df_list, ignore_index=True)
        required_cols = ['Sweep', 'I_inj', 'Nb_Spikes', 'V_steady']
        
        if not all(col in master_df.columns for col in required_cols):
            st.error(f"Colonnes requises manquantes : {required_cols}")
        else:
            # Sécurisation des formats de données
            for col in required_cols:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
            
            # Conversion mathématique nA -> pA (ex: 0.04883 nA devient 48.83 pA)
            master_df['I_inj_pA'] = master_df['I_inj'] * 1000
            
            # --- IMPLÉMENTATION DES DEUX OPTIONS DEMANDÉES ---
            if alignment_choice == T["align_sweep"][lang]:
                # OPTION 1 : Groupement strict par numéro de Sweep
                stats_df = master_df.groupby('Sweep').agg(
                    X_Value=('I_inj_pA', 'mean'), # L'axe X est la moyenne des courants réels pour ce sweep
                    N_Cells=('Cell_ID', 'nunique'),
                    Nb_Spikes_Mean=('Nb_Spikes', 'mean'),
                    Nb_Spikes_SEM=('Nb_Spikes', 'sem'),
                    V_steady_Mean=('V_steady', 'mean'),
                    V_steady_SEM=('V_steady', 'sem')
                ).reset_index()
                x_label = "Courant Injecté (pA) [Alignement par Sweep]" if lang == "Français" else "Injected Current (pA) [Aligned by Sweep]"
            else:
                # OPTION 2 : Groupement par valeur de courant arrondie à l'entier le plus proche
                if round_target == 1:
                    master_df['I_inj_Aligned'] = master_df['I_inj_pA'].round(0)
                else:
                    master_df['I_inj_Aligned'] = (master_df['I_inj_pA'] / round_target).round(0) * round_target
                
                stats_df = master_df.groupby('I_inj_Aligned').agg(
                    X_Value=('I_inj_Aligned', 'first'), # L'axe X prend la valeur discrète arrondie
                    N_Cells=('Cell_ID', 'nunique'),
                    Nb_Spikes_Mean=('Nb_Spikes', 'mean'),
                    Nb_Spikes_SEM=('Nb_Spikes', 'sem'),
                    V_steady_Mean=('V_steady', 'mean'),
                    V_steady_SEM=('V_steady', 'sem')
                ).reset_index()
                x_label = "Courant Injecté (pA) [Arrondi à l'entier]" if lang == "Français" else "Injected Current (pA) [Rounded to Integer]"

            # Remplissage des absences de variance et calcul des intervalles de confiance
            stats_df = stats_df.fillna(0)
            stats_df['Nb_Spikes_CI'] = stats_df['Nb_Spikes_SEM'] * 1.96
            stats_df['V_steady_CI'] = stats_df['V_steady_SEM'] * 1.96

            err_spikes = 'Nb_Spikes_SEM' if error_choice == T["err_sem"][lang] else 'Nb_Spikes_CI'
            err_vsteady = 'V_steady_SEM' if error_choice == T["err_sem"][lang] else 'V_steady_CI'

            st.success(f"✅ {cell_count} " + ("cellules moyennées et superposées." if lang == "Français" else "cells successfully averaged and aligned."))
            
            tab1, tab2 = st.tabs([T["tab_graphs"][lang], T["tab_data"][lang]])

            # --- ONGLET 1 : GRAPHIQUES ---
            with tab1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Isolation des courants dépolarisants (>= 0 pA) pour la courbe f-I
                stats_depol = stats_df[stats_df['X_Value'] >= -1e-2]
                
                # Graphe f-I
                ax1.plot(stats_depol['X_Value'], stats_depol['Nb_Spikes_Mean'], marker='o', color='firebrick', linewidth=2, label="Mean")
                ax1.fill_between(
                    stats_depol['X_Value'], 
                    stats_depol['Nb_Spikes_Mean'] - stats_depol[err_spikes], 
                    stats_depol['Nb_Spikes_Mean'] + stats_depol[err_spikes], 
                    color='firebrick', alpha=0.18, label=error_choice
                )
                ax1.set_title("Excitabilité (f-I)", fontweight='bold', fontsize=13)
                ax1.set_xlabel(x_label, fontsize=11)
                ax1.set_ylabel("Nombre de Potentiels d'Action", fontsize=11)
                ax1.legend(frameon=False)
                ax1.grid(True, linestyle='--', alpha=0.3)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                # Graphe I-V
                ax2.plot(stats_df['X_Value'], stats_df['V_steady_Mean'], marker='s', color='royalblue', linewidth=2, label="Mean")
                ax2.fill_between(
                    stats_df['X_Value'], 
                    stats_df['V_steady_Mean'] - stats_df[err_vsteady], 
                    stats_df['V_steady_Mean'] + stats_df[err_vsteady], 
                    color='royalblue', alpha=0.18, label=error_choice
                )
                ax2.set_title("Relation Courant-Voltage (I-V)", fontweight='bold', fontsize=13)
                ax2.set_xlabel(x_label, fontsize=11)
                ax2.set_ylabel("Voltage Stationnaire (mV)", fontsize=11)
                ax2.legend(frameon=False)
                ax2.grid(True, linestyle='--', alpha=0.3)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # --- ONGLET 2 : EXPÉDITION DES MATRICES ---
            with tab2:
                cols_to_show = ['X_Value', 'N_Cells', 'Nb_Spikes_Mean', 'Nb_Spikes_SEM', 'V_steady_Mean', 'V_steady_SEM']
                export_df = stats_df[cols_to_show].rename(columns={'X_Value': 'Injected_Current_pA'})
                st.dataframe(export_df.style.format(precision=2), use_container_width=True)
                
                csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=T["export_btn"][lang],
                    data=csv_bytes,
                    file_name="Averaged_Population_Data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
