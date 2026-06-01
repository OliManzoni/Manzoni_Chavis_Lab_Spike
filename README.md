🇫🇷 Version Française

pour nous citer:
https://github.com/OliManzoni/Manzoni_Chavis_Lab_Spike
https://doi.org/10.5281/zenodo.19912621

Ce dépôt contient le code source du Pipeline Expert Excitabilité, une application analytique haute performance conçue pour l'extraction automatisée des propriétés biophysiques neuronales à partir de fichiers .abf (Current-Clamp).
🚀 Déploiement & Accessibilité

L'outil est accessible via deux canaux selon vos besoins de calcul et de confidentialité :
1. Accès Web Immédiat (Streamlit Cloud)

Pour une analyse rapide de quelques cellules sans aucune installation technique :
👉 https://manzonichavislab-spikes.streamlit.app/
2. Version Expert (GitHub & Local)

Pour les chercheurs souhaitant intégrer l'outil dans des pipelines de calcul intensif ou modifier les algorithmes sources :

    Dépôt : [Lien GitHub du Manzoni Lab]

    Installation :
    Bash

    git clone https://github.com/votre-depot/manzoni-excitability.git
    pip install -r requirements.txt
    streamlit run app_spike.py

🔬 Pipeline Expert : Excitabilité & Propriétés Intrinsèques

Manzoni Lab | Neurobiologie de la Plasticité Synaptique

Ce pipeline est un environnement analytique haute résolution dédié à la caractérisation automatique des propriétés électrophysiologiques membranaires à partir de fichiers d'enregistrement .abf (Current-Clamp).
1. Bases Biophysiques & Signification Physiologique

L'excitabilité intrinsèque est le déterminant majeur de la fonction computationnelle du neurone. Elle définit la manière dont les entrées synaptiques (analogiques) sont converties en décharges de potentiels d'action (numériques).
A. Le Potentiel de Repos (Vrest​)

Le Vrest​ reflète l'état d'équilibre électrochimique de la cellule, principalement maintenu par les canaux potassiques de fuite (Leak channels) et la pompe Na+/K+-ATPase. Une dépolarisation du Vrest​ peut signaler un état de stress cellulaire ou une réduction des conductances potassiques basales.
B. Résistance d'Entrée (Rin​)

La Rin​ est inversement proportionnelle à la densité totale des canaux ioniques ouverts au repos.

    Signification : Un neurone avec une Rin​ élevée est plus "excitable" car une faible injection de courant suffira à générer une variation de tension importante (ΔV=Rin​⋅ΔI).

    Contexte : Dans les modèles de neuro-adaptation, une baisse de Rin​ est souvent corrélée à une hypertrophie de l'arborisation dendritique (loi de surface).

C. Capacitance Membranaire (Cm​) et Constante de Temps (τm​)

La membrane lipidique se comporte comme un condensateur (Cm​) en parallèle avec une résistance (Rin​).

    τm​ : Définit la "mémoire" électrique du neurone. Un τm​ long favorise la sommation temporelle des entrées synaptiques distantes.

    Cm​ : C'est un marqueur direct de la surface membranaire totale (~1 µF/cm²). Une augmentation de Cm​ traduit une croissance physique du neurone.

D. Courant Ih​ (Sag)

L'affaissement du voltage (Sag) lors d'une hyperpolarisation est la signature des canaux HCN. Ce courant "pacemaker" régule l'excitabilité sous-liminaire et la résonance du neurone.
2. Formalisme Mathématique & Algorithmes

Le pipeline utilise des méthodes de régression et de dérivation numérique couplées à des filtres topologiques pour garantir la reproductibilité des mesures et rejeter les artefacts de stimulation.
Extraction de la Résistance d'Entrée (Rin​)

Pour éviter les non-linéarités induites par l'activation de conductances voltage-dépendantes, Rin​ est calculée exclusivement sur le régime passif :

    Sélection des 4 premiers échelons hyperpolarisants (I<0).

    Extraction du voltage stationnaire (Vss​) en fin de pulse.

    Régression linéaire : La pente de la droite Vss​=f(Iinj​) donne Rin​.

        Note : Si l'unité est le nA, la pente est directement en MΩ. Si l'unité est le pA, le résultat est multiplié par 1000.

Calcul de la Capacitance (Cm​)

En Current-Clamp, Cm​ est dérivée de la cinétique de charge :
Cm​=Rin​τm​​

Le pipeline mesure τm​ sur le pulse hyperpolarisant le plus faible pour minimiser l'impact du courant Ih​. τm​ est défini par le temps nécessaire pour atteindre 63.2% de l'amplitude totale de la réponse.
Détection des Potentiels d'Action & Cinétique (dV/dt)

Contrairement aux méthodes basiques par seuil fixe, l'algorithme combine une analyse de dérivée et des contraintes biophysiques strictes :

    Seuil dV/dt : L'algorithme calcule la dérivée première du voltage. Le point de seuil est défini comme l'instant où l'accélération dépasse une valeur critique (par défaut 15 mV/ms). C'est le marqueur de l'ouverture massive des canaux sodiques.

    Proéminence : Pour être comptabilisé, un événement doit s'élever d'une certaine amplitude depuis sa ligne de base locale (ex: 20 mV), ce qui élimine les oscillations sous-liminaires ("spikelets").

    Période Réfractaire : Une distance temporelle minimale (ex: 2 ms) est imposée entre deux pics pour éviter le comptage multiple sur un potentiel d'action élargi.

Filtre Heuristique : Gestion du Bloc de Dépolarisation

Lors d'injections de courants très intenses, le neurone peut perdre sa capacité à repolariser la membrane, entrant en bloc de dépolarisation. Le pipeline intègre une porte logique d'exclusion :

    Si le seuil de déclenchement calculé s'effondre de façon aberrante (ex: Vthreshold​< -60 mV) ou si la phase de repolarisation n'est plus détectable (Decay = NaN), l'algorithme annule automatiquement le comptage pour ce palier.

    Cette sécurité empêche l'explosion stochastique du nombre de spikes due au bruit haute fréquence sur une membrane bloquée.

3. Guide d'Utilisation du Pipeline
Configuration Initiale

    Seuil dV/dt : Si vos potentiels d'action ont une phase d'ascension lente, abaissez ce seuil à 10 mV/ms. Pour des neurones très rapides (interneurones), montez à 20 mV/ms.

    Proéminence & Période Réfractaire : Ajustez ces contraintes biophysiques dans la barre latérale si le pipeline sur-détecte des événements (faux positifs) lors d'injections de courant extrêmes.

Exploration Visuelle

    Points Rouges : Ils permettent de synchroniser la trace visualisée en haut avec sa position sur les courbes I-V et f-I en bas. Cela permet d'identifier instantanément quel sweep a généré une valeur atypique (comme un bloc de dépolarisation).

    Overlay : Utilisez cette fonction pour comparer visuellement la morphologie du PA au seuil rhéobasique par rapport aux échelons de forte intensité.

Exportation

Le bouton "Exportation des Résultats" génère deux fichiers :

    _Global.csv : Résumé des propriétés intrinsèques (Vrest​, Rin​, Cm​, τm​, Rhéobase) pour la cellule (Tableau prêt pour l'analyse statistique).

    _Sweeps.csv : Matrice détaillée par échelon de courant (Points bruts de la courbe f-I, cinétiques fines du 1er PA telles que l'amplitude, Half-Width, Rise et Decay) prête à être importée sous GraphPad Prism.

🇬🇧 English Version
to cite us:
https://github.com/OliManzoni/Manzoni_Chavis_Lab_Spike
https://doi.org/10.5281/zenodo.19912621

This repository contains the source code for the Expert Excitability Pipeline, a high-performance analytical application designed for the automated extraction of neuronal biophysical properties from .abf (Current-Clamp) files.
🚀 Deployment & Accessibility

The tool is accessible via two channels depending on your computational and privacy needs:
1. Immediate Web Access (Streamlit Cloud)

For rapid analysis of a few cells without any technical installation:
👉 https://manzonichavislab-spikes.streamlit.app/
2. Expert Version (GitHub & Local)

For researchers wishing to integrate the tool into computationally intensive pipelines or modify the source algorithms:

    Repository: [Manzoni Lab GitHub Link]

    Installation:
    Bash

    git clone https://github.com/your-repo/manzoni-excitability.git
    pip install -r requirements.txt
    streamlit run app_spike.py

🔬 Expert Pipeline: Excitability & Intrinsic Properties

Manzoni Lab | Neurobiology of Synaptic Plasticity

This pipeline is a high-resolution analytical environment dedicated to the automatic characterization of membrane electrophysiological properties from .abf (Current-Clamp) recording files.
1. Biophysical Foundations & Physiological Significance

Intrinsic excitability is the major determinant of the computational function of the neuron. It defines how synaptic inputs (analog) are converted into action potential discharges (digital).
A. Resting Membrane Potential (Vrest​)

Vrest​ reflects the cell's electrochemical equilibrium state, primarily maintained by potassium leak channels and the Na+/K+-ATPase pump. Depolarization of Vrest​ may signal a state of cellular stress or a reduction in basal potassium conductances.
B. Input Resistance (Rin​)

Rin​ is inversely proportional to the total density of open ion channels at rest.

    Significance: A neuron with high Rin​ is more "excitable" because a small current injection is sufficient to generate a large voltage change (ΔV=Rin​⋅ΔI).

    Context: In models of neuroadaptation, a decrease in Rin​ is often correlated with hypertrophy of dendritic arborization (surface law).

C. Membrane Capacitance (Cm​) and Time Constant (τm​)

The lipid membrane behaves as a capacitor (Cm​) in parallel with a resistor (Rin​).

    τm​: Defines the electrical "memory" of the neuron. A long τm​ favors the temporal summation of distant synaptic inputs.

    Cm​: This is a direct marker of total membrane surface area (~1 µF/cm²). An increase in Cm​ reflects the physical growth of the neuron.

D. Ih​ Current (Sag)

Voltage sag during hyperpolarization is the signature of HCN channels. This "pacemaker" current regulates subthreshold excitability and neuronal resonance.
2. Mathematical Formalism & Algorithms

The pipeline utilizes regression and numerical derivation methods coupled with topological filters to ensure measurement reproducibility and reject stimulation artifacts.
Extraction of Input Resistance (Rin​)

To avoid nonlinearities induced by the activation of voltage-dependent conductances, Rin​ is calculated exclusively in the passive regime:

    Selection of the first 4 hyperpolarizing steps (I<0).

    Extraction of steady-state voltage (Vss​) at the end of the pulse.

    Linear Regression: The slope of the line Vss​=f(Iinj​) gives Rin​.

        Note: If the unit is nA, the slope is directly in MΩ. If the unit is pA, the result is multiplied by 1000.

Calculation of Capacitance (Cm​)

In Current-Clamp, Cm​ is derived from the charging kinetics:
Cm​=Rin​τm​​

The pipeline measures τm​ on the weakest hyperpolarizing pulse to minimize the impact of the Ih​ current. τm​ is defined by the time required to reach 63.2% of the total response amplitude.
Action Potential Detection & Kinetics (dV/dt)

Unlike basic fixed-threshold methods, the algorithm combines derivative analysis with strict biophysical constraints:

    dV/dt Threshold: The algorithm calculates the first derivative of the voltage. The threshold point is defined as the moment when acceleration exceeds a critical value (default 15 mV/ms). This is the marker of massive sodium channel opening.

    Prominence: To be counted, an event must rise by a certain amplitude from its local baseline (e.g., 20 mV), which eliminates subthreshold oscillations ("spikelets").

    Refractory Period: A minimum temporal distance (e.g., 2 ms) is enforced between two peaks to prevent multiple counts on a single broadened action potential.

Heuristic Filter: Depolarization Block Management

During very intense current injections, the neuron may lose its ability to repolarize the membrane, entering a depolarization block. The pipeline integrates an exclusion logic gate:

    If the calculated threshold drops aberrantly (e.g., Vthreshold​< -60 mV) or if the repolarization phase is no longer detectable (Decay = NaN), the algorithm automatically cancels the spike count for that sweep.

    This safeguard prevents the stochastic explosion of spike counts due to high-frequency noise on a blocked membrane.

3. Pipeline User Guide
Initial Configuration

    dV/dt Threshold: If your action potentials have a slow rising phase, lower this threshold to 10 mV/ms. For very fast neurons (interneurons), increase it to 20 mV/ms.

    Prominence & Refractory Period: Adjust these biophysical constraints in the sidebar if the pipeline over-detects events (false positives) during extreme current injections.

Visual Exploration

    Red Dots: These allow synchronization of the trace displayed at the top with its position on the I-V and f-I curves at the bottom. This instantly identifies which sweep generated an atypical value (such as a depolarization block).

    Overlay: Use this function to visually compare the AP morphology at the rheobase threshold versus high-intensity steps.

Export

The "Export Results" button generates two files:

    _Global.csv: Summary of intrinsic properties (Vrest​, Rin​, Cm​, τm​, Rheobase) for the cell (Table ready for statistical analysis).

    _Sweeps.csv: Detailed matrix per current step (Raw points for the f-I curve, fine kinetics of the 1st AP such as Amplitude, Half-Width, Rise, and Decay) ready to be imported into GraphPad Prism.

