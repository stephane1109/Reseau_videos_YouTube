# Analyse d'un réseaux de vidéos sur YouTube

**[www.codeandcortex.fr](http://www.codeandcortex.fr) – version 1.0 – 01/07/2025**

## Description

Ce tableau de bord Streamlit permet d'explorer les relations entre vidéos YouTube à partir de plusieurs dimensions d'analyse. Il utilise l’API YouTube pour récupérer des données en temps réel, et construit des graphes ou des clusters selon les critères choisis.

## Fonctionnalités

* **Réseau par commentaires communs**
  Relie deux vidéos si elles partagent des commentateurs.

* **Réseau par similarité sémantique**
  Relie deux vidéos si leurs titres et descriptions sont proches d’un point de vue sémantique (modèle de langue).

* **Réseau par métriques numériques**
  Relie deux vidéos si leurs vues, likes et commentaires sont similaires selon une distance cosinus.

* **Clustering K-Means sur les métriques**
  Regroupe automatiquement les vidéos en clusters selon leur profil d'engagement numérique (vues, likes, commentaires) avec visualisation en 2D.

## Filtres disponibles

* **Date de publication**
* **Nombre minimum de vues**
* **Nombre minimum de likes**
* **Nombre minimum de commentaires**

## Visualisations

* **Graphes interactifs** (Pyvis + NetworkX)
* **Nuage de points par cluster** (PCA + Plotly)

Les résultats sont **enregistrés automatiquement** sous forme de fichier `.xlsx` et `.html` dans le répertoire de travail.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
streamlit run main.py
```

## Configuration requise

* Une **clé API YouTube** valide (renseigner dans l’interface Streamlit).
* Python ≥ 3.8

## Auteur

**Stéphane Meurisse**
Site : [www.codeandcortex.fr](http://www.codeandcortex.fr)
