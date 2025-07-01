# www.codeandcortex.fr - version 1.0 - 01-07-2025
# Stéphane Meurisse

# python -m streamlit run main.py

# pip install streamlit pandas google-api-python-client XlsxWriter networkx pyvis
# pip install scikit-learn numpy
# pip install sentence_transformers
# pip install umap-learn plotly

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import networkx as nx
from networkx.algorithms import bipartite, community
from pyvis.network import Network
import tempfile
import os
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import html
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# import umap
# import plotly.express as px


st.set_page_config(layout="wide", page_title="Analyse YouTube par vidéo")
# En-tête de l’application
st.title("Analyse d'un réseau de vidéos sur YouTube")
# version
st.markdown("**www.codeandcortex.fr - version 1.0 - date : 01-07-2025**")


st.markdown("""
Ce tableau de bord vous permet de construire différents types de réseaux entre vidéos YouTube :

- **Par commentaires communs** : relie deux vidéos si elles partagent des commentateurs.
- **Par similarité sémantique** : relie deux vidéos si leurs titres/descriptions sont proches.
- **Par métriques numériques** : relie deux vidéos si leurs vues, likes, commentaires sont similaires.
- **Par clustering K-Means** : regroupe automatiquement les vidéos selon leur profil d'engagement (vues, likes, commentaires) numérique.

Vous pouvez appliquer des **filtres de date**, **de vues**, **de likes** et **de commentaires**.
Vous pouvez également télécharger les résultats sont enregistrés automatiquement sous forme de fichier Excel et visualiser un graphe interactif ou un nuage de points par cluster.
""")

# Paramètres utilisateur
cle_api = st.sidebar.text_input("Clé API YouTube", type="password")
mots_cles = st.sidebar.text_input("Mots-clés de recherche (séparés par des virgules)")
liste_mots = [mot.strip() for mot in mots_cles.split(",") if mot.strip()]
nb_videos = st.sidebar.slider("Nombre total de vidéos à analyser", 10, 1000, 30)
nb_commentaires = st.sidebar.slider("Commentaires max par vidéo", 10, 500, 100)
seuil_co = st.sidebar.slider("Seuil de commentaires communs (poids min)", 1, 20, 2)
seuil_sim = st.sidebar.slider("Seuil de similarité cosinus (0-1)", 0.0, 1.0, 0.8, step=0.05)

filtrer_date = st.sidebar.checkbox("Filtrer par date de publication")
if filtrer_date:
    date_min = st.sidebar.date_input(
        "Date minimale",
        value=datetime.date.today(),
        max_value=datetime.date.today()
    )

else:
    date_min = None

filtrer_stats = st.sidebar.checkbox("Filtrer par nombre de vues/likes/commentaires")
if filtrer_stats:
    vues_min = st.sidebar.number_input("Nombre minimum de vues", value=0)
    likes_min = st.sidebar.number_input("Nombre minimum de likes", value=0)
    commentaires_min = st.sidebar.number_input("Nombre minimum de commentaires", value=0)
else:
    vues_min = 0
    likes_min = 0
    commentaires_min = 0

run_social = st.sidebar.button("Construire un réseau par commentaires communs")
run_similarity = st.sidebar.button("Construire un réseau par similarité textuelle")
run_metrics = st.sidebar.button("Construire un réseau par métriques")
run_kmeans = st.sidebar.button("Construire un clustering par K-Means sur les métriques")


if run_social or run_similarity or run_metrics or run_kmeans:
    yt = build("youtube", "v3", developerKey=cle_api)

    video_ids, titres, descriptions, vues_d, likes_d, comments_d = [], {}, {}, {}, {}, {}

    for mot in liste_mots:
        page_token = None
        while len(video_ids) < nb_videos:
            try:
                res = yt.search().list(
                    q=mot,
                    part="snippet",
                    type="video",
                    maxResults=50,
                    pageToken=page_token
                ).execute()
            except HttpError:
                break

            for item in res.get("items", []):
                if item["id"].get("kind") != "youtube#video":
                    continue
                vid = item["id"].get("videoId")
                if not vid:
                    continue
                titre = html.unescape(item["snippet"].get("title", ""))
                desc = html.unescape(item["snippet"].get("description", ""))

                try:
                    stats = yt.videos().list(part="statistics", id=vid).execute()["items"][0]["statistics"]
                    vues = int(stats.get("viewCount", 0))
                    likes = int(stats.get("likeCount", 0))
                    comments = int(stats.get("commentCount", 0))
                except:
                    continue

                if vues < vues_min or likes < likes_min or comments < commentaires_min:
                    continue

                if vid not in video_ids:
                    video_ids.append(vid)
                    titres[vid] = titre
                    descriptions[vid] = desc
                    vues_d[vid] = vues
                    likes_d[vid] = likes
                    comments_d[vid] = comments

                if len(video_ids) >= nb_videos:
                    break

            page_token = res.get("nextPageToken")
            if not page_token:
                break

    st.write(f"{len(video_ids)} vidéos analysées.")
    df_videos = pd.DataFrame({
        "Titre": [titres[v] for v in video_ids],
        "URL": [f"https://www.youtube.com/watch?v={v}" for v in video_ids],
        "Vues": [vues_d[v] for v in video_ids],
        "Likes": [likes_d[v] for v in video_ids],
        "Commentaires": [comments_d[v] for v in video_ids]
    })
    st.dataframe(df_videos)

    G2 = nx.Graph()

    if run_social:
        B = nx.Graph()
        for vid in video_ids:
            B.add_node(vid, bipartite=0)
            try:
                rep = yt.commentThreads().list(
                    part="snippet", videoId=vid, maxResults=nb_commentaires, textFormat="plainText"
                ).execute()
                for item in rep.get("items", []):
                    snip = item["snippet"]["topLevelComment"]["snippet"]
                    auteur = snip.get("authorChannelId", {}).get("value")
                    if auteur:
                        B.add_node(auteur, bipartite=1)
                        B.add_edge(vid, auteur)
            except:
                continue
        videos = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
        G = bipartite.weighted_projected_graph(B, videos)
        for u, v, d in G.edges(data=True):
            if d["weight"] >= seuil_co:
                G2.add_edge(u, v, weight=d["weight"])

    elif run_similarity:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') # multilingue
        textes = [titres[v] + " " + descriptions[v] for v in video_ids]
        embeddings = model.encode(textes, convert_to_tensor=True)
        sim = util.cos_sim(embeddings, embeddings)
        for i in range(len(video_ids)):
            for j in range(i + 1, len(video_ids)):
                score = sim[i][j].item()
                if score >= seuil_sim:
                    G2.add_edge(video_ids[i], video_ids[j], weight=score)

    elif run_metrics:
        X = [[vues_d[v], likes_d[v], comments_d[v]] for v in video_ids]
        X_scaled = MinMaxScaler().fit_transform(X)
        sim = cosine_similarity(X_scaled)
        for i in range(len(video_ids)):
            for j in range(i + 1, len(video_ids)):
                score = sim[i][j]
                if score >= seuil_sim:
                    G2.add_edge(video_ids[i], video_ids[j], weight=score)

    elif run_kmeans:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import plotly.express as px

        X = [[vues_d[v], likes_d[v], comments_d[v]] for v in video_ids]

        if not X:
            st.warning(
                "Aucune donnée disponible pour effectuer l'analyse K-Means. Vérifiez les filtres ou les données récupérées.")
        else:
            # Normalisation
            X_scaled = MinMaxScaler().fit_transform(X)

            # Choix du nombre de clusters
            k = st.sidebar.slider("Nombre de clusters (K)", 2, 10, 4)

            # K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
            labels = kmeans.labels_

            # Affectation des clusters au DataFrame
            df_videos["Cluster"] = [labels[video_ids.index(v)] for v in video_ids]

            # Réduction de dimension pour visualisation
            coords = PCA(n_components=2).fit_transform(X_scaled)

            df_proj = pd.DataFrame(coords, columns=["x", "y"])
            df_proj["Titre"] = [titres[v] for v in video_ids]
            df_proj["URL"] = [f"https://www.youtube.com/watch?v={v}" for v in video_ids]
            df_proj["Cluster"] = labels

            # Affichage graphique
            fig = px.scatter(
                df_proj, x="x", y="y", color=df_proj["Cluster"].astype(str),
                hover_data=["Titre", "URL"], title="Clustering K-Means des vidéos"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Construction d’un graphe si besoin
            for i in range(len(video_ids)):
                for j in range(i + 1, len(video_ids)):
                    if labels[i] == labels[j]:
                        G2.add_edge(video_ids[i], video_ids[j], weight=1.0)

            # le bloc d'enregistrement
            fichier_excel_kmeans = "clustering_kmeans.xlsx"
            df_videos.to_excel(fichier_excel_kmeans, index=False)
            st.success(f"Fichier Excel enregistré sous : `{fichier_excel_kmeans}`")

            with open(fichier_excel_kmeans, "rb") as f:
                st.download_button("Télécharger les résultats (K-Means)", f, file_name=fichier_excel_kmeans)

            fichier_html_kmeans = "clustering_kmeans_graphique.html"
            fig.write_html(fichier_html_kmeans)
            st.success(f"Graphique interactif enregistré sous : `{fichier_html_kmeans}`")

    if not run_kmeans and G2.number_of_nodes() > 0:
        try:
            communities = list(community.greedy_modularity_communities(G2, weight="weight"))
            group_map = {node: idx for idx, comm in enumerate(communities) for node in comm}
        except:
            group_map = {n: 0 for n in G2.nodes()}

        net = Network(height="700px", width="100%", notebook=False)
        net.force_atlas_2based()

        for n in G2.nodes():
            titre = titres.get(n, n)
            deg = G2.degree(n, weight="weight")
            groupe = group_map.get(n, 0)
            net.add_node(
                n,
                label=titre[:80],
                size=15 + deg,
                title=titre,
                url=f"https://www.youtube.com/watch?v={n}",
                group=groupe
            )

        for u, v, d in G2.edges(data=True):
            net.add_edge(u, v, value=d["weight"], title=f"{d['weight']:.2f} lien")

        # Déterminer le nom selon le mode d’analyse
        if run_social:
            mode = "reseau_commentaires"
        elif run_similarity:
            mode = "reseau_semantique"
        elif run_metrics:
            mode = "reseau_metriques"
        else:
            mode = "clusters_kmeans"

        # Sauvegarde du graphe HTML
        chemin_html = f"{mode}.html"
        net.save_graph(chemin_html)
        st.success(f"Graphique sauvegardé sous : `{chemin_html}`")

        with open(chemin_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)

        # Export Excel avec les vidéos du graphe
        df_export = df_videos[df_videos["URL"].apply(lambda x: x.split("v=")[1] in G2.nodes())]
        fichier_excel = f"{mode}.xlsx"
        df_export.to_excel(fichier_excel, index=False)

        st.markdown(f"Résultats exportés dans le fichier : `{fichier_excel}`")

# Idées :
# - Grouper les vidéos par département
# - Créer un graphe inter-département (en utilisant les commentateurs communs)
# - Calculer les métriques de centralité, de connectivité, etc.
# - Afficher un tableau des métriques par département + une visualisation graphique.
