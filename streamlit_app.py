# ────────────────────────────────────────────────────────────────────────────────
# TITANIC BINARY CLASSIFICATION APP (Streamlit)
# ────────────────────────────────────────────────────────────────────────────────

# Import des librairies nécessaires
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Configuration de la page
st.set_page_config(page_title="Titanic ML App", layout="wide")

# Chargement des données
df = pd.read_csv('train.csv')

# Titre principal
st.title("🚢 Projet Titanic : Classification binaire des passagers")

# Barre latérale : navigation
st.sidebar.title("📑 Navigation")
pages = ["Exploration", "Visualisation", "Modélisation"]
page = st.sidebar.radio("Aller vers :", pages)

# ───────────────────────────────
# PAGE 1 : Exploration
# ───────────────────────────────
if page == "Exploration":
    st.header("🔎 Exploration des données")
    st.subheader("Aperçu du dataset Titanic")
    
    # Affiche les 10 premières lignes
    st.dataframe(df.head(10))

    # Affiche la forme du dataframe
    st.markdown(f"**Dimensions du jeu de données :** {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe())

    # Affichage des valeurs manquantes
    if st.checkbox("Afficher les valeurs manquantes"):
        st.subheader("Valeurs manquantes")
        st.dataframe(df.isna().sum())

# ───────────────────────────────
# PAGE 2 : Visualisation
# ───────────────────────────────
elif page == "Visualisation":
    st.header("📊 Data Visualisation")

    # Affichage de la distribution de la variable cible
    st.subheader("Répartition des survivants")
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)
    st.info("La majorité des passagers n'ont pas survécu. Les classes sont relativement équilibrées.")

    # Répartition par genre
    st.subheader("Répartition des passagers par genre")
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x='Sex', data=df)
    st.pyplot(fig)

    # Répartition par classe
    st.subheader("Répartition des classes")
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x='Pclass', data=df)
    st.pyplot(fig)

    # Distribution de l'âge
    st.subheader("Distribution des âges")
    fig = sns.displot(df['Age'].dropna(), kde=True, height=5, aspect=1.5)
    st.pyplot(fig)
    st.info("Les passagers sont principalement des hommes en 3ème classe, âgés entre 20 et 40 ans.")

    # Analyse de survie en fonction du sexe
    st.subheader("Survie selon le sexe")
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig)

    # Analyse selon la classe
    st.subheader("Survie selon la classe")
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point', height=5)
    st.pyplot(fig)

    # Régression entre l’âge et la survie
    st.subheader("Relation entre l'âge et la survie par classe")
    fig = sns.lmplot(x='Age', y='Survived', hue='Pclass', data=df, aspect=1.5)
    st.pyplot(fig)

    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ───────────────────────────────
# PAGE 3 : Modélisation
# ───────────────────────────────
elif page == "Modélisation":
    st.header("🤖 Modélisation")

    # Nettoyage et préparation
    st.subheader("Préparation des données")

    # Suppression de colonnes inutiles
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Séparation des variables
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    # Traitement des valeurs manquantes
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # Encodage et concaténation
    X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
    X = pd.concat([X_cat_encoded, X_num], axis=1)

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standardisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # Sélection du modèle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(model_name):
        if model_name == 'Random Forest':
            clf = RandomForestClassifier()
        elif model_name == 'SVC':
            clf = SVC()
        elif model_name == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    def evaluate_model(clf, metric):
        if metric == 'Accuracy':
            return clf.score(X_test, y_test)
        elif metric == 'Confusion Matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    st.subheader("Choix du modèle")
    model_options = ['Random Forest', 'SVC', 'Logistic Regression']
    model_choice = st.selectbox('Sélectionnez un modèle', model_options)

    clf = prediction(model_choice)

    st.subheader("Évaluation du modèle")
    metric = st.radio("Quel résultat souhaitez-vous afficher ?", ['Accuracy', 'Confusion Matrix'])

    if metric == 'Accuracy':
        score = evaluate_model(clf, metric)
        st.success(f"🔍 Précision du modèle **{model_choice}** : {round(score*100, 2)} %")
    else:
        matrix = evaluate_model(clf, metric)
        st.write("🔍 Matrice de confusion :")
        st.dataframe(pd.DataFrame(matrix, columns=["Prédit : Non", "Prédit : Oui"], index=["Réel : Non", "Réel : Oui"]))
