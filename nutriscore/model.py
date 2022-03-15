# Gestion dataframes
import pandas as pd

# Fonction pour nettoyer les données
from clean_data import clean

# Fonction pour sur-échantillonner les données
from imblearn.over_sampling import SMOTENC

# Fonctions sklearn pour le pré-processing et l'entraînement du modèle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Fonction pour exporter le modèle dans un fichier joblib
from joblib import dump


# Import des données
path_file = input()
df = pd.read_csv(path_file, sep='\t', low_memory=False)

# Nettoyage des données
df_cleaned = clean(df)

# Sur-échantillonnage des données
X = df_cleaned.drop(columns='nutriscore_grade')
y = df_cleaned['nutriscore_grade']
X_oversampled, y_oversampled = SMOTENC(
    categorical_features=[9]).fit_resample(X, y)

# Split données entraînement/test
X_train_oversampled, X_test_oversampled, y_train_oversampled, y_test_oversampled = train_test_split(
    X_oversampled, y_oversampled, test_size=0.2, random_state=42)

# Création du pipeline
cat_attribs = ['pnns_groups_2']
num_attribs = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
               'fiber_100g', 'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g']

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier()),
])

# Entraînement du pipeline
full_pipeline.fit(X_train_oversampled, y_train_oversampled)

# Export du modèle
dump(full_pipeline, 'model_rf_oversampled.joblib')
