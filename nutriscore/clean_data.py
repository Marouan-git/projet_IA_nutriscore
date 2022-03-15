import pandas as pd

'''
    Fonction clean() : 
    
    - prend en entrée un dataframe pandas contenant les données openfoodfacts non nettoyées 
    - renvoie un dataframe pandas contenant les données openfoodfacts nettoyées prêtes pour entraîner le modèle

    Paramètre : dataframe pandas

    Renvoi : dataframe pandas nettoyé
'''


def clean(df):

    # Suppression des lignes nulles de la valeur cible
    df_red = df.dropna(subset=['nutriscore_grade'])

    # Suppression des colonnes avec plus de 50% de valeurs nulles
    threshold_col = len(df_red) * 0.50
    df_reduit = df_red.dropna(axis=1, thresh=threshold_col)

    # Suppression des colonnes non pertinentes pour la prédiction
    col_to_keep = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                   'salt_100g', 'sodium_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'pnns_groups_2', 'nutriscore_grade']
    df_reduit = df_reduit[col_to_keep]

    # Suppression des lignes avec plus de 25% de valeurs nulles
    threshold_row = len(df_reduit.columns) * 0.75
    df_reduit_row = df_reduit.dropna(axis=0, thresh=threshold_row)

    # Suppression des lignes avec valeurs incohérentes

    # Les colonnes 100g ne doivent pas comporter de valeurs supérieures à 100
    col_100g = ['fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
                'proteins_100g', 'salt_100g', 'sodium_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g']
    df_reduit_100g = df_reduit_row[df_reduit_row['fat_100g'] < 100]
    for col in col_100g:
        df_reduit_100g = df_reduit_100g[(
            df_reduit_100g[col] < 100) & (df_reduit_100g[col] >= 0)]
    # La densité énergétique maximale pour 100g est de 3700 kJ, on supprime tout ce qui est supérieur à 3700kJ dans energy_100g
    df_reduit_100g = df_reduit_100g[df_reduit_100g['energy_100g'] < 3700]

    # Suppression de la colonne sodium qui est trop corrélée avec salt
    df_complet = df_reduit_100g.drop(columns=['sodium_100g'])

    return df_complet
