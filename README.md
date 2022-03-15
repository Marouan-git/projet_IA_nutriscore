Projet 1 - Groupe 7

Membres du groupe : Marouan et Nolan

# Présentation de l'application 

Application Web développée avec FastAPI

Contient un formulaire permettant de prédire le nutriscore d'un produit à partir de certaines caractéristiques pour 100g de produit (qui correspondent aux paramètres du modèle de prédiction utilisé):

- L'énergie (en kJ) --> energy_100g
- La quantité de gras (en g) --> fat_100g
- La quantité d'acides gras saturés (en g) --> saturated_fat_100g
- La quantité de carbohydrates (=glucides) (en g) --> carbohydrates_100g
- La quantité de sucre (en g) --> sugars_100g
- La quantité de fibres (en g) --> fibers_100g
- La quantité de protéines (en g) --> proteins_100g
- La quantité de sel (en g) --> salt_100g
- La quantité de fruits, légumes et noix (en g) --> fruits_vegetables_nuts_estimate_from_ingredients_100g
- La catégorie à laquelle appartient le produit (se basant sur la classification pnns groups 2) --> pnns_groups_2

Affichage du nutriscore prédit une fois les données du formulaire envoyées

Le modèle de prédiction a été entraîné à partir de la base de données openfoodfacts : https://fr.openfoodfacts.org/data  

# Modèles de prédiction utilisé

RandomForest et KNN avec les hyper-paramètres par défaut

# Utilisation du modèle joblib  
  
Note : le modèle joblib n'a pas été ajouté au répertoire git car trop volumineux. Pour le générer, il suffit d'exécuter le fichier nutriscore/model.py

Import du modèle à l'aide de la fonction load de la bibliothèque joblib:

`from joblib import load `

`model = load('model_rf_oversampled.joblib')`

Pour effectuer une prédiction, il faut fournir au modèle un dataframe contenant tous ses paramètres (mentionnés ci-dessus).
Il faut donc créer ce dataframe :

`df = pd.DataFrame(np.array([[energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_group_2]]),
                         columns=['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'pnns_groups_2'])`

Utiliser la fonction predict en plaçant le dataframe en paramètre, pour effectuer la prédiction :

`prediction = model.predict(df)`

# Installation et lancement de l'application en local

Cloner le répertoire git : 

`git clone https://gitlab.com/simplon-dev-ia/grenoble-2021-2022/projets/projet-1/projet-1-groupe-7.git`

Créer un environnement virtuel avec pipenv : 

`pipenv install`

Lancer l'application sur un serveur avec la commande : 

`uvicorn main:app --reload`

# Prédire le nutriscore via l'application Web

Remplir les champs du formulaire

Cliquer sur le bouton "Prédire"

# Prédire le nutriscore via l'API

### Test avec httpie

Installer httpie : 

`sudo apt install httpie`

Dans un terminal, run l'application sur un serveur: 

`uvicorn main:app --reload`

Dans un autre terminal, entrer la commande : (en spécifiant une valeur pour chaque paramètre)

`http http://127.0.0.1:8000/api/predict energy_100g=1 fat_100g=1 saturated_fat_100g=1 carbohydrates_100g=1 sugars_100g=1 fiber_100g=1 proteins_100g=1 salt_100g=1 fruits_vegetables_nuts_estimate_from_ingredients_100g=1 pnns_groups_2="Meat"`

Valeurs possibles pour pnns_groups_2 :

`['Dressings and sauces', 'Bread', 'Fruits', 'One-dish meals',
       'Vegetables', 'Dairy desserts', 'Pastries', 'unknown',
       'Salty and fatty products', 'Sweetened beverages',
       'Pizza pies and quiches', 'Sweets', 'Biscuits and cakes',
       'Fish and seafood', 'Cheese', 'Appetizers', 'Fats',
       'Chocolate products', 'Unsweetened beverages', 'Dried fruits',
       'Meat', 'Cereals', 'Eggs', 'Plant-based milk substitutes',
       'Processed meat', 'Sandwiches', 'Legumes', 'Breakfast cereals',
       'Soups', 'Milk and yogurt', 'Nuts',
       'Artificially sweetened beverages', 'Fruit juices', 'Ice cream',
       'Offals', 'Waters and flavored waters',
       'Teas and herbal teas and coffees', 'Potatoes', 'Fruit nectars',
       'Alcoholic beverages']`

# Lien vers l'application Web en ligne

Seul le modèle knn fonctionne, le modèle randomForest était trop volumineux pour pouvoir le déployer avec la version gratuite d'Heroku.

http://nutria.herokuapp.com/










