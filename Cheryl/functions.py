import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy.stats import chi2_contingency
from scipy.stats import skew, kurtosis
from collections import Counter

def find_categorical_variables(data, threshold):
    """
    Identifie les variables avec un nombre de valeurs uniques inférieur à un seuil.
    Retourne un DataFrame avec les variables et leurs comptes de valeurs uniques.
    """
    categorical_result = []
    for col in data.columns:
        unique_count = data[col].nunique()
        if unique_count < threshold:
            categorical_result.append([col, unique_count])

    result_df = pd.DataFrame(categorical_result, columns=["Variable", "Unique_Count"])

    print("-"*40)
    print(f"Nombre total de variables ayant moins de {threshold} modalités : {len(result_df)} \n")
    print("-"*40)
    print(result_df.sort_values(by="Unique_Count") if not result_df.empty
           else "Aucune variable trouvée avec un nombre de modalités unique inférieur au seuil donné.")
    print("-"*40)

    return result_df


def find_uniq_mod_variables(data, qualitative_vars, threshold=0.9):
    """
    Identifie les variables qualitatives ayant une modalité dominante dépassant un certain seuil.
    Retourne un DataFrame avec les variables, la modalité dominante, son pourcentage, et le nombre total de modalités.
    """
    unique_mod_result = []

    for var in qualitative_vars:
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences
        if value_counts.iloc[0] >= threshold:
            unique_mod_result.append({
                "Variable": var,
                "Nb_mod": data[var].nunique(),  # Nombre total de modalités
                "Mod_dominante": value_counts.index[0],
                "Frequence": value_counts.iloc[0]
            })

    result_df = pd.DataFrame(unique_mod_result)

    print("-"*40)
    print(f"Nombre de variables ayant une modalité dominante à plus de {threshold * 100}% : {len(result_df)}")
    print("-"*40)
    print(result_df if not result_df.empty else "Aucune variable trouvée avec une modalité dominante.")
    print("-"*40)

    return result_df

import matplotlib.pyplot as plt
import math

def plot_cat_vars_distributions(data, vars_list, cols=2):
    """
    Génère des graphiques montrant la distribution des modalités pour chaque variable dans une grille.
    Les barres sont annotées avec leurs valeurs.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten() 
    
    for i, var in enumerate(vars_list):
        value_counts = data[var].value_counts(normalize=True) 
        index_values = value_counts.index.to_flat_index()  
        index_values = [str(x) for x in index_values]   
        
        # Création du graphique
        bars = axes[i].bar(index_values, value_counts.values, color='skyblue')
        axes[i].set_ylabel('Proportion')
        axes[i].set_title(f'{var}')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Annotation des valeurs sur les barres
        for bar, value in zip(bars, value_counts.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                         f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Supprime les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Distribution des variables catégorielles", fontsize=10, x=0.0, y=1.02, ha='left') 
    
    plt.tight_layout()
    plt.show()


def tx_rsq_par_var(df, categ_vars, date, target, cols=2,sharey=True):
    """
    Génère une grille de graphiques montrant les taux d'événement moyens par modalité au fil du temps,
    pour une liste de variables catégorielles, en fonction des valeurs cibles fournies.
    """
    df = df.copy()

    # Vérification des colonnes
    missing_cols = [col for col in [date] + categ_vars if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Nettoyer les valeurs manquantes dans les colonnes nécessaires
    df = df.dropna(subset=[date] + categ_vars)

    num_vars = len(categ_vars)
    rows = math.ceil(num_vars / cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows * 4), sharex=False, sharey=sharey)
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, categ_var in enumerate(categ_vars):
        # Calcul des moyennes par date et catégorie
        df_times_series = (df.groupby([date, categ_var])[target].mean()).reset_index()
        df_pivot = df_times_series.pivot(index=date, columns=categ_var, values=target)

        # Création du graphique
        ax = axes[i]
        for category in df_pivot.columns:
            ax.plot(df_pivot.index, df_pivot[category], label=category)
        ax.set_title(f"{categ_var}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tx de défaut (%)")
        ax.legend(title="Modalités", fontsize='small', loc='upper left')

    # Supprimer les axes inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Taux de défaut par variable catégorielle",fontsize=10, x=0.0, y=1.02, ha='left')
    plt.tight_layout()
    plt.show()


def combined_barplot_lineplot(df, cat_vars, cible, cols=2):
    """
    Génère une grille de barplots combinés avec des lineplots pour une liste de variables catégorielles.
    """
    num_vars = len(cat_vars)
    rows = math.ceil(num_vars / cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, cat_col in enumerate(cat_vars):
        ax1 = axes[i]  # Axe pour le barplot

        # Vérifier si la variable est catégorielle et la convertir en chaîne si nécessaire
        if pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype(str)

        # Calcul du taux de risque
        tx_rsq = (df.groupby([cat_col])[cible].mean()).reset_index()

        # Calcul des effectifs
        effectifs = df[cat_col].value_counts(normalize=True).reset_index()
        effectifs.columns = [cat_col, "count"]

        # Fusion des données
        merged_data = effectifs.merge(tx_rsq, on=cat_col).sort_values(by=cible, ascending=True)

        # Création des graphiques
        ax2 = ax1.twinx()  # Deuxième axe pour le lineplot
        sns.barplot(data=merged_data, x=cat_col, y="count", color='grey', ax=ax1)
        sns.lineplot(data=merged_data, x=cat_col, y=cible, color='red', marker="o", ax=ax2)

        # Configuration des axes
        ax1.set_title(f"{cat_col}")
        ax1.set_xlabel("")
        ax1.set_ylabel("Fréquences des modalités")
        ax2.set_ylabel("Taux de risque (%)")
        ax1.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Titre général
    fig.suptitle("Barplots et Lineplots combinés pour les variables catégorielles",fontsize=10, x=0.0, y=1.02, ha='left')
    plt.tight_layout()
    plt.show()

def compare_distributions_grid(X_train, X_test, var_list, cols=2):
    """
    Compare les distributions des variables continues dans Train et Test et les affiche sous forme de grille.
    """
    num_vars = len(var_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows * 4))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(var_list):
        # Graphique pour chaque variable
        sns.kdeplot(X_train[var], label='Train', shade=True, ax=axes[i])
        sns.kdeplot(X_test[var], label='Test', shade=True, ax=axes[i])
        axes[i].set_title(f"{var}")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Comparaison des distributions dans l'echantillon Train et Test",fontsize=10, x=0.0, y=1.02, ha='left') 

    plt.tight_layout()
    plt.show()


def compare_distributions_summary(X_train, X_test, var_list):
    """
    Compare les distributions des variables continues dans Train et Test et retourne un tableau récapitulatif.
    Affiche les statistiques descriptives et les p-values des tests de Kolmogorov-Smirnov.
    """
    results = []

    for var in var_list:
        # Statistiques descriptives
        train_stats = X_train[var].describe()
        test_stats = X_test[var].describe()

        # Test de Kolmogorov-Smirnov
        ks_stat, ks_p_value = stats.ks_2samp(X_train[var], X_test[var])
        
        # Ajout des résultats dans la liste
        results.append({
            "Variable": var,
            "Train_Mean": train_stats["mean"],
            "Test_Mean": test_stats["mean"],
            "Train_Std": train_stats["std"],
            "Test_Std": test_stats["std"],
            "KS_Statistic": ks_stat,
            "KS_p_value": ks_p_value,
            "Similar_Distribution": "Yes" if ks_p_value > 0.05 else "No"
        })

    # Conversion des résultats en DataFrame
    result_df = pd.DataFrame(results)
    return result_df


def plot_modalities_over_time(X_train, date_col, categorical_vars, exclude_vars=None, cols=2):
    """
    Affiche l'évolution du nombre de modalités uniques par variable catégorielle au fil du temps.
    """
    if exclude_vars is None:
        exclude_vars = []

    # Filtrer les variables catégorielles
    cat_vars = [col for col in categorical_vars if col not in exclude_vars]

    # Créer un DataFrame pour stocker les informations pour la visualisation
    modalities_over_time = []

    # Itérer sur les dates d'observation
    for date in X_train[date_col].unique():
        filtered_data = X_train[X_train[date_col] == date]
        for col in cat_vars:
            modalities = filtered_data[col].unique()
            modalities_count = len(modalities)
            modalities_over_time.append({
                'date': date,
                'variable': col,
                'modalities_count': modalities_count
            })

    modalities_df = pd.DataFrame(modalities_over_time)

    # Déterminer le nombre de graphiques
    num_vars = len(cat_vars)
    rows = math.ceil(num_vars / cols)

    # Créer une grille de graphiques
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, var in enumerate(cat_vars):
        ax = axes[i]
        sns.lineplot(data=modalities_df[modalities_df['variable'] == var],
                     x='date', y='modalities_count', marker='o', ax=ax)
        ax.set_title(f"Évolution de {var}")
        ax.set_xlabel("Date d'observation")
        ax.set_ylabel("Nombre de modalités uniques")
        ax.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Evolution des modalités dans le temps",fontsize=10, x=0.0, y=1.02, ha='left') 

    plt.tight_layout()
    plt.show()


def plot_boxplots(data, vars_list, cols=2):
    """
    Affiche des boxplots pour chaque variable continue dans une grille.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(vars_list):
        sns.boxplot(data=data, y=var, ax=axes[i],showfliers=False)  # Création du boxplot
        axes[i].set_title(f"Boxplot de {var}")
        axes[i].set_xlabel("")  # Pas besoin d'étiquette pour x
        axes[i].set_ylabel("Valeurs")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution des variables continues",fontsize=10, x=0.0, y=1.02, ha='left')  # Positionner le titre légèrement au-dessus

    plt.tight_layout()
    plt.show()


import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_by_target(data, vars_list, target, cols=2):
    """
    Génère des boxplots (et KDE plots optionnellement) pour chaque variable continue en fonction des valeurs cibles fournies.
    
    Parameters:
    - data (DataFrame): Les données contenant les variables.
    - vars_list (list): Liste des variables continues à tracer.
    - target (str): Nom de la variable cible.
    - cols (int): Nombre de colonnes de la grille de subplots.
    - kde (bool): Si True, génère des KDE plots en plus des boxplots.
    """
    data = data.copy()
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour faciliter l'itération

    for i, var in enumerate(vars_list):
        sns.kdeplot(
            data=data, 
            x=var, 
            hue=target, 
            ax=axes[i], 
            fill=True, 
            palette="Set2", 
            common_norm=False, 
            alpha=0.5
        )
        axes[i].set_title(f"{var}")
        axes[i].set_xlabel(target)
        axes[i].set_ylabel(var)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"KDE des variables continues selon la variable cible",fontsize=10, x=0.0, y=1.02, ha='left')
    plt.tight_layout()
    plt.show()




def cramer_V(cat_var1,cat_var2):
    """
    Calcule le coefficient de Cramer's V pour une paire de variables catégorielles.
    """
    crosstab = np.array(pd.crosstab(cat_var1,cat_var2,rownames=None,colnames=None)) #tableau de contingence
    stat = chi2_contingency(crosstab)[0] #stat de test de khi-2
    obs=np.sum(crosstab) 
    mini = min(crosstab.shape)-1 #min entre les colonnes et ligne du tableau croisé ==> ddl
    return (np.sqrt(stat/(obs*mini)))

def compute_cramers_v(df, categorical_vars, target):
    """
    Calcule le coefficient de Cramer's V pour chaque combinaison paire de variables catégorielles dans la liste fournie.
    """
    results = []
    for var1 in categorical_vars :  # Unpack index and column name
        if var1 == target:
            continue  # Skip the calculation if the variable is the target itself
        cv = cramer_V(df[var1], df[target])  # Correctly pass the variable names
        results.append([var1, cv])  # Append the variable name, not the tuple

    # Create a DataFrame to hold the results
    result_df = pd.DataFrame(results, columns=['Columns', "Cramer_V"])
    return result_df

def stats_liaisons_var_quali(df,categorical_columns):
    """
    Calcule le test du chi-deux et le coefficient de cramer_v pour chaque paire de variables qualitatives
    """
    cramer_v_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    p_value_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    #tschuprow_t_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)

    #test de chi-deux pour chaque paire de variables quali
    for i, column1 in enumerate(categorical_columns):
        for j, column2 in enumerate(categorical_columns):
            if column1 != column2:
                contingency_table = pd.crosstab(df[column1], df[column2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                cramer_v = np.sqrt(chi2 / (df.shape[0] * (min(contingency_table.shape)-1) ))
                #tschuprow_t = np.sqrt(chi2 / (df.shape[0] * np.sqrt((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1))))
                cramer_v_df.loc[column1,column2] =cramer_v
                #tschuprow_t_df.loc[column1,column2] =tschuprow_t
                p_value_df.loc[column1,column2] = p
                
    return (p_value_df, cramer_v_df)

def test_freq_by_group(data, qualitative_vars, threshold=0.05):
    """
    Identifie les variables qualitatives qui ont au moins une modalité avec une fréquence relative
    inférieure ou égale au seuil spécifié.
    """
    # Liste pour stocker les variables correspondant au critère
    unique_mod_result = []

    for var in qualitative_vars:
        # Vérifie si la variable existe dans le DataFrame
        if var not in data.columns:
            print(f"Attention : la variable '{var}' n'existe pas dans le DataFrame.")
            continue
        
        # Calcul des fréquences relatives des modalités
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences

        # Vérifie si au moins une modalité a une fréquence <= threshold
        if (value_counts <= threshold).any():
            unique_mod_result.append(var)

    # Message si aucune variable ne satisfait le critère
    if len(unique_mod_result) == 0:
        print("Aucune variable n'a de modalités avec moins de 5% d'effectifs.")
    else :
        print(f"Les variables suivantes ont au moins une modalité avec une fréquence <= {threshold * 100}% :")
        print(unique_mod_result)

    return unique_mod_result


def group_by_rsq(df, cat_var,cible):
    """
    Groupe les modalités d'une variable catégorielle qui ont une fréquence inférieure à 5% 
    en fonction de leur taux de risque moyen.
    """
    grouped_classes = []
    cumulative_weight = 0
    group = []
    risk_rates = (df.groupby(cat_var)[cible].mean()).sort_values(ascending=False)
    freq_df = df.groupby(cat_var).size() / len(df)  # Fréquence des modalités
    freq_df = pd.DataFrame({'Frequence': freq_df, 'Taux de risque': risk_rates})
    freq_df = freq_df.sort_values(by='Taux de risque', ascending=False)
        
    # Affichage du DataFrame avec modalités, fréquences et taux de risque
    print(freq_df)
    for i, (interval, risk) in enumerate(risk_rates.items()):
        freq = df[df[cat_var] == interval].shape[0] / df.shape[0]
        group.append(interval)
        cumulative_weight += freq
        
        # Regrouper les classes pour que chaque groupe contienne au moins 5% de la population
        
        if cumulative_weight >= 0.05:
            grouped_classes.append(group)  # Ajouter le groupe aux groupes finals
            group = []  # Réinitialiser le groupe temporaire
            cumulative_weight = 0  # Réinitialiser le poids cumulatif
    # Gestion du dernier groupe (si existant)
    if group:
        last_group_weight = sum(df[df[cat_var] == g].shape[0] / df.shape[0] for g in group)
        if last_group_weight < 0.05 and grouped_classes:
            # Ajouter le dernier groupe au groupe précédent pour respecter la contrainte
            grouped_classes[-1].extend(group)
        else:
            # Ajouter le dernier groupe si la contrainte est respectée
            grouped_classes.append(group)
    return grouped_classes


def calculate_relative_difference(df, cat_var, cible):
    """
    Calcule l'écart relatif entre la modalité actuelle et celle précédente
    en fonction du taux de la variable cible.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        cat_var (str): Nom de la variable catégorielle.
        cible (str): Nom de la variable cible binaire (0 ou 1).
        
    Returns:
        pd.DataFrame: DataFrame avec les modalités, taux de la cible,
                      et écarts relatifs entre les modalités.
    """
    # Calculer le taux de la variable cible par modalité
    taux_cible = (
        df.groupby(cat_var)[cible]
        .mean()
        .reset_index()
        .rename(columns={cible: "taux_cible"})
    )
    
    
    # Trier les modalités par taux cible croissant
    taux_cible = taux_cible.sort_values("taux_cible").reset_index(drop=True)
    
    # Calculer l'écart relatif entre la modalité actuelle et la précédente
    taux_cible["ecart_relatif"] = taux_cible["taux_cible"].pct_change().fillna(0) *100
    
    return taux_cible


def group_by_rsq_relative_freq(df, cat_var, cible, seuil=0.3):
    """
    Groupe les modalités d'une variable catégorielle en fonction des écarts relatifs entre fréquences,
    calculés par rapport à la modalité précédente, et affiche les écarts relatifs.

    Arguments :
    - df : DataFrame contenant les données.
    - cat_var : str, nom de la colonne catégorielle.
    - cible : str, nom de la variable cible (ex. : taux de risque).
    - seuil : float, seuil d'écart relatif pour regrouper les modalités.

    Retourne :
    - grouped_classes : liste de groupes de modalités regroupées.
    """
    grouped_classes = []
    group = []
    risk_rates = (df.groupby(cat_var)[cible].mean()).sort_values(ascending=False)
    freq_df = df.groupby(cat_var).size() / len(df)  # Fréquence des modalités
    freq_df = pd.DataFrame({'Frequence': freq_df, 'Taux de risque': risk_rates})
    freq_df = freq_df.sort_values(by='Taux de risque', ascending=False)

    # Affichage du DataFrame avec modalités, fréquences et taux de risque
    print("\nTableau des fréquences et taux de risque :")
    print(freq_df)

    previous_freq = None  # Initialiser la fréquence précédente

    for interval, freq in freq_df['Taux de risque'].items():
        if previous_freq is None:
            # Première modalité : Démarrer un nouveau groupe
            group.append(interval)
        else:
            # Calculer l'écart relatif par rapport à la modalité précédente
            ecart_relatif = abs(freq - previous_freq) / previous_freq

            # Afficher l'écart relatif pour information
            print(f"\nÉcart relatif entre {interval} ({freq:.2%}) et la modalité précédente ({previous_freq:.2%}): {ecart_relatif:.2%}")

            if ecart_relatif <= seuil:
                # Ajouter la modalité au groupe si l'écart relatif est inférieur ou égal au seuil
                group.append(interval)
            else:
                # Terminer le groupe actuel et en commencer un nouveau
                grouped_classes.append(group)
                group = [interval]

        # Mettre à jour la modalité précédente
        previous_freq = freq

    # Gestion du dernier groupe (si existant)
    if group:
        freq = freq_df['Taux de risque'].iloc[-1]
        ecart_relatif = abs(freq - previous_freq) / previous_freq
        if ecart_relatif < 0.03 and grouped_classes:
            # Ajouter le dernier groupe au groupe précédent pour respecter la contrainte
            grouped_classes[-1].extend(group)
        else:
            # Ajouter le dernier groupe si la contrainte est respectée
            grouped_classes.append(group)

    return grouped_classes



import pandas as pd
def discretize_by_groups(df, cat_var, grouped_modalities,date,cible,id_client):
    """
    Discrétise une variable catégorielle selon les modalités regroupées.
    Le nom des groupes sera une concaténation des modalités regroupées.
    """
    temp_df = df[[date,cible,id_client,cat_var]].copy()
    # Créer un dictionnaire de mapping des groupes
    group_mapping = {}
    for group in grouped_modalities:
        group_name = f"[{','.join(map(str, group))}]"   # Concaténer les modalités pour former le nom du groupe
        for modality in group:
            group_mapping[modality] = group_name
    
    # Appliquer la discrétisation en fonction du dictionnaire
    temp_df[cat_var + "_dis"] = df[cat_var].map(group_mapping)
    temp_df[cat_var + "_dis"]  = temp_df[cat_var + "_dis"] .astype('category')
    return temp_df



#Pour discrétiser une variable continue Weighted of Evidence
def iv_woe(data,target,bins=5,show_woe=False,epsilon=1e-16):
    newDF,woeDF = pd.DataFrame(),pd.DataFrame()
    cols=data.columns

    #Run WOE and IV on all independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars],bins,duplicates="drop")
            d0=pd.DataFrame({'x':binned_x,'y':data[target]})
        else:
            d0=pd.DataFrame({'x':data[ivars],'y':data[target]})

        #calculate the nb of events in each group (bin)
        d=d0.groupby("x",as_index=False).agg({"y":["count", "sum"]})
        d.columns = ["Cutoff","N","Events"]

        #calculate % of events in each group
        d['% of Events']=np.maximum(d['Events'],epsilon)/(d['Events'].sum()+epsilon)

        #calculate the non events in each group
        d['Non-Events']=d['N'] - d['Events']
        #calculate % of non-events in each group
        d['% of Non-Events']=np.maximum(d['Non-Events'],epsilon)/(d['Non-Events'].sum()+epsilon)

        #calculate WOE by taking natural log of division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE']*(d['% of Events'] - d['% of Non-Events'])
        
        d.insert(loc=0,column="Variable",value=ivars)
        print("="*30+"\n")
        print("Information value of variable " + ivars + " is " + str(round(d["IV"].sum(),6)))
        temp=pd.DataFrame({"Variable":[ivars],"IV":[d["IV"].sum()]},columns=["Variable","IV"])
        newDF=pd.concat([newDF,temp],axis=0)
        woeDF=pd.concat([woeDF,d],axis=0)

        #show woe table
        if show_woe==True:
            print(d)
    return newDF,woeDF


from pprint import pprint

def discretize_with_iv_woe(X_train, cible,date, col, id_client,bins=5, epsilon=1e-16):
    discretized_data = X_train[[date,cible,id_client]].copy()

    # Appliquer la fonction iv_woe pour obtenir les points de coupure
    result = iv_woe(X_train[[col] + [cible]], cible, bins=bins, show_woe=False, epsilon=epsilon)

    pprint(result)

    cutoffs = result[1]["Cutoff"].unique()
    cutoffs = cutoffs
    # Si les cutoffs sont des intervalles, extraire les bornes
    if isinstance(cutoffs[0], pd.Interval):
        bins_edges = sorted(set([interval.left for interval in cutoffs] + [interval.right for interval in cutoffs]))
        bins_edges[0] = -np.inf  # Première borne : -inf
        bins_edges[-1] = np.inf 
    else:
        # Sinon, traiter les cutoffs comme des valeurs discrètes (par exemple pour des variables catégoriques)
        bins_edges = sorted(cutoffs)
        bins_edges = [-np.inf] + bins_edges + [np.inf]  # Ajouter -inf et +inf
    
    # Discrétiser la colonne en utilisant les bornes et ajouter la colonne discrétisée avec suffixe "_cut"
    discretized_data[col + "_dis"] = pd.cut(X_train[col].copy(), bins=bins_edges, include_lowest=True, duplicates='drop')

    print(f"\n Discrétisation de la colonne {col} avec les bornes: {bins_edges}")
    return discretized_data




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def select_variables_to_drop(corr_mat, kruskal_res,threshold=0.7):
    """
    Trie les variables très corrélées entre elles
    et sélectionne celles à supprimer en fonction des résultats de Kruskal-Wallis.
    """
    kruskal_res = kruskal_res.copy()

    kruskal_res.set_index('Columns', inplace=True)
    # Suppose that spearman_res is your DataFrame containing the Spearman correlations
    # Mask to keep only the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
    high_corr = corr_mat.where(mask)

    # Filter for correlations > threshold
    high_corr_pairs = high_corr[abs(high_corr) >threshold].stack().index

    #set() only to have unique values
    numerical_to_drop = set()
    for var1, var2 in high_corr_pairs:
        if var1==var2:
            continue
        if abs(kruskal_res.loc[var1, 'Stat']) > abs(kruskal_res.loc[var2, 'Stat']):
            numerical_to_drop.add(var2)
        else:
            numerical_to_drop.add(var1)
    

    kruskal_res.reset_index('Columns', inplace=True)
    return numerical_to_drop


def discretize_with_intervals(data, intervals_by_variable, date, cible):
    """
    Discrétise plusieurs colonnes d'un DataFrame en fonction des intervalles spécifiés dans le dictionnaire
    et retourne la liste des nouvelles variables créées.
    """
    df = data[[date, cible]].copy()
    new_variables = []  
    
    for entry in intervals_by_variable:
        variable = entry['variable']
        intervals = entry['intervals']
        labels = [
            f"[{intervals[i][0]}-{intervals[i][1]}]" if i == 0 else f"({intervals[i][0]}-{intervals[i][1]}]"
            for i in range(len(intervals))
        ] # Créer les labels pour chaque intervalle avec la borne inférieure exclue (sauf pour le premier intervalle) et la borne supérieure incluse
        
        # Nom de la nouvelle colonne
        new_col_name = f"{variable}_Dis"
        
        # Discrétisation
        df[new_col_name] = pd.cut(
            data[variable],
            bins=[intervals[0][0]] + [i[1] for i in intervals],  # Convertir intervalles en bornes
            labels=labels,
            include_lowest=True,
            right=True
        )
        
        new_variables.append(new_col_name)
    
    return df, new_variables


def perform_anova(df, continuous_var, target_name):
    anova_result = []
    for col in continuous_var :
        df_clean = df[[col,target_name]].dropna(axis=0)
        group=[group for _, group in df_clean.groupby(target_name)[col]]
        statistic, pvalue = stats.kruskal(*group)
        anova_result.append([col,statistic,pvalue])
    result_df = pd.DataFrame(anova_result,columns=["Columns","Stat","Pvalue"])
    return result_df

### test de Kruskall Wallis
def perform_kruskal_wallis(df, continuous_var,target_name):
    kruskal_result = []
    for col in continuous_var :
        df_clean = df[[col,target_name]].dropna(axis=0)
        group=[group for _, group in df_clean.groupby(target_name)[col]]
        statistic, pvalue = stats.kruskal(*group)
        kruskal_result.append([col,statistic,pvalue])
    result_df = pd.DataFrame(kruskal_result,columns=["Columns","Stat","Pvalue"])
    return result_df


import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm


def transf_logit_reg(cat_vars, cible, X_train_reg, X_test_reg=None):
    new_X_train = X_train_reg[cat_vars].copy()
    new_X_test = X_test_reg[cat_vars].copy() if X_test_reg is not None else None

    # Gestion des variables catégorielles
    modalites_reference = []
    for var in cat_vars:
        freq_defaut = (
            X_train_reg.groupby(var, as_index=True)[cible]
            .mean()
            .reset_index()
            .sort_values(by=cible, ascending=True)
            .reset_index(drop=True)
        )
        modalites_reference.append(var + "_" + str(freq_defaut[var].iloc[0]))

    X_train_encoded = pd.get_dummies(new_X_train, columns=cat_vars).copy()
    if new_X_test is not None:
        X_test_encoded = pd.get_dummies(new_X_test, columns=cat_vars).copy()

    # Supprimer les modalités de référence
    columns_to_drop = [col for col in modalites_reference if col in X_train_encoded.columns]
    X_train_encoded = X_train_encoded.drop(columns_to_drop, axis=1).copy()
    if new_X_test is not None:
        X_test_encoded = X_test_encoded.drop(columns_to_drop, axis=1).copy()

    # Ajouter une constante pour la régression
    X_train = sm.add_constant(X_train_encoded)
    if new_X_test is not None:
        X_test = sm.add_constant(X_test_encoded)

    # S'assurer que toutes les colonnes sont numériques
    X_train = X_train.astype(float)
    if new_X_test is not None:
        X_test = X_test.astype(float)
    else:
        X_test = None
    
    return X_train, X_test,modalites_reference


def logit_reg(cat_vars, cible, y_train, X_train_reg, y_test=None, X_test_reg=None):
    # Préparer les données d'entraînement
    X_train, X_test, modalites_reference = transf_logit_reg(cat_vars, cible, X_train_reg, X_test_reg)

    # Ajuster le modèle de régression logistique pour les données d'entraînement
    model_train = sm.Logit(y_train, X_train)
    result_train = model_train.fit(disp=False)

    # Prédictions et métriques pour les données d'entraînement
    y_pred_train = result_train.predict(X_train)
    auc_roc_train = roc_auc_score(y_train, y_pred_train)
    gini_index_train = 2 * auc_roc_train - 1
    precision, recall, _ = precision_recall_curve(y_train, y_pred_train)
    auc_pr_train = auc(recall, precision)

    if X_test is not None:
        # Ajuster le modèle pour les données de test (si disponibles)
        model_test = sm.Logit(y_test, X_test)
        result_test = model_test.fit(disp=False)

        # Prédictions et métriques pour les données de test
        y_pred_test = result_test.predict(X_test)
        auc_roc_test = roc_auc_score(y_test, y_pred_test)
        gini_index_test = 2 * auc_roc_test - 1
        precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
        auc_pr_test = auc(recall, precision)
    else:
        auc_roc_test = gini_index_test = auc_pr_test = None

    # Calcul des VIF pour les variables explicatives
    vif = pd.DataFrame()
    vif["Variable"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    # Vérification des p-valeurs et des VIF
    pvaleurs_coeffs = result_train.pvalues
    pvaleur_model = result_train.llr_pvalue

    odds_ratios = pd.DataFrame(
        {"OR": result_train.params, 
         "IC_inf": result_train.conf_int()[0],
         "IC_sup": result_train.conf_int()[1]}
    )
    odds_ratios = np.exp(odds_ratios)

    flag_significativite = 0
    if all(pvaleurs_coeffs <= 0.05) and pvaleur_model <= 0.05:
        flag_significativite = 1

    flag_VIF = 0
    if all(vif["VIF"].iloc[1:] < 10):
        flag_VIF = 1

    flag_OR = 0
    if all(odds_ratios["OR"].iloc[1:] > 0) and (
        all(odds_ratios["IC_inf"].iloc[1:] > 1) or all(odds_ratios["IC_sup"].iloc[1:] < 1)
    ):
        flag_OR = 1

    return gini_index_train, auc_pr_train, gini_index_test, auc_pr_test, flag_significativite, flag_VIF, flag_OR, result_train, modalites_reference


import itertools

def combinaisons(list_variables,nb_var):
    """
    Retourne toutes les combinaisons possibles de nb_var variables parmi la liste list_variables.
    """
    combinaisons = itertools.combinations(list_variables, nb_var)
    liste_combinaisons = [c for c in list(combinaisons)]

    return liste_combinaisons

def calculate_individual_scores(df_sc, df_individuals):
    """
    Calcule les scores des modalités associées à chaque individu et ajoute 
    la somme des scores dans le DataFrame des individus.

    Args:
        df_sc (pd.DataFrame): Une DataFrame contenant les scores par modalité (`SC(j, i)`).
            Elle doit inclure les colonnes `Modalities_merge` (concaténation variable_modalité) et `SC(j, i)`.
        df_individuals (pd.DataFrame): Une DataFrame contenant les données des individus,
            où chaque colonne est une variable et chaque valeur est une modalité.

    Returns:
        pd.DataFrame: Le DataFrame des individus avec une colonne supplémentaire `Total_Score`.
    """
    df = df_individuals.copy()
    # Créer un dictionnaire pour rechercher rapidement les scores des modalités
    modality_scores = df_sc.set_index("Modalities_merge")["SC(j, i)"].to_dict()

    # Calculer les scores totaux pour chaque individu
    total_scores = []
    for idx, row in tqdm(df_individuals.iterrows()):
        total_score = 0
        for variable, modality in row.items():
            # Construire la clé variable_modalité
            modality_merge = f"{variable}_{modality}"
            # Ajouter le score de la modalité correspondante si elle existe dans df_sc
            total_score += modality_scores.get(modality_merge, 0)
        total_scores.append(total_score)
    
    # Ajouter les scores totaux au DataFrame des individus
    df["score"] = total_scores
    return df

# Initialisation de SC

def compute_score(vars_selected,coefficients,X_train_reg,cible):
    SC = {}
    # Calcul des coefficients pour chaque variable
    for var in vars_selected:
        var_coeffs = coefficients[coefficients.index.str.startswith(var)]
        min_coef = var_coeffs.min()
        var_coeffs = var_coeffs - min_coef
        alpha_j = var_coeffs.max()

        # Compute the denominator Σ_j max(c(j, i))
        denominator = sum(coefficients[coefficients.index.str.startswith(var)].max() for var in vars_selected)
        score_max = coefficients[coefficients.index.str.startswith(var)].max()

        # Calculate SC and CTR
        CTR = score_max / 10  # Calcul de CTR
        
        # Afficher les proportions pour toutes les modalités
        prop_count = X_train_reg[var].value_counts(normalize=True)
        tx_defaut = X_train_reg.groupby(var)[cible].mean()
        
        # Identifier la modalité la plus risquée (avec le taux de défaut maximal)
        max_risk_modality = tx_defaut.idxmax()  # Modalité avec le taux de défaut maximal
        max_risk_value = tx_defaut.max()       # Taux de défaut de la modalité la plus risquée

        for modality, coef in zip(var_coeffs.index.str[len(var) + 1:], var_coeffs.values):
            relative_gap = abs(tx_defaut.get(modality, 0) - max_risk_value) / max_risk_value if max_risk_value != 0 else 0

            SC[modality] = {
                "Variables" : var,
                "Modalities_merge" : var + "_" + modality,
                "coef" : coef,
                "alpha_j": alpha_j,
                "SC(j, i)": 1000 * abs(coef - alpha_j) / denominator,
                "CTR": CTR  ,# Ajout de CTR,
                "p_j" : prop_count.get(modality, 0),
                "tx_defaut" : tx_defaut.get(modality, 0),
                "relative_gap": relative_gap,
                "m": prop_count.count(),
                "n": len(vars_selected),
            }

    # Convertir les résultats en DataFrame pour une meilleure visualisation
    SC_df = pd.DataFrame.from_dict(SC, orient="index").reset_index()
    SC_df.columns = ["Variable_Modality", "Variables","Modalities_merge","coef", "alpha_j","SC(j, i)", "CTR","p_k","tx_defaut","relative_gap","m","n"]
    SC_df = SC_df[["Variables", "Variable_Modality","Modalities_merge", "coef","alpha_j", "SC(j, i)", "CTR", "p_k", "tx_defaut", "relative_gap", "m", "n"]]


    #S_j (note moyenne pondérée) pour chaque variable
    SC_df['SC_j'] = SC_df.groupby('Variables')['SC(j, i)'].transform(lambda x: (x * SC_df.loc[x.index, 'p_k']).sum())

    # Calcul de q_j pour chaque variable
    q_j = {}
    denominator=0
    for var in vars_selected:
        # Sous-ensemble pour la variable
        var_data = SC_df[SC_df['Variables'] == var]
        
        # Numérateur : \(\sqrt{\sum_{k=1}^m p_k (SC(j, k) - SC_j)^2}\)
        numerator = np.sqrt(np.sum(var_data['p_k'] * (var_data['SC(j, i)'] - var_data['SC_j'])**2))
        denominator += numerator
        q_j[var] = numerator


    # diviser le numérateur par le dénominateur
    for var in q_j.keys():
        q_j[var] = q_j[var] / denominator

    # Ajout des contributions q_j dans le DataFrame
    SC_df['q_j'] = SC_df['Variables'].map(q_j)

    SC_df.sort_values(by=["Variables", "coef"], inplace=True)
    return SC_df


import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_and_boxplot(score,cible):
    """
    Crée un graphique combinant une KDE et un boxplot pour visualiser les distributions des scores.

    Args:
        sample (pd.DataFrame): Contient les données échantillonnées avec les colonnes `score` et `label`.
        sample_full (pd.DataFrame): Contient les données complètes avec les colonnes `score` et `label`.

    Returns:
        None: Affiche les graphiques.
    """
    sample = pd.DataFrame({
        "score": score["score"],  # Copie la colonne 'score' de score
        "label": score[cible].replace({0: 'sain', 1: 'défaut'})  # Remplace les valeurs 0 et 1 dans 'cible'
    })

    # Création des sous-graphiques
    fig, ax = plt.subplot_mosaic(
        """A
        A
        A
        A
        A
        A
        B
        """
    )

    # On plot une KDE de sample selon le label, avec le kde qui s'arrête à 1000
    sns.kdeplot(
        data=sample, x='score', hue='label', common_norm=False, 
        fill=True, ax=ax['A'], cut=0,legend=True
    )
    ax['A'].set_xticks([])
    ax['A'].grid(False)
    # On plot un boxplot de sample selon le label
    sns.boxplot(
        data=sample, x='score', hue='label', legend=False, ax=ax['B'], showfliers=False
    )
    # Labels et style final
    ax['B'].set(xlabel='Points/1000')
    sns.set_style('white')
    plt.show()

def plot_bar_stacked(score,cible):
    # Pour chaque note /1000, on calcule la part en pourcentage de sain et la part de défaut
    sample = pd.DataFrame({
    "score": score["score"],  # Copie la colonne 'score' de score
    "label": score[cible].replace({0: 'sain', 1: 'défaut'})  # Remplace les valeurs 0 et 1 dans 'cible'
    })

    # On arrondi le score au 5 le plus proche
    sample['score_round'] = np.round(sample['score'] / 5) * 5

    # Pour chaque note de score_round, on regarde la part de sain et la part de défaut
    # et vide initialisé
    part_df = pd.DataFrame(columns=['score_round', 'part_sain', 'part_dfo'])

    # Boucle sur les scores
    for score in sample['score_round'].unique():
        sub_df = sample[sample['score_round'] == score]
        # On calcule la part de sain
        part_sain = sub_df[sub_df['label'] == 'sain'].shape[0] / sub_df.shape[0]
        # On calcule la part de défaut
        part_dfo = sub_df[sub_df['label'] == 'défaut'].shape[0] / sub_df.shape[0]
        
        # On met dans le part_df
        part_df = pd.concat([part_df, pd.DataFrame({'score_round': [score], 
                                                    'part_sain': [part_sain], 
                                                    'part_dfo': [part_dfo]})])

    # On ordonne part_df
    part_df = part_df.sort_values(by='score_round')


    # On plot le graphique de part df en bar empilés
    fig, ax = plt.subplots()
    part_df[['part_sain', 'part_dfo']].plot(ax=ax, kind='bar', stacked=True, width=0.8)

    # On écrit les yticks en pourcentage
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])

    # On écrit les xticks en diagonale
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # On réécrit les xticks car ils sont illisibles
    ax.set_xticklabels([str(int(x)) for x in part_df['score_round']])

    # On affiche uniquement 1 tick sur 5
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))

    ax.set(xlabel='Points/1000', ylabel='Part')
    ax.legend(loc='upper left')
    plt.show()

