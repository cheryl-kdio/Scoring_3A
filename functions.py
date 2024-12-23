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

def plot_cat_vars_distributions(data, vars_list, cols=2):
    """
    Génère des graphiques montrant la distribution des modalités pour chaque variable dans une grille.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile
    
    for i, var in enumerate(vars_list):
        value_counts = data[var].value_counts(normalize=True) 
        index_values = value_counts.index.to_flat_index()  
        index_values = [str(x) for x in index_values]   
        
        # Création du graphique
        axes[i].bar(index_values, value_counts.values, color='skyblue')
        axes[i].set_ylabel('Proportion')
        axes[i].set_title(f'{var}')
        axes[i].tick_params(axis='x', rotation=45)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Distribution des modalités", fontsize=16, x=0.0, y=1.02, ha='left') 
    
    plt.tight_layout()
    plt.show()


def tx_rsq_par_var(df, categ_vars, date, target, cols=2):
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

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=False, sharey=False)
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, categ_var in enumerate(categ_vars):
        # Calcul des moyennes par date et catégorie
        df_times_series = (df.groupby([date, categ_var])[target].mean() * 100).reset_index()
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

    fig.suptitle("Taux de défaut par variable catégorielle",fontsize=16, x=0.0, y=1.02, ha='left')
    plt.tight_layout()
    plt.show()


def combined_barplot_lineplot(df, cat_vars, cible, cols=2):
    """
    Génère une grille de barplots combinés avec des lineplots pour une liste de variables catégorielles.
    """
    num_vars = len(cat_vars)
    rows = math.ceil(num_vars / cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, cat_col in enumerate(cat_vars):
        ax1 = axes[i]  # Axe pour le barplot

        # Vérifier si la variable est catégorielle et la convertir en chaîne si nécessaire
        if pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype(str)

        # Calcul du taux de risque
        tx_rsq = (df.groupby([cat_col])[cible].mean() * 100).reset_index()

        # Calcul des effectifs
        effectifs = df[cat_col].value_counts().reset_index()
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
        ax1.set_ylabel("Effectifs")
        ax2.set_ylabel("Taux de risque (%)")
        ax1.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Titre général
    fig.suptitle("Barplots et Lineplots combinés pour les variables catégorielles",fontsize=16, x=0.0, y=1.02, ha='left')
    plt.tight_layout()
    plt.show()

def compare_distributions_grid(X_train, X_test, var_list, cols=2):
    """
    Compare les distributions des variables continues dans Train et Test et les affiche sous forme de grille.
    """
    num_vars = len(var_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(var_list):
        # Graphique pour chaque variable
        sns.kdeplot(X_train[var], label='Train', shade=True, ax=axes[i])
        sns.kdeplot(X_test[var], label='Test', shade=True, ax=axes[i])
        axes[i].set_title(f"{var}")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Comparaison des distributions dans l'echantillon Train et Test",fontsize=16, x=0.0, y=1.02, ha='left') 

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
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=True, sharey=True)
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
    fig.suptitle("Evolution des modalités dans le temps",fontsize=16, x=0.0, y=1.02, ha='left') 

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

    fig.suptitle("Distribution des variables continues",fontsize=16, x=0.0, y=1.02, ha='left')  # Positionner le titre légèrement au-dessus

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
        axes[i].set_title(f"KDE Plot de {var} par {target}")
        axes[i].set_xlabel(target)
        axes[i].set_ylabel(var)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"KDE des variables continues selon la variable cible",fontsize=16, x=0.0, y=1.02, ha='left')
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
    temp_df = pd.DataFrame()

    temp_df[cible] = df[cible].values
    temp_df[date] = df[date].values
    temp_df[id_client] =df[id_client].values
    temp_df[cat_var] = df[cat_var].values

    # Créer un dictionnaire de mapping des groupes
    group_mapping = {}
    for group in grouped_modalities:
        group_name = f"[{','.join(map(str, group))}]"   # Concaténer les modalités pour former le nom du groupe
        for modality in group:
            group_mapping[modality] = group_name
    
    # Appliquer la discrétisation en fonction du dictionnaire
    temp_df[cat_var + "_dis"] = df[cat_var].map(group_mapping)
    
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
        print("--------------------------------------------\n")
        print("Information value of variable " + ivars + " is " + str(round(d["IV"].sum(),6)))
        temp=pd.DataFrame({"Variable":[ivars],"IV":[d["IV"].sum()]},columns=["Variable","IV"])
        newDF=pd.concat([newDF,temp],axis=0)
        woeDF=pd.concat([woeDF,d],axis=0)

        #show woe table
        if show_woe==True:
            print(d)
    return newDF,woeDF

def discretize_with_iv_woe(X_train, cible,date, numerical_columns, id_client,bins=5, epsilon=1e-16):
    discretized_data = X_train[[date,cible,id_client]].copy()
    discretized_columns = []
    non_discretized_columns = []
    cutoffs_dict = {}
    for col in numerical_columns:
        # Appliquer la fonction iv_woe pour obtenir les points de coupure
        result = iv_woe(X_train[[col] + [cible]], cible, bins=bins, show_woe=False, epsilon=epsilon)

        # Extract the cutoffs to a dict
        cutoffs_dict[col] = result[1]["Cutoff"].unique()

        if result[1]["IV"].sum() != 0:  # Si l'IV n'est pas nul, discrétiser
            # Extraire les cutoffs (intervalles)
            cutoffs = result[1]["Cutoff"].unique()
            cutoffs = cutoffs
            # Si les cutoffs sont des intervalles, extraire les bornes
            if isinstance(cutoffs[0], pd.Interval):
                bins_edges = sorted(set([interval.left for interval in cutoffs] + [interval.right for interval in cutoffs]))
            else:
                # Sinon, traiter les cutoffs comme des valeurs discrètes (par exemple pour des variables catégoriques)
                bins_edges = sorted(cutoffs)
            
            # Discrétiser la colonne en utilisant les bornes et ajouter la colonne discrétisée avec suffixe "_cut"
            discretized_data[col + "_dis"] = pd.cut(X_train[col].copy(), bins=bins_edges, include_lowest=True, duplicates='drop')
            discretized_columns.append(col + "_dis")

            print(f"Discrétisation de la colonne {col} avec les bornes: {bins_edges}")
        else:
            discretized_data[col ] = X_train[col].copy()
            non_discretized_columns.append(col)

    return discretized_data, discretized_columns, non_discretized_columns,cutoffs_dict


############# Discrétisation avec la méthode ChiMerge

class Discretization:
    ''' A process that transforms quantitative data into qualitative data '''
    
    def __init__(cls):
        print('Data discretization process started')
        
    def get_new_intervals(cls, intervals, chi, min_chi):
        ''' To merge the interval based on minimum chi square value '''
        
        min_chi_index = np.where(chi == min_chi)[0][0]
        new_intervals = []
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done:
                t = intervals[i] + intervals[i+1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        return new_intervals        
        
    def get_chimerge_intervals(cls, data, colName, label, max_intervals):
        '''
            1. Compute the χ 2 value for each pair of adjacent intervals
            2. Merge the pair of adjacent intervals with the lowest χ 2 value
            3. Repeat œ and  until χ 2 values of all adjacent pairs exceeds a threshold
        '''
        
        # Getting unique values of input column
        distinct_vals = np.unique(data[colName])
        
        # Getting unique values of output column
        labels = np.unique(data[label])
        
        # Initially set the value to zero for all unique output column values
        empty_count = {l: 0 for l in labels}
        intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
        while len(intervals) > max_intervals:
            chi = []
            for i in range(len(intervals)-1):
                
                # Find chi square for Interval 1
                row1 = data[data[colName].between(intervals[i][0], intervals[i][1])]
                # Find chi square for Interval 2
                row2 = data[data[colName].between(intervals[i+1][0], intervals[i+1][1])]
                total = len(row1) + len(row2)
                
                # Generate Contigency
                count_0 = np.array([v for i, v in {**empty_count, **Counter(row1[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(row2[label])}.items()])
                count_total = count_0 + count_1
                
                # Find the expected value by the following formula
                # Expected Value → ( Row Sum * Column Sum ) / Total Sum
                expected_0 = count_total*sum(count_0)/total
                expected_1 = count_total*sum(count_1)/total
                chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
                
                # Store the chi value to find minimum chi value
                chi_ = np.nan_to_num(chi_)
                chi.append(sum(chi_))
            min_chi = min(chi)
            
            intervals = cls.get_new_intervals(intervals, chi, min_chi)
        print(' Min chi square value is ' + str(min_chi))
        return intervals



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

def select_variables_and_plot_corr_matrix(pearson_res, kruskal_results, corr_threshold=0.6,plot=True,print_pairs=True):
    """
    Sélectionne des variables dont la corrélation est inférieure à un seuil donné, 
    en utilisant les résultats de Kruskal-Wallis pour déterminer la variable la plus pertinente
    parmi celles corrélées.
    
    Args:
    - pearson_res : DataFrame de la matrice de corrélation (Pearson)
    - kruskal_results : DataFrame avec les résultats de Kruskal-Wallis ('Columns' et 'Stat')
    - corr_threshold : Seuil de corrélation à respecter entre les variables
    
    Returns:
    - best_variables : Liste des variables retenues avec une corrélation inférieure au seuil
    """
        # Initialisation des paires et de la statistique de Kruskal-Wallis
    kruskal_stats = dict(zip(kruskal_results['Columns'], kruskal_results['Stat']))  # Convertir les résultats Kruskal-Wallis en dictionnaire

    # Étape 1 : Extraire les variables corrélées avec une corrélation > corr_threshold
    pearson_res_no_diag = pearson_res.where(np.triu(np.ones(pearson_res.shape), k=1).astype(bool))  # Supprimer la diagonale
    high_corr_pairs = pearson_res_no_diag.stack()  # Convertir en format colonne
    high_corr_pairs = high_corr_pairs[abs(high_corr_pairs) > corr_threshold]  # Filtrer les corrélations supérieures au seuil

    # Identifier les variables non corrélées
    all_vars = set(pearson_res.columns)  # Créer une liste de toutes les variables
    corr_vars = set(high_corr_pairs.index.get_level_values(0)).union(set(high_corr_pairs.index.get_level_values(1)))  # Variables dans les paires corrélées
    non_corr_vars = list(all_vars - corr_vars)  # Variables qui ne sont dans aucune paire corrélée
    if print_pairs :
        # Afficher les paires corrélées et leurs corrélations
        print("Paires de variables fortement corrélées (|ρ| > {:.1f}) :".format(corr_threshold))
        for pair, corr_value in high_corr_pairs.items():
            print(f"Paire {pair}: Corrélation = {corr_value:.2f}")

    # Étape 2 : Comparer les statistiques de test pour chaque paire
    best_variables = []  # Liste des variables à retenir
    for var1, var2 in high_corr_pairs.index:  # Parcourir les paires
        stat_var1 = kruskal_stats.get(var1, 0)  # Récupérer la statistique de test pour var1
        stat_var2 = kruskal_stats.get(var2, 0)  # Récupérer la statistique de test pour var2
        
        # Retenir la variable ayant la plus grande statistique
        if stat_var1 >= stat_var2:
            best_variables.append(var1)
        else:
            best_variables.append(var2)

    # Supprimer les doublons des variables retenues et ajouter les non corrélées
    best_variables = list(set(best_variables)) + non_corr_vars

    # Afficher les résultats
    print("\nVariables retenues après comparaison des statistiques de test Kruskal-Wallis :")
    print(best_variables)

    # Étape 3 : Tracer la matrice de corrélation pour les variables retenues
    if plot :
        selected_corr_matrix = pearson_res.loc[best_variables, best_variables]  # Filtrer la matrice de corrélation
        plt.figure(figsize=(12, 10))

        sns.heatmap(selected_corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)  # Heatmap
        plt.title('Matrice de Corrélation (Variables Sélectionnées)')
        plt.show()

    # Retourner les variables retenues
    return best_variables

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


import pandas as pd
from scipy.stats import f_oneway


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


# Application de modèles

import statsmodels.api as sm 
from sklearn.metrics import auc
from sklearn.metrics  import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor



def reg_logistique(df_train,var_cible,categorical_variables,numerical_variables):
    risk_drivers = categorical_variables + numerical_variables

    modalites_reference = []
    for var in categorical_variables :
        # La modalité de référence est celle avec le taux de défaut le plus élevé
        freq_defaut = df_train.groupby([var],as_index=True)[[var_cible]].mean().reset_index().sort_values(by=[var_cible],ascending=True).reset_index(drop=True)
        modalites_reference.append(var+'_'+str(freq_defaut[var][0]))

    # Transformer la base train
    # dummies_test = pd.get_dummies(df_test,columns=categorical_variables).drop(modalites_reference,axis=1).copy()
    dummies_train = pd.get_dummies(df_train,columns=categorical_variables,dtype='int').drop(modalites_reference,axis=1).copy()

    X_train = dummies_train.drop(var_cible,axis=1).copy()
    Y_train = dummies_train[var_cible].copy()

    # X_test = dummies_test.drop(var_cible,axis=1).copy()
    # Y_test = dummies_test[var_cible].copy()

    # Rajout de la constante dans le modèle
    X_train = sm.add_constant(X_train)
    # X_test = sm.add_constant(X_test)

    # Modèle
    model = sm.Logit(Y_train,X_train).fit(disp=False)
    y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)

    # Calculer les AUC
    fpr_train,tpr_train,thresholds_train = roc_curve(Y_train,y_train_pred)
    roc_auc_train = auc(fpr_train,tpr_train)

    # fpr_test,tpr_test,thresholds_test = roc_curve(Y_test,y_test_pred)
    #roc_auc_test = auc(fpr_test,tpr_test)

    # pvaleurs
    pvaleurs_coeffs = model.pvalues
    pvaleur_model = model.llr_pvalue

    flag_significativite = 0
    if (all(pvaleurs_coeffs<=0.05)) & (pvaleur_model<=0.05):
        flag_significativite = 1

    # VIF
    vif=pd.DataFrame()
    vif["variable"]=X_train.columns
    vif["VIF"]=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
    vif.drop(0,inplace=True)

    # flag pour vérifier si tous les VIF sont <10
    flag_VIF = 0
    if all(vif["VIF"]<10):
        flag_VIF = 1

    return risk_drivers, pvaleur_model, pvaleurs_coeffs.to_dict(),flag_significativite,vif,flag_VIF,roc_auc_train, fpr_train,tpr_train,model


