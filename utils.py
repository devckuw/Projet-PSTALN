import numpy as np
import requests
import pandas as pd 

specialties = {'Anesthésie-Réanimation': 0,
 'Cardiologie et maladies métaboliques': 10,
 'Confinement/Déconfinement': 5,
 'Dermatologie': 15,
 'Gynécologie Obstétrique': 8,
 'Gériatrie': 9,
 'Hygiène': 11,
 'Hématologie': 16,
 'Hépato-gastro-entérologie': 12,
 'Immunité': 3,
 'Infectiologie': 1,
 'Neurologie': 13,
 'Néphrologie': 18,
 'Pneumologie': 7,
 'Psychiatrie': 17,
 'Pédiatrie': 6,
 'Radiologie': 14,
 'Transversale': 2,
 'Virologie': 4}

litcovid_son = requests.get('https://pageperso.lis-lab.fr/benoit.favre/covid19-data/20201206/litcovid.json').json()
litcovid = pd.DataFrame(litcovid_son)

def get_data(data_dict, get_abstract=False):
    """
    A partir du dictionnaire des données, la fonction récupère les données que l'on veut
    get_abstract : ajouter les abstract dans les données, si False on a que les titres
    :return: texts, Y
    """
    texts, Y = [], []
    
    for article in data_dict:
        t = article['title']
        if get_abstract:
            if 'abstract' in article.keys():
                t += str(article['abstract'])
            else:  # on va chercher l'abstract dans les données litcovid
                ind = litcovid['title'] == t
                abstract = str(litcovid[ind]['abstract'])
                t += abstract
        texts.append(t)
        ind_spe = [specialties[spe['name']] for spe in article['specialties']]
        y = np.zeros(len(specialties))
        y[ind_spe] = 1
        Y.append(y)
    Y = np.array(Y)
    return texts, Y