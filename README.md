# PSTALN - Catégorisation automatique d’articles médicaux sur la COVID-19
Notebook et scripts utilisé pour la prédiction automatique de catégories

2 sources de données :
* [litcovid](https://pageperso.lis-lab.fr/benoit.favre/covid19-data/20201206/litcovid.json): prédiction de catégories (`topics`)
* [bibliovid](https://pageperso.lis-lab.fr/benoit.favre/covid19-data/20201206/): prédiction de catégories (`categories`) et de spécialités (`specialties`)

### Modèles utilisés
* LR, RF, KNN (`sklearn`)
* RNNs 
* BioBert

### Automatisation
Entrainement du modèle:
```
python litcovid.py --mode train --data_path litcovid.json 
```
Prédiction sur de nouvelles données:
```
python litcovid.py --mode predict --model_path biobert_model --data_path litcovid.json 
```

Paramètres du script:
```
python litcovid.py --help

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,test,predict}
                        How to use script.
  --model_path MODEL_PATH, -m MODEL_PATH
                        Path to load from / save to the model.
  --data_path DATA_PATH, -d DATA_PATH
                        Path to load training/predict data from.
  --predict_path PREDICT_PATH, -p PREDICT_PATH
                        Path to save predict data to.
  --feature {title,abstract,both}, -f {title,abstract,both}
                        What to use as prediction feature.
  --maxlen MAXLEN       Max length of features used - tokenizer parameter.
  --batch_size BATCH_SIZE
                        Batch size for model training.
  --epochs EPOCHS       Number of epochs for model training.
  --hidden_size HIDDEN_SIZE
                        GRU hidden size
  --embed_size EMBED_SIZE
                        Embedding size
  --lr_bio LR_BIO       Learning rate
  --lr_deci LR_DECI     Learning rate
  --freezebio           Freeze when training
  --pretrained_bert PRETRAINED_BERT
                        Pretrained model to load weights from for Bert.
  --decision_threshold DECISION_THRESHOLD, -th DECISION_THRESHOLD
                        Decision threshold for prediction.
```
