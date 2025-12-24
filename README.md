# xAI – Deepfake Audio + X-ray XAI (Streamlit)

DATA et modeles : https://drive.google.com/drive/folders/1-gEu2oPO9F5EmUfNcMH-tjc8vzzKkHRx

Ce repo contient un notebook d’entraînement (`project/training_models.ipynb`) et deux apps Streamlit :
- `project/app.py` : app principale (PyTorch) pour **Audio (FoR)** + **X-ray (CheXpert proxy)** avec XAI (**Grad‑CAM / LIME / SHAP**).
- `project/app_deepfake.py` : code “référence” issu du repo externe (TensorFlow/Keras) pour **Audio only** avec XAI (**LIME / Grad‑CAM**).

## Structure

- `project/app.py` : UI Streamlit + chargement modèles `.pt` + XAI.
- `project/app_deepfake.py` : UI Streamlit “legacy” (Keras) + spectrogramme via PNG + LIME + Grad‑CAM TF.
- `project/training_models.ipynb` : entraînement des modèles (PyTorch) et sauvegarde des poids.
- `project/weights/` : poids des modèles PyTorch exportés (`*.pt`).

## Lancer l’app (PyTorch)

Depuis la racine du repo :
```powershell
.\.venv\Scripts\python.exe -m streamlit run project\app.py
```

Pré‑requis : les packages doivent être installés dans `.venv` (voir les cellules `pip install` du notebook) et les poids doivent exister dans `project/weights/` (les clés sont définies dans `project/app.py`).

## Entraîner / générer les poids

Ouvrir et exécuter `project/training_models.ipynb`.
Les modèles sont sauvegardés dans `project/weights/` et ensuite chargés par `project/app.py`.

## Différences clés : `app.py` vs `app_deepfake.py`

### Framework & modèles
- `project/app.py` (PyTorch) :
  - charge des modèles torchvision (VGG16/ResNet50/MobileNetV2 pour audio, AlexNet/DenseNet121 pour X‑ray),
  - restaure les poids depuis `project/weights/*.pt`,
  - prédiction binaire via **logit** + `sigmoid`.
- `project/app_deepfake.py` (TensorFlow/Keras) :
  - charge un modèle Keras (chemin `saved_model/model`),
  - prédiction via `model.predict` (softmax multi‑classes dans leur code),
  - application centrée sur l’audio uniquement.

### Données supportées
- `project/app.py` : **2 modalités**
  - Audio `.wav` → image (mel‑spectrogram) → modèle image.
  - X‑ray (`png/jpg/jpeg`) → modèle image.
- `project/app_deepfake.py` : **audio uniquement**

### Pré‑traitement (important pour la “forme” des XAI)
- `project/app.py` :
  - Audio : génère le spectrogramme comme dans le repo de référence (**matplotlib → PNG → reload**) pour obtenir un rendu visuel plus proche.
  - X‑ray : conversion en grayscale puis RGB + normalisation ImageNet (comme dans `project/training_models.ipynb`).
- `project/app_deepfake.py` :
  - Audio : écrit un PNG `melspectrogram.png`, puis recharge l’image (pipeline disque).

### XAI disponibles
- `project/app.py` :
  - **Grad‑CAM** (pytorch‑grad‑cam) : couche cible adaptée pour X‑ray (ReLU final) et mode debug optionnel.
  - **LIME** (lime‑image) : segmentation superpixels + perturbations.
  - **SHAP** (masker image + superpixels) : affichage rouge/vert (positif/négatif) style “paper”.
- `project/app_deepfake.py` :
  - **LIME** et **Grad‑CAM** (implémentation TF “maison”).
  - Pas de SHAP.

### Interprétation des couleurs (SHAP rouge/vert dans `app.py`)
- **Rouge** : régions qui poussent la prédiction vers la classe expliquée (contribution positive).
- **Vert** : régions qui la tirent dans l’autre sens (contribution négative).

## Notes pratiques

- Les méthodes XAI peuvent donner des résultats différents (Grad‑CAM vs LIME vs SHAP) car elles n’expliquent pas exactement la même chose (gradients vs perturbations vs contributions moyennes).
- Pour Grad‑CAM X‑ray, une carte “vide” vient souvent d’une couche cible inadaptée (activations négatives + ReLU interne) : `project/app.py` cible donc une couche ReLU proche de la fin du backbone.

