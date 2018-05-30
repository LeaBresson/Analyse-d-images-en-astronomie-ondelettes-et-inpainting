# Analyse d'images en astronomie: ondelettes et inpainting

Project of compressed sensing - ENSAE | M2 Data science (lecturer: Guillaume Lecué)

## Contexte
Le problème des données manquantes est récurrent dans le domaine de l'astronomie. En effet, les images issues des microscopes souffrent souvent de "zones d'ombres" (mauvaise qualité de l'image, mauvaise calibration, etc.). Les techniques d'inpainting consistent à "remplir" ces zones masquées.

## Formalisation du problème
Soient X l'image complète (non observée), Y l'image observée souffrant de zone "masquées", et L un masque binaire. On a $Y=LX$. L'objectif est alors de retrouver X sachant que l'on connait Y et L. Pour résoudre ce problème, nous comparons deux techniques:

- la première méthode se place dans une base d'ondelettes et effectue une approximation par seuillage itératif (méthode de J.L Starck et J. Bobin dans leur article "Astronomical Data Analysis and Sparsity: from Wavelets to Compressed Sensing"). L'approche mathématique d'un tel algorithme est développée dans notre rapport.

- la seconde technique repose sur les techniques de complétion de matrices (ici matrices de pixels) par minimisation de la norme nucléaire étudiées en cours (http://lecueguillaume.github.io/assets/10_matrice_completion.pdf). Plus précisément, on comparera la formulation SDP du problème de minimisation de la norme nucléaire et la descente de gradient proximale.

L'objet de ce notebook est donc de vérifier si l'algorithme basé sur les ondelettes est, comme le suggérerait la théorie, plus efficace pour estimer des images présentant des structures fractales.

## Implémentation
Nous allons implémenter la technique de l'inpainting à une photo prise par le télescope Herschel (https://www.herschel.caltech.edu/images). Pour ce faire, nous appliquons un masque binaire à l'image considérée.

Nous nous sommes inspirés des codes de Jerome Bobin pour la première méthode (le code du fichier pyPW2 a été crée par J.Bobin - http://jbobin.cosmostat.org/master-2-mva).

## Auteurs
Léa Bresson (lea.bresson@polytechnique.edu), Arnaud Valladier (arnaud.valladier@polytechnique.edu)
