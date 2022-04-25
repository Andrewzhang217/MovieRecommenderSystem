Summary
=======
This dataset contains raw movie data for generating the Tag Genome dataset and results of the experiments conducted in [Kotkov et al., 2021]. The Tag Genome dataset contains movie-tag pair scores which indicate the degrees to which tags apply to movies. This dataset was introduced in [Vig et al., 2012]. To generate the dataset, the authors collected the following information about movies: metadata (title, directors, actors...), movie reviews from IMDB (https://imdb.com/), ratings, tags that users attached to movies in MovieLens (https://movielens.org/) and user judgements regarding the degrees to which tags apply to movies. The authors collected user judgements with a survey in MovieLens and used these data to train and evaluate their algorithm. The algorithm is based on regression and predicts the movie-tag scores. The authors of [Kotkov et al., 2021] prepared the raw movie data for publication, refactored the programming code of the algorithm and introduced TagDL, a novel algorithm for prediction of movie-tag scores. TagDL is based on neural networks and uses the same features as the regression algorithm of [Vig et al., 2012]. The code is available in the GitHub repository via the following link: https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem

Please note that this dataset is slightly different from those used in [Vig et al., 2012] and [Kotkov et al., 2021]. However, experimental results based in this dataset very closely match the results reported in the publications.

The structure of this dataset partly follows the structure of the data folder in the GitHub repository above. This dataset contains raw input data for the algorithms, features generated based on these data, evaluation results of the prediction algorithms and Tag Genome relevance scores. The structure of the dataset is as follows:

├─── raw – raw movie data
│   ├─metadata.json – movie data, such as cast, directors and average rating
│   ├─ratings.json – ratings user gave to movies in MovieLens
│   ├─reviews.json – movie reviews collected from IMDB
│   ├─survey_answers.json – user answers to the survey questions regarding the degree, to which a tag applies to a movie
│   ├─tag_count.json – numbers of times users attached tags to movies
│   └─tags.json – tags and tag ids
│
├─── scores – Tag Genome scores that indicate degree, with which tags apply to movies
│   ├─glmer.csv – scores generated based on the method from the regression algorithm of [Vig et al., 2012]
│   └─tagdl.csv - scores generated based on TagDL [Kotkov et al., 2021]
│
├─── processed
│   ├─features_r.csv - user survey answers along with movie-tag features
│   └─10folds – features_r.csv split into 10 folds
│     ├─test0.csv – test for fold 0
│     ├─...
│     ├─test9.csv – test for fold 9
│     ├─train0.csv – train for fold 0
│     ├─...
│     └─train9.csv – train for fold 9
│
└─── predictions – results of the regression [Vig et al., 2012] and the TagDL [Kotkov et al., 2021] prediction algorithms
    ├─performance_results_tenfolds.txt – mean absolute error summary
    ├─tagdl_predictions_fold_0.txt – predictions of TagDL for fold 0
    ├─...
    ├─tagdl_predictions_fold_9.txt – predictions of TagDL for fold 9
    ├─glmer_predictions_fold_0.txt – predictions of regression for fold 0
    ├─...
    └─ glmer_predictions_fold_9.txt – predictions of regression for fold 9



Usage License
=============

This work is licensed under the Creative Commons Attribution-NonCommercial 3.0 License.

Citation
========

To acknowledge use of the dataset in publications, please cite both of the following papers:

[Kotkov et al., 2021] Kotkov, D., Maslov, A., and Neovius, M. (2021). Revisiting the tag relevance prediction problem. In Proceedings of the 44th International ACM SIGIR conference on Research and Development in Information Retrieval. https://doi.org/10.1145/3404835.3463019

[Vig et al., 2012] Vig, J., Sen, S., and Riedl, J. (2012). The tag genome: Encoding community knowledge to support novel interaction. ACM Trans. Interact. Intell. Syst., 2(3):13:1–13:44. https://doi.org/10.1145/2362394.2362395

Acknowledgements
========================
We would like to thank GroupLens for providing us with the dataset and code for the regression algorithm [Vig et al., 2012]. We would also like to thank organizations that supported publication of this dataset: the Academy of Finland, grant #309495 (the LibDat project) and the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence FCAI.

Files
========================
The dataset files contain json objects (one line per object). The files are encoded as UTF-8. User ids (user_id), tag ids (tag_id) and movie ids (item_id) are consistent across all files of the dataset.

Folder `raw`
--------
The folder contains raw movie data.


raw/metadata.json
--------
The file contains information about movies from MovieLens - 84,661 lines of json objects that have the following fields:
title – movie title (84,484 unique titles)
directedBy – directors separated by comma (‘,’)
starring – actors separated by comma (‘,’)
dateAdded – date, when the movie was added to MovieLens
avgRating – average rating of a movie on MovieLens
imdbId – movie id on the IMDB website (84,661 unique ids)
item_id – movie id, which is consistent across files (84,661 unique ids)
Example line:
{"title": "Toy Story (1995)", "directedBy": "John Lasseter", "starring": "Tim Allen, Tom Hanks, Don Rickles, Jim Varney, John Ratzenberger, Wallace Shawn, Laurie Metcalf, John Morris, R. Lee Ermey, Annie Potts", "dateAdded": null, "avgRating": 3.89146, "imdbId": "0114709", "item_id": 1}

raw/reviews.json
--------
The file contains 2,624,608 lines of movie reviews collected from the IMDB website. The json objects have the following fields:
item_id – movie id (52,081 unique ids)
txt – review text
Example line:
{"item_id": 172063, "txt": "one-shot record of a belly dancer; \"Carmencita Dancing,\" one of a series of Edison short films featuring \r circus and vaudeville acts, displayed the... um... \"talents\" of a zaftig belly-dancer who agreed to undulate before the camera of the famous \"Black \r Maria\" studio. \r \r The dance was originally intended to be played in a Kinetoscope, a single -person arcade viewer connected to Edison's more famous invention, the phonograph. Through a pair of crude headphones, the latter device supplied an asynchronous soundtrack of \"hootchie-cootchie\" music. \r \r The Kinetograph camera here employed is so new -- even to its inventors \r -- that director Dickson has drastically \"overcranked\" the film, unintentionally producing one of the first examples of slow-motion.\r \r Carmencita's titillating movements were considered by many to be \r scandalous. Thus, the film prompted some of the earliest discussions of film censorship."}

raw/tags.json
--------
The file contains 1,094 lines of json objects with the following fields:
tag – tag string
id – tag id (1,094 unique)
Example line:
{"tag": "whitewash", "id": 1}

raw/tag_count.json
--------
The file contains numbers of times MovieLens users attached tags to movies. The file contains 212,704 lines of json objects with the following fields:
item_id – movie id (39,685 unique ids)
tag_id – tag id (1,094 unique ids)
num – number of times users have attached the tag to the movie
Example line:
{"item_id": 1, "tag_id": 2198, "num": 2}


raw/ratings.json
--------
The file contains ratings that users assigned to movies in MovieLens. Each rating represents the degree, to which the user enjoyed watching the movie. Each rating corresponds to a number of stars from 0.5 till 5 with the granularity of 0.5 star. The higher the rating, the higher the enjoyment. The file contains 28,490,116 lines of json objects with the following fields:
item_id – movie id (67,873 unique ids)
user_id – user id (247,383 unique ids)
rating – the number of stars
Example line:
{"item_id": 5, "user_id": 997206, "rating": 3.0}

raw/survey_answers.json
--------
The file contains ratings MovieLens users gave to movie-tag pairs in the survey. The users were asked to indicate the degree, to which a tag applies to a movie on a 5-point scale from the tag not applying at all (1 point) to applying very strongly (5 points). Users could also indicate that they are not sure about the degree (the -1 value). The file contains 58,903 lines of json objects with the following fields:
user_id – user id (679 unique ids)
item_id – movie id (5,546 unique ids)
tag_id – tag id (1,094 unique ids)
score – movie-tag rating, which takes values: 1 (`not at all`), 2, 3, 4, 5 (`very much`) and -1 (`not sure`)
Example line:
{"user_id": 978707, "item_id": 3108, "tag_id": 50126, "score": 3}


Folder `processed`
--------
The folder contains features generated based on the raw movie data.

processed/features_r.csv
--------
The file contains features of movie-tag pairs along with user survey answers with excluded `not sure` (-1) ratings. These features are calculated based on the raw data and used to generate movie-tag scores of Tag Genome by regression and TagDL. The file contains 51,162 lines and has the csv format with the following fields:
tag – tag string
item_id – movie id (5,192 unique ids)
log_IMDB – log of frequency with which the corresponding tag appears in text reviews of the movie
log_IMDB_nostem – the same as the previous field, but without stemming
rating_similarity - cosine similarity between ratings of the corresponding movie and aggregated ratings of movies tagged with the tag
avg_rating – average MovieLens rating of the movie
tag_exists - 1 if the tag has been applied to the movie and 0 otherwise
lsi_tags_75 - similarity between the tag and the movie using latent semantic indexing, where each document is the set of tags applied to the movie
lsi_imdb_175 - similarity between the tag and the movie using latent semantic indexing, where each document is the set of words in user reviews of the movie
tag_prob – the score predicted by a regression model using tag_exists as the output variable and the other features as the input variables
targets – user answers to survey questions
Example lines:
tag,item_id,log_IMDB,log_IMDB_nostem,rating_similarity,avg_rating,tag_exists,lsi_tags_75,lsi_imdb_175,tag_prob,targets
007,63113,1.75330436544309,2.27542404932756,1.71434828208978,-0.698401990607001,1.33220801542758,2.6985291947824,2.74139820850471,1.29043570541653,5
007,63113,1.75330436544309,2.27542404932756,1.71434828208978,-0.698401990607001,1.33220801542758,2.6985291947824,2.74139820850471,1.29043570541653,5
007,49272,1.90178729254183,2.4765957047332,0.710586261523279,0.457945907102739,1.33220801542758,2.82316470188989,2.37457040636021,1.16739829356456,5

Folder `processed/10folds`
--------
This folder contains 10-fold split of features_r.csv


Folder `scores`
--------
The folder contains Tag Genome scores that indicate degrees, with which tags apply to movies. To generate these files, we used ids of movies from the dataset released with the Tag Genome publication and tags used in the survey [Vig et al., 2012]. In case these files are missing necessary movies or tags, you can generate them with the code provided in the GitHub repository (see the link above).

scores/glmer.csv
--------
The file contains 10,551,656 Tag Genome scores between 0 and 1. The scores have been generated with the regression algorithm of [Vig et al., 2012]. The file has the csv format and contains the following fields:
tag - tag string (1,084 unique tags)
item_id - movie id (9,734 unique ids)
score - degree, with which the tag applies to the movie
Example lines:
tag,item_id,score
airplane,2,0.042390835398958
airplane,3,0.0506732208371692
airplane,4,0.0331609987530869

scores/tagdl.csv
--------
The file contains 10,551,656 Tag Genome scores between -0.07 and 1.13 (due to the prediction algorithm design). The scores have been generated with TagDL [Kotkov et al., 2021]. The file has the csv format and contains the following fields:
tag - tag string (1,084 unique tags)
item_id - movie id (9,734 unique ids)
score - degree, with which the tag applies to the movie
Example lines:
tag,item_id,score
airplane,2,0.006687045
airplane,3,0.0042806566
airplane,4,-0.00016063452

Folder `predictions`
--------
The folder contains predictions of algorithms based on movie-tag features. Files from tagdl_predictions_fold_0.txt to tagdl_predictions_fold_9.txt contain predictions of TagDL, while files from glmer_predictions_fold_0.txt to glmer_predictions_fold_9.txt contain predictions of the regression model. The prediction scores in these files correspond to test files in the 10folds folder. The file performance_results_tenfolds.txt contains mean absolute errors for each fold.
