# Arabic Propaganda Detection

Weakly supervised Arabic propaganda detection on X (Twitter formally). Propaganda is defined as the intent to deceive the audience towards a specific target following a predefined agenda utilizing means of language and vision. We focus on propaganda in the social media domain (especially X) and try to develop a model that can automatically detect this behavior. The project components are a fully supervised baseline and transformer model trained on 80% of the labeled dataset (2100 tweets in total), analysis of the users' behavior based on metadata (e.g. number of followers) and tweets (linguistics of the tweet), labeling functions which are the distant supervision signals that will be combined using a Probabilistic Graphical Model (Snorkel Label Model) and validated using 500 labeled tweets, and the weakly supervised model trained on the vast amount of the weakly labeled data labeled by the Snorkel Label Model.

## Table of Contents

- Programmatic Labeling
- Manually Labeled Dataset
- Unlabeled Dataset Analysis
- Labeling Functions
- Weakly Supervised Model
- Fully Supervised Model
- Conclusion

## Programmatic Labeling

Recent advances in Deep Learning have produced large models (large in the number of parameters) that can be trained to automatically learn the text features (characteristics) and map them to needed tasks, eliminating the need for feature engineering and researching the complete set of features that represent a piece of text to be understood by machines. While being a breakthrough, deep learning models require vast amounts of labeled data to be trained on.

Acquiring these large amounts of labeled data is costly in the following manners:

- **Data annotation (labeling)** is an exhaustive task that requires a lot of time and Subject Matter Experts to work on.
- **Subject Matter Experts (SMEs)** are rare to find, especially in complicated tasks like the one we are ahead of (propaganda detection). Even when available, their help is very expensive leading to a higher cost of the overall project.
- Most of the time, SMEs make **Human Errors** related to bias and personal perferences leading to complexity in vote aggregation and lowering the **Labeling Quality**.
- **Static Labeling** is the main drawback after all. At any time if the task is shifted or new rules need to be added while labeling, the dataset will be labeled again from scratch leading to even higher time and effort costs of an ongoing project.
- **Privacy Concerns** is yet another major issue in labeling as SMEs might not be allowed to have access to medical or classified documents. Which forces the loss of specialty in domains like health, finance, etc.

To address all these issues, we leverage **Programmatic Labeling** in which we ask SMEs to write down the rules and guides they would use to label a dataset. These rules and guides are then transformed into simple Python functions (**labeling Functions**) that can be applied to vast amounts of unlabeled data. The unlabeled data is assumed to be all-time available. In the case of classified datasets, the rules can be applied without worries about privacy concerns.

Rules and guides serving as **Labeling Functions** can be of any type. They can be regular expression patterns extracted from text, distant supervision signals from external databases like other annotated datasets (domain non-specific), or signals from third-party models like Zero-Shot models and task non-specific finetuned models. All these signals will be combined using a **Label Model** that learns the relationship between each and every labeling function and other labeling functions and the true (hidden or unknown) label. The trained Label Model will have a better understanding of the correlations and accuracies of the labeling functions in order to efficiently aggregate the labeling functions' votes. That is proved in the literature to surpass the majority vote by a large margin.

## Manually Labeled Dataset

The annotated propaganda detection dataset comes from previous research we did in this project. We have the tweets annotated as *propaganda* or *transparent*, an example will follow. The dataset consists of **2100** tweets manually labeled and it is published in its raw form as text without any processing. We intend to use 80% of it to train (training and validation) the fully supervised models with only **500** tweets stratifiedly selected from the 80% training data to develop the labeling functions and measure the performance of the **Label Model** before weakly labeling the unlabeled dataset. The left 20% of the data will serve as a gold test dataset that we will use to evaluate all the models.

The dataset is imbalanced with only **202** tweets labeled as propaganda. For that reason, we use stratified splitting throughout the selection process.

We intend to maximize the performance of the weakly supervised model to be on par with the fully supervised models with only **500** tweets (~29%) of the data compared to 80%.

```json
[
    {
        "tweet_text": "RT @AliKuemkh: النصر وجمهورة قصة فرح بصراحة يعجز اللسان عن وصف هالقصة بين عاشق ومعشوق #النصر_برازيل_اسيا https://t.co/jWKbYQVdi1",
        "tech": "name-calling - loaded language",
        "label": "propaganda"
    }
]
```

## Unlabeled Dataset Analysis

To get insights and ideas about the labeling functions that can be applied, we performed an analysis of the unlabeled dataset. The unlabeled dataset comes from two resources. The first is the X-published tweets of suspended propagandists from which we have extracted a 56K representative sample in previous research. The second is a list of legitimate users on X that we have scrapped their recent tweets. Based on the topics found in the dataset from the first resource, we collected 140K tweets to serve as tweets from genuine users. We aim to analyze the tweets from both sides to understand the propagandists' and genuine users' characteristics.

The number of unique user IDs of propagandist accounts exposed by X is **5929** while the number of genuine users we listed is **12145**. We aim to analyze the users' metadata (e.g. number of followers) for both worlds and the tweets' text.

### Meta Data

We have analyzed the users' metadata and compared the difference for each data type as follows:

- Bio length and whether an account has a bio or not.
- Number of words and emojis in the bio.
- Most used words in bio.
- The ratio of the followers count to the following count.
- Account creation year.
- Valid location reported in the account.

From the above analysis, we found that:

- Propagandist user accounts were created in `2018` and `2019`.
- Genuine user accounts were created in `2011` and `2012`.
- Genuine users use the words `الحساب` and `الرسمي` in their bios.

### Tweets

We have analyzed the linguistic features of the tweets for both worlds. We have also tested the presence of some of the propaganda techniques in both words. It is worth mentioning that the presence of a technique doesn't imply the tweet is propaganda. Propaganda is detected based on the intent of the user to deceive the audience and based on the context. We analyzed the following:

- Number of URLs, hashtags, mentions, and emojis.
- Number of and most used words with and without **Stop Words** (come from NLKT).
- Number of tweets containing questioning (serves the Doubt technique).
- Number of entities (serves both the Name Calling and the Smears techniques).
- Sarcasm and hate speech signals (serves the Loaded Language, Smears, and Glittering techniques).
- The presence of the Name Calling technique based on the combination of sarcasm and hate speech signal along with the presence of entities.

From the above analysis, we found that:

- Genuine users add `at least one URL` in their tweets.
- Propagandist users use `mentions` `at least one time` in their tweets.
- Some propagandist users tweet in less than 5 words.
- The word `حرب` is one of the `top 50` used words in the propagandists' tweets.
- The name-calling technique appears at least **1.2** times more in the propagandists' tweets.

The reported insights along with all the possible coding of the propaganda techniques will be used as labeling functions.

### Labeling Functions

The labeling functions are the distant supervision signals we rely on to label the tweets. We rely on the concept that a labeling function's signal is considered a *propaganda*/*transparent* label vote. They serve as crow-source labelers and we will aggregate their votes algorithmically via the **Label Model**. We have added labeling functions for each programmable technique of the propaganda techniques. Also, we utilized the help of a zero-shot model and a news-specific labeled dataset to serve as labeling functions. The complete list is as follows

- User Meta Data Labeling Functions:
  - Absence of a bio for an account (all tweets are propaganda).
  - Absence of a location for an account (all tweets are propaganda).
  - Account creation data in 2018-2019 (propaganda).
  - Account creation data in 2011-2012 (transparent).
  - Bio keywords, official, etc. (transparent).

- Text Labeling FUnctions
  - Text contains URL (transparent).
  - Text contains user mentions (propaganda).
  - Text contains pronouns (propaganda).
  - Text contains question keywords (doubt technique) (propaganda).
  - Labeling technique (ent + sarcasm) (propaganda).
  - Labeling technique (ent + hate) (propaganda).
  - Entities presence (labeling and smears techniques).
  - Loaded language, manual lexicon.
  - Loaded language, proppy lexicon.
  - Hate and sarcasm speech.
  - Labeled dataset distant supervision using cosine similarity.
  - Zero-Shot model vote as propaganda or transparent.
  - Presence of slogans using regex patterns (propaganda).
  - Presence of Reductio Ad Hitlerum technique lexicons (propaganda).
  - Presence of exaggerated tone technique using POS tags (propaganda).

In the case of unbalanced datasets, it is recommended by the Snorkel team to split labeling functions that produce multiple signals in order to understand and maximize the accuracy of each class. We follow this behavior and split the labeling functions that don't have a specific label appended to it from above into two labeling functions. We also have split each lexicon type in the proppy lexicons to use only the ones that maximized the label model performance.

The total number of resulting labeling functions is 50 LFs. We used only the ones that maximized the performance of the label model while eliminating the ones that have high coverage and high accuracy as they are suspected of providing a majority class signal rather than providing a correct one. It is also recommended by the Snorkel team to use only the labeling functions that we are sure to have at least a 50% precision score on the unlabeled dataset by generalizing the score of the 500 labeled examples. It is recommended but not required. We used all the labeling functions that maximized the label model performance regardless of their precision score on the 500 labeled examples up to a threshold (>= 20%).

In order to aggregate the labeling functions' votes, we trained a Probabilistic Graphical Model to learn the accuracies and correlation dependencies between the labeling functions and the true (*hidden*) label. We trained the label model using a constant learning rate scheduler and an Adam optimizer with a 0.05 warmup ratio. The number of training epochs picked is 2000 to ensure convergence. These settings along with tuning the L2 regularization parameters led to a label model that has 92 accuracy, 92 weighted-averaged F1, and 77 macro-averaged F1 scores on the 500 labeled examples. We then applied the label model on the entire unlabeled dataset and produced a weakly supervised one that has around 23% propaganda tweets. The weakly labeled dataset will be shared later.

## Weakly Supervised Model

Now is the time for the end model training. Since the job of the label model is to produce weakly labeled data, we need to train an end discriminative model that generalizes over the data and fuses the noisy labels in it while ensuring the coverage of the correct ones. It is recommended to train the end model using a noise-aware objective (loss) function. We aimed to use the Active-Passive Losses from (SOMEWHERE). This type of noise-aware objective doesn't support unbalanced datasets, so we reverted back to using the weighted cross entropy based on the accepted signal (performance) we have from the label model. We picked AraBERT version 2 models as Autoencoders as they have been adapted to accepted processed text from Farasa that we used as the text processing tool throughout the project. (PERFORMANCE)

## Fully Supervised Model

To validate the worthiness of our weakly supervised model, we trained a fully supervised model on 80% of the labeled dataset (1680 tweets not including the test set). Initially we trained a shallow Logistic Regression baseline and obtained 90 accuracy, 90 weighted-averaged F1, and 72 macro-averaged F1 scores. Note that we have already surpassed these results using the label model. For the main fully supervised model we used the AraBERT version family autoencoders as we did in the fully supervised model. Again, we trained using a weighted cross entropy loss to overcome the class imbalance. The fully supervised model achieved 90 accuracy, 91 weighted-averaged F1, and 78 macro-averaged F1 scores.

## Conclusion

We made. Thanks!
