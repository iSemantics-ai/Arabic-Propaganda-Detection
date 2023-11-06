# Arabic Propaganda Detection

Weakly supervised Arabic propaganda detection on X (Twitter formally). Propaganda is defined as the intent to decive audience towards a speciific target following a predefined agenda utilizing means of language and vision. We focus on propaganda in the social media domain (especially X) and try to develop a model that can automatically detect this behavior. The project components are fully supervised baseline and transformer model trained on the 80% of the labeled dataset (2100 tweets in total), analysis of the users behavior based on meta data (e.g. number of followers) and tweets (linguistics of the tweet), labeling functions which are the distant supervision signals that will be combined using a Probabilistic Graphical Model (Snorkel Label Model) and validated using 500 labeled tweets, and the weakly supervised model trained on the vast amount of the weakly labeled data labeled by the Snorkel Label Model.

## Table of Contents

- Programmatic Labeling
- Manually Labeled Dataset
- Unlabeled Dataset Analysis
- Labeling Functions
- Weakly Supervised Model
- Fully Supervised Model
- Conclusion

## Programmatic Labeling

Recent advances in Deep Learning have produced large models (lareg in number of parameters) that can be trained to autommatically learn the text features (characterisitcs) and map them to needed tasks, eliminating the need to feature-engineering and researching the complete set of features that represent a piece of text to be understood by machines. While being a breakthrough, deep learning models require vast amounts of labeled data to be trained on.

Aquiring these large amounts of labeled data is costly in the following manners:

- **Data annotation (labeling)** is an exhaustive task that requires a lot of time and Subject Matter Experts to work on.
- **Subject Matter Experts (SMEs)** are rare to find, especially in complicated tasks like the one we are ahead of (propaganda detection). Even when available, their help is very expensive leading to a higher cost of the overall project.
- Most of the time, SMEs make **Human Errors** related to bias and personal perferences leading to complexity in vote aggregation and lowering the **Labeling Quality**.
- **Static Labeling** is the main drawback after all. At any time if the task is shifted or new rules need to be added while labeling, the dataset will be labeled again from scratch leading to even higher time and effort costs of an on-going project.
- **Privacy Concerns** is yet another major issue in labeling as SMEs might not be allowed to have access to medical or classified document. Which forces loss of speciality in domains like health, finance, etc.

To address all these issues, we leverage **Programmatic Labeling** in which we ask SMEs to write down the rules and guides they would use to label a dataset. These rules and guides are then transformered to simple Python functions (**labeling Functions**) that can be applied over vast amounts of unlabeled data. The unlabeled data is assumed to be all-time available. And in case of classified datasets, the rules can be applied without worries about privacy concerns.

Rules and guides serving as **Labeling Functions** can be of any type. They can be regular expression patterns extracted from text, distant supervision signals from external databases like other annotated datasets (domain non-specific), or singals from third-party models like Zero-Shot models and task non-specific finetuned models. All these signals will be combined using a **Label Model** that learns the relationship between each and every labeling function and other labeling functions and the true (hidden or unknown) label. The trained Label Model will have a better understanding of the correlations and accuracies of the labeling functions in order to efficiently aggregate the labeling functions votes. That is proved in the literature to surpass majority vote by a large margin.

## Manually Labeled Dataset

The annotated propaganda detection dataset comes from a previous research we did in this project. We have the tweets annotated as *propaganda* or *transparent*, an example will follow. The dataset consists of **2100** tweets manullay labeled and it is published in its raw form as text without any processing. We intend to use 80% of it to train (training and validation) the fully supervised models with only **500** tweets stratifiedly selected from the 80% training data to develop the labeling functions and measure the performance of the **Label Model** before weakly labeling the unlabeled dataset. The left 20% of the data will serve as gold test dataset that we will use to evaluate all the models.

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

To get insights and ideas about the labeling functions that can be applied, we performaned analysis on the unlabeled dataset. The unlabeled dataset comes from two resources. The first is the X published tweets of suspended propagandists from which we have extracted a 56K representative sample in a previous research. The second is a list of legitimate users on X that we have scrapped their recent tweets. Based on the topics found in the dataset from the first resource, we collected 140K tweets to serve as tweets from genuine users. We aim to analyze the tweets from both sides to understand the propagandists and genuine users characteristics.

The number of unique user ids of propagandist accounts exposed by X is **5929** while the number of genuine users we listed is **12145**. We aim to analyze the users meta data (e.g. number of followers) for both worlds and the tweets' text.

### Meta Data

We have analyzed the users' meta data and compared the difference for each data type as follows:

- Bio length and whether an account has a bio or not.
- Number of words and emojis in the bio.
- Most used words in bio.
- The ration of the followers count to the following count.
- Account creation year.
- Valid location reported in the account.

From the above analysis we found that:

- Propagandist users accounts were created in `2018` and `2019`.
- Genuine users accounts were created in `2011` and `2012`.
- Genuine users use the words `الحساب` and `الرسمي` in their bios.

### Tweets

We have analyzed the linguistic features of the tweets for both worlds. We have also test the presence of some of the propaganda techniques in both words. It is worth mentioning that the presence of a technique doesn't imply the tweet is propaganda. Propaganda is detected based on the intent of the user to decive the audience and based on the context. We analyzed the following:

- Number of urls, hashtags, mentions, and emojis.
- Number of and mosted used words with and without **Stop Words** (come from NLKT).
- Number of tweets contianing questioning (serves the Doubt technique).
- Number of entities (serves both the Name Calling and the Smears techniques).
- Sarcasm and hate speech signals (serves the Loaded Language, Smears, and Glittering technique).
- The presence of the Name Calling technique based on the combination of sarcasm and hate speech signal along with the presence of entities.

From the above analysis we found that:

- Genuine users add `at least one URL` in their tweets.
- Propagandist users use `mentions` `at lest one time` in their tweets.
- Some propagandist users tweet in less than 5 words.
- The word `حرب` is of the `top 50` used words in the propagandists tweets.
- The name calling technique appears at least **1.2** times more in the propagandists tweets.

The reported insights along with all the possible coding of the propaganda techniques will be used as labeling functions.

### Labeling Functions
