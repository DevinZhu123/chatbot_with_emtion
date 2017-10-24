### 10707 Project Proposal

#### Problem statement
    > Dialogues generation with similary emotions:
    > Given a user's sentance/utterance as an initial input, generate dialogues with similar or proper emotions and correct word context.

#### Motivation
    > In daily conversations, like online chatting, users send messages with emotional interactions instead of simple Q&A pattern. A natural, fluent one-to-one conversation, generally, contains dialogues in which both users share similar emotion levels, such as happy, saddness. Our goal is to model a natural dialogue and the model can figure out users' emotions and response words with proper emotions.
   

#### Proposed approach
    > Our project contains three parts:
    > 1. Emotion extraction and feature embedding.
    #   In order to understand how emotions are presented in sentence and how can they be embedded into a metric space, we plan to train a classification model (sentence based RNN model) to do feature extraction and embedding. These features will be used later by our model to interpret user emotions. Also, this classification model will be used for adding possible emotino tags to our movie-subtile database. 
    #   Some statistical features will be used too.

    > 2. Generative model for sentence generation:
    #   To generate a response from an input sentence, a generate model is necessary to model the sentence generation. Bascailly, we plan to use Seq2Seq model with word embedding method like word2vec to acheive word embedding and language modeling.

    > 3. Reinforcement Learning Framework of dialogue generations:
    #   To take context information into consideration, we plan to fit our model into reinforcement learning framework. Specifically, we try to make our model predict a user's current emotion with optimized forward rewards (similarity of possible emotions) and backward rewards. 


#### Datasets
    >We collected 2.7 million sentences of subtitles from open subtitle database (http://opus.lingfil.uu.se/OpenSubtitles.php). The subtitles of movies and TV programs provide sufﬁcient conversation contents for training. Nevertheless, subtitles are slightly different from daily conversation due to theatrical and dramatic contents. We would ﬁlter long segment of words which is impractical in daily conversation, which could be monologues in movies.

    >To tag the sentences in OpenSubtitles dataset with emotion tags, we need to train a model to predict each sentence’s emotion. We would use a dataset of tweets with emotion tags from LiveJournal to train the prediction model. 


#### Other sections that you may find relevant for your particular project
    >
    >

#### Reference
    1. Deep Reinforcement Learning Dialogue generation
    2. Emotion CM
    3. Seq2Seq
    4. word2vec