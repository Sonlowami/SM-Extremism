# Overview
This project implements a machine learning solution for the Kaggle Social Media Extremism Detection Challenge. We conducted comprehensive quantitative analysis on the original dataset to understand class distribution, dataset size, and statistical patterns. In addition, we performed qualitative analysis through team-based annotation to gain deeper insights into labeling nuances. We built a baseline model to classify social media posts as EXTREMIST or NON_EXTREMIST, and performed systematic error analysis to identify performance gaps and improvement opportunities.

# Part A: Quantitative Analysis

# Dataset Statistics
**Total Records**: 2,777 posts
**Features**: 2 columns (Original_Message, Extremism_Label)
**Class Distribution**:
EXTREMIST: 47.64% with 1323 records
NON_EXTREMIST: 52.36% with 1454 records

# Data Cleaning
**Duplicates**: No duplicates found
**Missing Values**: Checked for missing data in text and labels and found that 1 row has a missing value in the "Original_Message" column. The missing value was handled by dropping. We applied dropping because the text itself serves as the primary input feature. It is not feasible to meaningfully impute or generate a missing social media post. Replacing it with an empty string or placeholder would introduce noise and potentially bias the model. Since only one record (0.04% of the dataset) is missing, removing it has a negligible impact on the overall dataset size.

Final shape: (2776, 2)
Final class distribution:
Extremism_Label
NON_EXTREMIST    1453
EXTREMIST        1323

# Text Characteristics
Added three new features to analyze text properties:
1. **text_length**: Character count per post
2. **word_count**: Number of words per post
3. **avg_word_length**: Average characters per word

**Key Findings**:
EXTREMIST posts tend to be slightly longer on average
Both classes show nearly similar word count distributions
Average word length is comparable across classes

=== Mean Comparison ===
                 text_length  word_count  avg_word_length
Extremism_Label                                          
EXTREMIST             134.56       24.50             5.47
NON_EXTREMIST         119.39       22.19             5.41


# Linguistic Analysis
**Top 20 Words (Overall)**: Identified most frequent terms across all posts

Top 20 Frequent Words
________________________________________
Word      Frequency
------  -----------
the            1802
a              1623
to             1589
you            1558
and            1350
of             1257
i              1166
is              959
are             922
bitch           832
not             809
in              708
that            677
it              663
with            628
they            557
all             530
kill            503
for             442
do              437



**Top 20 Words (Per Class)**: 
EXTREMIST: Words related to violence, hatred, specific groups
NON_EXTREMIST: More neutral, conversational language


Top 20 Frequent Words in Extremist Posts
________________________________________
Word      Frequency
------  -----------
the             899
to              840
and             786
of              688
you             677
a               636
are             588
i               548
is              471
kill            425
not             405
they            401
it              396
all             390
in              375
with            349
that            316
them            283
us              222
get             220



Top 20 Frequent Words in non Extremist Posts
________________________________________
Word       Frequency
-------  -----------
a                987
the              903
you              881
to               749
bitch            730
i                618
of               569
and              564
is               488
not              404
that             361
are              334
in               333
with             279
it               267
for              237
my               234
do               231
fucking          229
trump            228



**Word Clouds**: Visual representation of term frequency
Two most prominent words identified for each class  where words like kill and let are associated with extremism while words like bitch and fucking are associated with non-extremism.


# Part B: Qualitative Analysis

# Annotation Process
**Sample Size**: 30 posts (15 per class)
**Annotators**: 5 team members
**Random State**: 42
**Column added**: ID
**Method**: Independent annotation without discussion
**Result**: Team based annotated file saved to "mbd_annotation_sample_30.csv"

# Inter-Rater Reliability (IRR)
**Pairwise percentage Agreement**: Calculated for all annotator pairs including the original label and group members. This is simply done by dividing the number of instances where both annotators agreed by the total number of annotated samples.

The results were stored in a structured table showing the agreement score for each rater pair as follows.

Result:
______

Rater_1,Rater_2,Percent_Agreement
Extremism_Label,Group_member1,0.5
Extremism_Label,Group_member2,0.567
Extremism_Label,Group_member3,0.567
Extremism_Label,Group_member4,0.667
Extremism_Label,Group_member5,0.633
Group_member1,Group_member2,0.8
Group_member1,Group_member3,0.467
Group_member1,Group_member4,0.767
Group_member1,Group_member5,0.667
Group_member2,Group_member3,0.533
Group_member2,Group_member4,0.7
Group_member2,Group_member5,0.533
Group_member3,Group_member4,0.567
Group_member3,Group_member5,0.4
Group_member4,Group_member5,0.7

Interpretation: 
______________
There is moderate inter-annotator agreement overall, with several pairs ranging between 50% to 70%.

However, agreement is not uniformly high, indicating that extremism classification involves subjective judgment and potential ambiguity.

The variation in agreement highlights the importance of Clear annotation guidelines, Consensus discussions, Reliability metrics beyond simple percentage agreement.

**pairwise Krippendorff's Alpha**: To assess inter-annotator reliability beyond simple agreement, pairwise Krippendorff’s Alpha was computed for all combinations of annotators.
Krippendorff’s Alpha (α) is a statistical measure of inter-annotator reliability that evaluates how much agreement exists between raters beyond what would be expected by chance.

Result:
______

Rater_1,Rater_2,Krippendorff_Alpha
Extremism_Label,Group_member1,0.016
Extremism_Label,Group_member2,0.139
Extremism_Label,Group_member3,0.147
Extremism_Label,Group_member4,0.333
Extremism_Label,Group_member5,0.258
Group_member1,Group_member2,0.605
Group_member1,Group_member3,-0.049
Group_member1,Group_member4,0.536
Group_member1,Group_member5,0.317
Group_member2,Group_member3,0.066
Group_member2,Group_member4,0.376
Group_member2,Group_member5,0.078
Group_member3,Group_member4,0.123
Group_member3,Group_member5,-0.201
Group_member4,Group_member5,0.409


Interpretation:
______________


- Agreement with the Original Label:

Alpha values between the original dataset label and group members range from 0.016 to 0.333. α = 0.333 is considered weak reliability and α = 0.016 is essentially chance-level agreement.
This is very low reliability overall and it suggests that the original dataset labels do not strongly align with most group annotations when correcting for chance.


- Agreement Among Group Members:

Alpha values between group members range from -0.201 to 0.605. This includes Moderate reliability, fair reliability, weak reliability and semantic disagreement(shown by negative alpha values). However, only one pair exceeds 0.6 (moderate agreement) while several pairs are near zero or negative, indicating inconsistency.

This extremism labeling task appears to be subjective, ambiguous and sensitive to interpretation differences.

# Disagreement Analysis

**Inter-Human Disagreement Rate**:

===== Disagreement Analysis =====
Total Examples: 30
Disagreement Examples: 24
Inter-Human Disagreement Rate: 0.8

This means that annotators disagreed on 80% of the 30 examples, with disagreement occurring in 24 cases. It indicates very low consensus among human raters, suggesting that the extremism classification task is highly subjective.

**Examples with Most Disagreement**: Identified ambiguous cases.

ID       Original_Message                               
1  she isnt racist unlike your somalia bitch friend and you re always causing shit bye slut                 
2  if i ever saw kendall jones in person i d kill her on sight i hate that bitch more than anything in life               
4  i am in iraq as whatever speak your country is beautiful but thanks to terrorism still very unsafe and volatile but not that bad as it was before              
7  disabled people artfully the fucking scum of the world careless disgusting people                
8  if your fiance has a conversation with a bitch like this would you keep her                  
14  whatever do not blow up our wives and children in the name an allah nor did whatever murder in darfur let us not compare apples and oranges              
15  never a doubt that our founders saw the writing on the wall when they revolted against the british uk supports the muslim global caliphate christian refugees not welcome               
21  in my opinion whatever should be afraid of them even in captivity they are unpredictable savage animals you just never know and you should not count on anyone jumping in to save you if something happens and you got the daylights stomped out of you by a pack of actual raging baboons from the zoo                
24  keyshia key whore is a bird gucci should have give that bitch back to year old gotta             
26  ceasefire let s see how long those towel heads can go without trying to attack israel then cry to national media when they get popped again


The reason for this: These posts sit in the gray area between hate speech, offensive language, and it can cause the challenge of applying the definitions of EXTREMIST and NON-EXTREMIST consistently. While according to the provided definitions, EXTREMIST texts are classified as those that promote, call for, or justify violence, hatred, or extremist ideology and/or dehumanize a group, many posts contain strong offensive or threatening language without a clear ideological agenda. Annotators may differ on whether posts that insult, use profanity, or express anger toward individuals or groups constitute extremist content, since such language can be dehumanizing but not explicitly promoting extremist ideology. Additionally, ambiguity in intent and lack of contextual information makes it difficult to distinguish between political commentary, personal anger, or genuine extremist advocacy. These factors, combined with personal thresholds for what qualifies as extremist, contributed to the high disagreement rates observed.


# Comparison with Original Labels
**Majority Vote Agreement**: Not all the percentages agree with original dataset
Majority agreement count: 16
Total examples: 30
Majority vs Original Agreement Rate: 0.533
**Disagreements**: Identified cases where team annotations differed from original

===== Examples Where Group Disagreed With Original Label =====

    ID                                   Original_Message  Extremism_Label  majority_vote
    2  if i ever saw kendall jones in person i d kill...                1   0
    4  i am in iraq as whatever speak your country is...                1   0
    7  disabled people artfully the fucking scum of t...                1   0
    8  if your fiance has a conversation with a bitch...                1   0
   10  yet calling president trump and republicans na...                0   1
   13  leader and i use that word entirely wrongly cl...                0   1
   14  whatever do not blow up our wives and children...                0   1
   15  never a doubt that our founders saw the writin...                0   1
   17                         jihadi john is a terrorist                0   1
   18  i really sympathize for the kid at do not care...                0   1
   22  the fucking scum of the world non whites sense...                1   0
   24  keyshia key whore is a bird gucci should have ...                1   0
   25  the fucking scum of the world half breeds sens...                1   0
   27  love let not all men be equally yoked to the a...                0   1



**Implications**: 
- Original labels often disagree with group consensus, indicating that some labels may have been applied subjectively or without strictly following the provided definition of extremism.

- High disagreement and discrepancies suggest that annotators interpreted offensive or threatening language differently, confusing general insults with ideologically motivated extremism.

- The task itself is subjective and context-dependent, which makes consistent labeling challenging without clear guidelines or training.

About annotation quality, the dataset contains label noise, which can impact model performance, and highlights the importance of majority voting, adjudication, or more rigorous annotation protocols to improve reliability.


# Part C: ML Baseline Model
### Data Splitting
**Training Set**: 70% with 1943 records
**Validation Set**: 15% with 416 records
**Test Set**: 15% with 417 records
**Random State**: 42

# Feature Engineering
**Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
**Parameters**: Default scikit-learn settings


# Model Training
**Algorithm**: Logistic Regression
**Parameters**: Default settings, max_iter=1000
**Convergence**: Successfully converged

### Validation Results

Classification Report (Validation Set):

               precision    recall  f1-score   support

    EXTREMIST       0.83      0.79      0.81       198
NON_EXTREMIST       0.81      0.85      0.83       218

     accuracy                           0.82       416
    macro avg       0.82      0.82      0.82       416
 weighted avg       0.82      0.82      0.82       416

### Confusion Matrix

         EXTREMIST             156                    42

Actual:

         NON-EXTREMIST         33                     185

                            EXTREMIST             NO-EXTREMIST

                                       Predicted

       
       Observation:

EXTREMIST class: 156 correctly predicted, 42 misclassified as NON_EXTREMIST.
NON_EXTREMIST class: 185 correctly predicted, 33 misclassified as EXTREMIST.


## Part D: Error Analysis

### Systematic Error Review

ALL misclassifications were identified, False Positives and False Negatives were separated,
 and a csv file of misclassified records was saved as "errors_validation_10.csv".

**Total Misclassifications**: 75 errors

### Error Type Breakdown
1. **False Positives (FP)**: 33 cases
   Predicted EXTREMIST, actually NON_EXTREMIST
   Common pattern: Profanity without extremist intent

2. **False Negatives (FN)**: 42 cases
   Predicted NON_EXTREMIST, actually EXTREMIST
   Common pattern: Subtle/coded extremist language


### 10 misclassified examples

 **False Positive examples:
 - twelve oclock is the most important thing all this temporary bullshit and lies is fa the birds kill that
 - how long will merkel hang onto power a heart of a country welcomed migrants and that heart has been torn apart immigration will go
 - leaving a pot on the stove can burn more than just your dinner always stay in the kitchen while cooking more tips here
 - illegal alien accused of trying to rob stab woman in home invasion breitbart stoptheinvasion
 - burn your bra s

 **False Negative examples:
 - the quran teaches terrorism
 - far right cell planned attack on jewish event in paris resist resist trump fuck trump
 - pink champagne for my boot hang she is a bad bitch i hope that she do not trade me
 - you are a traitor to you own people country stick your tests up your ass go to hell
 - take action and behead those mudslime trailer rise up english lions

### Error Patterns Identified

**False Positive Patterns**:
1. Posts with profanity but no extremist ideology
2. Sarcastic or hyperbolic statements
3. Discussions about extremism without promoting it
4. Strong emotional language without violent intent


**False Negative Patterns**:
1. Subtle dog-whistles and coded language
2. Indirect calls to violence or hatred
3. Extremist content in neutral-sounding language
4. Context-dependent extremism

### Why the Model Fails

1. **Lack of Context Understanding**
   TF-IDF treats words independently
   Cannot distinguish discussing vs. promoting extremism

2. **Profanity Confusion**
   Model associates offensive language with extremism
   Many non-extremist posts use profanity

3. **Subtle Extremism**
   Coded language and dog-whistles missed
   Indirect incitement not recognized

4. **Feature Limitations**
   - Bag-of-words loses word order and semantics
   - No understanding of negation or sarcasm

5. **Annotation Noise**
   - Label noise reduces model reliability and makes it harder for the classifier to learn clear patterns distinguishing EXTREMIST from NON-EXTREMIST posts.

### Proposed Improvements

1. **Use N-grams (Bigrams/Trigrams)**
   Capture word sequences for better context
   Implementation: `TfidfVectorizer(ngram_range=(1, 3))`
   Expected Impact: Better phrase understanding

2. **Add Profanity Filter Feature**
   Separate profanity from extremist language
   Implementation: Binary feature for profanity presence
   Expected Impact: Reduce false positives

3. **Use Pre-trained Embeddings (BERT)**
   Leverage semantic understanding
   Implementation: Replace TF-IDF with BERT embeddings
   Expected Impact: Capture subtle meanings and context

### Implemented Improvement: TF-IDF + LinearSVC with Raw Term Frequency and Bigram Features

To enhance classification performance, we upgraded the baseline model by combining TF-IDF vectorization with raw term frequencies and bigrams (n-gram range = 1–2) and using a Linear Support Vector Classifier (LinearSVC) with class balancing.

- Vectorization: TF-IDF captured both unigrams and bigrams, including repeated and structured phrases, while ignoring extremely rare (min_df=2) and overly common (max_df=0.95) terms.

- Classifier: LinearSVC with class_weight="balanced" gave proportional importance to the minority EXTREMIST class, improving detection of rare but critical examples.


**Results**:


Total misclassifications: 66
False Positives: 36
False Negatives: 30


Classification Report (Validation Set):

               precision    recall  f1-score   support

    EXTREMIST       0.82      0.85      0.84       198
NON_EXTREMIST       0.86      0.83      0.85       218

     accuracy                           0.84       416
    macro avg       0.84      0.84      0.84       416
 weighted avg       0.84      0.84      0.84       416


 ### Confusion Matrix

         EXTREMIST             168                    30

Actual:

         NON-EXTREMIST         36                     182

                            EXTREMIST             NO-EXTREMIST

                                       Predicted

       
       Observation:
EXTREMIST class: 168 correctly predicted, 30 misclassified as NON_EXTREMIST.
NON_EXTREMIST class: 182 correctly predicted, 36 misclassified as EXTREMIST.


**Improvement and Analysis**: 
Overall Accuracy: The model’s accuracy increased from 82% to 84%, representing a modest yet meaningful improvement on the same validation set.

EXTREMIST Class: The most notable change occurred here. While precision slightly decreased (from 0.83 to 0.82), recall improved significantly from 0.79 to 0.85. This means the updated model now identifies more true extremist posts that were previously missed—a critical improvement for content safety. The F1 score also increased from 0.81 to 0.84, reflecting better overall performance.

NON-EXTREMIST Class: Here, the trend is reversed. Recall slightly decreased (0.85 → 0.83), indicating the model is somewhat less conservative in labeling posts as non-extremist. However, precision improved (0.81 → 0.86), meaning when the model predicts a post as non-extremist, it does so with greater confidence.

Trade-Off: Overall, the adjustments shifted the decision boundary, making the model more aggressive in detecting extremist content. This led to higher EXTREMIST recall but slightly more false positives, while NON-EXTREMIST recall dropped slightly but with improved precision.


# Key Insights

1. **Dataset Quality**: Balanced dataset with clear class separation in word usage
2. **Annotation Challenge**: Moderate inter-rater agreement indicates task difficulty
3. **Model Performance**: Baseline achieves reasonable accuracy but struggles with subtle cases


## Recommendations

1. **Advanced Models**: Implement transformer-based models (BERT, RoBERTa)
2. **Feature Engineering**: Add sentiment analysis, hate speech lexicons
3. **Data Augmentation**: Generate synthetic examples for edge cases
4. **Ensemble Methods**: Combine multiple models for robust predictions
5. **Active Learning**: Focus annotation efforts on difficult examples


## References

1. Kaggle Dataset: Digital Extremism Detection Curated Dataset
2. Krippendorff's Alpha: https://www.appen.com/blog/krippendorffs-alpha
3. Scikit-learn Documentation: https://scikit-learn.org/

