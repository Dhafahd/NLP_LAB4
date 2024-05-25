### MST AIDS 2023-2024 (Département Génie Informatique)
**Subject : The main purpose behind this lab is to get familiar with ule-based techniques, Regex, and NLP Word Embedding.**\
**Realize by : Chibani Fahd**\
**web source : Aljarida24r**\
**Course : NLP**\

## Introduction
In this lab, we will embark on a comprehensive journey through the field of Natural Language Processing (NLP) using advanced machine learning techniques. Our primary objective is to develop a robust NLP pipeline and evaluate various language models. The process begins with collecting text data from several Arabic websites on a specific topic using web scraping libraries such as Scrapy and BeautifulSoup. This data will be organized into a dataset where each text is assigned a relevance score between 0 to 10, indicating its pertinence to the topic. Following data collection, we will preprocess the dataset through a series of NLP techniques, including tokenization, stemming, lemmatization, stop words removal, and discretization, to prepare it for model training. We will then train four different architectures—RNN, Bidirectional RNN, GRU, and LSTM—while tuning hyper-parameters to optimize their performance. Finally, we will evaluate the models using standard metrics and additional measures like the BLEU score to determine their efficacy in handling the Arabic language dataset. This lab aims to provide a deep understanding of the capabilities and limitations of various NLP models in processing and analyzing text data in Arabic.
## 1. Scraping data :
### 1.1 Scraping using Selenuim :
We used the Selenium library to scrape links from the Al Jazeera website related to the Palestinian war. By automating our web scraping process with Selenium, we efficiently navigated through various web pages, dynamically loaded content, and extracted the necessary link
```python
Exemple of links scraped : ['https://www.aljazeera.net/news/2024/5/25/%D8%A3%D9%88%D9%84%D9%85%D8%B1%D8%AA-2']
```
Using this library we scrapted more than 304 links related to the Palestinian war
### 1.2 Scraping using BeautifulSoup :
We used the Selenium library to scrape links from the Al Jazeera website related to the Palestinian war. After gathering the links, we employed the BeautifulSoup library to extract the content of each article. BeautifulSoup allowed us to parse the HTML structure of the web pages, efficiently retrieving text, images, and other relevant data. Once the content was extracted, we organized it and stored it in a JSON file, ensuring a structured and easily accessible format for further analysis. This combination of Selenium for link extraction and BeautifulSoup for content retrieval enabled us to create a comprehensive and well-organized dataset of articles on the Palestinian war.
```python
Exemple of an article scraped : ['قرر يوفنتوس الإيطالي الطعن في حكم كريستيانو رونالدو لاعب النصر السعودي بعد فوزه رسميا في المعركة القانونية مع النادي بشأن الراتب.  وأكد الصحفي الإيطالي الشهير فابريزيو رومانو عبر حسابه على منصة \"إكس\" أن يوفنتوس سيعمل بشكل قانوني لتجنب دفع مبلغ 9.8 ملايين يورو رواتب متأخرة لرونالدو عن موسم 2020-2021، بالإضافة إلى الفوائد.    ويصر نادي \"السيدة العجوز\" على موقفه بأن رونالدو ليس له الحق في الأموال المتنازع عليها، لأنه لم يتم توقيع أي عقود جديدة تتعلق بالتنازلات عن الرواتب خلال جائحة كورونا، بينما حصل اللاعبون الآخرون الذين وقعوا على تلك الاتفاقيات على مستحقاتهم.  ووافق لاعبو يوفنتوس على تأجيل رواتبهم لمدة 4 أشهر خلال الفترة من مارس/آذار 2020 وحتى أبريل/نيسان 2021، حيث كان النادي يعاني ماليا، ولكن تم أيضا إبرام اتفاقيات فردية مع اللاعبين.  وزعم يوفنتوس أيضا أن رونالدو تنازل عن أمواله المستحقة عندما غادر إلى مانشستر يونايتد الإنجليزي في صيف عام 2021.  وطالب رونالدو، الذي لعب في صفوف يوفنتوس 3 مواسم (2018-2021) قبل العودة إلى مان يونايتد (2021-2022) ومنه إلى النصر السعودي، بـ19.5 مليون يورو، لكن المحكمة خفّضت المبلغ بنسبة 50%.  ووفقا لمجلة الأعمال الأميركية فوربس، كان رونالدو، الفائز بجائزة الكرة الذهبية 5 مرات، الرياضي الأعلى أجرا في العالم خلال عام 2023، بمبلغ 136 مليون دولار، من بينها 46 مليون دولار رواتب.    وسبق أن أعلن يوفنتوس، الأكثر تتويجا في إيطاليا، في أكتوبر/تشرين الأول الماضي، عن خسائر بلغت 123.7 مليون يورو في السنة المالية 2022-2023، التي امتدت حتى نهاية يونيو/حزيران.  وأمضى رونالدو 3 سنوات في تورينو بعد مغادرته ريال مدريد في 2018، وسجل خلالها 101 هدف في 134 مباراة مع يوفنتوس، وساهم مع \"السيدة العجوز\" في الفوز بلقبين في الدوري الإيطالي وكأس إيطاليا.
']
```
## 2. Score data :
Once the content was extracted, we organized it and stored it in a JSON file, ensuring a structured and easily accessible format for further analysis. To evaluate the relevance of each article to our topic, we implemented a scoring system using a classification pipeline with a pre-trained model for zero-shot classification specifically designed for Arabic text. This model provided a rating based on the similarity of each article to the Palestinian war topic, allowing us to prioritize the most pertinent articles for our research.
```python
Exemple of an article scraped :"content": ['قرر يوفنتوس الإيطالي الطعن في حكم كريستيانو رونالدو لاعب النصر السعودي بعد فوزه رسميا في المعركة القانونية مع النادي بشأن الراتب.  وأكد الصحفي الإيطالي الشهير فابريزيو رومانو عبر حسابه على منصة \"إكس\" أن يوفنتوس سيعمل بشكل قانوني لتجنب دفع مبلغ 9.8 ملايين يورو رواتب متأخرة لرونالدو عن موسم 2020-2021، بالإضافة إلى الفوائد.    ويصر نادي \"السيدة العجوز\" على موقفه بأن رونالدو ليس له الحق في الأموال المتنازع عليها، لأنه لم يتم توقيع أي عقود جديدة تتعلق بالتنازلات عن الرواتب خلال جائحة كورونا، بينما حصل اللاعبون الآخرون الذين وقعوا على تلك الاتفاقيات على مستحقاتهم.  ووافق لاعبو يوفنتوس على تأجيل رواتبهم لمدة 4 أشهر خلال الفترة من مارس/آذار 2020 وحتى أبريل/نيسان 2021، حيث كان النادي يعاني ماليا، ولكن تم أيضا إبرام اتفاقيات فردية مع اللاعبين.  وزعم يوفنتوس أيضا أن رونالدو تنازل عن أمواله المستحقة عندما غادر إلى مانشستر يونايتد الإنجليزي في صيف عام 2021.  وطالب رونالدو، الذي لعب في صفوف يوفنتوس 3 مواسم (2018-2021) قبل العودة إلى مان يونايتد (2021-2022) ومنه إلى النصر السعودي، بـ19.5 مليون يورو، لكن المحكمة خفّضت المبلغ بنسبة 50%.  ووفقا لمجلة الأعمال الأميركية فوربس، كان رونالدو، الفائز بجائزة الكرة الذهبية 5 مرات، الرياضي الأعلى أجرا في العالم خلال عام 2023، بمبلغ 136 مليون دولار، من بينها 46 مليون دولار رواتب.    وسبق أن أعلن يوفنتوس، الأكثر تتويجا في إيطاليا، في أكتوبر/تشرين الأول الماضي، عن خسائر بلغت 123.7 مليون يورو في السنة المالية 2022-2023، التي امتدت حتى نهاية يونيو/حزيران.  وأمضى رونالدو 3 سنوات في تورينو بعد مغادرته ريال مدريد في 2018، وسجل خلالها 101 هدف في 134 مباراة مع يوفنتوس، وساهم مع \"السيدة العجوز\" في الفوز بلقبين في الدوري الإيطالي وكأس إيطاليا.
']
"rating":7.3990756273
```

### 3.1 Data pre-procecing 
Our workflow commences with the essential task of text refinement, focusing on the removal of stop words and adjectives such as 'new,' 'cool','fresh'... from the bill text. This meticulous cleansing is pivotal in simplifying the subsequent regex pattern matching process. By eliminating these unnecessary elements, we enhance the text's suitability for regex pattern matching, enabling more efficient and accurate extraction of relevant information. This initial cleaning phase optimizes the text for seamless integration into regex-based algorithms
```python
Before cleaning : ['I bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ each and four kilos of fresh banana for 2,4 dollar a kilogram']
After cleaning : ['bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ four kilos banana 2,4 dollar kilogram']
```
### 3.2 Regex pattern :
fter meticulously cleaning our text by removing extraneous elements such as stop words and adjectives, we proceed to employ regex for pattern matching. Utilizing regex, we define a pattern with three groups: the first capturing the price, the second identifying the product name, and the third representing the unit price. This systematic approach enables us to efficiently extract relevant information from the text and generate the bill.
```python
 pattern = r"((?:" + '|'.join(numbers) + r"|\d)(?:\s(?:" + '|'.join(numbers) + r"|\d|and))*)(.*?)(\d+[\.|\,]?\d*)\b\s*(\$|dollar)"
```
To convert textual numbers into their numerical values, we create a Python script (Word2num.py File )capable of intelligently parsing numbers up to 999,999,999,999. This script efficiently take the sentence, identifies numerical representations, and transforms them into their corresponding numeric values. 
```python
Sentence : I bought three hundred two thousand twenty seven Samsung smartphones 150,333 $ each, acquired five smartphones for 145$ each four kilos of fresh banana for 2,4 dollar a kilogram and one Hamburger with 4,5 dollar, ten boxes of tisseues for 2.5 $ each
---------------------------------
Results : Quantity: 302027
Product:  Samsung smartphones 
Price: 150,333
---------------------------------
Quantity: 5
Product:  smartphones 
Price: 145
---------------------------------
Quantity: 4
Product:   banana 
Price: 2,4
---------------------------------
Quantity: 1
Product:  Hamburger
Price: 4,5
---------------------------------
Quantity: 10
Product:  boxes tisseues
Price: 2.5
```

## 4.  word Embedding :
To better understand word embedding in Arabic text, we attempt to apply it to the paragraphs that we scraped in Lab 1.
### 4.1. one hot encoding :
One-hot encoding is a technique used in machine learning to represent categorical data numerically. Each category is represented as a binary vector, where only one bit is activated (set to 1) for the corresponding category, and all other bits are set to 0.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
binary representation : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```
### 4.2. bag of words :
The Bag of Words model represents text by counting the occurrence of words without considering grammar or order, forming a sparse numerical matrix used in various natural language processing tasks.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Our vocabulary :{'طالب': 60, 'نواب': 72, 'المعارضة': 30, 'البرلمانية': 16, 'بالكشف': 42, 'سبل': 57, 'تيسير': 50, 'الحصول': 19, 'الدعم': 20, 'الاجتماعي': 10, 'المباشر': 29, 'ظهور': 61, 'عدد': 62, 'الاشكاليات': 14, 'ووجه': 83, 'إدريس': 5, 'السنتيسي': 21, 'رئيس': 53, 'الفريق': 27, 'الحركي': 18, 'بمجلس': 46, 'النواب': 37, 'فوزي': 64, 'لقجع': 67, 'الوزير': 39, 'المنتدب': 34, 'المكلف': 33, 'بالميزانية': 43, 'سؤالا': 55, 'كتابيا': 66, 'مفاده': 70, 'سحب': 58, 'المواطنين': 36, 'شروعهم': 59, 'الاستفادة': 12, 'وأكد': 74, 'سؤاله': 56, 'العديد': 25, 'المواطنات': 35, 'والمواطنين': 76, 'فوجئوا': 63, 'بسحب': 44, 'ومعها': 82, 'نظام': 71, 'أمو': 2, 'تضامن': 48, 'الشروع': 23, 'أشهر': 1, 'الأمر': 8, 'أثار': 0, 'امتعاض': 40, 'الفئات': 26, 'المعنية': 31, 'تقرر': 49, 'إبعادها': 4, 'البرنامج': 17, 'بمبرر': 45, 'ارتفاع': 6, 'مؤشرهم': 69, 'رغم': 54, 'أنه': 3, 'يمكن': 84, 'تتغير': 47, 'وضعيتهم': 79, 'الاجتماعية': 11, 'والاقتصادية': 75, 'الظرف': 24, 'الوجيز': 38, 'وساءل': 78, 'حقيقة': 51, 'الإجراء': 9, 'وأسبابه': 73, 'ودوافعه': 77, 'وعدد': 80, 'المعنيين': 32, 'وكذا': 81, 'انعكاساته': 41, 'الأشخاص': 7, 'قدرة': 65, 'الانخراط': 15, 'الشامل': 22, 'للأشخاص': 68, 'القادرين': 28, 'دفع': 52, 'الاشتراكات': 13}
BoW representation : [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
```
### 4.3. TF-IDF :
TF-IDF (Term Frequency-Inverse Document Frequency) measures word importance in a document by considering both the frequency of the term within the document and its rarity across the entire document collection, aiding in tasks like text mining and information retrieval.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Our vocabulary :'['أثار' 'أشهر' 'أمو' 'أنه' 'إبعادها' 'إدريس' 'ارتفاع' 'الأشخاص' 'الأمر''الإجراء' 'الاجتماعي' 'الاجتماعية' 'الاستفادة' 'الاشتراكات' 'الاشكاليات''الانخراط' 'البرلمانية' 'البرنامج' 'الحركي' 'الحصول' 'الدعم' 'السنتيسي''الشامل' 'الشروع' 'الظرف' 'العديد' 'الفئات' 'الفريق' 'القادرين' 'المباشر''المعارضة' 'المعنية' 'المعنيين' 'المكلف' 'المنتدب' 'المواطنات''المواطنين' 'النواب' 'الوجيز' 'الوزير' 'امتعاض' 'انعكاساته' 'بالكشف''بالميزانية' 'بسحب' 'بمبرر' 'بمجلس' 'تتغير' 'تضامن' 'تقرر' 'تيسير''حقيقة' 'دفع' 'رئيس' 'رغم' 'سؤالا' 'سؤاله' 'سبل' 'سحب' 'شروعهم' 'طالب''ظهور' 'عدد' 'فوجئوا' 'فوزي' 'قدرة' 'كتابيا' 'لقجع' 'للأشخاص' 'مؤشرهم''مفاده' 'نظام' 'نواب' 'وأسبابه' 'وأكد' 'والاقتصادية' 'والمواطنين''ودوافعه' 'وساءل' 'وضعيتهم' 'وعدد' 'وكذا' 'ومعها' 'ووجه' 'يمكن']
TFIDF representation : [[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5 0. 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.  0. 0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]
```
### 4.4. Word2Vec with skip Gram :
Word2Vec with skip-gram is a popular algorithm used to generate word embeddings by predicting the context words given a target word in a sentence. It learns distributed representations of words based on their co-occurrence patterns, capturing semantic similarities and relationships between words in high-dimensional vector spaces.
```python
Sentence : ['طالب نواب المعارضة البرلمانية']
Cost after epoch 4750: 1.3515463363565614
skip-grams: ['البرلمانية', 'نواب', 'طالب', 'المعارضة'] طالب
skip-grams: ['البرلمانية', 'طالب', 'نواب', 'المعارضة'] نواب
skip-grams: ['البرلمانية', 'نواب', 'طالب', 'المعارضة'] المعارضة
skip-grams: ['المعارضة', 'نواب', 'طالب', 'البرلمانية'] المعارضة
```
### 4.5. Glove and FastText :
#### 4.5.1. FastText :
FastText is a word embedding technique developed by Facebook AI Research that extends the Word2Vec model by also considering subword information. It breaks words into character n-grams and learns embeddings for these subwords, enabling it to handle out-of-vocabulary words and capture morphological similarities effectively, making it particularly useful for tasks with large vocabularies and morphologically rich languages.
```python
Word vector for : ملك
Similar words to ( ملك ) : [('الملك', 0.33138006925582886), ('للملك', 0.2525012493133545)]
```
#### 4.5.1. Glove :
GloVe (Global Vectors for Word Representation) is a word embedding model that learns vector representations of words based on their co-occurrence statistics in a corpus. It aims to capture global word co-occurrence patterns by optimizing a global objective function, producing dense word vectors that encode semantic relationships between words, which are useful for various natural language processing tasks such as word similarity and analogy detection.

## 9. What I learned
In summary, my reflections from this lab underscore the intricacies inherent in Arabic, surpassing those of its Latin counterparts. The exercise has equipped me with strategies for discerning patterns among words and discerning similarities between them. Moreover, the exploration revealed a significant research gap in Arabic NLP, resulting in a paucity of available libraries tailored to Arabic text processing, thereby posing challenges to practitioners seeking comprehensive tools in this domain.
## 10. References
<a id="1">[1]</a>Jeet. (2021, December 15). One Hot encoding of text data in Natural Language Processing. Medium. https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148 \
<a id="2">[2]</a> Alkhatib, R. M., Zerrouki, T., Shquier, M. M. A., & Balla, A. (2023). Tashaphyne0.4: A new Arabic light stemmer based on rhizome modeling approach. Information Retrieval Journal, 26(14). doi: https://doi.org/10.1007/s10791-023-09429-y \
<a id="3">[3]</a>  notebook.community. (n.d.). https://notebook.community/arcyfelix/Courses/18-03-07-Deep%20Learning%20With%20Python%20by%20Fran%C3%A7ois%20Chollet/.ipynb_checkpoints/Chapter%206.1.1%20-%20One-hot%20encoding%20of%20words%20and%20characters-checkpoint/ \
<a id="4">[4]</a> MagedSaeed. (n.d.). GitHub - MagedSaeed/farasapy: A Python implementation of Farasa toolkit. GitHub. https://github.com/MagedSaeed/farasapy?tab=readme-ov-file#want-to-cite \
<a id="5">[5]</a> Munther, I. (2021, December 30). Sentiment Analysis of Arabic Text Data (Tweets) - Analytics Vidhya - Medium. Medium. https://medium.com/analytics-vidhya/sentiment-analysis-of-arabic-text-data-tweets-4e96c8da892b \
<a id="6">[6]</a> freeCodeCamp.org. (2019, July 24). How to process textual data using TF-IDF in Python. https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/ \
<a id="7">[7]</a>  GeeksforGeeks. (2024, January 3). Word Embedding using Word2Vec. GeeksforGeeks. https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

