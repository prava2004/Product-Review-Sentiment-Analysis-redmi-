import pandas as pd
import re                    # for regular expressions (nlp,text cleaning)
import nltk                  # for NLP 
from sklearn.feature_extraction.text import TfidfVectorizer # to convert text to no for machine to understand
from sklearn.model_selection import train_test_split # To split the data
from sklearn.linear_model import LogisticRegression # To build a ml mlodel
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix # To evaluate Model
import seaborn as sns  #great for making beautiful statistical plots (like heatmaps, bar charts, etc.)
import matplotlib.pyplot as plt  #basic plotting library, seaborn actually builds on top of it


file_path=r"C:\Users\Pinnamaneni\Downloads\archive (1)\redmi6.csv"
df = pd.read_csv(file_path,encoding = 'latin1')
print(df.head())
print(df.tail())                                                    

print("shape:",df.shape)
print(df.columns)
print(df.isnull().sum())

print(df['Comments'].head(10))
df['Lower_case_Comments'] = df['Comments'].str.lower()
print("Lower_case_comments:\n",df['Lower_case_Comments'].head(10))

#here comes NLP(reg exp part)
def clean_text(text):
    text = re.sub(r'[^a-z0-9\s]','',text)#Keep only lowercase letters (a-z), numbers (0-9), and spaces
    return text
df['Clean_text']=df['Lower_case_Comments'].apply(clean_text)
print("Clean_text:",df['Clean_text'])
 
# Download and load stopwords (common English words like "is", "the", "and", etc.)
nltk.download('stopwords')
from nltk.corpus import stopwords

Stop_words = set(stopwords.words('english'))
def rem_stopwords(text):
    words = text.split()
    words = [x for x in words if x not in Stop_words] #It checks every word (i) in that list and keeps it only if it’s not a stopword.
    return ' '.join(words)
df['new_words']=df['Clean_text'].apply(rem_stopwords)
print(df['new_words'])

def get_sentiment(x): # (or both are same)-> df['Sentiment'] = df['Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')

    if x > 3:
        return 'Positive'
    else:
        return 'Negative'

df['Rating_Num'] = df['Rating'].str.extract(r'(\d+\.?\d*)').astype(float) #Extract the first number (actual rating) from strings like "4.0 out of 5 stars"

df['Sentiment'] = df['Rating_Num'].apply(get_sentiment)
print(df['Sentiment'])
print(df[['Rating', 'Rating_Num', 'Sentiment']].head(10))

# scikit learn comes used cuz ml cant understand text we need to convery text to no(tfid using here like bag of words importance)
numbers = TfidfVectorizer(max_features=500) #feature_extraction
x = numbers.fit_transform(df['Clean_text'])
print(x.shape)

# fit() → learns what words exist and how frequent they are (builds a vocabulary).
# transform() → converts each text (review) into a numeric vector.
# fit_transform() does both steps in one line.

# split the data into train test(80and 20%)
y = df['Sentiment']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# build ml model
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# x_train → 80% of reviews (features) → used to teach the model
# y_train → 80% of labels → tells the model what the “correct answer” is
# x_test → 20% of reviews → unseen by the model → used to check if it learned properly
# y_test → 20% of actual labels → we compare with predictions
# The model now looks at x_test (reviews it hasn’t seen) and guesses the sentiment(y_pred = predicted sentiment)

#Evaluate model
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classificataion_Report",classification_report(y_test,y_pred))


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# For bar graph
pred_counts = pd.Series(y_pred).value_counts()
plt.bar(pred_counts.index, pred_counts.values, color=['red','green'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Predicted Sentiment Counts')
plt.show()

# bar again
sns.countplot(x='Sentiment', data=df)
plt.title("Count of Positive vs Negative Feedback")
plt.show()
