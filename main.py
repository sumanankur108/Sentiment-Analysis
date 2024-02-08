import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import cmudict

# Download required NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('cmudict')

url = "https://insights.blackcoffer.com/rising-it-cities-and-its-impact-on-the-economy-environment-infrastructure-and-city-life-by-the-year-2040-2/"
response = requests.get(url)

if response.status_code == 200:
    webpage_content = response.text
else:
    print(f"Failed to fetch webpage. Status code: {response.status_code}")

soup = BeautifulSoup(webpage_content, 'html.parser')
text_content = soup.get_text()

# Tokenize text
words = word_tokenize(text_content)

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Calculate sentiment scores
positive_score = sum(sid.polarity_scores(word)['pos'] for word in words)
negative_score = sum(sid.polarity_scores(word)['neg'] for word in words)
polarity_score = sum(sid.polarity_scores(word)['compound'] for word in words)
subjectivity_score = sum(sid.polarity_scores(word)['pos'] + sid.polarity_scores(word)['neg'] for word in words)

# Calculate average sentence length
sentences = nltk.sent_tokenize(text_content)
average_sentence_length = len(words) / len(sentences)

# Calculate percentage of complex words
d = cmudict.dict()
complex_word_count = sum(1 for word in words if word.lower() in d)
percentage_complex_words = (complex_word_count / len(words)) * 100

# Calculate FOG Index
fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_word_count / len(words)))

# Calculate average number of words per sentence
average_words_per_sentence = len(words) / len(sentences)

# Calculate average syllables per word
syllable_count = sum(len(d[word.lower()][0]) for word in words if word.lower() in d)
average_syllables_per_word = syllable_count / len(words)

# Count personal pronouns
personal_pronouns = sum(1 for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'myself',
                                                               'you', 'your', 'yours', 'yourself', 'yourselves',
                                                               'he', 'him', 'his', 'himself',
                                                               'she', 'her', 'hers', 'herself',
                                                               'it', 'its', 'itself',
                                                               'we', 'us', 'our', 'ours', 'ourselves',
                                                               'they', 'them', 'their', 'theirs', 'themselves'])

# Calculate average word length
average_word_length = sum(len(word) for word in words) / len(words)

# Display results
print(f"Positive score: {positive_score}")
print(f"Negative score: {negative_score}")
print(f"Polarity score: {polarity_score}")
print(f"Subjectivity score: {subjectivity_score}")
print(f"Average Sentence Length: {average_sentence_length}")
print(f"Percentage of Complex Words: {percentage_complex_words}")
print(f"FOG Index: {fog_index}")
print(f"Average Number of Words per Sentence: {average_words_per_sentence}")
print(f"Complex Word Count: {complex_word_count}")
print(f"Word Count: {len(words)}")
print(f"Syllable per word: {average_syllables_per_word}")
print(f"Personal Pronouns: {personal_pronouns}")
print(f"Average Word Length: {average_word_length}")
