import pandas as pd
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.probability import FreqDist

# Download required NLTK resources
resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words"
]

for resource in resources:
    nltk.download(resource, quiet=True)

# 1. Load the dataset
df = pd.read_csv("Project5_NLP_Text.csv")

print("=" * 60)
print("          NLP TEXT ANALYZER USING NLTK")
print("          K. Hansika | B230258 | Project 5")
print("=" * 60)

print("\n--- DATASET ---")
print(df[["Document_ID", "Category"]])
print("\nTotal number of documents:", len(df))

# Lists to store results from all documents
all_sentences = []
all_words = []
all_clean_words = []
all_entities = []

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Analyze every document
for _, row in df.iterrows():
    document_id = row["Document_ID"]
    text = str(row["Text"])

    print(f"\n{'=' * 50}")
    print("DOCUMENT:", document_id)
    print("CATEGORY:", row["Category"])
    print("TEXT:", text)

    # 2. Sentence tokenization
    sentences = sent_tokenize(text)
    all_sentences.extend(sentences)

    print("\nSentences:")
    print(sentences)

    # 3. Word tokenization
    words = word_tokenize(text)
    all_words.extend(words)

    print("\nWord tokens:")
    print(words)

    # 4. Remove English stop words and punctuation
    clean_words = [
        word.lower()
        for word in words
        if word.lower() not in stop_words
        and not all(char in string.punctuation for char in word)
    ]

    all_clean_words.extend(clean_words)

    print("\nWords after stop-word removal:")
    print(clean_words)

    # 5. Stemming
    print("\nStemming:")
    for word in clean_words:
        print(f"{word} -> {stemmer.stem(word)}")

    # 5. Lemmatization
    print("\nLemmatization:")
    for word in clean_words:
        print(f"{word} -> {lemmatizer.lemmatize(word)}")

    # 6. POS tagging
    tagged_words = pos_tag(words)

    print("\nPOS tagging:")
    print(tagged_words)

    # 7. Named Entity Recognition
    entities = ne_chunk(tagged_words)

    print("\nNamed entities:")
    document_entities = []

    for entity in entities:
        if hasattr(entity, "label"):
            name = " ".join(word for word, tag in entity.leaves())
            label = entity.label()
            document_entities.append((name, label))
            all_entities.append((name, label))

    if document_entities:
        for name, label in document_entities:
            print(f"{name} -> {label}")
    else:
        print("No named entities identified.")

# 8. Calculate dataset-wide statistics
print("\n" + "=" * 60)
print("           OVERALL NLP STATISTICS")
print("=" * 60)

print("Total number of documents:", len(df))
print("Total number of sentences:", len(all_sentences))
print("Total number of word tokens:", len(all_words))
print("Number of unique words:", len(set(all_clean_words)))

# Calculate top 10 frequently occurring words
frequency = FreqDist(all_clean_words)

print("\nTop 10 most frequently occurring words:")
for word, count in frequency.most_common(10):
    print(f"{word}: {count}")

# 9. Display named entities identified
print("\n--- NAMED ENTITIES IDENTIFIED ---")

if all_entities:
    for name, label in sorted(set(all_entities)):
        print(f"{name} -> {label}")
else:
    print("No named entities identified.")

# Category-wise summary
print("\n--- CATEGORY-WISE SUMMARY ---")
category_summary = df.groupby("Category")["Document_ID"].count()

for category, count in category_summary.items():
    print(f"{category}: {count} documents")

print("\n--- ANALYSIS COMPLETED ---")