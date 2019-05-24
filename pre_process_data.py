import pandas as pd
from nltk.corpus import words as nltk_words
import csv
import re


dataset_path = './data-all-annotations/trainingdata-all-annotations.txt'
cleaned_dataset_path = './new_dataset.csv'


def read_dataset_File(path):
    data = pd.read_csv(path, sep="\t", encoding='ISO-8859-1')
    return data


def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "./slang.txt"

        # File Access mode [Read Mode]
        with open(fileName, "r") as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9]+', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    return ' '.join(user_string)


pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")


# def split_hashtag(tag):
#     return


def get_hash_tag(x):
    words = x.split()
    hashtag_cleaned_tweet = ""
    for w in words:
        if w.startswith("#"):
            hashtag_cleaned_tweet = hashtag_cleaned_tweet + " " + " ".join(pattern.findall(w.replace("#", "")))
        else:
            hashtag_cleaned_tweet = hashtag_cleaned_tweet + " " + w
    # hashes = [split_hashtag(w.replace("#", "").lower()).join(" ") for w in words if w.startswith('#')]
    return hashtag_cleaned_tweet.lower()


def preprocess_data(data):
    """
     Encoding categorical data
    """
    stance_mapping = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
    data['Stance'] = data['Stance'].map(stance_mapping)

    """
    Replace abbreviations    
    """
    data['Tweet'] = data['Tweet'].apply(lambda x: translator(x))

    """
    Remove @mentions from the tweet data
    """
    data['Tweet'] = data['Tweet'].apply(lambda x: re.sub(r'@[A-Za-z0-9]+', '', x))

    """
    Remove URL links 
    """
    data['Tweet'] = data['Tweet'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))

    """
    Removing the hashtag and keeping the text
    """
    data['Tweet'] = data['Tweet'].apply(get_hash_tag)
    data['Tweet'] = data['Tweet'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

    """
    Removing extra whitespaces
    """

    data['Tweet'] = data['Tweet'].apply(lambda x: re.sub("\s+", " ", x))

    """
    Removing "sem st" from tweets
    """
    data['Tweet'] = data['Tweet'].apply(lambda x: x.replace("sem st", ""))

    print("Data cleaning process successfully completed...")
    return data.drop(columns=['ID', 'Opinion towards', 'Sentiment'])


# data_df = read_dataset_File('./data-all-annotations/trainingdata-all-annotations.txt')
# data_df = read_dataset_File('./data-all-annotations/testdata-taskA-all-annotations.txt')
data_df = read_dataset_File('./data-all-annotations/testdata-taskB-all-annotations.txt')
# data_df_1 = read_dataset_File('./data-all-annotations/trialdata-all-annotations.txt')
# data_df = pd.concat([data_df, data_df_1])
data_df_new = preprocess_data(data_df)
print(data_df_new)
data_df_new.to_csv('./test_taskB.csv')
