import numpy as np
from scipy import sparse
import scipy
import sys
import sqlite3


def get_features(database, feature_type='binary', max_songs=4000):
    """
        Read in songs lyrics and genres and extract certain features.

        Input: database: string, name of the database
               feature_type: string, optional types of features are
                           ['binary', 'frequency', 'tf-idf'] 
               max_songs: int, the number of songs extracted 
                        (used for debugging)
        Return: train_x: list of sparse matrix, each sparse matrix stores 
                        non-zero entries for the feature vector of a song in 
                        the training set
                train_y: list of ints, each integer corresponds to the genre 
                        of one song in the testing set.
                test_x: list of sparse matrix, each sparse matrix stores 
                        non-zero entries for the feature vector of a song in 
                        the training set
                test_y: list of ints, each integer corresponds to the genre 
                        of one song in the testing set.
                vocabulary: python dictionary, where keys are words and 
                            values are indices of words.
                wordList:  python list, could be considered as a reverse
                            dictionary of vocabulary
                genreDict: python dictionary, where keys are genres and 
                            values are indices of genres.
                genreList:  python list, could be considered as a reverse
                            dictionary of genreDict 
                            
    """
    print("Extracting training data and testing data. {} features are applied"\
            .format(feature_type))
    connection = sqlite3.connect(database)
    vocabulary, wordList = _get_vocabulary(connection)
    genreDict, genreList = _get_unique_genres(connection)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM shared_lyrics;")
    lyrics = cursor.fetchall()
    songs = {}
    print("---Reading lyrics and genres---")
    num_songs = 0
    for lyric in lyrics:
        if lyric[0] not in songs:
            if num_songs != 0 and num_songs % 1000 == 0:
                print("{} songs processed.".format(num_songs))
                if num_songs == max_songs:
                    break
            num_songs += 1
            songs[lyric[0]] = {}
            songs[lyric[0]]['lyrics'] = {'words':[], 'count':[]}
            sql_command = "SELECT * FROM shared_genres WHERE track_id='{}';"\
                    .format(lyric[0])
            cursor.execute(sql_command)
            labels = cursor.fetchone()
            songs[lyric[0]]['genre'] = genreDict[labels[1]]
            songs[lyric[0]]['is_test'] = labels[2]
        songs[lyric[0]]['lyrics']['words'].append(vocabulary[lyric[1]])
        if feature_type == 'binary':
            songs[lyric[0]]['lyrics']['count'].append(1)
        else:
            songs[lyric[0]]['lyrics']['count'].append(lyric[2])
    print(len(songs))
    if feature_type == 'tf-idf':
        df, idf = _get_idf(songs)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for trackid, song in songs.items():
        if feature_type == 'tf-idf':
            count = np.array(song['lyrics']['count']) * \
                    idf[song['lyrics']['words']]
        else:
            count = np.array(song['lyrics']['count'])
        words = np.array(song['lyrics']['words'])
        z = np.zeros(words.shape)
        # sparse matrix representation
        sm = sparse.coo_matrix((count, (z, words)), shape=(1, 5000))
        if song['is_test'] == 1:
            test_x.append(sm)
            test_y.append(song['genre'])
        else:
            train_x.append(sm)
            train_y.append(song['genre'])
    print(len(train_x), len(train_y), len(test_x), len(test_y))
    return train_x, train_y, test_x, test_y,\
        vocabulary, wordList, genreDict, genreList


def _get_idf(songs):
    print('---Calculating inverse term frequency---')
    df = np.zeros(5000)
    N = 0
    for trackid, song in songs.items():
        if song['is_test'] == 0:
            for word in song['lyrics']['words']:
                df[word] += 1
            N += 1
    return df, np.log(N/df)/np.log(2)



def _get_vocabulary(connection):
    """
        Get all unique words in the lyrics that will be ocnsidered.

        Input: connection:  a sql connectoin of mxm_dataset
        Return: vocabulary: python dictionary, where keys are words and 
                            values are indices of words.
                word_list:  python list, could be treated as a reverse
                            dictionary of vocabulary
    """
    print('---Getting vocabulary---')
    vocabulary = {}
    word_list = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM words;")
    res = cursor.fetchall()
    num_words = 0
    for word in res:
        vocabulary[word[0]] = num_words
        word_list.append(word[0])
        num_words += 1
    return vocabulary, word_list


def _get_unique_genres(connection):
    """
        Get all unique genres in the database.

        Input: connection:  a sql connectoin of mxm_dataset
        Return: genreDict:  python dictionary, where keys are genres and 
                            values are indices of genres.
                genreDict:  python list, could be treated as a reverse
                            dictionary of vocabulary
    """
    print('---Getting unique genres---')
    genreDict = {}
    genreList = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM shared_genres;")
    res = cursor.fetchall()
    num_genres = 0
    for genre in res:
        if genre[1] not in genreDict:
            genreDict[genre[1]] = num_genres
            genreList.append(genre[1])
            num_genres += 1
    return genreDict, genreList


if __name__ == '__main__':
    get_features(sys.argv[1], feature_type='tf-idf')
