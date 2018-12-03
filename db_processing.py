"""
This file is used to rewrite the .db file to a txt file with much smaller
size so that the time consumed on reading in all data would be greately 
reduced.
"""
import numpy as np
from scipy import sparse
import scipy
import sys
import sqlite3

def write_new_database(songs):
    f = open('data/frequency_features.txt', 'w')
    for trackid, song in songs.items():
        line = '{}+{}+'.format(song['genre'], song['is_test'])
        words = song['lyrics']['words']
        count = song['lyrics']['count']
        wc_pairs = []
        for i in range(len(words)):
            wc_pairs.append('{},{}'.format(words[i],count[i]))
        f.write(line+' '.join(wc_pairs)+'\n')
    f.close()


def write_list(filename, d, idf=[]):
    f = open(filename, 'w')
    for key, index in d.items():
        if len(idf) > 0:
            f.write('{},{}\n'.format(key, idf[index]))
        else:
            f.write(key+'\n')
    f.close()


def db_processing(database):
    """
        Read in songs lyrics and genres and extract certain features.
        And write into smaller txt files includeing:
            (1) binary.txt
            (2) frequency.txt
            (3) tf-idf.txt

        Input: database: string, name of the database
    """
    print("---Processing database---")
    connection = sqlite3.connect(database)
    vocabulary = _get_vocabulary(connection)
    genreDict = _get_unique_genres(connection)
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
        songs[lyric[0]]['lyrics']['count'].append(lyric[2])
    df, idf = _get_idf(songs)
    write_new_database(songs)
    write_list('data/vocabulary.txt', vocabulary, idf)
    write_list('data/genresList.txt', genreDict)


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
    """
    print('---Getting vocabulary---')
    vocabulary = {}
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM words;")
    res = cursor.fetchall()
    num_words = 0
    for word in res:
        vocabulary[word[0]] = num_words
        num_words += 1
    return vocabulary


def _get_unique_genres(connection):
    """
        Get all unique genres in the database.

        Input: connection:  a sql connectoin of mxm_dataset
        Return: genreDict:  python dictionary, where keys are genres and 
                            values are indices of genres.
    """
    print('---Getting unique genres---')
    genreDict = {}
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM shared_genres;")
    res = cursor.fetchall()
    num_genres = 0
    for genre in res:
        if genre[1] not in genreDict:
            genreDict[genre[1]] = num_genres
            num_genres += 1
    return genreDict


if __name__ == '__main__':
    db_processing(sys.argv[1])
