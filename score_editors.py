import pandas as pd
import csv
from string import ascii_letters
from nltk.corpus import words, stopwords
from collections import Counter, defaultdict
import re, hashlib, os
from pandas.tseries.offsets import DateOffset
import numpy as np
import math
from tqdm import tqdm
import pdb

""" This script scores editors based on the proportion of edits that last
after a discussion ends

Input:
* CSV with talk page contributions, 1 per line
* Directory with article revision histories TSV files, 1 per row

Output:
* CSV of multiple CSVs with all article edits scored (edscores_path)
* CSV with all talk page contributions in a conversation for an editor, 1 per 
line, scored (outpath)

"""

""" I/O vars """
# input
infile = 'enwiki_talk/talk_filtered_2.csv'
diff_dir = 'enwiki_diffs/'
#ardict_path = '/home/michael/school/research/wp/ar/arabic-wordlist-1.6.txt'

# output
edscores_path = 'enwiki_edit_scores/'
outpath = 'enwiki_talk/enwiki_talk_scores.csv'

# settings
mult_files = True # whether or not want edscores to be printed to multiple files (if will be scoring lots of edits) or a single file


def dict_diff(orig, chg):
    """ Calculates diff between dictionary a and b based on keys from dict a
        Returns: (edit_score, #tokens changed in edit)
    """
    
    if len(orig) == 0: # no change in original except for stopwords
        return (1.0, 0)
    
    chg_abs_sum = 0 # relevant word changes
    orig_abs_sum = 0
    for k in orig:
        orig_abs_sum += abs(orig[k])
        if orig[k] * chg[k] <= 0: # signs differ
            chg_abs_sum += abs(chg[k])
            
    if orig_abs_sum == 0: 
        if chg_abs_sum == 0: # no sign difference, so revert successful
            return 1.0, abs(sum(orig.values()))
        else:
            pdb.set_trace()
            
    return max(1-chg_abs_sum/orig_abs_sum, 0.0), orig_abs_sum



def score_editors(stops, talk):
    """ Score editors
        Print out article edit file with score for each edit 

        Args:
            stops: list of stopwords
            talk: DataFrame of talk data
    """

    print("Scoring editors ...")

    # Load, initialize data
    assert os.path.exists(diff_dir)
    thread_durs = []

    # SCORE THREAD-WISE WINNERS FOR EACH DISCUSSION AND EDIT PARTICIPANT
    art_data = [] # Final article edit spreadsheet with scores, will be organized by article edit keys
    art_data_colnames = ['article', 'edit_timestamp', 
                        'additions', 'deletions',
                        'editor', 'edit_comment',
                         'edit_score', 'thread_title',
                        'comparison_timestamp',
                        'editor_thread_score']

    threads = sorted(list(set(zip(talk['article'], talk['thread']))))

    prev_art = ''

    # Get edits by thread editors in between initial revert and session end
    # session end: end of thread or last revert 7 days after last thread entry by thread participants

    no_article_ctr = 0
    for i, (art, thread) in enumerate(tqdm(threads[:30])):
        
        # Talk page participants
        talk_parts = set(talk[(talk['article']==art) & (talk['thread']==thread)]['username'].values)
        
        talk_rows = talk[(talk['article'] == art) 
                                & (talk['thread'] == thread)
                                ].loc[:, ['article', 'thread', 'timestamp']]
        thread_end = max(talk['timestamp'])
        thread_beg = min(talk['timestamp'])
        
        # Build edit history from beg to end of thread
        if prev_art != art:
            artfp = os.path.join(diff_dir, art.replace(' ', '_').replace('/', '_').lower() + '_diff.csv')
            if not os.path.exists(artfp):
                no_article_ctr += 1
                tqdm.write("{:d} No article".format(no_article_ctr))
                pdb.set_trace()
                continue
            diff_data = pd.read_csv(artfp, parse_dates=['timestamp'])
        sess_beg = thread_beg - DateOffset(days=1)
        sess_end = thread_end + DateOffset(days=1)

        # Find edits that are in same timeframe as thread and which thread participants make
        sess_edits = diff_data.loc[(diff_data['timestamp'] >= sess_beg) & (diff_data['timestamp'] < sess_end)
                             & diff_data['editor'].isin(talk_parts)] # could be intervening edits by non-talk participants
        sess_parts = set(sess_edits['editor'].tolist())
        
        if sess_edits.empty:
            #print('No diffs')
            continue
        sess_beg = min(sess_edits['timestamp'])
        sess_end = max(sess_edits['timestamp'])
        
        edscores = defaultdict(lambda: [0,0]) # [n_wds_successful, n_wds_changed]
        
        # Calculate success score for each edit compared with end revision
        new_rows = []
        for row in sess_edits.itertuples():
            row_timestamp = row[2]
            row_editor = row[5]
            edit_text = diff_data.loc[diff_data['timestamp']==row_timestamp]
            diffs = diff_data.loc[(diff_data['timestamp'] > row_timestamp) & 
                (diff_data['timestamp'] <= (sess_end + DateOffset(days=1)))] 
                # includes edits by non-talk participants, as should
            
            # Unigram counter for edit
            if (len(edit_text['deletions'].values) > 0) and (isinstance(edit_text['deletions'].values[0], str)):
                positive_dels = Counter([w.lower() for w in edit_text['deletions'].values[0].split() if w.lower() not in stops])
                edit_diff = Counter({key:-1*positive_dels[key] for key in positive_dels})
            else:
                edit_diff = Counter()
            if (len(edit_text['additions'].values) > 0) and (isinstance(edit_text['additions'].values[0], str)):
                adds = Counter([w.lower() for w in edit_text['additions'].values[0].split() if w.lower() not in stops])
            else:
                adds = Counter()
            edit_diff.update(adds)
            edit_diff = {k: edit_diff[k] for k in edit_diff if edit_diff[k] != 0}
            
            if diffs.empty: # No revisions after thread end
                edit_score = 1.0
                n_wds = abs(sum(edit_diff.values()))
                edscores[row_editor][0] += edit_score * n_wds
                edscores[row_editor][1] += n_wds
                    
            else: 
                
                # Unigram counter for revision diffs in window
                next_dels = ' '.join(d.lower() for d in diffs['deletions'].values.tolist() if isinstance(d, str))
                changes = Counter([w for w in next_dels.split() if w not in stops])
                changes = Counter({key:-1*changes[key] for key in changes})
                next_adds = ' '.join(a.lower() for a in diffs['additions'].values.tolist() if isinstance(a, str))
                next_addwds = Counter([w for w in next_adds.split() if w not in stops])

                changes.update(next_addwds)
                
                edit_score, n_wds = dict_diff(edit_diff, changes)
                edscores[row_editor][0] += edit_score * n_wds
                edscores[row_editor][1] += n_wds

            new_row = list(row[1:]) # don't want index
            new_row.append(edit_score)
            new_row.append(thread) # thread_title
            if diffs.empty:
                new_row.append(np.nan)
            else:
                new_row.append(max(diffs['timestamp'])) # comparison_timestamp
            #art_data.append(new_row)
            new_rows.append(new_row)

        prev_art = art
        
        # Calculate editor thread scores
#        art_data['editor_thread_score'] = pd.Series()
#        for ed in sess_parts:
#            sess_finalrows = art_data[(art_data['article']==art) & 
#                            (art_data['thread_title']==thread) &
#                            (art_data['editor']==ed)]
#            if edscores[ed][1] == 0: # editor whose only edit was the final one and was of no words
#                ed_threadscore = 1.0
#            else:
#                ed_threadscore = edscores[ed][0]/edscores[ed][1]
#            if ed_threadscore > 1.0 or ed_threadscore < 0.0:
#                debug_here()
#            for idx in sess_finalrows.index:
#                art_data.loc[idx, 'editor_thread_score'] = ed_threadscore
        
        for ed in sess_parts:
            sess_finalrows = [r for r in new_rows \
                                if r[0]==art and r[-2]==thread and \
                                r[4]==ed]
            if edscores[ed][1] == 0: # editor whose only edit was the final one and was of no words
                ed_threadscore = 1.0
            else:
                ed_threadscore = edscores[ed][0]/edscores[ed][1]
            if ed_threadscore > 1.0 or ed_threadscore < 0.0:
                pdb.set_trace()
            for i in range(len(sess_finalrows)):
                sess_finalrows[i].append(ed_threadscore)
        
            art_data.extend(sess_finalrows)

    # Sort art_data
    art_data = pd.DataFrame(art_data, columns=art_data_colnames)
    art_data.sort_values(['article', 'thread_title', 'edit_timestamp'], inplace=True)

    # Select columns
    cols = ['article', 'thread_title', 'edit_timestamp', 'editor', 'edit_comment',
           'edit_score', 'editor_thread_score', 'comparison_timestamp', 'additions', 'deletions']
    art_data = art_data[cols]

    # Remove reverts that don't have diffs (just removed Wikipedia metadata, for instance)
    mask = [isinstance(tup[0], str) or isinstance(tup[1], str) for tup in zip(art_data['additions'], art_data['deletions'])]
    art_data = art_data[mask]
        
    art_data.drop_duplicates(inplace=True)

    # Print edit scores
    if mult_files:
        art_data.to_csv(os.path.join(edscores_path, art.replace(' ', '_').replace('/', '_').lower() + '_edit_scores.csv'), index=False)
    else:
        art_data.to_csv(edscores_path, index=False)
    print('Wrote edscores to {:s}'.format(edscores_path))

def build_out(talk):
    """ Assemble editor scores with talk page discussion 
        Args:
            talk: DataFrame of talk
    """

    if not mult_files:
        art_data = pd.read_csv(edscores_path)

        edthreads = sorted(set((zip(art_data['article'], art_data['thread_title'], art_data['editor'], art_data['editor_thread_score']))))

    else:
        edthreads = []
        for fname in os.listdir(edscores_path):
            art_data = pd.read_csv(os.path.join(edscores_path, fname))
            ind_edthreads = set((zip(art_data['article'], art_data['thread_title'], art_data['editor'], art_data['editor_thread_score'])))
            edthreads.extend(list(ind_edthreads))

        edthreads = sorted(edthreads)

    # Build input corpora of editors' text + others' text, get labels
    # Assemble input text of just editors' text
    edtalk = defaultdict(str)
    n_edturns = {}
    n_otherturns = {}
    othertalk = defaultdict(str)

    # Filter out usernames from text
    usernames = set(talk['username'].tolist())

    # Check usernames against an English dictionary
    endict = set(words.words())
    #print("Username overlap with English dictionary: {:.1%}".format(
            #len(usernames.intersection(endict))/len(usernames)))

    # Check usernames against an Arabic dictionary
    #with open(ardict_path) as f:
        #ardict = set([w for w in f.read().splitlines()])
    #print("Username overlap with Arabic dictionary: {:.1%}".format(
    #        len(usernames.intersection(ardict))/len(usernames)))

    # Filter out dict words from usernames
    #usernames = set([u for u in usernames if not u in ardict and not u in endict])
    usernames = set([u for u in usernames if not u in endict])

    print("Assembling editor and other text ...")
    for i, el in enumerate(tqdm(edthreads)):
        rows = talk[(talk['article']==el[0]) &
                        (talk['thread']==el[1])]

        # Editor rows
        edrows = rows[rows['username']==el[2]]

        # Screen out usernames--TODO separate preprocess fn like for ar
        screened = []
        for t in edrows['text'].tolist():
            new_t = ' '.join([w for w in str(t).split() if not w in usernames and not '--' in w]) 
                # handle --username, doesn't handle multi-word usernames
            screened.append(new_t)
        edtalk[el] = ' '.join(screened) # need +=?

        n_edturns[el] = len(edrows)
        
        # Other text
        other_rows = rows[rows['username']!=el[2]]

        # Screen out usernames        
        screened = []
        for t in other_rows['text'].tolist():
            new_t = ' '.join([w for w in str(t).split() if not w in usernames and not '--' in w])
            screened.append(new_t)

        othertalk[el] = ' '.join(screened) # need +=?
        #othertalk[el] = ' '.join([str(t) for t in other_rows['post_text'].tolist()])
        n_otherturns[el] = len(other_rows)
        
    # Build one relevant dataframe
    #print("Assembling dataframe ...")
    outrows = []
    for i, el in enumerate(edthreads):
        outrows.append([el[0], el[1], el[2], edtalk[el], othertalk[el], n_edturns[el], n_otherturns[el],
                        el[3]])
        
    talk_scores = pd.DataFrame(outrows, columns=['article', 'thread_title', 'editor', 'editor_talk', 'other_talk',
                                  '#editor_turns', '#other_turns', 'editor_score'])
    talk_scores.to_csv(outpath, index=False)
    print('Wrote thread info with editor scores to {:s}'.format(outpath))



def main():

    talk = pd.read_csv(infile, parse_dates=['timestamp'])

    ## Filter out usernames from text
    #usernames = set(talk['username'].tolist())

    ## Check usernames against an English dictionary
    #endict = set(words.words())
    #print("Username overlap with English dictionary: {:.1%}".format(
    #        len(usernames.intersection(endict))/len(usernames)))

    ## Check usernames against an Arabic dictionary
    #with open(ardict_path) as f:
    #    ardict = set([w for w in f.read().splitlines()])
    #print("Username overlap with Arabic dictionary: {:.1%}".format(
    #        len(usernames.intersection(ardict))/len(usernames)))

    #stops = stopwords.words('arabic') 
    with open("stopwords_en.txt") as f:
        stops = f.read().splitlines()

    score_editors(stops, talk)
    build_out(talk)

if __name__ == '__main__':
    main()
