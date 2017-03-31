import pandas as pd
import csv
from string import ascii_letters
from nltk.corpus import words, stopwords
import nltk
from collections import Counter, defaultdict
import re, hashlib, os
from pandas.tseries.offsets import *
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
* CSV with all article edits scored (edscores_path)
* CSV with all talk page contributions in a conversation for an editor, 1 per 
line, scored (outpath)

"""

""" I/O vars """
# input
infile = '/home/michael/school/research/wp/ar/artalk_cspages_segmented.csv'
diff_dir = '/home/michael/school/research/wp/ar/ar_article_diffs/'
ardict_path = '/home/michael/school/research/wp/ar/arabic-wordlist-1.6.txt'

# output
edscores_path = '/home/michael/school/research/wp/ar/editor_thread_scores.csv'
outpath = '/home/michael/school/research/wp/ar/cs_talk_scores.csv'


""" Check for conversations with reverts """

def check_reverts():

    # Load talk data
    cs_threads = pd.read_csv(infile, parse_dates=['timestamp'])

    # Organize into threads
    threadtimes = {} # (article_title, thread_title): [beg_timestamp, end_timestamp, [users]]

    threads = set(zip(cs_threads['article_title'], cs_threads['thread_title']))

    # Collect thread info
    for art_name, t in threads:
        rows = cs_threads[(cs_threads['article_title']==art_name) & (cs_threads['thread_title']==t)]
        beg = min(rows['timestamp'])
        end = max(rows['timestamp'])
        users = set(rows['username'].tolist())
        threadtimes[(art_name, t)] = [beg, end, users]

    del cs_threads


    # Get threads with reverts in timeframe
    rev_dirpath = '/home/michael/school/research/wp/ar/ar_articles/with_reverts'
    n_matches = 0
    n_posts = 0
    match_threads = []

    for i, (art_name, t) in enumerate(threadtimes):
    #     print(art_name)
        
        beg = threadtimes[art_name, t][0] - DateOffset(days=7)
        end = threadtimes[art_name, t][1] + DateOffset(days=7)
        users = threadtimes[art_name, t][2]
        
        if art_name.startswith('Discussion:'):
            art_name = art_name[len('Discussion:'):]
        
        # Load revisions
        rev_path = os.path.join(rev_dirpath, art_name + '.csv')
        if os.path.isfile(rev_path):
            print(n_matches, end="/")
            print(i)
            chunks = pd.read_csv(rev_path, parse_dates=[1], header=None, iterator=True, chunksize=1000)
            revs = pd.concat(chunks, ignore_index=True)
            del chunks
            reverts = revs[revs[5] == 1]
        
            time_mask = [ts >= beg and ts <= end for ts in reverts[1]]
            selected_rows = reverts[time_mask]
            if not selected_rows.empty:
                selected_rows = selected_rows[selected_rows[3].isin(users)]
            if not selected_rows.empty:
                # Check if reverted also in discussion
                sel_posts = 0 # actually are article edits, not posts
                for ind in selected_rows.index: 
                    if revs.loc[ind-1, 3] in users:
                        sel_posts += 1
                n_posts += sel_posts
                if sel_posts > 0:
                    n_matches += 1
                    match_threads.append((art_name, t))
                    
            del revs

    print('Number of threads with reverts: {:d}'.format(n_matches))


    # Make csv of discussion posts
    cs_threads = pd.read_csv(infile, parse_dates=['timestamp'])
    rev_threads_inds = []
    for art_name, t in match_threads:
        rows = cs_threads[(cs_threads['article_title']==art_name) & (cs_threads['thread_title']==t)]
        rev_threads_inds.extend(rows.index)

    # Will want to remove duplicates, which come from the original cs_threads having 'Discussion' or not
    cs_revert_talk = cs_threads.loc[rev_threads_inds, :].sort_index()
    cs_revert_talk.to_csv(rv_outpath, index=None)
    print('Wrote revert info')



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

def score_editors():
    """ Score editors
        Print out article edit file with score for each edit 
    """

    print("Scoring editors ...")

    # Load, initialize data
    talk = pd.read_csv(infile, parse_dates=['timestamp'])
    assert os.path.exists(diff_dir)
    thread_durs = []

    # SCORE THREAD-WISE WINNERS FOR EACH DISCUSSION AND EDIT PARTICIPANT
    art_data = pd.DataFrame() # Final article edit spreadsheet with scores, will be organized by article edit keys
    art_data['edit_timestamp'] = pd.Series()
    art_data['edit_score'] = pd.Series()
    art_data['editor_thread_score'] = pd.Series()
    art_data['additions'] = pd.Series()
    art_data['deletions'] = pd.Series()
    art_data['comparison_timestamp'] = pd.Series()

    threads = sorted(list(set(zip(talk['article_title'], talk['thread_title']))))

    stops = stopwords.words('arabic') 
    prev_art = ''

    # Get edits by thread editors in between initial revert and session end
    # session end: end of thread or last revert 7 days after last thread entry by thread participants
    art_data = pd.DataFrame()

    no_article_ctr = 0
    #for i, (art, thread) in enumerate(threads):
    for i, (art, thread) in enumerate(tqdm(threads)):
        
        # Talk page participants
        talk_parts = set(talk[(talk['article_title']==art) & (talk['thread_title']==thread)]['username'].values)
        
        talk_rows = talk[(talk['article_title'] == art) 
                                & (talk['thread_title'] == thread)
                                ].loc[:, ['article_title', 'thread_title', 'post_timestamp']]
        thread_end = max(talk['timestamp'])
        thread_beg = min(talk['timestamp'])
        
        # Build edit history from beg to end of thread
        if prev_art != art:
            artfp = os.path.join(diff_dir, art.replace(' ', '_').replace('/', '_') + '_diff.csv')
            if not os.path.exists(artfp):
                no_article_ctr += 1
                tqdm.write("{:d} No article".format(no_article_ctr))
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
        for row in sess_edits.itertuples():
            edit_text = diff_data.loc[diff_data['timestamp']==row.timestamp]
            diffs = diff_data.loc[(diff_data['timestamp'] > row.timestamp) & 
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
                edscores[row.editor][0] += edit_score * n_wds
                edscores[row.editor][1] += n_wds
                    
            else: 
                
                # Unigram counter for revision diffs in window
                next_dels = ' '.join(d.lower() for d in diffs['deletions'].values.tolist() if isinstance(d, str))
                changes = Counter([w for w in next_dels.split() if w not in stops])
                changes = Counter({key:-1*changes[key] for key in changes})
                next_adds = ' '.join(a.lower() for a in diffs['additions'].values.tolist() if isinstance(a, str))
                next_addwds = Counter([w for w in next_adds.split() if w not in stops])

                changes.update(next_addwds)
                
                edit_score, n_wds = dict_diff(edit_diff, changes)
                edscores[row.editor][0] += edit_score * n_wds
                edscores[row.editor][1] += n_wds

            new_row = pd.DataFrame([row]) # SLOW, OPTIMIZE
            new_row['edit_score'] = edit_score
    #             new_row['winner'] = winner
            new_row['thread_title'] = thread
            if not diffs.empty:
                new_row['comparison_timestamp'] = max(diffs['timestamp'])
            new_row.rename(columns={'timestamp': 'edit_timestamp', 'article_name': 'article'}, inplace=True)
            new_row.drop('Index', axis=1, inplace=True)
            art_data = art_data.append(new_row, ignore_index=True)
                
        art_data.reset_index(drop=True)
        prev_art = art
        
        # Calculate editor thread scores
        for ed in sess_parts:
            sess_finalrows = art_data[(art_data['article']==art) & 
                            (art_data['thread_title']==thread) &
                            (art_data['editor']==ed)]
            if edscores[ed][1] == 0: # editor whose only edit was the final one and was of no words
                ed_threadscore = 1.0
            else:
                ed_threadscore = edscores[ed][0]/edscores[ed][1]
            if ed_threadscore > 1.0 or ed_threadscore < 0.0:
                debug_here()
            for idx in sess_finalrows.index:
                art_data.loc[idx, 'editor_thread_score'] = ed_threadscore
        
    # Sort art_data
    art_data.sort_values(['article', 'thread_title', 'edit_timestamp'], inplace=True)

    # Select columns
    cols = ['article', 'thread_title', 'edit_timestamp', 'editor', 'edit_comment',
           'edit_score', 'editor_thread_score', 'comparison_timestamp', 'additions', 'deletions']
    art_data = art_data[cols]

    # Remove reverts that don't have diffs (just removed Wikipedia metadata, for instance)
    mask = [isinstance(tup[0], str) or isinstance(tup[1], str) for tup in zip(art_data['additions'], art_data['deletions'])]
    art_data = art_data[mask]
        
    art_data.drop_duplicates(inplace=True)
    art_data.to_csv(edscores_path, index=False)
    len(art_data)
    print('Wrote edscores to {:s}'.format(edscores_path))

def build_out():
    """ Assemble editor scores with talk page discussion """
    art_data = pd.read_csv(edscores_path)
    talk = pd.read_csv(infile, parse_dates=['timestamp'])

    # Build input corpora of editors' text + others' text, get labels
    edthreads = sorted(set((zip(art_data['article'], art_data['thread_title'], art_data['editor'], art_data['editor_thread_score']))))
    edthreads

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
    with open(ardict_path) as f:
        ardict = set([w for w in f.read().splitlines()])
    #print("Username overlap with Arabic dictionary: {:.1%}".format(
    #        len(usernames.intersection(ardict))/len(usernames)))

    # Filter out dict words from usernames
    usernames = set([u for u in usernames if not u in ardict and not u in endict])

    print("Assembling editor and other text ...")
    for i, el in enumerate(tqdm(edthreads)):
        rows = talk[(talk['article_title']==el[0]) &
                        (talk['thread_title']==el[1])]

        # Editor rows
        edrows = rows[rows['username']==el[2]]

        # Screen out usernames        
        screened = []
        for t in edrows['post_text'].tolist():
            new_t = ' '.join([w for w in str(t).split() if not w in usernames and not '--' in w]) 
                # handle --username, doesn't handle multi-word usernames
            screened.append(new_t)
        edtalk[el] = ' '.join(screened) # need +=?

        n_edturns[el] = len(edrows)
        
        # Other text
        other_rows = rows[rows['username']!=el[2]]

        # Screen out usernames        
        screened = []
        for t in other_rows['post_text'].tolist():
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

    #talk = pd.read_csv(infile, parse_dates=['timestamp'])

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

    #score_editors()
    build_out()

if __name__ == '__main__':
    main()
