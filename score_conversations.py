
# coding: utf-8

# In[1]:

import pandas as pd
import pickle
import csv
import random
from tqdm import tqdm_notebook as tqdm
from collections import Counter, defaultdict
from pandas.tseries.offsets import DateOffset
from scipy.stats import pearsonr
import numpy as np
import re
import os


# # Calculate outcome measure: number of edits and diff sizes during and after a discussion

# In[6]:

# samp = pd.read_csv('/home/michael/school/research/wp/enwiki_qualitative_analysis.csv', parse_dates=['timestamp'])
# talk = samp
talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/all_talk.csv', 
                   names=['id', 'article', 'thread', 'timestamp', 'editor', 'thread_type', 'text'],
                  parse_dates=['timestamp'])
talk.drop(['id'], axis=1, inplace=True)

keys = set(zip(talk['article'], talk['thread']))
talk


# In[8]:

# Filter all_talk
conv_sizes = talk.groupby(['article', 'thread']).size()
conv_sizes.name = 'conv_size'
talk = talk.join(conv_sizes, on=['article', 'thread'])
talk


# In[9]:

talk.sort_values(['article', 'thread'], inplace=True)
talk


# In[10]:

print(len(talk))
talk = talk[talk['conv_size']>=2]
print(len(talk))


# In[11]:

talk.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/talk_conversational_outcomes.csv', index=False)


# In[12]:

def diff_filename(art):
    return "{}_diff.csv".format(art.lower().replace(' ', '_').replace('/', '_'))


# In[51]:

# Calculate n_edits
diff_dirpath = '../enwiki_diffs/'
# sess_outcome = {'n_article_edits': {}, 'diff_size': {}}
sess_outcome = {}
zero_ctr = 0

for a,t in tqdm(sorted(keys)):
    fullpath = os.path.join(diff_dirpath, diff_filename(a))
#     print(fullpath)
    diff = pd.read_csv(fullpath, parse_dates=['timestamp'])

    sess_talk = talk[(talk['article']==a) & (talk['thread']==t)]
    sess_ts = {}
    sess_ts['beg'] = min(sess_talk['timestamp'])
    sess_ts['end'] = max(sess_talk['timestamp']) + DateOffset(days=7)
    talk_participants = set(sess_talk['editor'].unique())

    sess_diffs = diff[(diff['article_name']==a) &                       (diff['editor'].isin(talk_participants)) &                      (diff['timestamp'] > sess_ts['beg']) & (diff['timestamp'] < sess_ts['end'])]
    if len(sess_diffs)==0:
        zero_ctr += 1

    n_article_edits = len(sess_diffs)
    diff_size = sum(sess_diffs['additions'].map(lambda x: len(str(x).split()))) +                         sum(sess_diffs['deletions'].map(lambda x: len(str(x).split())))
    
    sess_outcome[a,t] = [n_article_edits, diff_size]

print(zero_ctr)


# In[56]:

# Create sess_outcome dataframe, merge in
outcome_colnames = ['n_article_edits', 'diff_size']
sess_outcome_df = pd.DataFrame([[k[0], k[1]] + val for k,val in sess_outcome.items()], columns=['article', 'thread'] + outcome_colnames)
sess_outcome_df


# In[57]:

# merged = pd.merge(samp, sess_outcome_df)
merged = pd.merge(new_df, sess_outcome_df)
merged


# In[23]:

conv_sizes


# In[24]:

merged.columns


# In[58]:

# new_df.to_csv('../enwiki_sample_outcomes.csv', index=False)
merged.to_csv('../enwiki_sample_outcomes_7days.csv', index=False)


# In[5]:

# Write out sample article names
samp_arts = sorted(samp['article'].unique().tolist())
print(len(samp_arts))

with open('../enwiki_sample_articles.txt', 'w') as out:
    for art in samp_arts:
        out.write("{}_diff.csv\n".format(art.lower().replace(' ', '_').replace('/', '_')))


# In[37]:

merged


# ## Correlations

# In[38]:

pearsonr(merged['conv_size'], merged['n_article_edits'])


# In[39]:

pearsonr(merged['n_article_edits'], merged['diff_size'])


# ## Face validity check

# In[40]:

# Face validity for n_edits
n_edits_sorted = merged.sort_values(['n_article_edits'], ascending=False)
n_edits_sorted


# In[45]:

# Look into 0s
zeros = merged[merged['n_article_edits']==0]
print(len(set(zip(zeros['article'], zeros['thread']))))
zeros


# In[44]:

for a,t in sorted(set(zip(zeros['article'], zeros['thread']))):
    print(a)
    print(t)
    fullpath = os.path.join(diff_dirpath, diff_filename(a))
    diff = pd.read_csv(fullpath, parse_dates=['timestamp'])

    sess_talk = talk[(talk['article']==a) & (talk['thread']==t)]
    sess_ts = {}
    sess_ts['beg'] = min(sess_talk['timestamp'])
    sess_ts['end'] = max(sess_talk['timestamp']) + DateOffset(days=1)
    talk_participants = set(sess_talk['editor'].unique())
    print(sess_ts)

    sess_diffs = diff[(diff['article_name']==a) &                       (diff['editor'].isin(talk_participants)) &                      (diff['timestamp'] > sess_ts['beg']) & (diff['timestamp'] < sess_ts['end'])]
    print(len(sess_diffs))

    sess_outcome[a,t] = sum(sess_diffs['additions'].map(lambda x: len(str(x).split()))) +                         sum(sess_diffs['deletions'].map(lambda x: len(str(x).split())))
    print()


# # Try to predict editor score based on talk by myself--see what's going on in conversations

# In[3]:

talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/all_talk.csv', header=None, names=['id','article', 'thread', 'timestamp', 'editor', 'thread_starter', 'text'])
print(len(talk))

data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_scores_roles.csv')

merged = pd.merge(talk, data, how='left')
print(len(merged))
merged.columns


# In[4]:

merged = merged.loc[:, ['article', 'thread', 'editor', 'timestamp', 'thread_starter', 'text', 'editor_score']]
scored = merged[np.isfinite(merged['editor_score'])]

threads = set(zip(scored['article'], scored['thread']))
s = random.sample(threads, 100)

mask = list(map(lambda tup: tup in s, zip(merged['article'], merged['thread'])))
sort = merged[mask].sort_values(['article', 'thread', 'timestamp'], inplace=False)
sort


# In[5]:

sort.to_csv('/home/michael/school/research/wp/enwiki_qualitative_analysis.csv', index=False)


# In[30]:

pd.set_option('display.max_colwidth', 999)


# # Check into #editors discrepancy

# In[ ]:

data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_scores_roles.csv')


# In[3]:

data[(data['article']=='Western Sahara conflict') & (data['thread']=='Merge')]


# In[4]:

data[(data['article']=='Western Sahara conflict')]


# ## Recalculate #editors cols

# In[6]:

gped = data.groupby(['article', 'thread'])
gped.size()[('Western Sahara conflict', 'Merge')]


# In[7]:

convo_size = pd.DataFrame([(row[0][0], row[0][1], row[1]) for row in gped.size().iteritems()], columns=['article', 'thread', '#editors'])
len(convo_size)


# In[9]:

data.drop('#editors', axis=1, inplace=True)


# In[10]:

merged = pd.merge(data, convo_size)
print(len(merged))
merged.columns


# In[11]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_scores_roles.csv', index=False)


# In[12]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
feats.drop('#editors', axis=1, inplace=True)
len(feats)


# In[13]:

merged = pd.merge(feats, convo_size)
print(len(merged))
merged.columns


# In[14]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Examine role distributions in conversations

# In[2]:

data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_scores_roles.csv')
print(len(data))
data.columns


# In[3]:

data_roles = data[np.isfinite(data['TargetRole_0'])]
print(len(data_roles))
role_threads = list(set(zip(data_roles['article'], data_roles['thread'])))
# n = 0
# t = role_threads[n]
# t


# In[4]:

pd.set_option('display.max_colwidth', 999)


# In[7]:

# Split based on conversation size
threads_bysize = {}

for i in range(2,5):
    sel = data_roles[data_roles['#editors']==i]
    threads_bysize[i] = list(set(zip(sel['article'], sel['thread'])))
    
sel = data_roles[data_roles['#editors']>=5]
threads_bysize[5] = list(set(zip(sel['article'], sel['thread'])))

df_samples = {}

sel_cols = ['article', 'thread', 'editor', '#editors', 'editor_talk'] + ['TargetRole_{}'.format(i) for i in range(5)]
#     + ['NontargetRole_{}'.format(i) for i in range(5)]
for i in range(2,6):
    s = random.sample(threads_bysize[i], 10)
    mask = list(map(lambda tup: tup in s, zip(data['article'], data['thread'])))
    df_samples[i] = data[mask].loc[:, sel_cols]


# In[8]:

df_samples[2]


# In[10]:

df_samples[4]


# In[16]:

sel_cols = ['article', 'thread', 'editor', '#editors', 'editor_talk'] + ['TargetRole_{}'.format(i) for i in range(5)] +     ['NontargetRole_{}'.format(i) for i in range(5)]
selected = data[(data['article']==t[0]) & (data['thread']==t[1])].loc[:, sel_cols]
selected


# # Look into some dev and test rows not having role distros

# In[16]:

devtest = data[data['split'].isin(['dev', 'test'])]
print(len(devtest))
devtest_threads = list(set(zip(devtest['article'], devtest['thread'])))
n = 0
t = devtest_threads[n]
t


# In[13]:

sel_cols = ['split', 'article', 'thread', 'editor', '#editors', 'editor_talk'] + ['TargetRole_{}'.format(i) for i in range(5)] +     ['NontargetRole_{}'.format(i) for i in range(5)]
selected = data[(data['article']==t[0]) & (data['thread']==t[1])].loc[:, sel_cols]
selected


# In[15]:

devtest['TargetRole_0'].count()


# In[28]:

eg = dev.loc[0]
eg


# In[31]:

# devtest[(devtest['article']==eg['article']) & (devtest['thread']==eg['thread']) & (devtest['editor']==eg['editor'])]
# devtest[(devtest['article']==eg['article']) & (devtest['thread']==eg['thread'])]
devtest[(devtest['article']==eg['article'])]


# In[35]:

# Make sure all dev keys present in combo dataset
combo_dev = devtest[devtest['split']=='dev']
combined_keys = sorted(set(zip(combo_dev['article'], combo_dev['thread'], combo_dev['editor'])))
print(len(combined_keys))

dev_keys = sorted(set(zip(dev['article'], dev['thread'], dev['editor'])))
print(len(dev_keys))
# extra_dev = dev_keys.intersection(combined_keys)
# all(k in combined_keys  dev_keys)

print(combined_keys[:10])
print(dev_keys[:10])


# In[42]:

# Make sure all test keys present in combo dataset
combo_test = devtest[devtest['split']=='test']
# combined_keys = sorted(set(zip(combo_test['article'], combo_test['thread'], combo_test['editor'])))
combined_keys = set(zip(combo_test['article'], combo_test['thread'], combo_test['editor']))
print(len(combined_keys))

# test_keys = sorted(set(zip(test['article'], test['thread'], test['editor'])))
test_keys = set(zip(test['article'], test['thread'], test['editor']))
print(len(test_keys))
inter = test_keys.intersection(combined_keys)
# all(k in combined_keys  dev_keys)
print(len(inter))

# print(combined_keys[:10])
# print(dev_keys[:10])


# In[36]:

devtest


# In[37]:

data[data['article']=='1-800-FREE-411']


# In[39]:

# Double-check fold info
fold_info = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/fold_info.tsv', sep='\t', header=None)
fold_info[fold_info[0]=='1-800-FREE-411']


# In[40]:

dev['fold'].unique()


# In[41]:

test['fold'].unique()


# In[33]:

# Make sure all dev keys present in combo dataset
combined_keys = set(zip(devtest['article'], devtest['thread'], devtest['editor']))
len(combined_keys)

test_keys = set(zip(test['article'], test['thread'], test['editor']))
extra_dev = dev_keys.intersection(combined_keys)
# all(k in combined_keys  dev_keys)
len(extra_dev)


# In[17]:

dev = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/role_weights/PRPM_K5-dev_all.csv')
print(len(dev))


# In[22]:

test = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/role_weights/PRPM_K5-test_all.csv')
print(len(test))


# In[23]:

len(test) + len(dev)


# In[24]:

dev['TargetRole_0'].count()


# In[25]:

test['TargetRole_0'].count()


# In[26]:

data['TargetRole_0'].count()


# In[44]:

data_roles = data[np.isfinite(data['TargetRole_0'])]
data_roles['fold'].unique()


# # Normalize features for Carolyn

# ## Merge in PRPM 5-role weights

# In[2]:

dev = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/role_weights/PRPM_K5-dev_all.csv')
print(len(dev))
test = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/role_weights/PRPM_K5-test_all.csv')
print(len(test))


# In[3]:

merged = pd.merge(dev, test, how='outer')
print(len(merged))


# In[4]:

merged.columns.tolist()


# In[5]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')


# In[8]:

allmerged = pd.merge(feats, merged, how='left')
len(allmerged)


# In[10]:

selected = allmerged[feats.columns.tolist() + ['TargetRole_{}'.format(i) for i in range(5)] + ['NontargetRole_{}'.format(i) for i in range(5)]]
selected.columns


# In[11]:

cols = selected.columns.tolist()
new_cols = cols[:cols.index('editor_score')] + cols[cols.index('editor_score')+1:] + ['editor_score']
new_cols


# In[12]:

selected = selected[new_cols]


# In[13]:

selected['TargetRole_2'].count()


# In[14]:

selected.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_scores_roles.csv', index=False)


# ## Rename allfeats columns, create normalized features

# In[2]:

feats.columns


# In[4]:

feats['talk_len'] = feats['editor_talk'].map(lambda x: len(x.split()))
feats['talk_len']


# In[5]:

cols = list(feats.columns)
count_featnames = cols[cols.index('ART::DEF'):cols.index('URL')+1]
count_featnames


# In[6]:

for col in count_featnames:
    feats['{}_norm'.format(col)] = feats[col]/feats['talk_len']


# In[8]:

feats.columns


# In[9]:

feats['ed_old_pred'].count()


# In[10]:

feats['ed_ftopic_pred'].count()


# In[11]:

feats['all_nonbow_pred'].count()


# In[12]:

feats['all_pred'].count()


# In[13]:

for col in ['ed_old_pred', 'ed_ftopic_pred', 'all_nonbow_pred', 'all_pred']:
    feats.drop(col, axis=1, inplace=True)


# In[14]:

feats.columns


# In[15]:

feats['ed_pred'].count()


# In[17]:

feats.drop('ed_pred', axis=1, inplace=True)


# In[18]:

feats.drop('ed_dialog_act_pred', axis=1, inplace=True)


# In[20]:

feats.drop('all_dialog_act_pred', axis=1, inplace=True)


# In[16]:

feats['ed_dialog_act_pred'].count()


# In[21]:

feats.columns


# In[24]:

cols = feats.columns.tolist()
new_cols = ['split', 'fold'] + cols[:cols.index('editor')+1] + ['#editors'] +             cols[cols.index('editor_talk'):cols.index('#other_turns')+1] +             cols[cols.index('ftopic0'):cols.index('other_ftopic19')+1] +             cols[cols.index('ART::DEF'):cols.index('URL')+1] +             cols[cols.index('ART::DEF_norm'):] + ['talk_len', 'editor_score'] 
new_cols


# In[25]:

print(len(cols))
print(len(new_cols))


# In[26]:

feats = feats[new_cols]
feats.columns


# In[27]:

feats.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Get preds from Diyi's model, split by conversation size

# In[15]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')

n_roles = 5
with open('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/diyi_model/ijcnlp_{0}/result/preds_{0}.txt'.format(n_roles)) as f:
    lines = [x for x in f.read().splitlines()]
    preds = [float(x.split('\t')[0]) for x in lines]
    scores = [float(x.split('\t')[1]) for x in lines]
    
print(len(preds))

test = feats[feats['split']=='test']
test.columns


# In[16]:

# Get Diyi alignment of test data
summary = '/home/michael/school/research/wp/wikipedia/data/talk/enwiki/diyi_model/ijcnlp_{}/summary_test.txt'.format(n_roles)
with open(summary) as f:
    lines = f.read().splitlines()[4:]
    
# parse lines
conv_ids = [line.split('\t') for line in lines]

diyi_scores = pd.DataFrame(conv_ids, columns=['article', 'thread', 'editor'])
diyi_scores['diyi_edscore'] = scores
diyi_scores['diyi_{}_pred'.format(n_roles)] = preds
diyi_scores


# In[17]:

merged = pd.merge(feats, diyi_scores)
print(len(merged))
print(len(test))
union = pd.merge(test, diyi_scores, how='outer')
print(len(union))
nomatch = union[union.isnull()['ftopic0']]
print(len(nomatch))
print(len(union) - len(test)) # good match


# In[18]:

# Check if RMSE correct (aligned predictions)--seems to be off for some reaso
np.sqrt(np.mean((merged['diyi_{}_pred'.format(n_roles)] - merged['editor_score']) ** 2))


# In[19]:

# Split by conversation size
for i in range(2,5):
    print(i, end=': ')
    selected = merged[merged['#editors']==i]
    print(np.sqrt(np.mean((selected['diyi_{}_pred'.format(n_roles)] - selected['editor_score']) ** 2)))
    
print("5+: ", end="")
selected = merged[merged['#editors']>=5]
print(np.sqrt(np.mean((selected['diyi_{}_pred'.format(n_roles)] - selected['editor_score']) ** 2)))


# ## Investigate misalignment, old prediction merging

# In[11]:

# Check if scores aligned
test['diyi_edscore']==test['editor_score']


# In[6]:

# Check if RMSE correct (aligned predictions)--seems to be off for some reason
np.sqrt(np.mean((test['diyi_{}_pred'.format(n_roles)] - test['editor_score']) ** 2))


# In[53]:

merged = pd.merge(feats, test, how='left')
print(len(merged))
merged.columns


# In[67]:

merged['diyi_2_pred'].count()


# In[73]:

# Split by conversation size

for i in range(2,5):
    print(i, end=': ')
    selected = test[test['#editors']==i]
    print(np.sqrt(np.mean((selected['diyi_2_pred'] - selected['editor_score']) ** 2)))
    
print("5: ", end="")
selected = test[test['#editors']>=5]
print(np.sqrt(np.mean((selected['diyi_2_pred'] - selected['editor_score']) ** 2)))


# In[34]:

# Add in test, train, dev
fold_mapping = {0: 'test', 
                1: 'test',
                2: 'dev',
                3: 'dev',
                4: 'train',
                5: 'train',
                6: 'train',
                7: 'train',
                8: 'train',
                9: 'train',
               }

feats['split']=[fold_mapping[n] for n in feats['fold']]
feats['split']


# In[35]:

feats.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Merge in role weight data

# In[16]:

settings = ['dev', 'test']
n_roles = range(2,6)
role_wts = {c: {} for c in n_roles}

for c in n_roles:
    for s in settings:
        role_wts[c][s] = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/role_weights/PRPM_K{}-{}_theta.csv'.format(c,s))
        role_wts[c][s].rename(columns={'page': 'article', 'user': 'editor'}, inplace=True)


# In[17]:

role_wts[2]['test']


# In[12]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
print(len(feats))
feats.columns


# In[13]:

# Add in test, train, dev
fold_mapping = {0: 'test', 
                1: 'test',
                2: 'dev',
                3: 'dev',
                4: 'train',
                5: 'train',
                6: 'train',
                7: 'train',
                8: 'train',
                9: 'train',
               }

feats['split']=[fold_mapping[n] for n in feats['fold']]
feats['split']


# In[14]:

role_wts.keys()


# In[23]:

merged =  {n: {} for n in n_roles}

for c in n_roles:
    for s in settings:
        merged[c][s] = pd.merge(feats, role_wts[c][s], how='inner')
        print(len(merged[c][s]))


# In[31]:

for c in n_roles:
    for s in settings:
        print(len(role_wts[c][s]))


# In[29]:

feats[np.isfinite(feats['ed_ftopic_pred'])]


# In[24]:

merged[3]['test']


# In[30]:

for c in n_roles:
    for s in settings:
        merged[c][s].to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/PRPM_K{}-{}_all.csv'.format(c, s), index=False)


# # Explicitly divide train, dev, test data sets

# In[2]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
len(feats)


# In[3]:

test = feats[feats['fold'].isin([0,1])]
print(len(test))

dev = feats[feats['fold'].isin([2,3])]
print(len(dev))

train = feats[feats['fold'].isin(range(4,10))]
print(len(train))


# In[4]:

test.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/diyi_model/data/test.csv', index=False)
train.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/diyi_model/data/train.csv', index=False)
dev.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/diyi_model/data/dev.csv', index=False)


# # Merge in old Yohan features

# In[2]:

oldfeats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_oldfeats.csv')
oldfeats


# In[3]:

len(oldfeats)


# In[5]:

keith_datapts = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/instances.tsv', sep='\t', header=None, names=['article', 'thread', 'editor'])
keith_keys = set(zip(keith_datapts['article'], keith_datapts['thread'], keith_datapts['editor']))

# Check for Keith's keys
current = oldfeats
current_keys = set(zip(current['article'], current['thread_title'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[6]:

oldfeats.rename(columns={'thread_title': 'thread'}, inplace=True)
oldfeats.columns


# In[7]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')


# In[9]:

merged = pd.merge(feats, oldfeats)
print(len(merged))
merged.columns


# In[11]:

# Check for Keith's keys
current = merged
current_keys = set(zip(current['article'], current['thread'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[12]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Align dataset with Keith's instances

# ## Rebuild allfeats to regain missing # data

# ### Merge folds and ftopics

# In[23]:

old_allfeats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
old_allfeats.columns


# In[24]:

# Check for Keith's keys
current = old_allfeats
current_keys = set(zip(current['article'], current['thread'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[25]:

fold_df = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_folds.csv')
fold_df.columns


# In[26]:

ftopics_df = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_ftopics.csv')
ftopics_df.columns


# In[27]:

merged = pd.merge(fold_df, ftopics_df)
merged.columns


# In[28]:

# Check for Keith's keys
current = merged
current_keys = set(zip(current['article'], current['thread'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[30]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# ### Add in #editors

# In[31]:

feats=merged


# In[32]:

# Group test folds by conversation size
gped = feats.groupby(['article', 'thread'])
convo_size = pd.DataFrame([(row[0][0], row[0][1], row[1]) for row in gped.size().iteritems()], columns=['article', 'thread', '#editors'])
convo_size


# In[33]:

merged = pd.merge(feats, convo_size)
merged.columns


# In[36]:

print(min(merged['#editors']))
print(max(merged['#editors']))


# In[37]:

# Remove singletons
merged = merged[merged['#editors']>1]


# In[46]:

# Check for Keith's keys
current = restricted
current_keys = set(zip(current['article'], current['thread'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[39]:

len(merged)


# In[40]:

nodups = merged.drop_duplicates(keep='first')
len(nodups)


# In[41]:

len(set(zip(merged['article'], merged['thread'], merged['editor'])))


# In[42]:

# Restrict to just Keith's keys
restricted = pd.merge(keith_datapts, nodups)
restricted.columns


# In[43]:

len(restricted)


# In[44]:

restricted.drop_duplicates(inplace=True, keep='first')
len(restricted)


# In[45]:

restricted.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# ### Get folds

# In[3]:

fold_info = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/fold_info.tsv', sep='\t', header=None, names=['article', 'fold'])
fold_info


# In[6]:

keith_datapts = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/instances.tsv', sep='\t', header=None, names=['article', 'thread', 'editor'])
keith_keys = set(zip(keith_datapts['article'], keith_datapts['thread'], keith_datapts['editor']))
len(keith_keys)


# In[10]:

keith_articles = set(keith_datapts['article'].unique())
keith_articles


# In[13]:

# Check to make sure Keith's articles still in there
len(keith_articles - keith_articles.intersection(set(fold_info['article'])))


# In[14]:

# Merge with edit scores
scores = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores.csv')
scores.columns


# In[15]:

merged = pd.merge(scores, fold_info)
print(len(merged))
merged.columns


# In[18]:

# Check for Keith's keys
current = merged
current_keys = set(zip(current['article'], current['thread_title'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[19]:

merged.rename(columns={'thread_title': 'thread'}, inplace=True)
merged.columns


# In[20]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_folds.csv', index=False)


# ### Get ftopics

# #### Editor DAs

# In[21]:

# Load sentence-level DAs
sent_das = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/HSTTM8W-talk0-S5-FT20-BT100-FA0.1-B0.001-FG1.0-K0.1-E0.75-N0.75-I10000-InstSentAssign.csv')
sent_das.columns


# In[24]:

# Group things by seqid
grouped = sent_das.groupby(['SeqId', 'User'])

ftopic_counts = {}

for (thread, user), inds in tqdm(grouped.groups.items()):
    ftopics = Counter(sent_das.iloc[inds]['FTopic'].tolist())
    ftopic_counts[(thread, user)] = ftopics
    
thread_user_topics = pd.DataFrame(ftopic_counts).transpose()
thread_user_topics.fillna(0.0, inplace=True)
thread_user_topics = thread_user_topics[[float(x) for x in range(20)]]
thread_user_topics['editor'] = [item[1] for item in thread_user_topics.index]
thread_user_topics['article_name'] = [item[0].split('#', 1)[0] for item in thread_user_topics.index]
thread_user_topics['thread'] = [item[0].split('#', 1)[1] for item in thread_user_topics.index]
thread_user_topics.rename(columns={n: 'ftopic{:.0f}'.format(n) for n in range(20)}, inplace=True)
thread_user_topics

cols = thread_user_topics.columns.tolist()
new_cols = cols[-2:] + cols[-3:-2] + cols[:-3]
thread_user_topics


# In[25]:

thread_user_topics = thread_user_topics[new_cols]
thread_user_topics


# In[26]:

keith_keys = set(zip(keith_datapts['article'], keith_datapts['thread'], keith_datapts['editor']))
len(keith_keys)


# In[29]:

ftopic_keys = set(zip(thread_user_topics['article_name'], thread_user_topics['thread'], thread_user_topics['editor']))
print(len(keith_keys.intersection(ftopic_keys)))


# In[30]:

thread_user_topics.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_dialog_acts.csv', index=False)


# #### Other DAs

# In[2]:

# Load sentence-level DAs
sent_das = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/HSTTM8W-talk0-S5-FT20-BT100-FA0.1-B0.001-FG1.0-K0.1-E0.75-N0.75-I10000-InstSentAssign.csv')
sent_das.columns


# In[3]:

# Group things by seqid and user
user_grouped = sent_das.groupby(['SeqId', 'User'])
grouped = sent_das.groupby(['SeqId'])


# In[8]:

# Build up other ftopic counts
ftopic_other_counts = {}

for (thread, user), inds in tqdm(user_grouped.groups.items()):
    thread_inds = grouped.groups[thread]
    other_inds = list(set(thread_inds) - set(inds))
    ftopics_other = Counter(sent_das.iloc[other_inds]['FTopic'].tolist())
    ftopic_other_counts[(thread,user)] = ftopics_other
    
thread_other_topics = pd.DataFrame(ftopic_other_counts).transpose()
thread_other_topics = thread_other_topics[[float(x) for x in range(20)]]
thread_other_topics.fillna(0.0, inplace=True)
thread_other_topics


# In[9]:

# Manipulate columns
thread_other_topics['editor'] = [item[1] for item in thread_other_topics.index]
thread_other_topics['article'] = [item[0].split('#', 1)[0] for item in thread_other_topics.index]
thread_other_topics['thread'] = [item[0].split('#', 1)[1] for item in thread_other_topics.index]
thread_other_topics


# In[11]:

keith_datapts = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/instances.tsv', sep='\t', header=None, names=['article', 'thread', 'editor'])
len(keith_datapts)


# In[12]:

keith_keys = set(zip(keith_datapts['article'], keith_datapts['thread'], keith_datapts['editor']))
len(keith_keys)


# In[19]:

ftopic_keys = set(zip(thread_other_topics['article'], thread_other_topics['thread'], thread_other_topics['editor']))


# In[20]:

print(len(keith_keys.intersection(ftopic_keys)))


# #### Merge editor DAs and other DAs

# In[31]:

print(thread_user_topics.columns)
print(thread_other_topics.columns)


# In[32]:

thread_other_topics.rename(columns={x: 'other_ftopic{:d}'.format(int(x)) for x in range(20)}, inplace=True)
thread_other_topics.columns


# In[33]:

thread_user_topics.rename(columns={'article_name': 'article'}, inplace=True)
thread_user_topics.columns


# In[34]:

merged = pd.merge(thread_user_topics, thread_other_topics)
merged.columns


# In[35]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_dialog_acts.csv', index=False)


# In[36]:

# Merge with edit scores
scores = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores.csv')
scores.columns


# In[37]:

scores.rename(columns={'thread_title': 'thread'}, inplace=True)
scores.columns


# In[38]:

merged_all = pd.merge(scores, merged)
print(merged_all.columns)
len(merged_all)


# In[39]:

current = merged_all
current_keys = set(zip(current['article'], current['thread'], current['editor']))
print(len(keith_keys.intersection(current_keys)))


# In[40]:

current.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_ftopics.csv', index=False)


# ## Investigate missing data

# In[10]:

keith_datapts = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/instances.tsv', sep='\t', header=None, names=['article', 'thread', 'editor'])
len(keith_datapts)


# In[5]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
len(feats)


# In[6]:

merged = pd.merge(feats, keith_datapts, how='right')
print(len(merged))
merged.columns


# In[7]:

merged = pd.merge(feats, keith_datapts, how='inner')
print(len(merged))
merged.columns


# In[8]:

merged_keys = set(zip(merged['article'], merged['thread'], merged['editor']))
len(merged_keys)


# In[9]:

keith_keys = set(zip(keith_datapts['article'], keith_datapts['thread'], keith_datapts['editor']))
len(keith_keys)


# In[10]:

feats_keys = set(zip(feats['article'], feats['thread'], feats['editor']))
len(feats_keys)


# In[11]:

# Find rows that Keith has that I don't
missing = keith_keys-merged_keys
print(len(missing))
missing


# In[12]:

# Find missing rows in data
scores = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores.csv')
scores.columns


# In[23]:

# Search for missing row
egs = random.sample(list(missing), 10)
egs


# In[16]:

scores_keys = set(zip(scores['article'], scores['thread_title'], scores['editor']))
len(scores_keys)


# In[24]:

all(eg in scores_keys for eg in egs)


# In[26]:

pd.set_option('display.max_colwidth', 999)


# In[27]:

mask = list(map(lambda tup: tup in egs, zip(scores['article'], scores['thread_title'], scores['editor'])))
scores[mask]


# ## Check in ftopics

# In[29]:

df = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_ftopics.csv')
df.columns


# In[37]:

fields = ['article', 'thread_title', 'editor']
keys = set(zip(df[fields[0]], df[fields[1]], df[fields[2]]))
print(len(keys))
print(all(eg in keys for eg in egs))
print(any(eg in keys for eg in egs))
print()

for eg in egs:
    print(eg in keys, end=':\t')
    print(eg)


# In[34]:

mask = list(map(lambda tup: tup in egs, zip(df[fields[0]], df[fields[1]], df[fields[2]])))
selected = df[mask]
print(len(selected))
selected


# ## Check in dialogue acts

# In[51]:

df = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_dialog_acts.csv')
df.columns


# In[54]:

fields = ['article_name', 'thread', 'editor']
keys = set(zip(df[fields[0]], df[fields[1]], df[fields[2]]))
print(len(keys))
print(all(eg in keys for eg in egs))
print(any(eg in keys for eg in egs))
print()

for eg in egs:
    print(eg in keys, end=':\t')
    print(eg)


# ## Check in folds

# In[38]:

df = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_folds.csv')
df.columns


# In[39]:

fields = ['article', 'thread_title', 'editor']
keys = set(zip(df[fields[0]], df[fields[1]], df[fields[2]]))
print(len(keys))
print(all(eg in keys for eg in egs))
print(any(eg in keys for eg in egs))
print()

for eg in egs:
    print(eg in keys, end=':\t')
    print(eg)


# In[40]:

mask = list(map(lambda tup: tup in egs, zip(df[fields[0]], df[fields[1]], df[fields[2]])))
selected = df[mask]
print(len(selected))
selected


# ## Check if singletons

# In[45]:

eg_threads = [(tup[0],tup[1]) for tup in egs]
eg_threads


# In[50]:

gped = scores.groupby(['article', 'thread_title'])
thread_sizes = gped.size()

for t in eg_threads:
    print(thread_sizes[t])


# # Check dates of data

# In[2]:

talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/talk_filtered_2.csv', parse_dates=['timestamp'])
max(talk['timestamp'])


# # Add conversation size to features

# In[44]:

# Load features
feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
feats


# In[46]:

# Group test folds by conversation size
gped = feats.groupby(['article', 'thread'])
gped.size()


# In[59]:

convo_size = pd.DataFrame([(row[0][0], row[0][1], row[1]) for row in gped.size().iteritems()], columns=['article', 'thread', '#editors'])
convo_size


# In[61]:

merged = pd.merge(feats, convo_size, on=['article', 'thread'])
merged.columns


# In[62]:

len(merged)


# In[63]:

min(merged['#editors'])


# In[64]:

max(merged['#editors'])


# In[65]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Add new fold information to allfeats dataset

# In[27]:

# Open tsv of fold info in Pandas
fold_info = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/fold_info2.tsv', sep='\t', header=None, names=['article', 'fold'])
fold_info


# In[24]:

fold_artnames = set(fold_info[0])
len(fold_artnames)


# In[25]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_ftopics.csv')
len(feats['article'].unique())


# In[33]:

# Merge in fold info
merged = pd.merge(feats, fold_info, on=['article'])
merged


# In[31]:

cols = merged.columns.tolist()
cols


# In[34]:

merged = merged[['fold', 'article', 'thread'] + cols[cols.index('editor'):cols.index('article_name')] +                 cols[cols.index('ftopic0'):cols.index('fold')]]
# merged.drop(['article_name', 'thread_title'], axis=1, inplace=True)
merged


# In[35]:

len(merged)
# Need to then remove singleton conversations


# In[38]:

merged.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)


# # Check for any pages with only singleton conversations

# In[4]:

no_singletons = set(pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')['article'])
print(len(no_singletons))

singletons = set(pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores.csv')['article'])
print(len(singletons))


# In[5]:

# Load fold info
with open('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/ijcnlp-rawfolds.pickle', 'rb') as f:
    folds = pickle.load(f, encoding='latin1')
    
folds


# In[12]:

# Load fold info
with open('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/ijcnlp-rawfolds.pickle', 'rb') as f:
    folds = pickle.load(f, fix_imports)
    
folds


# In[6]:

len(folds)


# In[7]:

fold_articles = set(folds.keys())
intersect = fold_articles.intersection(singletons)
len(intersect)


# In[8]:

fold_articles = set(folds.keys())
intersect = fold_articles.intersection(no_singletons)
len(intersect)


# In[9]:

nomatch = [art for art in fold_articles if not art in singletons]
nomatch


# In[10]:

len(nomatch)


# In[11]:

search = 'Mauritania'
[art for art in singletons if search in art]


# In[17]:

for art in nomatch:
    print(art)


# # Remove singleton conversations from features

# In[39]:

feats = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv')
print(len(feats))


# In[41]:

gped = feats.groupby(['article', 'thread'])
thread_sizes = gped.size()

thread_sizes.values >= 2

twoplus_threads = thread_sizes[thread_sizes.values >= 2].index.tolist()
twoplus_threads


# In[42]:

# Filter conversations
mask = list(map(lambda tup: tup in twoplus_threads, zip(feats['article'], feats['thread'])))
filtered = feats[mask]
len(filtered)


# In[43]:

filtered.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores_allfeats.csv', index=False)

