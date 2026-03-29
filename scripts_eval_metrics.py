import pandas as pd
from collections import defaultdict, Counter

train_path = r'd:\coursera\学习\ISE\项目\推荐系统\data\data_raw\train_click_log.csv'
articles_path = r'd:\coursera\学习\ISE\项目\推荐系统\data\data_raw\articles.csv'

# Load and sort
print('loading train...')
df = pd.read_csv(train_path, usecols=['user_id','click_article_id','click_timestamp'])
df = df.sort_values(['user_id','click_timestamp'])

# Build leave-one-out split
hist = defaultdict(list)
gt = {}
for uid, g in df.groupby('user_id'):
    items = g['click_article_id'].tolist()
    if len(items) < 2:
        continue
    hist[uid] = items[:-1]
    gt[uid] = items[-1]

users = list(hist.keys())[:5000]
hist = {u: hist[u] for u in users}
gt = {u: gt[u] for u in users}

# Popularity
pop_counter = Counter()
for u in users:
    pop_counter.update(hist[u])
pop_list = [i for i,_ in pop_counter.most_common(500)]

# Transition-based itemcf proxy
next_counter = defaultdict(Counter)
for u in users:
    seq = hist[u]
    for a,b in zip(seq[:-1], seq[1:]):
        next_counter[a][b] += 1

# Category-based embedding proxy
art = pd.read_csv(articles_path, usecols=['article_id','category_id'])
cat_map = dict(zip(art['article_id'], art['category_id']))
cat_pop = defaultdict(Counter)
for u in users:
    for it in hist[u]:
        c = cat_map.get(it)
        if c is not None:
            cat_pop[c][it] += 1

def dedup_keep_order(arr):
    s=set(); out=[]
    for x in arr:
        if x in s:
            continue
        s.add(x)
        out.append(x)
    return out

def rec_pop(u, k=20):
    seen=set(hist[u])
    return [i for i in pop_list if i not in seen][:k]

def rec_itemcf(u, k=20):
    seen=set(hist[u])
    score=Counter()
    for it in hist[u][-10:]:
        for j,c in next_counter.get(it, {}).items():
            if j not in seen:
                score[j]+=c
    rec=[i for i,_ in score.most_common(k*3)]
    if len(rec)<k:
        rec += [i for i in pop_list if i not in seen and i not in set(rec)]
    return rec[:k]

def rec_embed(u, k=20):
    seen=set(hist[u])
    recent = hist[u][-1]
    c = cat_map.get(recent)
    rec=[]
    if c is not None:
        rec = [i for i,_ in cat_pop[c].most_common(k*3) if i not in seen]
    if len(rec)<k:
        rec += [i for i in pop_list if i not in seen and i not in set(rec)]
    return rec[:k]

def rec_ranker(u, k=20):
    seen=set(hist[u])
    c1 = rec_itemcf(u,50)
    c2 = rec_embed(u,50)
    c3 = rec_pop(u,50)
    cand = dedup_keep_order(c1 + c2 + c3)
    recent = hist[u][-1]
    recent_cat = cat_map.get(recent)
    score={}
    for it in cand:
        s=0.0
        if it in c1:
            s += 2.0/(1+c1.index(it))
        if it in c2:
            s += 1.2/(1+c2.index(it))
        if it in c3:
            s += 0.8/(1+c3.index(it))
        if recent_cat is not None and cat_map.get(it)==recent_cat:
            s += 0.3
        score[it]=s
    ranked = [i for i,_ in sorted(score.items(), key=lambda x:x[1], reverse=True)]
    ranked = [i for i in ranked if i not in seen]
    return ranked[:k]

def metrics(rec_func, k=20):
    hit=0
    rr=0.0
    for u in users:
        rec = rec_func(u,k)
        g = gt[u]
        if g in rec:
            hit += 1
            rr += 1.0/(rec.index(g)+1)
    n=len(users)
    hitrate = hit/n
    recall = hitrate
    mrr = rr/n
    return recall,mrr,hitrate

models = {
    'Popularity baseline': rec_pop,
    'ItemCF': rec_itemcf,
    'Embedding recall': rec_embed,
    'Ranker (hybrid)': rec_ranker,
}

print('users_eval',len(users))
rows=[]
for name,fn in models.items():
    r,m,h = metrics(fn,20)
    rows.append((name,r,m,h))
    print(name, f'Recall@20={r:.4f}', f'MRR@20={m:.4f}', f'HitRate@20={h:.4f}')

out = pd.DataFrame(rows, columns=['Model','Recall@20','MRR@20','HitRate@20'])
out.to_csv(r'd:\coursera\学习\ISE\项目\推荐系统\recommender_system\offline_metrics_summary.csv', index=False)
print('saved offline_metrics_summary.csv')
