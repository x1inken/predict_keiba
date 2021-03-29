#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[ ]:


# データベースに接続
engine = create_engine('mysql+mysqlconnector://root@localhost/jra?charset=utf8')


# In[ ]:


# 学習用データ取得SQL

learnsql = '''
    
select 
	cast(KakuteiJyuni as signed) as junni,
	cast(Umaban as signed) as umaban,
	case when cast(Umaban as signed) <= 3 then 1 else 0 end as inside,
	case when cast(Umaban as signed) >= 10 then 1 else 0 end as outside,
	case when ZogenFugo = '' then '=' else ZogenFugo end as fugo,
	1-(cast(ZogenSa as signed) / cast(BaTaijyu as signed)) as sapoint,
    cast(Ninki as signed) as Ninki
FROM
	n_uma_race
where
	KakuteiJyuni != '00'
and
	Odds != '0000'
and 
	ZogenSa != ''
order by
	Year desc,
	MonthDay desc


'''
df = pd.read_sql_query(learnsql, engine)


# In[ ]:


# 欠損値の削除
df.dropna(how='any', inplace=True)


# In[ ]:


# 3着以内フラグのカラムを追加
cf = lambda x: 1 if x in[1,2,3] else 0
df['3着以内'] = df['junni'].map(cf)


# In[ ]:


# 増減符号をダミー変数化
df = pd.get_dummies(df,columns=['fugo'])


# In[ ]:


# 学習に必要じゃないデータを削除
df.drop(['junni','umaban','fugo_-','fugo_='],axis=1, inplace=True)


# In[ ]:


# 説明変数のdataframe
X = df.drop(['3着以内'], axis=1)


# In[ ]:


# 目的変数のdataframe
Y = df['3着以内']


# In[ ]:


# データを学習用データと検証・テストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X,Y ,test_size=0.2, stratify=Y)


# In[ ]:


# 標準化関数
stdsc = StandardScaler()


# In[ ]:


# 学習データを標準化
X_train_std = stdsc.fit_transform(X_train)


# In[ ]:


# 学習データからテストデータを標準化
X_test_std = stdsc.transform(X_test)


# In[ ]:


# ロジスティック回帰で学習
clf = LogisticRegression()
clf.fit(X_train_std, y_train)


# In[ ]:


# 変数の寄与度を見る
clf.score(X_train_std,y_train)
clf_df = pd.DataFrame([X_train.columns, clf.coef_[0]]).T


# In[ ]:


# テストデータを予測
x_pred = clf.predict(X_test_std)


# In[ ]:


# モデルの精度を見る
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


# 分類したマトリクス
print('confusion matrix = \n', confusion_matrix(y_test, x_pred))


# In[ ]:


# 正解率
print('accuracy = ', accuracy_score(y_test, x_pred))


# In[ ]:


# 適合率
print('precision = ', precision_score(y_test, x_pred))


# In[ ]:


# 再現率
print('recall = ', recall_score(y_test, x_pred))


# In[ ]:


# F1値
print('f1 = ', f1_score(y_test, x_pred))


# In[ ]:


# 学習データの結果を見る
pred = pd.Series(x_pred, index=X_test.index, name='3着以内')
result = pd.concat([pred, X_test], axis=1)


# In[ ]:


# 実際の予測


# In[ ]:


# 中山記念の出走馬取得
sql_pred = '''

select 
	cast(KakuteiJyuni as signed) as junni,
	cast(Umaban as signed) as umaban,
	case when cast(Umaban as signed) <= 3 then 1 else 0 end as inside,
	case when cast(Umaban as signed) >= 10 then 1 else 0 end as outside,
	case when ZogenFugo = '' then '=' else ZogenFugo end as fugo,
	1-(cast(ZogenSa as signed) / cast(BaTaijyu as signed)) as sapoint,
	cast(Ninki as signed) as Ninki
FROM
	n_uma_race
where
 Year = '2021' and MonthDay = '0228'
and
 JyoCD = '06' and RaceNum = '11'
order by
 Umaban;

'''
df_pred = pd.read_sql_query(sql_pred, engine)


# In[ ]:


# ダミー変数化
df_pred = pd.get_dummies(df_pred, columns=['fugo'])

# 説明変数のdataframe
Xp = df_pred.drop(['junni','umaban','fugo_-','fugo_='],axis=1)


# In[ ]:


# 標準化
Xp_std = stdsc.transform(Xp)


# In[ ]:


# 予測
x_pred = clf.predict(Xp_std)

# 結果を見やすいように整形
pred = pd.Series(x_pred, index=df_pred.index, name='3着以内')
result = pd.concat([pred, df_pred], axis=1)


# In[ ]:


result


# In[ ]:




