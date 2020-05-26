#-*- coding:utf-8 -*-

#import python packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import *
from sklearn.cluster import *

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from operator import itemgetter
from operator import attrgetter

from pyjarowinkler import distance
from collections import Counter

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import nltk

import math
import time

import csv
import sys

import re
import io
import os

start_time = time.time()

#전처리 함수 정의부
def remove_string_special_characters(s):
    
    stripped = re.sub('[^a-zA-z\s]', '', s)
    
    stripped = re.sub('_', '', stripped)
    
    stripped = re.sub('\s+', ' ', stripped)
    
    stripped = stripped.strip()
    
    if stripped != '':
        return stripped.lower()
    
#클래스 정렬 함수 정의부
def multisort(xs, specs):
    
    for key, reverse in reversed(specs):
        
        xs.sort(key=attrgetter(key), reverse=reverse)
        
    return xs

#속성집합 추출 함수 정의부
#키워드 매개변수(입력csv path, 속선집합 포함 출력csv path, 추출할 단어 수)
def extractive_keyword(path,database_update_path,extract_word_num=20):
    
    reviewee = pd.read_csv(path, encoding='latin1')
    count,temp = len(reviewee),[]

    for i in range(count):
        
        temp_intro = reviewee['submitter_intro'][i]
        temp_sent = summarize(reviewee['submitter_intro'][i], ratio=0.05)

        textrank_textsent_mearge = ''
        textrank_text,textrank_sent = '',''

        for c in (keywords(temp_intro, words=extract_word_num-(extract_word_num//4), lemmatize=True).split('\n')):
            
            textrank_text += (c+ " ")

        for cc in (keywords(temp_sent, words=(extract_word_num//4), lemmatize=True).split('\n')):
            
            textrank_sent += (cc+ " ")

        temp.append(textrank_text + " " + textrank_sent)

    reviewee['submitter_attribute']=temp
    
    reviewee.iloc[:,1:].to_csv(database_update_path)
    
    #return type : pandas.dataframe
    return reviewee


#전문성 검사 함수 정의부
#키워드 매개변수(입력csv path, 투고원고 DataFrame, i번째 투고 원고, 추천할 심사자 수, 실루엣값 계산 범위 지정)
def professionalism(path,extractive_keyword_result,reviewee_index,top_limit,silhouette_range=25):
    
    reviewee=extractive_keyword_result
    index=reviewee_index
    top=top_limit
    
    temp_id,temp_doi = 0,''
    
    temp_title = reviewee.loc[index]['submitter_title']
    temp_attribure = reviewee.loc[index]['submitter_attribute']

    reviewer_attr = pd.read_csv(path, encoding='latin1')
    
    reviewer_attr.loc[-1]=[str(temp_id),temp_doi,temp_title,temp_attribure]
    
    reviewer_attr.index += 1
    
    reviewer_attr.sort_index(inplace=True)
    
    reviewer=reviewer_attr['reviewer_paper_attribure']
    
    jac_token,jac,cos,avg=[],[],[],[]

    for t in range(len(reviewer)):
        
        jac_token.append(set(nltk.ngrams((nltk.word_tokenize(reviewer[t])), n=1)))
        
    for j in range(len(reviewer)):
        
        jac.append(1-(nltk.jaccard_distance(jac_token[0], jac_token[j])))

    count_vectorizer = CountVectorizer(stop_words='english')
    
    count_vectorizer = CountVectorizer()

    sparse_matrix = count_vectorizer.fit_transform(reviewer)
    
    doc_term_matrix = sparse_matrix.todense()

    df = pd.DataFrame(doc_term_matrix, 
                      columns=count_vectorizer.get_feature_names(), 
                      index=[i for i in reviewer])

    cos=cosine_similarity(df, df)[0].tolist()

    for i in range(len(jac)):
        
        avg.append((jac[i] + cos[i])/2)
        
    reviewer_attr['sim']=avg

    vectorizer = TfidfVectorizer(stop_words='english')
    
    Y = vectorizer.fit_transform(reviewer)
    
    YY = Y.toarray()
    
    X = StandardScaler().fit_transform(YY)

    top_avg,top_k=0,0
    silhouette,k_mean,k_mean2=[],[],[]

    for i in range(2,silhouette_range+1,1):

        model = SpectralClustering(n_clusters=i, affinity="nearest_neighbors")
        
        cluster_labels = model.fit_predict(X)

        sample_silhouette_values = silhouette_samples(YY, cluster_labels)
        
        silhouette_avg = sample_silhouette_values.mean()

        if top_avg < silhouette_avg:
            
            top_avg = silhouette_avg
            
            top_k = i

        silhouette_temp=[]
        
        silhouette_temp.append('k=' + str(i) + '일때 : ')
        
        silhouette_temp.append(silhouette_avg)
        
        silhouette.append(silhouette_temp)

    model = KMeans(n_clusters=(top_k), init='k-means++', max_iter=100, n_init=1)
    
    model.fit(Y)
    
    for k in range(len(reviewer)):
        
        YYY = vectorizer.transform([reviewer[k]])
        
        prediction = model.predict(YYY)
        
        k_mean.append(prediction)

    for k in range(len(reviewer)):
        
        k_mean2.append(int(k_mean[k][0]))
        
    reviewer_attr['k_mean']=k_mean2
    
    kmean_reviewer = reviewer_attr[reviewer_attr['k_mean'] == reviewer_attr.loc[0]['k_mean']]
    
    kmean_reviewer2 = kmean_reviewer.sort_values(by=['sim'], axis=0, ascending=False)
    
    professionalism=kmean_reviewer2.iloc[1:top+1]
    
    #return type : pandas.dataframe
    return professionalism

#이해관계 검사 함수 정의부
#키워드 매개변수(심사후보자_공저자csv path, 심사후보자_정보csv path, 심사후보자_공저자네트워크csv path,전문성검사결과_DataFrame, 투고원고_DataFrame, i번째 투고 원고, 추천할 심사자 수, 심사후보자_공저자네트워크_곱셈횟수)
def interest(co_author_path, reviewer_information_path, co_author_network_path, professionalism_result, extractive_keyword_result, reviewee_index,top_limit,matrix_multifly_count):
    
    crash_result,reviewee_list=[],[]
    reviewer_list1,reviewer_co_list=[],[]
    
    path1=co_author_path
    path2=reviewer_information_path
    network_path=co_author_network_path
    
    temp = professionalism_result
    reviewee=extractive_keyword_result
    
    index=reviewee_index
    top=top_limit
    multifly=matrix_multifly_count
    
    co_author_csv = pd.read_csv(path1, encoding='latin1')
    
    co_author_df = co_author_csv.merge(temp, on=['reviewer_orcid'])
    
    tt = co_author_df.iloc[:]['reviewer_name'].tolist()

    reviewee_list=[]
    reviewee.fillna(0, inplace=True)

    for i in range(1,11):
        col_index = (i*3)+5
        if reviewee.loc[index][col_index] != 0:
            reviewee_list.append(reviewee.loc[index][col_index])

    reviewer_list,reviewer_co_list=[],[]

    for j in range(len(co_author_csv)):

        co_list_temp=[]
        reviewer_list.append(co_author_csv['reviewer_name'][j])
        co_list_temp.append(co_author_csv['reviewer_name'][j])

        for i in range(1,11):
            col_index = (i*2)
            if co_author_csv.loc[j][col_index] != 0:
                co_list_temp.append(co_author_csv.loc[j][col_index])

        reviewer_co_list.append(co_list_temp)

    co_rel_df = pd.DataFrame(
        columns=[i for i in reviewer_list],
        index=[j for j in reviewee_list])

    for j in range(len(reviewee_list)):
        for i in range(len(reviewer_list)):
            for k in range(len(reviewer_co_list[i])):
                if reviewee_list[j] == reviewer_co_list[i][k]:
                    co_rel_df.iat[j, i] = 1

    co_rel_df.fillna(0, inplace=True)

    try :

        matrix_df = pd.read_csv(co_author_network_path, encoding='latin1', index_col=0)

    except FileNotFoundError :

        index = co_author_csv['reviewer_orcid'].index[co_author_csv['reviewer_orcid'].apply(np.isnan)]

        df_index = co_author_csv.index.values.tolist()

        nan_range =[df_index.index(i) for i in index]

        try :

            import_csv2=co_author_csv.iloc[:nan_range[0]]
            id_list=import_csv2['reviewer_name'].tolist()

        except IndexError :

            import_csv2=co_author_csv
            id_list = co_author_csv.iloc[:]['reviewer_name'].tolist()

        matrix_df = pd.DataFrame(

            columns=[i for i in id_list],
            index=[j for j in id_list])

        for i in range(len(id_list)):

            for j in range(len(id_list)):

                index=[1,]

                index.extend([(j*2) for j in range(1,11)])

                for k in range(11):

                    if (id_list[i]) == (import_csv2.iloc[j][index[k]]) :

                        print(id_list[i], import_csv2.iloc[j][index[k]])

                        print(i)

                        matrix_df.iat[j, i] = 1
                        matrix_df.iat[i, j] = 1

                if str(id_list[i]) == str(id_list[j]):

                    matrix_df.iat[i, j] = 0

        matrix_df.fillna(0, inplace=True)
        matrix_df.to_csv(co_author_network_path)
    
    for i in range(multifly):
        
        matrix_df = matrix_df.dot(matrix_df)

    a=matrix_df.values
    b=co_rel_df.values

    aaa = b.dot(a)

    aaa2=pd.DataFrame(data=aaa,
                 index=(co_rel_df.index).tolist(),
                 columns=(matrix_df.index).tolist())

    a_series = (aaa2 != 0).any(axis=1)
    
    new_df = aaa2.loc[a_series]
    
    ccc=(new_df.index).tolist()
    
    ddd=co_author_df['reviewer_name'].tolist()
    
    reviewer_list1 = list(set(ddd).difference(ccc))
    
    co_inst_csv = pd.read_csv(path2, encoding='latin1')
    
    co_inst_df = co_inst_csv.merge(temp, on=['reviewer_orcid'])

    reviewee_list2,reviewer_list2,reviewer_inst_list=[],[],[]
    
    reviewee.fillna(0, inplace=True)

    for i in range(1,11):
        
        col_index = (i*3)+6
        
        if reviewee.loc[index][col_index] != 0:
            
            reviewee_list2.append(reviewee.loc[index][col_index])

    for j in range(len(co_inst_df)):
        
        inst_list_temp=[]
        
        reviewer_list2.append(co_inst_df['reviewer_name'][j])
        
        reviewer_inst_list.append(co_inst_df['reviewer_institution'][j])

    inst_rel_df = pd.DataFrame(
        columns=[i for i in reviewee_list2],
        index=[j for j in reviewer_list2])

    for i in range(len(reviewee_list2)):
        
        for j in range(len(reviewer_list2)):
            
            if reviewee_list2[i] == reviewer_inst_list[j]:
                
                inst_rel_df.iat[j, i] = 1

    for i in range(len(reviewer_list2)):
        
        if (inst_rel_df.sum(axis=1)[i]) > 0:
            
            reviewer_list2.remove(inst_rel_df.index[i])
            
            crash_result.append(inst_rel_df.index[i])

    reviewer_list1,reviewer_list2 = reviewer_list1[0:top*2],reviewer_list2[0:top*2]
    
    reviewer_rank = list(set(reviewer_list1).intersection(reviewer_list2))
    
    id_index,sim_index,count_index=[],[],[]
    
    reviewer_rank = pd.DataFrame({'reviewer_name': reviewer_rank})

    for i in range(len(reviewer_rank)):
        
        for j in range(len(co_author_df)):
            
            if reviewer_rank.loc[i]['reviewer_name'] == co_author_df.loc[j]['reviewer_name'] :
                
                id_index.append(int(co_author_df.iloc[j]['reviewer_orcid']))
                
                sim_index.append(co_author_df.iloc[j]['sim'])
            
            if reviewer_rank.loc[i]['reviewer_name'] == co_inst_df.loc[j]['reviewer_name'] :
                
                count_index.append(co_inst_df.iloc[j]['count'])
                
    reviewer_rank['reviewer_orcid']=id_index
    
    reviewer_rank['sim']=sim_index
    
    reviewer_rank['count']=count_index
                
    #return type : pandas.dataframe
    return reviewer_rank


#csv 저장 함수 정의부
#키워드 매개변수(save_path, 투고원고_DataFrame, 전문성검사_DataFrame, i번째 투고 원고, 추천할 심사자 수)
def save_csv(output_path,extractive_keyword_result,professionalism_result,reviewee_index,top_limit):
    
    path=output_path
    
    reviewee=extractive_keyword_result
    
    reviewer_rank_name=professionalism_result
    
    ee_num=reviewee_index
    
    top=top_limit
    
    export_data=[]

    for i in range((top*2)):
        
        temp=[]
        
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['submitter_title'])
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['date'])
        temp.append(reviewee.iloc[(1//top*2)+ee_num]['submitter_name'])
        
        temp.append(reviewer_rank_name.iloc[i]['reviewer_name'])
        temp.append(reviewer_rank_name.iloc[i]['reviewer_orcid'])
        temp.append(reviewer_rank_name.iloc[i]['sim'])
        temp.append(reviewer_rank_name.iloc[i]['count'])
        
        export_data.append(temp)
        
    try :
            
        export_csv = pd.read_csv(path,index_col=0)
        
    except FileNotFoundError :
            
        export_csv = pd.DataFrame([],columns=[
            'submitter_title','date','submitter_name','reviewer_name','reviewer_orcid','sim','count'])
    
    for i in range(len(export_data)):
        
        export_csv.loc[len(export_csv)] = export_data[i]
        
    export_csv.to_csv(path)

#균등할당 함수 정의부
#키워드 매개변수(입력 path)
def equl_distribution(input_csv_path, output_csv_path):
    
    final_list=[]
    
    export_csv2 = pd.read_csv(input_csv_path,index_col=0)

    class Paper:
        
        def __init__(self, title, date, submitter, reviwer_name, reviwer_orcid, sim, count):
            
            self.title = title
            self.date = date
            self.submitter = submitter
            self.reviwer_name = reviwer_name
            self.reviwer_orcid = reviwer_orcid
            self.sim = sim
            self.count = count

        def __repr__(self):
            
            return repr((self.title, self.date, self.submitter, self.reviwer_name, self.reviwer_orcid, self.sim, self.count))

    papers,objs=[export_csv2.iloc[i].tolist() for i in range(len(export_csv2))],[]

    for paper in papers:
        
        objs.append(Paper(*paper))
    
    o = (multisort(list(objs), (('date', False), ('sim', True))))

    for i in range(0,len(export_csv2),6) :
        
        temp_list=[]
        
        for t in range(6):
            
            if len(temp_list) == 3:
                break
            else :
                temp = i + t

                if (o[temp].count) < 3 :

                    o[temp].count += 1

                    for j in range(0+temp, len(export_csv2)) :

                        if (o[temp].reviwer_name == o[j].reviwer_name) :

                            o[j].count += 1

                    o[temp].count -= 1
                    
                    temp_list.append(o[temp])
                    
        final_list.extend(temp_list)
            
    final=pd.DataFrame(final_list,columns=['result'])
    
    final.to_csv(output_csv_path)

#디폴트 실행 함수 정의부
def main():
    
    #투고원고에 대한 속성집합 추출
    #키워드 매개변수(입력csv path, 속선집합 포함 출력csv path, 추출할 단어 수)
    reviewee=extractive_keyword(path='../reviewee/submitter_10.csv',
                                database_update_path='../reviewee/reviwupdate.csv',
                                extract_word_num=20)
    #return type : pandas.dataframe
    
    
    #투고원고 수 만큼의 검사세트 진행
    for i in range(len(reviewee)):
        
        #전문성검사
        #키워드 매개변수(입력csv path, 투고원고 DataFrame, i번째 투고 원고, 추천할 심사자 수, 실루엣값 계산 범위 지정)
        reviewer=professionalism(path='../reviewer_pool/reviewer_attribute_5.csv',
                                 extractive_keyword_result=reviewee,
                                 reviewee_index=i,
                                 top_limit=10,
                                 silhouette_range=25)
        #return type : pandas.dataframe
        
        
        #이해관계검사
        #키워드 매개변수(심사후보자_공저자csv path, 심사후보자_정보csv path, 심사후보자_공저자네트워크csv path,
        #전문성검사결과_DataFrame, 투고원고_DataFrame, i번째 투고 원고, 추천할 심사자 수, 심사후보자_공저자네트워크_곱셈횟수)
        reviewer_rank = interest(
            co_author_path='../reviewer_pool/reviewer_coauthor_5.csv',
            reviewer_information_path='../reviewer_pool/reviewer_information_5.csv',
            co_author_network_path='../reviewer_pool/co_author_network_0525.csv',
            professionalism_result=reviewer,
            extractive_keyword_result=reviewee,
            reviewee_index=i,
            top_limit=6,
            matrix_multifly_count=1)
        #return type : pandas.dataframe
        
        
        #csv저장
        #키워드 매개변수(save_path, 투고원고_DataFrame, 전문성검사_DataFrame, i번째 투고 원고, 추천할 심사자 수)
        save_csv(output_path='../system_output/export_csv_0525_10.csv',
                 extractive_keyword_result=reviewee,
                 professionalism_result=reviewer_rank,
                 reviewee_index=i,
                 top_limit=3)
        
    #균등할당
    #키워드 매개변수(입력 path)
    equl_distribution(input_csv_path='../system_output/export_csv_0525_10.csv',
                     output_csv_path='../system_output/final_csv_0525_10.csv')
    
if __name__ == '__main__':
    #디폴트 실행 함수
    main()