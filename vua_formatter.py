"""
This files functions as a VUA formatter tool for the original HateExplain and 
ETHOS hate-speech datasets.
"""
import pandas as pd
import json
import numpy as np

##################################
# Hate explain conversion        #
##################################

# "24439295_gab": {"post_id": "24439295_gab", 
# "annotators": [
#     {"label": "offensive", "annotator_id": 222, "target": ["Homosexual"]}, 
#     {"label": "offensive", "annotator_id": 209, "target": ["Other"]}, 
#     {"label": "hatespeech", "annotator_id": 201, "target": ["Homosexual"]}], 
#     "rationales": [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]], 
#     "post_tokens": ["my", "rhymes", "pass", "any", "bar", "exam", "they", "call", "me", "kincannon", "<number>", "shots", "to", "your", "think", "tank", "im", "wavin", "twin", "cannons", "you", "a", "fag", "that", "loves", "it", "up", "the", "ass", "you", "are", "a", "ken", "kaniff", "and", "yo", "bitch", "gimme", "sloppy", "toppy", "they", "call", "her", "steve", "bannon", "the", "g.o.a.t."]},

def get_rationales_words():
    with open('Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain.json', 'r') as fp:
        data = json.load(fp)
    ration_data = pd.DataFrame(columns=["Id","Word","Ratio"])
    for count,key in enumerate(data):
        Id = str(key)
        df_temp = pd.DataFrame()
        temp={}
        temp["rationales"] = list(data[key]["rationales"])
        temp['text']=list(data[key]['post_tokens'])

        if len(temp["rationales"])>0:
            # it = iter(temp["rationales"])
            # the_len = len(next(it))
            # if all(len(l) == the_len for l in it):
            summed_list = sum(map(np.array, temp["rationales"]))/len(temp["rationales"])
            # else:
            #     print(Id,temp["rationales"])
            #     continue

        else:
            summed_list = np.zeros(len(temp["text"]))
        
        df_temp["Word"] = temp["text"]
        df_temp["Ratio"] = summed_list
        df_temp.insert(0, 'Id', Id)

        ration_data = pd.concat([ration_data,df_temp])
        # print(count)
        # if count > 100:
        #     break
    ration_data.to_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain_rationales.csv",index=False,sep="\t")
    print(ration_data.head())
    print(ration_data.info())

# get_rationales_words()



def get_annotated_data(classes):
    #temp_read = pd.read_pickle(params['data_file'])
    with open('Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain.json', 'r') as fp:
        data = json.load(fp)
    dict_data=[]
    for key in data:
        temp={}
        temp['post_id']=key
        temp['text']=data[key]['post_tokens']
        final_label=[]
        for i in range(1,4):
            temp['annotatorid'+str(i)]=data[key]['annotators'][i-1]['annotator_id']
#             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target'+str(i)]=data[key]['annotators'][i-1]['target']
            temp['label'+str(i)]=data[key]['annotators'][i-1]['label']
            final_label.append(temp['label'+str(i)])

        final_label_id=max(final_label,key=final_label.count)
        temp['rationales']=data[key]['rationales']
            
        if(classes == "two"):
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                if(final_label_id in ['hatespeech','offensive']):
                    final_label_id='toxic'
                else:
                    final_label_id='non-toxic'
                temp['final_label']=final_label_id

        
        else:
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                temp['final_label']=final_label_id

        
        
        
        dict_data.append(temp)    
    temp_read = pd.DataFrame(dict_data)  
    return temp_read    


vua_hateexplain = False
if vua_hateexplain == True:
    # data_he = get_annotated_data("else")

    # data_he.to_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain_2.csv")

    data_full = pd.read_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain_2.csv")


    data_full = data_full[["post_id","text","final_label"]]

    # data_full["Text"] = data_full["text"].apply(lambda x: ' '.join(x))
    colomn = []
    for string in data_full["text"]:
        punc = '''[]','''
    
        # Removing punctuations in string
        # Using loop + punctuation string
        for ele in string:
            if ele in punc:
                string = string.replace(ele, "")
        colomn.append(string)
    data_full["text"] = colomn

    data_full = data_full.rename(columns={"post_id": "Id", "text": "Text","final_label":"Label"})

    data_full.to_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\HateExplain\hateexplain_2_VUA.csv",index=False,sep="\t")
    print(data_full.head())


##################################
# Ethos conversion               #
##################################

vua_ethos = False
if vua_ethos == True:
    data_ethos = pd.read_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\ETHOS\ethos.csv",sep=";")
    data_ethos["Label"] = np.where(data_ethos['isHate']== 1, 'hate', 'noHate')


    data_ethos = data_ethos[["comment","Label"]]
    data_ethos = data_ethos.rename(columns={"comment": "Text"})

    headers = ["Id","Text","Label"]

    data_ethos.to_csv("Pipeline\ma-course-subjectivity-mining\pynlp\data\ETHOS\ethos_VUA.csv", index_label="Id",sep="\t")
    print(data_ethos.head())
