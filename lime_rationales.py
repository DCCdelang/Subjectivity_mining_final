from os import sep
import pandas as pd
import rbo

model = "RoBerta" # ["roBerta","Bert","HateBert"]
prediction_data = pd.read_csv("Predictions\prediction_HateExplain_"+model+".csv", sep="\t")
lime_data = pd.read_csv("Lime values\Values_HateExplain_"+model+".csv",sep=",")
ratio_data = pd.read_csv("HateExplain_data\hateexplain_rationales.csv", sep="\t")


"""Take only rows from full rationales file that are in lime file"""
# Filter predictions on hate labelled cases
prediction_data_hate = prediction_data.loc[prediction_data["predictions"] == 1]
prediction_data_nohate = prediction_data.loc[prediction_data["predictions"] == 0]

# Get tweet ids from hatefull cases
prediction_hate_ids = prediction_data_hate["Id"].unique()
prediction_nohate_ids = prediction_data_nohate["Id"].unique()

# Filter full rationalset on only tweets predicted as hate
ratio_data_filtered = ratio_data[ratio_data["Id"].isin(prediction_hate_ids)]

# Filter Lime data on predicted as hate
lime_data_filtered = lime_data[lime_data["Tweet_id"].isin(prediction_hate_ids)]
lime_data_filtered_no = lime_data[lime_data["Tweet_id"].isin(prediction_nohate_ids)]

print(lime_data_filtered.shape)
print(lime_data_filtered_no.shape)

def similarity_RBO(id):
    ratio_tweet = ratio_data_filtered.loc[ratio_data_filtered['Id'] == id].drop("Id", axis=1)
    lime_tweet = lime_data_filtered.loc[lime_data_filtered['Tweet_id'] == id].drop("Tweet_id", axis=1)
    df1 = ratio_tweet.sort_values(by=["Ratio"],ascending=False)
    
    df1 = df1.groupby("Word").mean().sort_values(by=["Ratio"],ascending=False)
    df2 = lime_tweet.sort_values(by=["Value"],ascending=False)
 
    list1 = list(df1[df1.Ratio > 0].index)
    list2 = list(df2.Word)[:len(list1)]

    return round(rbo.RankingSimilarity(list1,list2).rbo(),3)

def IOU(lst1, lst2):
    overlap = list(set(lst1+lst2))
    intersection = list(set(lst1) & set(lst2))
    # print(intersection, overlap)
    if len(intersection) > 0 and len(overlap) > 0:
        return len(intersection)/ len(overlap)
    else:
        return 0

def similarity_IOU(id):
    ratio_tweet = ratio_data_filtered.loc[ratio_data_filtered['Id'] == id].drop("Id", axis=1)
    lime_tweet = lime_data_filtered.loc[lime_data_filtered['Tweet_id'] == id].drop("Tweet_id", axis=1)
    df1 = ratio_tweet.sort_values(by=["Ratio"],ascending=False)
    
    df1 = df1.groupby("Word").mean().sort_values(by=["Ratio"],ascending=False)
    df2 = lime_tweet.sort_values(by=["Value"],ascending=False)
 
    list1 = list(df1[df1.Ratio > 0.5].index)
    list2 = list(df2[:len(list1)].Word)

    return IOU(list1,list2)

sim_list = []
for id in prediction_hate_ids:
    sim_list.append(similarity_IOU(id))

df_sim = pd.DataFrame({"Id": prediction_hate_ids, "Similarity":sim_list})

# RBO
# print("Average RBO similarity "+model, round(df_sim.Similarity.mean(),3))

# IOU 
print("Average IOU similarity "+model, round(df_sim.Similarity.mean(),3))

# df_sim.to_csv("Similarity/"+model+"_hate_sim_IOU.csv")