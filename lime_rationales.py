from os import sep
import pandas as pd
import rbo

prediction_data = pd.read_csv("Predictions\prediction_HateExplain_Bert.csv", sep="\t")
lime_data = pd.read_csv("Lime values\Values_HateExplain_Bert.csv",sep=",")
ratio_data = pd.read_csv("HateExplain_data\hateexplain_rationales.csv", sep="\t")


"""Take only rows from full rationales file that are in lime file"""
# Filter predictions on hate labelled cases
prediction_data_hate = prediction_data.loc[prediction_data["predictions"] == 1]

# Get tweet ids from hatefull cases
prediction_hate_ids = prediction_data_hate["Id"].unique()

# Filter full rationalset on only tweets predicted as hate
ratio_data_filtered = ratio_data[ratio_data["Id"].isin(prediction_hate_ids)]

# Filter Lime data on predicted as hate
lime_data_filtered = lime_data[lime_data["Tweet_id"].isin(prediction_hate_ids)]


def similarity(id):
    ratio_tweet = ratio_data_filtered.loc[ratio_data_filtered['Id'] == id].drop("Id", axis=1)
    lime_tweet = lime_data_filtered.loc[lime_data_filtered['Tweet_id'] == id].drop("Tweet_id", axis=1)
    df1 = ratio_tweet.sort_values(by=["Ratio"],ascending=False)
    
    df1 = df1.groupby("Word").mean()
    df2 = lime_tweet.sort_values(by=["Value"],ascending=False)
 
    list1 = list(df1.index)
    list2 = list(df2.Word)
    if len(list1) > len(list2):
        list1 = list1[:len(list2)]
    elif len(list1) < len(list2):
        list2 = list2[:len(list1)]

    # print(len(list1),len(list2))

    return round(rbo.RankingSimilarity(list1,list2).rbo(),3)

sim_list = []
for id in prediction_hate_ids:
    sim_list.append(similarity(id))

df_sim = pd.DataFrame({"Id": prediction_hate_ids, "Similarity":sim_list})

print("Average similarity:", round(df_sim.Similarity.mean(),3))

df_sim.to_csv("Similarity/bert_hate_sim.csv")