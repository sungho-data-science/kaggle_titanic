import pandas as pd

def preprocess(df):

    # generate a feature called "last_name" from "Name"
    df["last_name"] = df["Name"].apply(lambda x: x.split(',')[0].strip())


    # generate a feature called "title" from "Name"
    title_list = ["Mr.","Miss.","Mrs.","Master.", "Dr.", "Rev."]

    def extract_title(name):
        for title in title_list:
            if title in name:
                return title
        return None

    # title from Name
    df["title"] = df["Name"].apply(lambda x: extract_title(x))
    df.loc[df["title"].isna(),:]
    print("'title' feature is created")

    # Age imputation
    age_reference_df = df[["Pclass","Sex","title","Age","count"]].groupby(["Pclass","Sex","title"]).agg({"count":"sum","Age":["mean"]}).reset_index()
    age_reference_df.columns = [c[0] + "_" + c[1] for c in age_reference_df.columns]
    age_reference_df

    def age_imputation(cols):
        Pclass = cols["Pclass"]
        Sex = cols["Sex"]
        title = cols["title"]
        Age = cols["Age"]
        value = age_reference_df.loc[(age_reference_df["Pclass_"]==Pclass)&(age_reference_df["Sex_"]==Sex)&(age_reference_df["title_"]==title),"Age_mean"]
        if pd.isnull(Age):
            if len(value) == 1:
                return value.item()
            else:
                return Age
        else:
            return Age
        
    df["Age"] = df[["Pclass","Sex","title","Age"]].apply(age_imputation,axis=1)


    # relatives from SibSp & Parch
    df["relatives"] = df["SibSp"] + df["Parch"] + 1
    print("'relative' feature is created")

    # alone from SibSp & Parch
    df.loc[df["SibSp"] + df["Parch"] == 0, "alone"] = "Yes"
    df.loc[df["SibSp"] + df["Parch"] > 0, "alone"] = "No"
    print("'alone' feature is created")

    # Ticket_len from Ticket
    df["Ticket_len"] = df["Ticket"].apply(lambda x: len(x))
    print("'Ticket_len' feature is created")

    # Cabin_len from Ticket
    df["Cabin"] = df["Cabin"].fillna("unknown")
    df["Cabin_len"] = df["Cabin"].apply(lambda x: len(x))

    # Pclass dtype to category
    df["Pclass"]=df["Pclass"].astype('category')
    print("dtype of Pclass has been changed to category")

    # Sex dtype to category
    df["Sex"]=df["Sex"].astype('category')
    print("dtype of Sex has been changed to category")

    # Cabin dtype to category
    df["Cabin"]=df["Cabin"].astype('category')
    print("dtype of Cabin has been changed to category")

    # Embarked dtype to category
    df["Embarked"]=df["Embarked"].astype('category')
    print("dtype of Embarked has been changed to category")

    # last_name dtype to category
    df["last_name"]=df["last_name"].astype('category')
    print("dtype of last_name has been changed to category")

    # title dtype to category
    df["title"]=df["title"].astype('category')
    print("dtype of title has been changed to category")

    # alone dtype to category
    df["alone"]=df["alone"].astype('category')
    print("dtype of alone has been changed to category")

    # alone dtype to category
    df["Survived"]=df["Survived"].astype('Int64', errors='ignore')
    print("dtype of Survived has been changed to int")

    return df