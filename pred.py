"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random
import numpy as np
import pandas as pd
import re

label_mapping = {0: 'Pizza', 1: 'Shawarma', 2: 'Sushi'}

feature_names = [

    "complexity",

    "num_ingredients",

    "s_wday_lunch",
    "s_wday_dinner",
    "s_wkend_lunch",
    "s_wkend_dinner",
    "s_party",
    "s_night_snack",

    "price",

    "m_cloudy_with_meatballs",
    "m_home_alone",
    "m_spiderman",
    "m_teenage_mutant",
    "m_avengers",
    "m_ratatouille",
    "m_aladdin",
    "m_kung_fu_panda",
    "m_jiro_dreams_of_sushi",
    "m_spirited_away",

    "d_water",
    "d_soda",
    "d_juice",
    "d_tea",
    "d_soup",
    "d_lemonade",
    "d_ginger_ale",

    "r_parents",
    "r_siblings",
    "r_siblings",
    "r_teachers",
    "r_strangers",

    "hot_sauce"

]

# maps how much hot sauce someone would add to an integer
hot_sauce_mapping = {
    "None":                                                   0,
    "A little (mild)":                                        1,
    "A moderate amount (medium)":                             2,
    "A lot (hot)":                                            3,
    "I will have some of this food item with my hot sauce":   4
}

weights = [
    [-2.1652163295987525e-01, -4.1025668306452438e-03,  2.2062419979052034e-01],
    [ 4.3894750690754852e-02,  8.2920734726549261e-02, -1.2681548541730425e-01],
    [-3.8010055787633855e-02,  2.3804609925968109e-01, -2.0003604347204759e-01],
    [ 1.7012436627438059e-02, -7.6765591631230992e-02,  5.9753155003792857e-02],
    [ 5.6449342124081326e-03,  9.8767418715655052e-03, -1.5521676083973705e-02],
    [ 1.0051538881186600e-01, -3.3213909962534488e-01,  2.3162371081347766e-01],
    [ 8.1487658907044336e-01, -6.3850621839190469e-01, -1.7637037067853881e-01],
    [ 3.1275813926857199e-01, -1.1688008147097061e-01, -1.9587805779760165e-01],
    [-3.4570257475787680e-02, -2.5773068340065829e-02,  6.0343325815853194e-02],
    [ 7.0057382952304847e-02, -2.9108298332234030e-02, -4.0949084620070776e-02],
    [ 1.5399926668111549e-01, -9.3661563035335474e-02, -6.0337703645780061e-02],
    [ 1.5248979005883193e-01, -7.0934702468327615e-02, -8.1555087590504213e-02],
    [ 1.5973232049622066e-01, -8.9591188406387343e-02, -7.0141132089833275e-02],
    [-2.7047174675740465e-01,  5.0860226951186416e-01, -2.3813052275445895e-01],
    [ 5.1724281519591998e-02, -3.3943281347160209e-02, -1.7781000172431730e-02],
    [-3.9108136737413077e-02,  6.4385558812621993e-02, -2.5277422075209020e-02],
    [-1.3266382511984505e-02, -1.1559806973509388e-02,  2.4826189485493862e-02],
    [-7.0520978717836469e-02, -6.6889094830086510e-02,  1.3741007354792295e-01],
    [-4.0379078495648833e-02, -3.4881409978643002e-02,  7.5260488474291787e-02],
    [-2.4055622537820207e-01,  2.4923144263135754e-02,  2.1563308111506665e-01],
    [ 6.1914752026125563e-01,  7.4770339267014809e-04, -6.1989522365392602e-01],
    [-3.4765352590692078e-02,  7.5546849241514655e-02, -4.0781496650822618e-02],
    [-1.9817455151014685e-01, -1.0198888976378924e-01,  3.0016344127393679e-01],
    [-2.8775767753632255e-02, -2.4058067208458635e-02,  5.2833834962090875e-02],
    [-1.4396780514603632e-02,  2.7893573185957584e-02, -1.3496792671353964e-02],
    [ 2.5503823338519024e-02,  1.0751042410613546e-03, -2.6578927579580369e-02],
    [-5.1830166559263639e-02, -1.4329202281365869e-01,  1.9512218937292219e-01],
    [ 5.8448087143276990e-02, -1.2732253787377684e-01,  6.8874450730499845e-02],
    [ 1.9134475199033590e-01, -1.4945770581994214e-01, -4.1887046170393356e-02],
    [ 2.7072149940841678e-01, -1.8272304160599553e-01, -8.7998457802420882e-02],
    [-1.6051739367019371e-01,  1.7432152466056766e-01, -1.3804130990373725e-02],
    [-1.0980225306707624e-01,  5.1175338929636183e-01, -4.0195113622928591e-01]
]

bias = [ 0.05844437817638963, -0.07822520295768688,  0.019780824781297102 ]


def get_num_ingredients(text):
    """
    Helper function to extract number of ingredients from survey responses.
    """
    if pd.isna(text):
        return 0
    text = str(text)
    # attempts to extract the last number in the response
    extracted_num = re.findall(r'\d+', text)
    if extracted_num:
        return int(extracted_num[-1])
    # if there is no number, returns the number of elements in the list split by commas
    # (assuming the response listed the ingredients instead of giving number)
    return len(text.split(','))

def process_data(filename):
    """
    Process the training data to extract features and labels.
    This code is here for demonstration purposes only.
    """
    data = pd.read_csv(filename)
    data.rename(columns={"Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "complexity"}, inplace=True)
    data.rename(columns={"Q2: How many ingredients would you expect this food item to contain?": "num_ingredients"}, inplace=True)
    data.rename(columns={"Q3: In what setting would you expect this food to be served? Please check all that apply": "setting"}, inplace=True)
    data.rename(columns={"Q4: How much would you expect to pay for one serving of this food item?": "price"}, inplace=True)
    data.rename(columns={"Q5: What movie do you think of when thinking of this food item?": "movie"}, inplace=True)
    data.rename(columns={"Q6: What drink would you pair with this food item?": "drink"}, inplace=True)
    data.rename(columns={"Q7: When you think about this food item, who does it remind you of?": "remind"}, inplace=True)
    data.rename(columns={"Q8: How much hot sauce would you add to this food item?": "hot_sauce"}, inplace=True)
    data_fets = np.stack([
        data["complexity"].astype(int),
        data["num_ingredients"].apply(get_num_ingredients),
        data["setting"].str.contains("Week day lunch", na=False).astype(int),
        data["setting"].str.contains("Week day dinner", na=False).astype(int),
        data["setting"].str.contains("Weekend lunch", na=False).astype(int),
        data["setting"].str.contains("Weekend dinner", na=False).astype(int),
        data["setting"].str.contains("At a party", na=False).astype(int),
        data["setting"].str.contains("Late night snack", na=False).astype(int),
        data["price"].astype(str).str.extract('(\d+)').fillna(0).astype(int).iloc[:, 0],
        data["movie"].str.lower().str.contains("cloudy|chance|meatball", na=False).astype(int),
        data["movie"].str.lower().str.contains("home alone", na=False).astype(int),
        data["movie"].str.lower().str.contains("spiderman|spider", na=False).astype(int),
        data["movie"].str.lower().str.contains("teenage|mutant|nunja|turtle", na=False).astype(int),
        data["movie"].str.lower().str.contains("avenger|avengers|endgame", na=False).astype(int),
        data["movie"].str.lower().str.contains("ratatouille", na=False).astype(int),
        data["movie"].str.lower().str.contains("aladdin|aladin|alladin", na=False).astype(int),
        data["movie"].str.lower().str.contains("kung|panda", na=False).astype(int),
        data["movie"].str.lower().str.contains("jiro|sushi", na=False).astype(int),
        data["movie"].str.lower().str.contains("spirited", na=False).astype(int),
        data["drink"].str.lower().str.contains("water", na=False).astype(int),
        data["drink"].str.lower().str.contains("pop|soft|coke|cola|pepsi|sprite|soda|carbonated|crush", na=False).astype(int),
        data["drink"].str.lower().str.contains("juice", na=False).astype(int),
        data["drink"].str.lower().str.contains("tea", na=False).astype(int),
        data["drink"].str.lower().str.contains("soup", na=False).astype(int),
        data["drink"].str.lower().str.contains("lemonade|lemon", na=False).astype(int),
        data["drink"].str.lower().str.contains("ginger ale|ginger", na=False).astype(int),
        data["remind"].str.contains("Parents", na=False).astype(int),
        data["remind"].str.contains("Siblings", na=False).astype(int),
        data["remind"].str.contains("Friends", na=False).astype(int),
        data["remind"].str.contains("Teachers", na=False).astype(int),
        data["remind"].str.contains("Strangers", na=False).astype(int),
        data["hot_sauce"].map(hot_sauce_mapping).fillna(0).astype(int)
    ])
    return data_fets

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, weights, bias):
    """
    predict class labels for input X
    """
    logits = np.dot(X.T, weights) + bias
    probs = softmax(logits)
    predicted_indices = np.argmax(probs, axis=1)  # Use argmax to get the index of the max probability
    predicted_labels = np.array([label_mapping[idx] for idx in predicted_indices])
    return predicted_labels

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    X = process_data(filename)

    predictions = predict(X, weights, bias)

    return predictions


