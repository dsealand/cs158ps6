import numpy as np

######################################################################
# test data.py
######################################################################

def test_process_record(features) :
    nan = np.nan
    exp_features = {
        'Age': 54.0,
        'Gender': 0.0,
        'Height': nan,
        'ICUType': 4.0,
        'Weight': nan,
        'mean_ALP': nan,
        'mean_ALT': nan,
        'mean_AST': nan,
        'mean_Albumin': nan,
        'mean_BUN': 10.5,
        'mean_Bilirubin': nan,
        'mean_Cholesterol': nan,
        'mean_Creatinine': 0.75,
        'mean_DiasABP': nan,
        'mean_FiO2': nan,
        'mean_GCS': 14.923076923076923,
        'mean_Glucose': 160.0,
        'mean_HCO3': 27.0,
        'mean_HCT': 32.5,
        'mean_HR': 70.8108108108108,
        'mean_K': 4.2,
        'mean_Lactate': nan,
        'mean_MAP': nan,
        'mean_Mg': 1.7,
        'mean_NIDiasABP': 50.14705882352941,
        'mean_NIMAP': 71.55911764705883,
        'mean_NISysABP': 114.38235294117646,
        'mean_Na': 136.5,
        'mean_PaCO2': nan,
        'mean_PaO2': nan,
        'mean_Platelets': 203.0,
        'mean_RespRate': 17.428571428571427,
        'mean_SaO2': nan,
        'mean_SysABP': nan,
        'mean_Temp': 37.357142857142854,
        'mean_TroponinI': nan,
        'mean_TroponinT': nan,
        'mean_Urine': 171.05263157894737,
        'mean_WBC': 10.3,
        'mean_pH': nan
    }
    
    assert features.keys() == exp_features.keys()
    for key in features.keys() :
        np.testing.assert_almost_equal(features[key], exp_features[key])


def test_Vectorizer(df_features, df_labels) :
    import icu_featurize
    
    rid = df_labels['RecordID'][0] # 132539
    avg_vect = icu_featurize.Vectorizer()
    one_df = df_features[df_features['RecordID'] == rid]
    test = avg_vect.fit_transform(one_df)
    
    nan = np.nan
    exp = np.array(
        [[ 54.        ,   0.        ,          nan,   4.        ,
                    nan,          nan,          nan,          nan,
                    nan,  10.5       ,          nan,          nan,
            0.75      ,          nan,          nan,  14.92307692,
            160.        ,  27.        ,  32.5       ,  70.81081081,
            4.2       ,          nan,          nan,   1.7       ,
            50.14705882,  71.55911765, 114.38235294, 136.5       ,
                    nan,          nan, 203.        ,  17.42857143,
                    nan,          nan,  37.35714286,          nan,
                    nan, 171.05263158,  10.3       ,          nan]])
    
    np.testing.assert_almost_equal(test, exp)



######################################################################
# test icu_practice.py
######################################################################

def test_score() :
    import icu_practice

    # np.random.seed(1234)
    # y_true = 2 * np.random.randint(0,2,10) - 1
    # np.random.seed(2345)
    # y_pred = (10 + 10) * np.random.random(10) - 10
    
    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    #y_pred = [ 1, -1,  1, -1,  1,  1, -1, -1,  1, -1]
    # confusion matrix
    #          pred pos     neg
    # true pos      tp (2)  fn (4)
    #      neg      fp (3)  tn (1)
    y_pred      = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
                    2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics     = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    exp_scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]
    
    for i, metric in enumerate(metrics) :
        test = icu_practice.score(y_true, y_pred, metric)
        np.testing.assert_almost_equal(test, exp_scores[i])

