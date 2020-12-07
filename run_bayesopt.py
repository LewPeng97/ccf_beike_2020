import warnings
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import numpy as np

def load_features(feature_path,model_name,flag=True):
    features = []
    if flag:
        for choose in model_name:
            train_x = np.load(feature_path + str(choose)+'_train_features.npz')
            train_features = train_x['arr_0']
            train_labels = train_x['arr_1']
            features.append(train_features)
        features = np.concatenate(features, axis=-1)
        labels = train_labels
        return features,labels
    else:
        for choose in model_name:
            train_x = np.load(feature_path + str(choose)+'_test_features.npz')
            test_features = train_x['arr_0']
            features.append(test_features)
        features = np.concatenate(features, axis=-1)
        return features

def process():
    base_path = "/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main"
    """
    目前共6个模型
    """
    model_choose = ['chinese-bert-wwm-ext', 'chinese_roberta_wwm_large_ext', 'roberta-zh-large', 'chinese-xlnet-mid',
                    'chinese-xlnet-base', 'nezha-large']  # ,'nezha-large'
    model_name = '/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main/bert_pretrain/hfl/' + str(
        model_choose) + '/'
    save_model_dir = base_path + '/save_model'

    train_data, train_labels = load_features(base_path + '/save_model/', model_choose)
    test_data = load_features(base_path + '/save_model/', model_choose, False)

    def lgb_cv(num_leaves,
               min_data_in_leaf,
               learning_rate,
               min_sum_hessian_in_leaf,
               feature_fraction,
               lambda_l1,
               lambda_l2,
               min_gain_to_split,
               max_depth
               ):
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)
        params = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'learning_rate': learning_rate,
            'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_depth': max_depth,
            'save_binary': True,
            'max_bin': 63,
            'bagging_fraction': 0.4,
            'bagging_freq': 5,
            'seed': 2017,
            'objective': 'multiclass',
            'num_class': 2,
            'boosting_type': 'dart',
            'verbose': -1,
            'metric': 'multi_error',
        }

        scores = []

        fold_num = 10  # 10 cross validation
        per_fold_num = len(train_data) // 11
        train_x = train_data
        test_x = test_data

        for fold in range(fold_num):

            x_train = np.concatenate((train_x[0:fold * per_fold_num], train_x[(fold + 1) * per_fold_num:]), axis=0)
            y_train = np.hstack((train_labels[0:fold * per_fold_num], train_labels[(fold + 1) * per_fold_num:]))
            # deal valid features
            x_val = train_x[fold * per_fold_num:(fold + 1) * per_fold_num]
            y_val = train_labels[fold * per_fold_num:(fold + 1) * per_fold_num]
            # lgb deal the dataset
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
            #  lightgbm train the features
            gbm = lgb.train(params, lgb_train, num_boost_round=150, valid_sets=lgb_eval, early_stopping_rounds=30, verbose_eval=0)
            # lightgbm pred the test dataset
            val_pred = gbm.predict(x_val)
            val_pred = np.argmax(val_pred, axis=1)
            confusion = np.zeros((2, 2))
            for i in range(len(val_pred)):
                confusion[val_pred[i], y_val[i]] += 1
            correct = np.sum(val_pred == y_val)
            marco_f1 = []
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")
            for i in range(2):
                try:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                    f1 = 2 * p * r / (p + r)  # 分母+1防止除数为0，理论上不应该+1
                except ZeroDivisionError:
                    f1 = 0
                marco_f1.append(f1)
            scores.append(np.mean(marco_f1))
            return np.mean(scores)

    bounds = {
        'num_leaves': (20, 90),
        'min_data_in_leaf': (5, 100),
        'learning_rate': (0.005, 0.5),
        'min_sum_hessian_in_leaf': (0.00001, 20),
        'feature_fraction': (0.001, 0.5),
        'lambda_l1': (0, 10),
        'lambda_l2': (0, 10),
        'min_gain_to_split': (0, 1.0),
        'max_depth': (3, 200),
    }

    lgb_bo = BayesianOptimization(lgb_cv,bounds)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        lgb_bo.maximize(n_iter=5)
    print('***************')
    print(lgb_bo.max)

if __name__ == '__main__':
    process()