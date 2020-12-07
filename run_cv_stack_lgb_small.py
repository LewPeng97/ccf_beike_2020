import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from transformers import BertModel, AutoModel, BertForNextSentencePrediction, BertTokenizer, BertForQuestionAnswering,XLNetModel,XLNetTokenizer
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelWithLMHead, get_linear_schedule_with_warmup
import torch.nn.functional as F
import random
from sklearn.model_selection import KFold
# from log import Logger
from datetime import datetime
import lightgbm as lgb
# from bayes_opt import BayesianOptimization
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/text_experiment')
'''
使用具有NSP任务的预训练模型
'''
seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed) # Numpy module.
random.seed(seed) # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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


if __name__ == '__main__':
    base_path = "/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main"
    """
    目前共6个模型
    """
    model_choose = ['chinese-bert-wwm-ext','chinese_roberta_wwm_large_ext','roberta-zh-large','chinese-xlnet-mid','chinese-xlnet-base','nezha-large']#,'nezha-large'
    model_name = '/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main/bert_pretrain/hfl/' + str(model_choose) + '/'
    save_model_dir = base_path+'/save_model'


    train_data, train_labels = load_features(base_path+'/save_model/',model_choose)
    test_data = load_features(base_path+'/save_model/',model_choose,False)

    # 记录多个模型的预测结果
    total_res_prob = np.zeros((len(test_data), 2))
    # 初始化lgm模型
    params = {
        'task': 'train',
        'boosting_type': 'dart',  # 设置提升类型
        'objective': 'multiclass',
        'num_class': 2,  
        'metric': 'multi_error',
    
        'save_binary': True,
        'max_bin': 63,
        'bagging_fraction': 0.4,
        'bagging_freq': 5,
    
        'feature_fraction': 0.4716469149933727,
        'lambda_l1':  3.109640893457194,
        'lambda_l2': 4.993050450697515,
        'learning_rate': 0.49773024766558066,
        'max_depth': 104,
        'min_data_in_leaf': 31,
        'min_gain_to_split': 0.6227038412868907,
        'min_sum_hessian_in_leaf': 4,
        'num_leaves': 65,
        'rounds': 30
    }
    # feature_num = len(Multi_lr)*2
    fold_num = 10          # 10 cross validation
    per_fold_num = len(train_data) // 11
    total_train_features = []
    total_test_features = []
    labels = list(train_labels)
    train_x = train_data
    test_x = test_data

    print("Begin train the lightgbm model!!")
    # print('train shape: {}, label shape: {}, test shape: {}'.format(train_x.shape, train_labels.shape, train_labels.shape))
    # lightgbm train the features by multi different model
    for fold in range(fold_num):
        # predeal train features
        x_train = np.concatenate((train_x[0:fold* per_fold_num], train_x[(fold+1)*per_fold_num:]), axis = 0)
        y_train = np.hstack((train_labels[0:fold*per_fold_num], train_labels[(fold+1)*per_fold_num:]))
        # deal valid features
        x_val = train_x[fold*per_fold_num:(fold+1)*per_fold_num]
        y_val = train_labels[fold*per_fold_num:(fold+1)*per_fold_num]
        # lgb deal the dataset
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val, reference = lgb_train)
        #  lightgbm train the features
        gbm = lgb.train(params, lgb_train, num_boost_round=int(params['rounds']), valid_sets= lgb_eval, early_stopping_rounds=20, verbose_eval=0)
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
            p = confusion[i, i].item() / confusion[i, :].sum().item()
            r = confusion[i, i].item() / confusion[:, i].sum().item()
            f1 = 2*p*r/(p+r)#分母+1防止除数为0，理论上不应该+1
            marco_f1.append(f1)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}.".format(i, p, r, f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(val_pred), correct, len(val_pred)))
        total_res_prob += gbm.predict(test_x)
    res = np.argmax(total_res_prob, axis=1)
    submit = pd.DataFrame()
    train_data = pd.read_csv(base_path+'/data/train.csv')
    test_data = pd.read_csv(base_path+'/data/test.csv')
    submit['id'] = test_data['id']
    submit['reply_sort'] = test_data['reply_sort']
    submit['label'] = res
    str_time = datetime.now().strftime('%Y%m%d%H%M%S')
    print('save submit files is : {}'.format(str_time))
    submit.to_csv(base_path+'/data/submit_cv_stack_{}.tsv'.format(str_time), sep='\t', header=False, index=False)
