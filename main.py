import pandas as pd
import numpy as np
import os.path as osp
import argparse
import os
import time
import torch
import logging
import random
import joblib
import sklearn
import scipy

from sklearn.svm import SVR
from collections import OrderedDict
from datetime import datetime
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

config = OrderedDict()

config['root_project'] = '/home/m24linyc/DL/project_hormones/'
config['root_data'] = config['root_project'] + 'data/train.csv'
config['root_test_data'] = config['root_project'] + 'data/test.csv'
config['label'] = 'EPA_titer'  #'EPA_titer', 'EPA(%)', 'lipid_titer'
config['features'] = ['Bet', 'CPPU', 'ETA', 'DEA', 'GA', 'MI', 'MT']

# config['model'] = 'XGBoost'
# config['model'] = 'RandomForest'
# config['model'] = 'SVM'
# config['model'] = 'LightGBM'
# config['model'] = 'CatBoost'
config['model'] = 'Tabpfn'

# config['mode'] = 'train_valid'
config['mode'] = 'test'
# config['mode'] = 'predict'

# config['kfold'] = True
config['kfold'] = False

if config['kfold']:
    config['n_folds'] = 5  # 5/10
    if config['n_folds'] == 5:
        config['fold'] = [0, 1, 2, 3, 4]
    elif config['n_folds'] == 10:
        config['fold'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

config['save_metrics'] = True
config['save_model'] = True

# ============================================================
HYPER_PARAMS = {
    'EPA_titer': {
        'XGBoost': {
            'max_depth': 8,
            'min_child_weight': 1,
            'learning_rate': 0.012491901625729832,
            'n_estimators': 934,
            'subsample': 0.7596039443375517,
            'colsample_bytree': 0.8992855504166716,
            'colsample_bylevel': 0.7005690944602841,
            'reg_lambda': 3.689452545562362,
            'reg_alpha': 0.033871730055488845,
            'gamma': 0.00045751854945076187,
        },
        'SVM': {
            'kernel': 'rbf',
            'C': 848.4800410911824,
            'gamma': 0.0008314942864320529,
            'epsilon': 0.010258676247432607,
            'degree': 5,
        },
        'RandomForest': {
            'n_estimators': 471,
            'max_features': None,
            'max_depth': 78,
            'min_samples_split': 4,
            'min_samples_leaf': 2,
        },
        'LightGBM': {
            'num_leaves': 71,
            'max_depth': 9,
            'learning_rate': 0.15314960761118335,
            'n_estimators': 281,
            'subsample': 0.7979387222948192,
            'colsample_bytree': 0.8467406599433578,
            'reg_alpha': 0.2550309345897846,
            'reg_lambda': 1.236299752334129,
            'min_child_samples': 9,
            'min_child_weight': 1.0235928475258194e-05,
        },
        'CatBoost': {
            'iterations': 481,
            'depth': 4,
            'learning_rate': 0.17904890835649712,
            'l2_leaf_reg': 8.965458791007574,
        },
        'Tabpfn': {
            'n_estimators': 31,
            'softmax_temperature': 0.41286972973122793,
        },
    },
    'lipid_titer': {
        'XGBoost': {
            'max_depth': 7,
            'min_child_weight': 5,
            'learning_rate': 0.010700288026815047,
            'n_estimators': 372,
            'subsample': 0.6350165354251748,
            'colsample_bytree': 0.8751167348032087,
            'colsample_bylevel': 0.8450000327603486,
            'reg_lambda': 0.3445268729963732,
            'reg_alpha': 0.22225589700233916,
            'gamma': 0.3558365783742249,
        },
        'SVM': {
            'kernel': 'linear',
            'C': 1.155442983166083,
            'gamma': 0.001059779516169882,
            'epsilon': 0.09856809309646702,
            'degree': 3,
        },
        'RandomForest': {
            'n_estimators': 622,
            'max_features': None,
            'max_depth': 100,
            'min_samples_split': 11,
            'min_samples_leaf': 1,
        },
        'LightGBM': {
            'num_leaves': 34,
            'max_depth': 8,
            'learning_rate': 0.010031028994291478,
            'n_estimators': 311,
            'subsample': 0.8752067690763183,
            'colsample_bytree': 0.8762082345140251,
            'reg_alpha': 0.5107613914861606,
            'reg_lambda': 0.6079164806392277,
            'min_child_samples': 8,
            'min_child_weight': 0.00019709596198878182,
        },
        'CatBoost': {
            'iterations': 462,
            'depth': 4,
            'learning_rate': 0.10892255308033824,
            'l2_leaf_reg': 1.7369303943243293,
        },
        'Tabpfn': {
            'n_estimators': 1,
            'softmax_temperature': 1.8276027831785728,
        },
    },
    'EPA(%)': {
        'XGBoost': {
            'max_depth': 12,
            'min_child_weight': 2,
            'learning_rate': 0.03262075915978754,
            'n_estimators': 903,
            'subsample': 0.8847617624973194,
            'colsample_bytree': 0.5690472025115232,
            'colsample_bylevel': 0.5682141775117585,
            'reg_lambda': 1.0059368613394202,
            'reg_alpha': 0.014070661265883494,
            'gamma': 0.0016035181605752126,
        },
        'SVM': {
            'kernel': 'linear',
            'C': 5.717863273506239,
            'gamma': 5.182075366634185,
            'epsilon': 0.03413620289453398,
            'degree': 4,
        },
        'RandomForest': {
            'n_estimators': 256,
            'max_features': 'sqrt',
            'max_depth': 33,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
        },
        'LightGBM': {
            'num_leaves': 89,
            'max_depth': 5,
            'learning_rate': 0.07969854665414111,
            'n_estimators': 709,
            'subsample': 0.8545542854363837,
            'colsample_bytree': 0.8311124089213767,
            'reg_alpha': 0.021159681577900716,
            'reg_lambda': 0.5879946294294206,
            'min_child_samples': 13,
            'min_child_weight': 0.6062302790112071,
        },
        'CatBoost': {
            'iterations': 492,
            'depth': 4,
            'learning_rate': 0.19650974060208964,
            'l2_leaf_reg': 7.92135427854693,
        },
        'Tabpfn': {
            'n_estimators': 18,
            'softmax_temperature': 0.9134562731480682,
        },
    },
}


def get_model_params(config):
    label = config['label']
    model = config['model']
    if label not in HYPER_PARAMS:
        raise ValueError(f"Unsupported prediction target: {label}. Please select 'EPA_titer' / 'EPA(%)' / 'lipid_titer' in config['label']")
    if model not in HYPER_PARAMS[label]:
        raise ValueError(f"Unsupported model: {model}. Please select from the keys of HYPER_PARAMS['{label}']")
    return HYPER_PARAMS[label][model]


if config['mode'] == 'test':
    checkpoint_root = config['root_project'] + 'results/checkpoints/'
    config['checkpoint_dir'] = checkpoint_root + config['label'] + '/' + config['model']


#########################################################################DATA###########################################################################
def load_dataframe(args, config):
    if config['kfold']:
        df = pd.read_csv(config['root_data'])
        df.rename(columns={config['label']: 'label'}, inplace=True)
        df.fillna(0, inplace=True)
        df["fold"] = -1
        kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
            df.loc[val_idx, "fold"] = fold_idx
        return df
    else:
        df = pd.read_csv(config['root_data'])
        df.rename(columns={config['label']: 'label'}, inplace=True)
        df.fillna(0, inplace=True)
        return df


class Trainer(object):
    def __init__(self, args, config):
        self.logger = args.logger

        if config['mode'] == 'train_valid' or config['mode'] == 'test':
            self.df_all = load_dataframe(args, config)
        elif config['mode'] == 'predict':
            self.df_predict = load_dataframe(args, config)
        else:
            exit()

        params = get_model_params(config)

        if config['model'] == 'XGBoost':
            self.model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                booster='gbtree',
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                colsample_bylevel=params['colsample_bylevel'],
                reg_lambda=params['reg_lambda'],
                reg_alpha=params['reg_alpha'],
                gamma=params['gamma'],
                device=args.device,
                sampling_method='uniform',
                n_jobs=args.num_workers,
                random_state=args.seed,
            )
        elif config['model'] == 'SVM':
            self.model = SVR(
                kernel=params['kernel'],
                C=params['C'],
                gamma=params['gamma'],
                degree=params['degree'],
                epsilon=params['epsilon'],
            )
        elif config['model'] == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_features=params['max_features'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf']
            )
        elif config['model'] == 'LightGBM':
            self.model = LGBMRegressor(
                num_leaves=params['num_leaves'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_samples=params['min_child_samples'],
                min_child_weight=params['min_child_weight']
            )
        elif config['model'] == 'CatBoost':
            self.model = CatBoostRegressor(
                iterations=params['iterations'],
                depth=params['depth'],
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
            )
        elif config['model'] == 'Tabpfn':
            self.model = TabPFNRegressor(
                model_path="/home/m24linyc/.cache/tabpfn/.cache/huggingface/download/tabpfn-v2-regressor.ckpt",
                device="cuda:0",
                n_estimators=params['n_estimators'],
                softmax_temperature=params['softmax_temperature'],
            )

        self.args = args
        self.config = config

    def run(self, fold=0):
        if self.config['kfold']:
            df_train = self.df_all[self.df_all.fold != fold].reset_index(drop=True).drop('fold', axis=1)
            df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True).drop('fold', axis=1)
            x_train = df_train[config['features']].to_numpy()
            y_train = df_train['label'].to_numpy()
            x_valid = df_valid[config['features']].to_numpy()
            y_valid = df_valid['label'].to_numpy()
        else:
            df = self.df_all.sample(frac=1, random_state=self.args.seed).reset_index(drop=True)
            split_idx = int(len(df) * 0.8)
            df_train = df[:split_idx]
            df_valid = df[split_idx:]
            x_train = df_train[config['features']].to_numpy()
            y_train = df_train['label'].to_numpy()
            x_valid = df_valid[config['features']].to_numpy()
            y_valid = df_valid['label'].to_numpy()

        start_time = time.time()

        if config['label'] == 'EPA_titer':
            y_train = y_train * 100

        if self.config['model'] in ['SVM'] and self.config['kfold']:
            self.args.logger.info('Scaling...')
            scaler = StandardScaler().fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_valid_scaled = scaler.transform(x_valid)
            self.model.fit(x_train_scaled, y_train)
            y_predict_proba = self.model.predict(x_valid_scaled)
            if config['label'] == 'EPA_titer':
                y_predict_proba = y_predict_proba / 100
            val_metrics = self.compute_metrics(y_valid, y_predict_proba)
        else:
            self.model.fit(x_train, y_train)
            y_predict_proba = self.model.predict(x_valid)
            if config['label'] == 'EPA_titer':
                y_predict_proba = y_predict_proba / 100
            val_metrics = self.compute_metrics(y_valid, y_predict_proba)

        elapsed = time.time() - start_time

        if config['kfold']:
            self.logger.info("Fold[{}] Time {}s || Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                            .format(
                                fold+1,
                                round(elapsed,3),
                                round(val_metrics['pearsonr_corr_coefficient'], 4),
                                round(val_metrics['pearsonr_p'], 4),
                                round(val_metrics['spearmanr_corr_coefficient'], 4),
                                round(val_metrics['spearmanr_p'], 4),
                                round(val_metrics['mse'], 4),
                            )
            )
        else:
            self.logger.info("Time {}s || Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                            .format(
                                round(elapsed,3),
                                round(val_metrics['pearsonr_corr_coefficient'], 4),
                                round(val_metrics['pearsonr_p'], 4),
                                round(val_metrics['spearmanr_corr_coefficient'], 4),
                                round(val_metrics['spearmanr_p'], 4),
                                round(val_metrics['mse'], 4),
                            )
            )

        if self.config['save_model']:
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)

            if self.config['kfold']:
                joblib.dump(self.model, os.path.join(self.args.checkpoint_dir, "last"+str(fold)+".pkl"))
            else:
                joblib.dump(self.model, os.path.join(self.args.checkpoint_dir, "last.pkl"))
            self.logger.info("Dumped last model.")

        if self.config['save_metrics']:
            df_metrics = pd.DataFrame([val_metrics]).T
            df_metrics.columns = ['val_metrics']
            if self.config['kfold']:
                df_metrics.to_csv(osp.join(self.args.metrics_dir, 'metrics'+str(fold)+'.csv'), sep=',', index=True, header=True)
            else:
                df_metrics.to_csv(osp.join(self.args.metrics_dir, 'metrics.csv'), sep=',', index=True, header=True)

        return val_metrics

    def test(self, fold=0):
        if self.config['kfold']:
            df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True).drop('fold', axis=1)
            x_valid = df_valid[config['features']].to_numpy()
            y_valid = df_valid['label'].to_numpy()
            self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last"+str(fold)+".pkl"))
        else:
            test_df = pd.read_csv(config['root_test_data'])
            test_df.rename(columns={config['label']: 'label'}, inplace=True)
            test_df.fillna(0, inplace=True)
            x_valid = test_df[config['features']].to_numpy()
            y_valid = test_df['label'].to_numpy()
            self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last.pkl"))

        self.logger.info("Loaded model.")

        start_time = time.time()
        y_predict = self.model.predict(x_valid)
        if config['label'] == 'EPA_titer':
            y_predict = y_predict / 100
        val_metrics = self.compute_metrics(y_valid, y_predict)
        elapsed = time.time() - start_time

        if config['kfold']:
            self.logger.info("Fold[{}] Time {}s || Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                        .format(
                            fold,
                            round(elapsed,3),
                            round(val_metrics['pearsonr_corr_coefficient'], 4),
                            round(val_metrics['pearsonr_p'], 4),
                            round(val_metrics['spearmanr_corr_coefficient'], 4),
                            round(val_metrics['spearmanr_p'], 4),
                            round(val_metrics['mse'], 4)
                        )
            )
        else:
            self.logger.info("Time {}s || Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                        .format(
                            round(elapsed,3),
                            round(val_metrics['pearsonr_corr_coefficient'], 4),
                            round(val_metrics['pearsonr_p'], 4),
                            round(val_metrics['spearmanr_corr_coefficient'], 4),
                            round(val_metrics['spearmanr_p'], 4),
                            round(val_metrics['mse'], 4)
                        )
            )

        df_compare = pd.DataFrame({
            'y_true': y_valid,
            'y_predict': y_predict
        })

        if self.config['kfold']:
            save_name = 'pred_vs_true_fold{}.csv'.format(fold)
        else:
            save_name = 'pred_vs_true.csv'
        save_path = osp.join(self.args.metrics_dir, save_name)
        df_compare.to_csv(save_path, sep=',', index=False)
        self.logger.info("Prediction results saved to: {}".format(save_path))

        if self.config['save_metrics']:
            df_metrics = pd.DataFrame([val_metrics]).T
            df_metrics.columns = ['val_metrics']
            if self.config['kfold']:
                df_metrics.to_csv(osp.join(self.args.metrics_dir, 'metrics'+str(fold)+'.csv'), sep=',', index=True, header=True)
            else:
                df_metrics.to_csv(osp.join(self.args.metrics_dir, 'metrics.csv'), sep=',', index=True, header=True)

        return val_metrics

    def predict(self):
        x_valid = self.df_predict.to_numpy()
        self.model = joblib.load(os.path.join(self.config['checkpoint_dir'], "last.pkl"))
        self.logger.info("Loaded model.")

        start_time = time.time()
        y_predict = self.model.predict(x_valid)
        elapsed = time.time() - start_time

        df = pd.DataFrame(y_predict[:,-1])
        df.to_csv(osp.join(self.args.log_dir, 'predict.csv'), sep=',', index=False, header=False)
        return

    def compute_metrics(self, y_true, y_pred):
        pearsonr_corr_coefficient, pearsonr_p = scipy.stats.pearsonr(y_true, y_pred)
        spearmanr_corr_coefficient, spearmanr_p = scipy.stats.spearmanr(y_true, y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        return {
            'pearsonr_corr_coefficient': pearsonr_corr_coefficient,
            'pearsonr_p': pearsonr_p,
            'spearmanr_corr_coefficient': spearmanr_corr_coefficient,
            'spearmanr_p': spearmanr_p,
            'mse': mse,
        }


def get_current_time():
    current_time = str(datetime.fromtimestamp(int(time.time())))
    d = current_time.split(' ')[0]
    t = current_time.split(' ')[1]
    d = d.split('-')[0] + d.split('-')[1] + d.split('-')[2]
    t = t.split(':')[0] + t.split(':')[1] + t.split(':')[2]
    return d+ '_' +t

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logging_config(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'out.log')),
                                  logging.StreamHandler(os.sys.stdout)])

def set_logger(args, config):
    set_logging_config(args.log_dir)
    logger = logging.getLogger(' ')
    logger.info('Generated logs will be saved to: {}'.format(args.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args.checkpoint_dir))
    logger.info('')
    logger.info('------------------------------command line arguments------------------------------')
    logger.info(args)
    logger.info('')
    logger.info('--------------------------------------configs-------------------------------------')
    logger.info(config)
    return logger

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./results/logs', help='path that log will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints', help='the path of checkpoint')
    parser.add_argument('--gpu_ids', type=list, default='0')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    args = parser.parse_args()
    return args

def init_configs():
    args = parse()
    current_time = get_current_time()
    args.log_dir = os.path.join(args.log_dir, current_time)
    args.metrics_dir = os.path.join(args.log_dir, 'metrics')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, config['label'], config['model'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.metrics_dir):
        os.makedirs(args.metrics_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    args.logger = set_logger(args, config)
    set_seed(args)
    return args, config

def main():
    args, config = init_configs()
    if config['mode'] == 'predict':
        trainer = Trainer(args, config)
        trainer.predict()
    elif config['mode'] == 'train_valid' or config['mode'] == 'test':
        if config['kfold']:
            results_valid = {'Pearsonr_corr': [], 'Pearsonr_p': [], 'Spearmanr_corr': [], 'Spearmanr_p': [], 'MSE': []}
            for fold in range(config['n_folds']):
                args.logger.info('')
                args.logger.info("======================================== Starting fold {}: ========================================".format(fold))
                trainer = Trainer(args, config)
                if config['mode'] == 'train_valid':
                    res = trainer.run(fold)
                else:
                    res = trainer.test(fold)
                args.logger.info("Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                                 .format(
                                    res['pearsonr_corr_coefficient'],
                                    res['pearsonr_p'],
                                    res['spearmanr_corr_coefficient'],
                                    res['spearmanr_p'],
                                    res['mse']
                                )
                )
                results_valid['Pearsonr_corr'].append(res['pearsonr_corr_coefficient'])
                results_valid['Pearsonr_p'].append(res['pearsonr_p'])
                results_valid['Spearmanr_corr'].append(res['spearmanr_corr_coefficient'])
                results_valid['Spearmanr_p'].append(res['spearmanr_p'])
                results_valid['MSE'].append(res['mse'])

            args.logger.info("{}Fold-Mean - Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                            .format(
                                config['n_folds'],
                                np.mean(results_valid['Pearsonr_corr']),
                                np.mean(results_valid['Pearsonr_p']),
                                np.mean(results_valid['Spearmanr_corr']),
                                np.mean(results_valid['Spearmanr_p']),
                                np.mean(results_valid['MSE'])
                            )
            )
        else:
            trainer = Trainer(args, config)
            if config['mode'] == 'train_valid':
                res = trainer.run()
            else:
                res = trainer.test()
            args.logger.info("Val - Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}  MSE {}"
                            .format(
                                res['pearsonr_corr_coefficient'],
                                res['pearsonr_p'],
                                res['spearmanr_corr_coefficient'],
                                res['spearmanr_p'],
                                res['mse']
                            )
            )

if __name__ == '__main__':
    main()
