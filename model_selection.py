import os
import numpy as np
import pandas as pd
import random
import argparse
import logging
import time
import sys
import traceback

import optuna

from datetime import datetime
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor



config = OrderedDict()
config['root'] = '/home/m24linyc/DL/project_hormones/'
config['root_data'] = config['root'] + 'data/train.csv'

# ------------------------------------------ Data ------------------------------------------
config['label'] = 'lipid_titer'
# ------------------------------------------ Model ------------------------------------------
config['convert_type'] = 'CA_D'  # CA_D / AP_D / 
# config['convert_type'] = 'AP_D' 

# config['model'] = 'XGBoost' 
config['model'] = 'RandomForest' 
# config['model'] = 'Ridge' 
# config['model'] = 'SVM'
# config['model'] = 'LightGBM'
# config['model'] = 'CatBoost'
# config['model'] = 'TabPFN'
config['tabpfn_model_path'] = '/home/m24linyc/.cache/tabpfn/.cache/huggingface/download/tabpfn-v2-regressor.ckpt'  
# ---------------------------------------- Training ----------------------------------------

config['n_folds'] = 5  
config['n_trials'] = 500  
#############################################################################################################################
#############################################################################################################################

class OptunaLoggingCallback:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, study, trial):
        params_str = " | ".join([f"{k}={v}" for k, v in trial.params.items()])
        s = trial.user_attrs.get('spearman_mean', 0.0)
        p = trial.user_attrs.get('pearson_mean', 0.0)
        m = trial.user_attrs.get('mse_mean', 0.0)
        self.logger.info(f"[Trial {trial.number}] Spearman={s:.4f} | Pearson={p:.4f} | MSE={m:.6f} | {params_str}")


def load_dataframe(args, config):
    df = pd.read_csv(config['root_data'], delimiter=',')
    df.rename(columns={config['label']: 'label'}, inplace=True)
    df.fillna(0, inplace=True)

    # KFold
    df["fold"] = -1
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
        df.loc[val_idx, "fold"] = fold_idx
    
    return df
    
class Trainer(object):
    def __init__(self, args, config):
        self.logger = args.logger
        
        # load data
        self.df_all = load_dataframe(args, config)
        
        self.args = args
        self.config = config

    def run(self):
        df_full = self.df_all.drop('fold', axis=1)
        x_full = df_full.drop('label', axis=1)
        y_full = df_full['label']
        
        inner_kfold = KFold(n_splits=self.config['n_folds'], shuffle=True, random_state=self.args.seed)
        
        def cv_score(model_factory, use_scaler=False):
            scores = {'mse': [], 'pearson': [], 'spearman': []}
            for cv_train_idx, cv_val_idx in inner_kfold.split(x_full):
                x_cv_train = x_full.iloc[cv_train_idx]
                x_cv_val = x_full.iloc[cv_val_idx]
                y_cv_train = y_full.iloc[cv_train_idx]
                y_cv_val = y_full.iloc[cv_val_idx]
                
                if use_scaler:
                    scaler = StandardScaler()
                    x_cv_train = scaler.fit_transform(x_cv_train)
                    x_cv_val = scaler.transform(x_cv_val)
                
                model = model_factory()
                model.fit(x_cv_train, y_cv_train)
                preds = model.predict(x_cv_val)
                scores['mse'].append(mean_squared_error(y_cv_val, preds))
                
                try:
                    p = pearsonr(y_cv_val, preds)[0]
                    if np.isnan(p):
                        p = 0.0
                except Exception:
                    p = 0.0
                scores['pearson'].append(p)
                
                try:
                    s = spearmanr(y_cv_val, preds)[0]
                    if np.isnan(s):
                        s = 0.0
                except Exception:
                    s = 0.0
                scores['spearman'].append(s)
            
            return {
                k: {'mean': np.mean(v), 'folds': v} for k, v in scores.items()
            }
        
        optuna_callback = OptunaLoggingCallback(self.args.logger)
        
        if config['model'] == 'XGBoost':
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                }
                metrics = cv_score(lambda: XGBRegressor(
                    **params, objective='reg:squarederror', eval_metric='rmse',
                    booster='gbtree', device=self.args.device,
                    sampling_method='uniform', random_state=self.args.seed, n_jobs=1,
                ))
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.args.seed))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            final_model = XGBRegressor(**best_params, objective='reg:squarederror', eval_metric='rmse',
                booster='gbtree', device=self.args.device, sampling_method='uniform', random_state=self.args.seed, n_jobs=1)
            final_model.fit(x_full, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'Ridge':
            def objective(trial):
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 1000, log=True),
                    'max_iter': trial.suggest_int('max_iter', 50, 2000),
                    'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
                    'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                }
                metrics = cv_score(lambda: Ridge(**params, random_state=self.args.seed))
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.args.seed))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            final_model = Ridge(**best_params, random_state=self.args.seed)
            final_model.fit(x_full, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'SVM':
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-5, 1e1, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid', 'linear']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                    'degree': trial.suggest_int('degree', 2, 5),
                }
                metrics = cv_score(lambda: SVR(**params, tol=1e-4, max_iter=10000, cache_size=2000), use_scaler=True)
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, seed=self.args.seed),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, gc_after_trial=True, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            scaler = StandardScaler().fit(x_full)
            x_full_scaled = scaler.transform(x_full)
            final_model = SVR(**best_params, tol=1e-4, max_iter=10000, cache_size=2000)
            final_model.fit(x_full_scaled, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'RandomForest':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'max_depth': trial.suggest_int('max_depth', 10, 100),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 200),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
                }
                metrics = cv_score(lambda: RandomForestRegressor(**params, random_state=self.args.seed, verbose=0, n_jobs=1))
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, seed=self.args.seed),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            final_model = RandomForestRegressor(**best_params, random_state=self.args.seed, verbose=0, n_jobs=1)
            final_model.fit(x_full, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'LightGBM':
            device_search = 'cpu'
            device_final = 'gpu' if 'cuda' in self.args.device else 'cpu'
            
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
                }
                
                def factory():
                    model = LGBMRegressor(**params, objective='regression', metric='rmse', boosting_type='gbdt',
                                          device=device_search, random_state=self.args.seed, verbose=-1, n_jobs=1)
                    return model
                
                metrics = cv_score(factory)
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.args.seed))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            final_model = LGBMRegressor(**best_params, objective='regression', metric='rmse', boosting_type='gbdt',
                                        device=device_final, random_state=self.args.seed, verbose=-1, n_jobs=1)
            final_model.fit(x_full, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'CatBoost':
            gpu_id = int(self.args.device.split(':')[-1]) if 'cuda' in self.args.device else None
            task_type = 'GPU' if 'cuda' in self.args.device else 'CPU'
            
            def objective(trial):
                params = {
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                }
                metrics = cv_score(lambda: CatBoostRegressor(**params, loss_function='RMSE', eval_metric='RMSE',
                                                          task_type=task_type, devices=[gpu_id] if gpu_id is not None else None,
                                                          random_state=self.args.seed, verbose=0, thread_count=1))
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.args.seed))
            catboost_n_jobs = 1  
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            final_model = CatBoostRegressor(**best_params, loss_function='RMSE', eval_metric='RMSE',
                                            task_type=task_type, devices=[gpu_id] if gpu_id is not None else None,
                                            random_state=self.args.seed, verbose=0, thread_count=1)
            final_model.fit(x_full, y_full, verbose=0)
            return final_model, best_params, best_metrics, best_fold_metrics
        
        elif config['model'] == 'TabPFN':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 32),
                    'softmax_temperature': trial.suggest_float('softmax_temperature', 0.1, 2.0, log=True),
                }
                
                def factory():
                    tabpfn_kwargs = {
                        **params,
                        'device': self.args.device,
                        'random_state': self.args.seed,
                    }
                    if 'tabpfn_model_path' in self.config and self.config['tabpfn_model_path']:
                        tabpfn_kwargs['model_path'] = self.config['tabpfn_model_path']
                    return TabPFNRegressor(**tabpfn_kwargs)
                
                scores = {'mse': [], 'pearson': [], 'spearman': []}
                for cv_train_idx, cv_val_idx in inner_kfold.split(x_full):
                    x_cv_train = x_full.iloc[cv_train_idx]
                    x_cv_val = x_full.iloc[cv_val_idx]
                    y_cv_train = y_full.iloc[cv_train_idx]
                    y_cv_val = y_full.iloc[cv_val_idx]
                    model = factory()
                    model.fit(x_cv_train, y_cv_train)
                    preds = model.predict(x_cv_val)
                    if np.any(np.isnan(preds)):
                        self.args.logger.warning(f"[Trial {trial.number}] NaN predictions detected, returning -inf")
                        return -float('inf')
                    scores['mse'].append(mean_squared_error(y_cv_val, preds))
                    try:
                        p = pearsonr(y_cv_val, preds)[0]
                        if np.isnan(p):
                            p = 0.0
                    except Exception:
                        p = 0.0
                    scores['pearson'].append(p)
                    try:
                        s = spearmanr(y_cv_val, preds)[0]
                        if np.isnan(s):
                            s = 0.0
                    except Exception:
                        s = 0.0
                    scores['spearman'].append(s)
                
                metrics = {
                    k: {'mean': np.mean(v), 'folds': v} for k, v in scores.items()
                }
                trial.set_user_attr('mse_mean', metrics['mse']['mean'])
                trial.set_user_attr('pearson_mean', metrics['pearson']['mean'])
                trial.set_user_attr('spearman_mean', metrics['spearman']['mean'])
                trial.set_user_attr('mse_folds', metrics['mse']['folds'])
                trial.set_user_attr('pearson_folds', metrics['pearson']['folds'])
                trial.set_user_attr('spearman_folds', metrics['spearman']['folds'])
                return metrics['spearman']['mean']
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.args.seed))
            study.optimize(objective, n_trials=self.config['n_trials'], n_jobs=1, callbacks=[optuna_callback])
            best_params = study.best_params
            best_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_mean', 0.0),
                'pearson': study.best_trial.user_attrs.get('pearson_mean', 0.0),
                'spearman': study.best_value,
            }
            best_fold_metrics = {
                'mse': study.best_trial.user_attrs.get('mse_folds', []),
                'pearson': study.best_trial.user_attrs.get('pearson_folds', []),
                'spearman': study.best_trial.user_attrs.get('spearman_folds', []),
            }
            self.args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
            self.args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
            self.args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
            self.args.logger.info("Best Hyperparameters:")
            for k, v in best_params.items():
                self.args.logger.info(f"  {k}: {v}")
            self.args.logger.info("Fold-wise Metrics:")
            self.args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
            self.args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
            self.args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
            
            tabpfn_kwargs = {
                **best_params,
                'device': self.args.device,
                'random_state': self.args.seed,
            }
            if 'tabpfn_model_path' in self.config and self.config['tabpfn_model_path']:
                tabpfn_kwargs['model_path'] = self.config['tabpfn_model_path']
            final_model = TabPFNRegressor(**tabpfn_kwargs)
            final_model.fit(x_full, y_full)
            return final_model, best_params, best_metrics, best_fold_metrics
        
    
def get_current_time():
    current_time = str(datetime.fromtimestamp(int(time.time())))
    d = current_time.split(' ')[0]
    t = current_time.split(' ')[1]
    d = d.split('-')[0] + d.split('-')[1] + d.split('-')[2]
    t = t.split(':')[0] + t.split(':')[1] + t.split(':')[2]
    return d+ '_' +t

def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
def set_logging_config(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logger = logging.getLogger(' ')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(os.path.join(logdir, 'out.log'), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_logger(args, config):
    logger = set_logging_config(args.log_dir)
    
    logger.info(f'Model: {config["model"]} | Label: {config["label"]} | Folds: {config["n_folds"]} | Trials: {config["n_trials"]}')
    return logger

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='', help='the files of config')

    parser.add_argument('--log_dir', type=str, default='./results/logs_ml', help='path that log will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints_ml', help='the path of checkpoint')
    
    parser.add_argument('--gpu_ids', type=list, default='0')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers (-1 is max.)')
    
    args = parser.parse_args()
    return args

def init_configs():
    args = parse()
    
    current_time = get_current_time()
    args.log_dir = os.path.join(args.log_dir, current_time)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
       
    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    
    args.logger = set_logger(args, config)
    
    set_seed(args)

    return args, config

def main():
    start_time = time.time()
    
    try:
        args, config = init_configs()
        
        trainer = Trainer(args, config)
        res = trainer.run()
        
        final_model, best_params, best_metrics, best_fold_metrics = res
        args.logger.info(f"\n========== Final Result ==========")
        args.logger.info(f"Best CV Spearman: {best_metrics['spearman']:.4f}")
        args.logger.info(f"Best CV Pearson: {best_metrics['pearson']:.4f}")
        args.logger.info(f"Best CV MSE: {best_metrics['mse']:.6f}")
        args.logger.info("Best Hyperparameters:")
        for k, v in best_params.items():
            args.logger.info(f"  {k}: {v}")
        args.logger.info("Fold-wise Metrics:")
        args.logger.info(f"  MSE folds: {best_fold_metrics['mse']}")
        args.logger.info(f"  Pearson folds: {best_fold_metrics['pearson']}")
        args.logger.info(f"  Spearman folds: {best_fold_metrics['spearman']}")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        args.logger.info(f"\n========== Total Time: {hours}h {minutes}m {seconds}s ==========")
        args.logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger = logging.getLogger(' ')
        if not logger.handlers:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(' ')
        
        logger.error(f"\n========== Experiment Failed ==========")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()
