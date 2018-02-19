# param_bounds = {
#     'n_estimators': (20, 5, 1, 1, 50),
#     'min_samples_split': (14, 12, 1, 1, 32),
#     'max_features': (0.2, 0.15, 0.05, 0.05, 1)
# }
import copy
import random


class ParamSearch(object):
    def __init__(self, model, scoring, seed=None):
        self.model = copy.copy(model)
        self.scoring = scoring
        if seed is None:
            self.seed = random.randint(1, 2**32 - 2)
        else:
            self.seed = seed
        self.all_scores = {}
        self.best_score = -1000
        self.best_params = 0
        self.new_scores_count = 1
        self.best_score_path = []
        self.best_param_path = []

    def search(self, param_bounds, verbose=False):
        step_sizes = []
        param_vals = []
        min_step_sizes = []
        lower_lims = []
        upper_lims = []
        param_names = param_bounds.keys()
        param_names.sort()
        self.param_names = param_names
        for p in param_names:
            p_bounds = param_bounds[p]
            if type(p_bounds) == dict:
                llim = p_bounds['lims'][0]
                ulim = p_bounds['lims'][1]
                if 'step_size' in p_bounds:
                    step_size = p_bounds['step_size']
                else:
                    step_size = (ulim - llim) / 2
                if 'min_step_size' in p_bounds:
                    min_step_size = p_bounds['min_step_size']
                else:
                    min_step_size = step_size
                if 'start' in p_bounds:
                    start = p_bounds['start']
                else:
                    start = (ulim + llim) / 2
            else:
                llim = p_bounds[0]
                ulim = p_bounds[1]
                step_size = (ulim - llim) / 2
                if len(p_bounds) > 2:
                    min_step_size = p_bounds[2]
                else:
                    min_step_size = step_size
                if len(p_bounds) > 3:
                    start = p_bounds[3]
                else:
                    start = (ulim + llim) / 2
            param_vals.append(start)
            step_sizes.append(step_size)
            lower_lims.append(llim)
            upper_lims.append(ulim)
            min_step_sizes.append(min_step_size)
        self.iter = 0
        if verbose:
            print(param_names)
            print(param_vals)
            print('\tstep_size: %s' % step_sizes)
        while self.iter < 30:
            param_grid = {}
            for i in range(len(param_names)):
                p = param_names[i]
                if param_vals[i] < lower_lims[i]:
                    param_vals[i] = lower_lims[i]
                if param_vals[i] > upper_lims[i]:
                    param_vals[i] = upper_lims[i]
                if param_vals[i] - step_sizes[i] < lower_lims[i]:
                    param_grid[p] = [lower_lims[i], param_vals[i], param_vals[i] + step_sizes[i]]
                elif param_vals[i] + step_sizes[i] > upper_lims[i]:
                    param_grid[p] = [param_vals[i] - step_sizes[i], param_vals[i], upper_lims[i]]
                else:
                    param_grid[p] = [
                        param_vals[i] - step_sizes[i], param_vals[i], param_vals[i] + step_sizes[i]
                    ]
            self.new_scores_count = 0
            res = self.process(param_grid, param_names=param_names)
            if verbose:
                print(param_names)
                print(param_vals)
                print('\tstep_sizes: %s' % step_sizes)
                print('\tnew_scores_count: %s' % self.new_scores_count)
                print('\tbest_score: %s' % self.best_score)
            self.best_score_path.append(self.best_score)
            self.best_param_path.append(self.best_params)
            if self.new_scores_count == 0:
                # cut step_sizes in half
                step_sizes_old = step_sizes
                step_sizes = []
                for i in range(len(param_names)):
                    if type(step_sizes_old[i]) == str:
                        continue
                    if type(step_sizes_old[i]) == int:
                        new_step = step_sizes_old[i] / 2
                    if type(step_sizes_old[i]) == float:
                        new_step = step_sizes_old[i] / 2
                    if new_step < min_step_sizes[i]:
                        new_step = step_sizes_old[i]
                    step_sizes.append(new_step)
                if step_sizes == step_sizes_old:
                    break
            param_vals = self.best_params
            self.iter += 1
        res['param_names'] = param_names
        return res

    def get_score(self, params, param_names):
        if params in self.all_scores:
            return self.all_scores[params]
        else:
            self.new_scores_count += 1
            params_dict = dict(zip(param_names, params))
            self.model.clf.set_params(**params_dict)
            # select the correct proba
            proba = True
            if type(self.scoring) == str and self.scoring in ('accuracy'):
                proba = False
            if type(self.scoring) == dict:
                if self.scoring.values()[0].__name__ in ('accuracy_score'):
                    proba = False
            eval_results = self.model.evaluate(scoring=self.scoring, proba=proba, seed=self.seed)
            score_type = eval_results._scores.keys()[0]
            if score_type in ['brier']:
                score = -eval_results._scores[score_type]['mean']
            else:
                score = eval_results._scores[score_type]['mean']
            self.all_scores[params] = score
            return score

    def process(self, param_grid, param_values=(), param_names=None):
        param_grid = param_grid.copy()
        keys = param_grid.keys()
        keys.sort()
        k = keys[0]
        param_v = param_grid.pop(k)
        if len(param_grid) > 0:
            # process next batch
            params = []
            values = []
            for px in param_v:
                px2 = param_values + (px,)
                results = self.process(param_grid, param_values=px2, param_names=param_names)
                params.append(results['params'])
                values.append(results['values'])
            return {'params': params, 'values': values}
        else:
            params = []
            values = []
            for px in param_v:
                p_cell = param_values + (px,)
                params.append(p_cell)
                score = self.get_score(p_cell, param_names)
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = p_cell
                values.append(score)
            return {'params': params, 'values': values}
