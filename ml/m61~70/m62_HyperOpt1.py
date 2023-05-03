# BayesianOptimization -> 최대값

# hyperopt -> 최솟값
import numpy as np
import hyperopt
import pandas as pd
print(hyperopt.__version__)         # 0.2.7

from hyperopt import hp, fmin, tpe, Trials

search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),
    'x2' : hp.quniform('x2', -15, 15, 1),   
}
print(search_space)

def objective_function(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 - 20*x2
    
    return return_value
    # 권장리턴방식 return {'loss': return_value, 'status': STATUS_OK}

trial_val = Trials()

best = fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,   # 디폴트
    max_evals=20,
    trials=trial_val,
    rstate= np.random.default_rng(seed=10)
)

print('best : ', best)
print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'},
#  {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'},
#  {'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'},
#  {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'}, {'loss': -300.0, 'status': 'ok'},
#  {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

print(trial_val.vals)
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0],
#  'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}



######### pandas 데이터프레임에 trial_val.vals 넣기 ############
# a = pd.DataFrame(trial_val.vals)

results = [i['loss'] for i in trial_val.results]

df = pd.DataFrame({'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2'],
                   'results' : results})

print(df)


# for aaa in trial_val.results:
#     losses.append(aaa['loss'])        와 동일