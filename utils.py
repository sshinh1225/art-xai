import pickle
#from visdom import Visdom
import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing

def save_obj(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print("Saved object to %s." % filename)


def load_obj(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f, encoding='latin1')
    f.close()
    return obj

def load_csv_as_dict(csv_path):
    with open(csv_path, mode='r', encoding='utf8') as infile:
        reader = csv.reader(infile)
        
        mydict = {rows[1]:rows[0] for rows in reader}
    
    return mydict

def emd_to_csv(emd_path):
    name_csv = ''.join(emd_path.split('.')[:-1]) + '.csv'
    emd_data = pd.read_csv(emd_path, skiprows=1, sep=' ', header=None, index_col=0)
    emd_data = emd_data.sort_index()
    emd_data.columns = np.arange(emd_data.shape[1])


    emd_data.to_csv(name_csv)

def save_att_as_csv(data, csv_path):
    aux = pd.DataFrame.from_dict(data, orient='index')
    aux['key'] = aux.index
    aux.to_csv(csv_path, header=None, index=None)
    

#### GRAPH MATCHING AND COMPARISON
def graph_similarity(g1, g2):
    return 1 - np.abs(g1-g2).sum().sum() / (g1.shape[0] * g1.shape[1])

#### DEEP FEATURES TO UNIQUE CSV
def format_features(path='./DeepFeatures/', task='author', embedds='bow'):
    import os

    get_file_num = lambda a: int(a.split('_')[2])

    def get_max_number(path1):
        
        max_num = 0
        for file in os.listdir(path1):
            file_number = get_file_num(file)
            if file_number > max_num:
                max_num = file_number
        
        return max_num
    
    def get_X(path1, dataset='train', task='author', embedds='bow'):
        ix = 0
        if dataset == 'test':
            x_file = pd.read_csv(path + dataset + '_' + 'x_' + task + '_' + embedds + '.csv', index_col=0)
            def foo_bar(x):
                return float(x[7:-1])
                
            res = x_file #.applymap(foo_bar)
        else:
            for file in os.listdir(path1):
                if dataset == 'train':
                    if (file.split('_')[1] == 'x') and (file.split('_')[0] == dataset) and (file.split('_')[3] == task) and (embedds in file):
                        x_file = pd.read_csv(path + file, index_col=0)

                if ix == 0:
                    res = x_file
                    ix += 1
                else:
                    res = pd.concat([res, x_file])
            '''elif (dataset == 'test') and (file.split('_')[1] == 'x') and (file.split('_')[0] == dataset) and (file.split('_')[2] == task) and (file.split('_')[3].split('.')[0] == model):
                x_file = pd.read_csv(path + file, index_col=0)

                if ix == 0:
                    res = x_file
                    ix += 1
                else:
                    res = pd.concat([res, x_file])'''
        return res

    def get_y(path1, dataset, task, embedds='bow'):
        ix = 0
        if dataset == 'test':
            x_file = pd.read_csv(path + dataset + '_' + 'y_' + task + '_' + embedds + '.csv', index_col=0)
            res = x_file

        else:
            for file in os.listdir(path1):
                if (file.split('_')[1] == 'y') and (file.split('_')[0] == dataset) and (file.split('_')[3] == task) and (embedds in file):
                    x_file = pd.read_csv(path + file, index_col=0)
        for file in os.listdir(path1):
            if (dataset == 'train') and (file.split('_')[1] == 'y') and (file.split('_')[-1].split('.')[0] == model) and (file.split('_')[3] == task) and (file.split('_')[0] == dataset):
                x_file = pd.read_csv(path + file, index_col=0)

                if ix == 0:
                    res = x_file
                    ix += 1
                else:
                    res = pd.concat([res, x_file])
            elif (dataset == 'test') and (file.split('_')[1] == 'y') and (file.split('_')[-1].split('.')[0] == model) and (file.split('_')[2] == task) and (file.split('_')[0] == dataset):
                x_file = pd.read_csv(path + file, index_col=0)

                if ix == 0:
                    res = x_file
                    ix += 1
                else:
                    res = pd.concat([res, x_file])
        
        return res

    X = get_X(path, 'train', task=task, model=model)
    X_test = get_X(path, 'test', task=task, model=model)
    X = get_X(path, 'train', task=task, embedds=embedds)
    X_test = get_X(path, 'test', embedds=embedds)

    y_author = get_y(path, 'train', 'author', embedds=embedds)
    y_type = get_y(path, 'train', 'type', embedds=embedds)
    y_time = get_y(path, 'train', 'time', embedds=embedds)
    y_school = get_y(path, 'train', 'school', embedds=embedds)

    y_author_test = get_y(path, 'test', 'author', embedds=embedds)
    y_type_test = get_y(path, 'test', 'type', embedds=embedds)
    y_time_test = get_y(path, 'test', 'time', embedds=embedds)
    y_school_test = get_y(path, 'test', 'school', embedds=embedds)


    y_final = np.zeros((y_type.iloc[:, 0].shape[0], 4))
    y_final[:, 0] = np.squeeze(y_type.values)
    y_final[:, 1] = np.squeeze(y_time.values)
    y_final[:, 2] = np.squeeze(y_author.values)
    y_final[:, 3] = np.squeeze(y_school.values)
    y_final = pd.DataFrame(y_final)
    
    y_final_test = np.zeros((y_type_test.iloc[:, 0].shape[0], 4))
    y_final_test[:, 0] = np.squeeze(y_type_test.values)
    y_final_test[:, 1] = np.squeeze(y_time_test.values)
    y_final_test[:, 2] = np.squeeze(y_author_test.values)
    y_final_test[:, 3] = np.squeeze(y_school_test.values)
    y_final_test = pd.DataFrame(y_final_test)

    return X, y_final, X_test, y_final_test
    
def performance_classifier_tasks(clf0, path, feature_select=True, embedds='bow', og_task='author'):
    from sklearn.feature_selection import SelectFromModel

    X_train0, y_train, X_test0, y_test = format_features(path, embedds=embedds, task=og_task)
    tasks = ['Type', 'Time', 'Author', 'School']
    final_features_mask = np.zeros((len(tasks), X_train0.shape[1]))
    scaler = preprocessing.StandardScaler().fit(X_train0)
    X_train0 = np.array(X_train0)

    res = np.zeros((4, ))

    for task in range(4):
        clf = clf0
        
        clf.fit(X_train0, y_train.to_numpy()[:,task])
        if feature_select:
            finish = False
            threshold = 0.4
            while not finish:
                model = SelectFromModel(clf, prefit=True, threshold=threshold)
                final_features_mask[task, :] = model.get_support()

                X_train = model.transform(X_train0)
                X_test = model.transform(X_test0)

                if X_train.shape[1] > 2:
                    finish = True
                else:
                    threshold -= 0.1
            
            print('New features are size: ' + str(X_train.shape[1]))
            clf2 = clf0
            clf2.fit(X_train, y_train.to_numpy()[:,task])       
            clf = clf2
        
        else:
            X_train = X_train0
            X_test = X_test0
        print('Task ' + tasks[task])    
        print('Task train performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_train), y_train.to_numpy()[:,task]))))
        print('Task test performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_test), y_test.to_numpy()[:,task]))))
        res[task] = np.mean(np.equal(clf.predict(X_test), y_test.to_numpy()[:,task]))

    return final_features_mask, res

def performance_classifier_tasks_append(clf, path):
    X_train_author, y_train_author, X_test_author, y_test_author = format_features(path, task='author')
    X_train_type, y_train_type, X_test_type, y_test_type = format_features(path, task='type')
    X_train_time, y_train_time, X_test_time, y_test_time = format_features(path, task='time')
    X_train_school, y_train_school, X_test_school, y_test_school = format_features(path, task='school')

    X_train = np.concatenate([X_train_author.T, X_train_type.T, X_train_time.T, X_train_school.T]).T
    X_test = np.concatenate([X_test_author.T, X_test_type.T, X_test_time.T, X_test_school.T]).T

    tasks = ['Type', 'Time', 'Author', 'School']
    y_train = y_train_author #All the ys are the same
    y_test = y_test_author

    for task in range(4):
        clf.fit(X_train, y_train.to_numpy()[:,task])

        print('Task train performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_train), y_train.to_numpy()[:,task]))))
        print('Task test performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_test), y_test.to_numpy()[:,task]))))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = None #Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

if __name__ == '__main__':
    # path = '/home/javierfumanal/Documents/DeepFeatures/'
    path = '/Users/javier.fumanal/Downloads/DeepFeatures20/DeepFeatures/'
    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import SelectFromModel
    import matplotlib.pyplot as plt
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    embedds = 'clip'
    # X_train, y_train, X_test, y_test = format_features(path, embedds=embedds)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression() #SGDClassifier(max_iter=1000, tol=1e-3)
    """print('Clip')
    print('Non optimized features')
    print('Author')
    features_used, res_author = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='author')
    print('Type')
    features_used, res_type = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='type')
    print('Time')
    features_used, res_time = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='time')
    print('School')
    features_used, res_school = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='school')"""

    '''print()
    print('Optimized features')
    print('Author')
    features_used, res_author = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='author')
    print('Type')
    features_used, res_type = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='type')
    print('Time')
    features_used, res_time = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='time')
    print('School')
    features_used, res_school = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='school')'''

    embedds = 'clip'
    print()
    print('CLIP')
    """print('Non optimized features')
    print('Author')
    features_used, res_author = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='author')
    print('Type')
    features_used, res_type = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='type')
    print('Time')
    features_used, res_time = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='time')
    print('School')
    features_used, res_school = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=False, og_task='school')"""

    print()
    print('Optimized features')
    print('Author')
    features_used, res_author = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='type')
    print('Type')
    features_used, res_type = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='school')
    print('Time')
    features_used, res_time  = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='time')
    print('School')
    features_used, res_school = performance_classifier_tasks(clf, path, embedds=embedds, feature_select=True, og_task='author')
    ola = pd.DataFrame(np.zeros((4, 4)), index=['Author', 'Type', 'Time', 'School'], columns=['Type', 'School', 'Time', 'Author'])
    ola.iloc[0, :] = res_author
    ola.iloc[1, :] = res_type
    ola.iloc[2, :] = res_time
    ola.iloc[3, :] = res_school
    print(ola)
    '''aux = [np.sum(np.sum(features_used, axis=0) >= x) for x in range(4)]
    plt.bar([1, 2, 3, 4], np.sum(aux, axis=0)); plt.xticks([1, 2, 3, 4]); plt.xlabel('Number of tasks'); plt.ylabel('Features used'); plt.title('Features used in at least x tasks');plt.show()
    print('Done!')'''
    