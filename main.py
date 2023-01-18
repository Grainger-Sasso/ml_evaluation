# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
from macpath import split
import numpy as np
import math
from sklearn.model_selection import GroupKFold

def groupFold():
    data = {
        "subj_1": [i for i in range(1)],
        "subj_2": [i for i in range(1)],
        "subj_3": [i for i in range(1)],
        "subj_4": [i for i in range(2)],
        "subj_5": [i for i in range(2)],
        "subj_6": [i for i in range(2)],
        "subj_7": [i for i in range(3)],
        "subj_8": [i for i in range(3)],
        "subj_9": [i for i in range(3)],
    }
    # data = {
    #     "subj_1": 1000,
    #     "subj_2": 3000,
    #     "subj_3": 2000,
    #     "subj_4": 3000,
    #     "subj_5": 2000,
    #     "subj_6": 1000,
    #     "subj_7": 1000,
    #     "subj_8": 2000,
    #     "subj_9": 3000
    # }
    y=np.array([0,0,0,0,0,1,1,1,1])
    group_kfold=GroupKFold(n_splits=3)
    group_idx=[i for i in range(len(data.items()))]#assign group index, which has the same length as the number of keys, or patient number
    print('group_idx',group_idx)
    X=np.array([V for K,V in data.items()])
    for train_index, test_index in group_kfold.split(X, y, group_idx):
        print("TRAIN:", train_index.astype(int), "TEST:", test_index.astype(int))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train, X_test, y_train, y_test)
    # total_samples=sum(data.values())
    # nSplit=3
    # if sum(data.values())%nSplit==0:
    #     offset=0
    # else:
    #     offset=1000#(total_samples%nSplit+total_samples//nSplit)*nSplit-total_samples
    # print('offset: ',offset,'total_sample',total_samples)#math.ceil(min(data.values())/2))
    # group_kfold=GroupKFold(n_splits=3)
    # kv=[]
    # key_to_index={}
    # items=data.items()
    # idx=0
    # for k,v in items:
    #     kv.append([k,v])
    #     key_to_index[k]=idx
    #     idx+=1
    # groups=split_group(kv,3,offset)
    # print('final',groups)
    # idx=0
    # group_idx=[0]*len(items)
    # for g in groups:
    #     for e in g:
    #         group_idx[key_to_index[e[0]]]=idx
    #     idx+=1
    
    # group_kfold.get_n_splits([(k,v) for k,v in data.items()], y, group_idx)
    # print(group_kfold)
    

def split_group(nums,k,offset):
    total = sum([n[1] for n in nums])

        # if total % k:
        #     return False

    reqSum = total // k
    subSets = [0] * k
    #nums.sort(reverse = True)
    elements=[[] for i in range(k)]

    def recurse(i):
        if i == len(nums):    
            return True

        for j in range(k):
            if subSets[j] + nums[i][1] <= reqSum+offset:
                
                elements[j].append(nums[i])
                #print(subSets,elements)
                subSets[j] += nums[i][1]

                if recurse(i + 1):
                    
                    return True

                subSets[j] -= nums[i][1]

                # Important line, otherwise function will give TLE
                if subSets[j] == 0:
                    break

                """
                Explanation:
                If subSets[j] = 0, it means this is the first time adding values to that subset.
                If the backtrack search fails when adding the values to subSets[j] and subSets[j] remains 0, it will also fail for all subSets from subSets[j+1:].
                Because we are simply going through the previous recursive tree again for a different j+1 position.
                So we can effectively break from the for loop or directly return False.
                """

        return False
    recurse(0)
    return elements
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    groupFold()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
