import pandas as pd
from pandas import DataFrame 
df_tennis = DataFrame.from_csv('tennis.csv')
print("\n Given Play Tennis Data Set:\n\n",df_tennis)
df_tennis.keys()[0]
print(df_tennis.keys()[0])
def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )
def entropy_of_list(a_list):  
    from collections import Counter
    cnt = Counter(x for x in a_list)   # Counter calculates the proportion of class
    num_instances = len(a_list)*1.0   # = 14
    print("\n Number of Instances of the Current Sub Class is {0}:".format(num_instances ))
    probs = [x / num_instances for x in cnt.values()]  # x means no of YES/NO
    print("\n Classes:",min(cnt),max(cnt))
    print(" \n Probabilities of Class {0} is {1}:".format(min(cnt),min(probs)))
    print(" \n Probabilities of Class {0} is {1}:".format(max(cnt),max(probs)))
    return entropy(probs) # Call Entropy :

print("\n  INPUT DATA SET FOR ENTROPY CALCULATION:\n", df_tennis['PlayTennis'])
total_entropy = entropy_of_list(df_tennis['PlayTennis'])
print("\n Total Entropy of PlayTennis Data Set:",total_entropy)

def information_gain(df, split_attribute_name, target_attribute_name):
    print("Information Gain Calculation of ",split_attribute_name)
    
    df_split = df.groupby(split_attribute_name)
    for name,group in df_split:
            print("Name:\n",name)
            print("Group:\n",group)
    nobs = len(df.index) * 1.0
    print("NOBS",nobs)
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
   
    print("DFAGGENT",df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    print("New entropy is:",new_entropy)
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy 


print('Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'PlayTennis')),"\n")
print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis, 'Humidity', 'PlayTennis')),"\n")
print('\n Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'PlayTennis')),"\n")
print('\n Info-gain for Temperature is:' + str( information_gain(df_tennis, 'Temperature','PlayTennis')),"\n")

def id3(df, target_attribute_name, attribute_names, default_class=None):
    
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])# class of YES /NO
             
    if len(cnt) == 1:
        return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
    
    elif df.empty or (not attribute_names):
        return default_class  # Return None for Empty Data Set
    
    else:
        default_class = max(cnt.keys()) #No of YES and NO Class
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
        index_of_max = gainz.index(max(gainz)) # Index of Best Attribute
        best_attr = attribute_names[index_of_max]
        
        tree = {best_attr:{}} # Iniiate the tree with best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

attribute_names = list(df_tennis.columns)
print("List of Attributes:", attribute_names) 
attribute_names.remove('PlayTennis') #Remove the class attribute 
print("Predicting Attributes:\n", attribute_names)

tree = id3(df_tennis,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
print(tree)
