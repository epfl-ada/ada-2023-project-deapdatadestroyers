import json
import statsmodels.formula.api as smf
import networkx as nx
import time
import numpy as np
import pandas as pd

def get_columns(Name, df, col):
        out = df[df['Wikipedia ID']==Name][col].values[0]
        return out

def normalize(df, var):
    return (df[var] - df[var].mean())/df[var].std()


def analyse(df, dependent_var, matching_vars, independent_var, row_id_index, onehot_vars):
    df_matching = df.copy(deep=True).dropna(subset=matching_vars + [dependent_var, independent_var] + onehot_vars)
  
    for v in matching_vars:
        df_matching[v] = df_matching[v].astype(float)
        df_matching[v] = normalize(df_matching, v)
        
        

    
    mod = smf.logit(formula=f'{dependent_var} ~  {" + ".join(matching_vars + onehot_vars)}', data=df_matching)
    res = mod.fit(maxiter=100)
    df_matching['Propensity_score'] = res.predict()

    #display(df_matching)

    # We start by creating the two groups
    treatment_group = df_matching[df_matching[dependent_var]==1]
    control_group = df_matching[df_matching[dependent_var]==0]

    # We print the number of element in each group to check that their sum is 1538 and make sure that our matching has
    # the same size as the smallest of the two groups (sanity check)
    print(len(treatment_group), len(control_group))


    G = nx.Graph()

    # Add nodes for each paper in the treatment and control groups
    G.add_nodes_from(treatment_group[row_id_index], bipartite=0)
    G.add_nodes_from(control_group[row_id_index], bipartite=1)

    # Calculate dissimilarity scores using vectorization
    treatment_scores = np.array(treatment_group['Propensity_score'])
    control_scores = np.array(control_group['Propensity_score'])

    dissimilarity_scores = np.abs(treatment_scores[:, None] - control_scores)

    start = time.time()

    for i, t_node in enumerate(treatment_group[row_id_index]):
        for j, c_node in enumerate(control_group[row_id_index]):
            G.add_edge(t_node, c_node, weight=dissimilarity_scores[i, j])

        
    end = time.time()

    print(f'Time : {end-start}')
    # This function is to help us print the final dataframe (sanity check)
    
    col = 'Propensity_score'

    # Perform minimum weight full matching
    matched_pairs = nx.bipartite.minimum_weight_full_matching(G, weight='weight', top_nodes=treatment_group[row_id_index])
    #matched_pairs = nx.max_weight_matching(G)

    # The matching function from Networkx gives a symmetric dictionary (2 times too long), so we filter it here
    filtered_edges = dict([(u, v) for u, v in matched_pairs.items() if G.nodes[u]['bipartite'] == 0 and G.nodes[v]['bipartite'] == 1])

    # Create the sanity check dataframe (shows score and track)
    df_check_matched = pd.DataFrame([(u, v, 
                                get_columns(u, treatment_group, col),
                                get_columns(v, control_group, col),
                            ) 
                            for u, v in filtered_edges.items()], 
                            columns=['Name_t', 'Name_c', col + ' u', col + ' v'])
    
    df_check_matched['matching_value'] = abs(df_check_matched['Propensity_score u']-df_check_matched['Propensity_score v'])
    
    df_check_matched = df_check_matched[df_check_matched['matching_value'] < 0.0001]

    # Create the simple dataframe with matched papers
    df_matched = df_check_matched[['Name_t', 'Name_c']]
    
    

    #display(df_check_matched)

    #print(len(df_check_matched[df_check_matched['track u'] != df_check_matched['track v']]))
    df_matched_treatment = df_matching[(df_matching[dependent_var] == 1) & 
                                    (df_matching[row_id_index].isin(df_matched['Name_t']))]

    df_matched_control = df_matching[(df_matching[dependent_var] == 0) & 
                                    (df_matching[row_id_index].isin(df_matched['Name_c']))]


    #display(df_matched_treatment)

    paired_ttest = smf.ols(formula=f'{independent_var} ~ {dependent_var}', data=pd.concat([df_matched_treatment, 
                                                                                    df_matched_control])).fit()
    print(f'Test for {independent_var} ~ {dependent_var}')
    print('')
    print(paired_ttest.summary())
    print('')
    
    return df_matched_treatment, df_matched_control
