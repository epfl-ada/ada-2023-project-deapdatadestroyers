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
    """Simple normalization function
    """
    return (df[var] - df[var].mean())/df[var].std()

#propensity score matching function 
def analyse(df, independent_var, matching_vars, dependent_var, row_id_index, onehot_vars, print_ols=False, print_effect=True):
        """
    Perform propensity score matching on a dataframe to create matched treatment and control groups
    
    Args:
        df: The input dataframe containing the dataset
        independent_var: treatment variable
        matching_vars: list of variables used for propensity score matching
        dependent_var: dependent variable for analysis
        row_id_index: column name representing unique identifiers for each row
        onehot_vars : variables to one-hot encode for modeling
        print_ols: whether to print the OLS summary. 
        print_effect: whether to print the treatment effect information.
    
    Returns:
        df_matched_treatment: df of matched treatment group
        df_matched_control: df of matched control group
        p_values: p-value from the paired t-test on the matched groups
        effect_sizes: Estimated treatment effect size
        intercept: Intercept term from the paired t-test model
    """
    df_matching = df.copy(deep=True).dropna(subset=matching_vars + [dependent_var, independent_var] + onehot_vars)

    for v in matching_vars:
        df_matching[v] = df_matching[v].astype(float)
        df_matching[v] = normalize(df_matching, v)
        
        
    print(len(df_matching[df_matching[independent_var]==1]), len(df_matching[df_matching[independent_var]==0]))
    
    mod = smf.logit(formula=f'{independent_var} ~  {" + ".join(matching_vars + onehot_vars)}', data=df_matching)
    control = df_matching[df_matching[independent_var]==0]
    treatment = df_matching[df_matching[independent_var]==1]
    try:
        res = mod.fit(maxiter=100, disp=False)
        df_matching['Propensity_score'] = res.predict()
        assert(len(control)>0)
        assert(len(treatment)>0)
    except:
        fake_effect_size = -control[dependent_var].mean() + treatment[dependent_var].mean()
        return pd.DataFrame({'A' : []}), pd.DataFrame({'A' : []}), 1, fake_effect_size, pd.DataFrame({'A' : []}) 

    #we start by creating the two groups
    treatment_group = df_matching[df_matching[independent_var]==1]
    control_group = df_matching[df_matching[independent_var]==0]

    #we print the number of element in each group to check that their sum is 1538 and make sure that our matching has
    #the same size as the smallest of the two groups (sanity check)
    print(len(treatment_group), len(control_group))

    G = nx.Graph()

    #add nodes for each paper in the treatment and control groups
    G.add_nodes_from(treatment_group[row_id_index], bipartite=0)
    G.add_nodes_from(control_group[row_id_index], bipartite=1)

    #calculate dissimilarity scores using vectorization
    treatment_scores = np.array(treatment_group['Propensity_score'])
    control_scores = np.array(control_group['Propensity_score'])

    dissimilarity_scores = np.abs(treatment_scores[:, None] - control_scores)

    start = time.time()

    for i, t_node in enumerate(treatment_group[row_id_index]):
        for j, c_node in enumerate(control_group[row_id_index]):
            G.add_edge(t_node, c_node, weight=dissimilarity_scores[i, j])

        
    end = time.time()
    #this function is to help us print the final dataframe (sanity check)
    
    col = 'Propensity_score'

    #perform minimum weight full matching
    matched_pairs = nx.bipartite.minimum_weight_full_matching(G, weight='weight', top_nodes=treatment_group[row_id_index])
    #matched_pairs = nx.max_weight_matching(G)

    #the matching function from Networkx gives a symmetric dictionary (2 times too long), so we filter it here
    filtered_edges = dict([(u, v) for u, v in matched_pairs.items() if G.nodes[u]['bipartite'] == 0 and G.nodes[v]['bipartite'] == 1])

    #create the sanity check dataframe (shows score and track)
    df_check_matched = pd.DataFrame([(u, v, 
                                get_columns(u, treatment_group, col),
                                get_columns(v, control_group, col),
                            ) 
                            for u, v in filtered_edges.items()], 
                            columns=['Name_t', 'Name_c', col + ' u', col + ' v'])
    
    df_check_matched['matching_value'] = abs(df_check_matched['Propensity_score u']-df_check_matched['Propensity_score v'])
    
    df_check_matched = df_check_matched[df_check_matched['matching_value'] < 0.0001]

    #create the simple dataframe with matched papers
    df_matched = df_check_matched[['Name_t', 'Name_c']]
    
    df_matched_treatment = df[(df[independent_var] == 1) & 
                                    (df[row_id_index].isin(df_matched['Name_t']))]

    df_matched_control = df[(df[independent_var] == 0) & 
                                    (df[row_id_index].isin(df_matched['Name_c']))]


    try:
        assert(len(df_matched_treatment) > 0)
        assert(len(df_matched_control) > 0)
    except:
        control = df_matching[df_matching[independent_var]==0]
        treatment = df_matching[df_matching[independent_var]==1]
        fake_effect_size = -control[dependent_var].mean() + treatment[dependent_var].mean()
        return pd.DataFrame({'A' : []}), pd.DataFrame({'A' : []}), 1, fake_effect_size, pd.DataFrame({'A' : []})

    paired_ttest = smf.ols(formula=f'{dependent_var} ~ {independent_var}', data=pd.concat([df_matched_treatment, 
                                                                                    df_matched_control])).fit()

    effect_sizes = paired_ttest.params[independent_var]
    p_values = paired_ttest.pvalues[independent_var]
    intercept = paired_ttest.params["Intercept"]
    
    print(f'Test for {dependent_var} ~ {independent_var}')
    if print_ols:
        print(paired_ttest.summary())
    if print_effect:
        print(f'The effect of {independent_var} is {effect_sizes} with a p-value of {p_values}')
        print(f'The base effect is {intercept}')
    print('')
    
    return df_matched_treatment, df_matched_control, p_values, effect_sizes, intercept






