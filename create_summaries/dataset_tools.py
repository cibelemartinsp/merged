from create_summaries.arousal import compute_arousal_masks, check_presence_locomotion_values

def get_summary_prefix_name(quantity, filtering_cond_name, special_dFoF_params) :

    if quantity == 'dFoF' : 
        summary_prefix_name = ''
    else : 
        summary_prefix_name = quantity + '_'

    if filtering_cond_name is not None : 
        summary_prefix_name += filtering_cond_name  + '_' #Run or Rest for exemple

    if special_dFoF_params is not None: 
        for key in special_dFoF_params.keys(): 
            summary_prefix_name += key + '_' + str(special_dFoF_params[key]) + '_'

    return summary_prefix_name


def build_filtering_cond_quantities(filtering_cond_name, data, quantities):

    if filtering_cond_name == 'Run' or filtering_cond_name == 'Rest' :
        if data.has_running():
            quantities += ['running']
            data.build_running(verbose=False)
            check_presence_locomotion_values(data)

    return data, quantities


def get_filtering_cond(filtering_cond_name, Episodes):

    if filtering_cond_name == None : 
        filtering_cond = None
    
    elif filtering_cond_name == 'Run' : 
        filtering_cond =  compute_arousal_masks(Episodes)[0]

    elif filtering_cond_name == 'Rest' : 
        filtering_cond =  compute_arousal_masks(Episodes)[1]

    return filtering_cond


