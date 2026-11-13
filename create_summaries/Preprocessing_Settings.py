def get_dFoF_params(dataset, special_dFoF_params = None):

    dFoF_parameters = dict(\
                roi_to_neuropil_fluo_inclusion_factor=1.15, # no factor here
                neuropil_correction_factor = 0.7,
                #roi_specific_neuropil_correction_factor = False,
                method_for_F0 = 'sliding_percentile',
                percentile=5., # percent
                sliding_window = 5*60, # seconds
                computed_neuropil_fact_method = None, 
        )
    
    if dataset[:4]=='PYR-':
        # means pyramidal cells
        dFoF_parameters['roi_to_neuropil_fluo_inclusion_factor'] = 0 

    if special_dFoF_params is not None :
        for key in special_dFoF_params : 
            dFoF_parameters[key] = special_dFoF_params[key]

    return dFoF_parameters

def get_stat_test_props(dataset): 

    stat_test_props=dict(interval_pre=[-1.,0],
                        interval_post= [0.,1.],                                   
                        test='ttest',                                            
                        sign='positive')

    if dataset[:4]=='SST-' : 
        stat_test_props['interval_post'] = [1.,2.]

    return stat_test_props

