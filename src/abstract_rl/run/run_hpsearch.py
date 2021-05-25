import datetime

def run_hpsearch(run_algo, basic_run_config, hp_search_config):
    for keys, values in zip(hp_search_config.keys(), hp_search_config.values()):
        hp_search_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
        keys_split = keys.split(",")
        assert set(keys_split).issubset(basic_run_config.keys())
        for value in values:
            run_config = basic_run_config.copy()
            run_config['hp_search'] = {'varied_parameter': keys, 'values': values, 'concrete_run_value': value, 'hp_search_id': hp_search_id}
            for key in keys_split:
                run_config[key] = value
            run_algo(run_config)
