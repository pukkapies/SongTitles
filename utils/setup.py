import json


DEFAULT_SETTINGS = {
    "cuda_batch": 1,
    "verbose": False,
    'arch': 'CharRNN',
    'max_epoch': 2000,
    "learning_rate": 0.0001,
    'validation_frequency': 0,
    'shuffle_before_batching': True,
    "resume_training": None,
    "max_patience": 50,
    "shuffle_after_every_epoch": True
}


def make_settings(cl_args, model_folder):
    settings = DEFAULT_SETTINGS
    settings.update(cl_args)

    with open(model_folder + 'char_rnn_settings.json') as json_file:
        char_rnn_settings = json.load(json_file)
    settings['CharRNN_settings'] = char_rnn_settings

    settings.update({'model_folder': model_folder})

    return settings


def create_json(settings_file, json_dict):
    with open(settings_file, 'w') as json_file:
        json.dump(json_dict, json_file, sort_keys=True, indent=4)