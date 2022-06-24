# import dotenv
import hydra
import os
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.stage_1.main import train_stage_1
    from src.stage_1.eval import eval_stage_1
    from src.stage_1.create_high_qual_training_data import create_high_qual_training_set
    from src.stage_1 import utils
    from src.stage_2.main import train_stage_2
    from src.stage_3.sentence_level.main import train_stage_3_sentence_re
    from preprocess.data_trimmer import create_debug_dataset
    from preprocess.preprocess import preprocess_data
    from preprocess.preprocess_erica import preprocess_erica

    # Create debug dataset
    if config.create_debug_dataset_mode:
        return create_debug_dataset(config)

    # Create high-quality training dataset mode
    if config.create_high_qual_training_set_mode:
        return create_high_qual_training_set(config)


    # Preprocess data mode
    if config.preprocess_mode:
        if config.erica_pretrain_data:
            return preprocess_erica(config)
        else:
            return preprocess_data(config)

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)


    # Train
    if config.name == 'stage_1_experiment':
        print('Running stage 1 training.')
        return train_stage_1(config)
    elif config.name == 'stage_2_experiment':
        print('Running stage 2 training.')
        return train_stage_2(config)
    elif config.name == 'stage_3_sentence_re':
        print('Running stage 3 training -- sentence re.')
        return train_stage_3_sentence_re(config)

    # Eval
    if config.name == 'stage_1_eval':
        if config.wcl_method:
            print('Running stage 1 training -- WCL METHOD!')
            return eval_stage_1(config)
        else:
            print('Running stage 1 eval.')
            return eval_stage_1(config)

if __name__ == "__main__":
    main()
    print('Script finished.')
