import os
from dataclasses import dataclass, field

from TTS.trainers import Trainer, TrainerArgs
from TTS.config import load_config, register_config
from TTS.tts.models import setup_model
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.xtts_config import XttsConfig

register_config("xtts", XttsConfig)

@dataclass
class TrainXttsArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})

def main():
    args = TrainXttsArgs()
    parser = args.init_argparse()
    parsed_args, config_overrides = parser.parse_known_args()
    args.parse_args(parsed_args)

    config = load_config(args.config_path)
    config.parse_known_args(config_overrides)

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = setup_model(config, train_samples + eval_samples)

    trainer = Trainer(
        args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )

    trainer.fit()

if __name__ == "__main__":
    main()
