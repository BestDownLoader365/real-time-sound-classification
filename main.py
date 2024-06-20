from sound_detector import main
import yaml

with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)


main(config['model'])