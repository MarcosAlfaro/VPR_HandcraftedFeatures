import yaml
import os

current_directory = os.path.dirname(os.path.realpath(__file__))


class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'parameters.yaml')):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.device = config.get('device')

            self.csvDir = config.get('csvDir')
            self.figuresDir = config.get('figuresDir')
            self.datasetDir = config.get('datasetDir')
            self.modelsDir = config.get('modelsDir')

            self.lr = config.get('train').get('lr')
            self.loss = config.get('train').get('loss')
            self.margin = config.get('train').get('margin')
            self.batch_size = config.get('train').get('batch_size')


            # CSV creation parameters
            self.csv_path = config.get('directories').get('csv_path')
            self.cold_path = config.get('directories').get('cold_path')
            self._360loc_path = config.get('directories').get('_360loc_path')
            self.train_length = config.get('train').get('train_length')
            self.num_validations = config.get('train').get('num_validations')
            self.r_pos = config.get('train').get('pos_neg_thresholds').get('r_pos')
            self.r_neg = config.get('train').get('pos_neg_thresholds').get('r_neg')
            self.r_pos_360loc = config.get('train').get('pos_neg_thresholds').get('r_pos_360loc')
            self.r_neg_360loc = config.get('train').get('pos_neg_thresholds').get('r_neg_360loc')

            # Image processing parameters
            self.eq = config.get('img_processing').get('eq')
            self.inv = config.get('img_processing').get('inv')
            self.sh = config.get('img_processing').get('sh')
            self.color_rep = config.get('img_processing').get('color_rep')

            # Model parameters
            self.model = config.get('model').get('model')
            self.backbone = config.get('model').get('backbone')
            self.embedding_size = config.get('model').get('embedding_size')
            self.saved_models_path = config.get('model').get('saved_models_path')

            # DAv2
            self.enc = config.get('dav2').get('enc')

            # Perceptron architecture
            self.perceptron_architecture = config.get('mlp').get('perceptron_architecture')
            self.num_iterations_mlp = config.get('mlp').get('num_iterations_mlp')

            # PCA
            self.pca_values = config.get('pca').get('values')

            self.override = config.get('override')

PARAMS = Config()
