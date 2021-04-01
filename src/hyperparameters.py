class ModelHyperparameters(object):
    def __init__(self, data: dict):
        self.seq_len = data["seq_len"]
        self.n_time_features = data["n_time_features"]
        self.n_linear_features = data["n_linear_features"]
        self.n_out_features = data["n_out_features"]
        self.d_time_embed = data["d_time_embed"]
        self.d_linear = data["d_linear"]
        self.n_head = data["n_head"]
        self.num_encoder_layers = data["num_encoder_layers"]
        self.dropout = data["dropout"]


class TrainingHyperparameters(object):
    def __init__(self, data: dict):
        self.batch_size = data["batch_size"]
        self.n_epochs = data["n_epochs"]
        self.learning_rate = data["learning_rate"]
