class ModelHyperparameters(object):
    def __init__(self, data: dict):
        self.type = data["type"]
        self.src_window_len = data["src_window_len"]
        self.tgt_window_len = data["tgt_window_len"]
        self.n_time_features = data["n_time_features"]
        self.n_linear_features = data["n_linear_features"]
        self.n_out_features = data["n_out_features"]
        self.d_time_embed = data["d_time_embed"]
        self.d_linear_embed = data["d_linear_embed"]
        self.n_head = data["n_head"]
        self.n_encoder_layers = data["n_encoder_layers"]
        self.n_decoder_layers = data["n_decoder_layers"]
        self.dropout = data["dropout"]
        self.use_pos_enc = data["use_pos_enc"]

    def create_meta_string(self):
        r"""The string is used for tensorboard and generally model saving."""
        return f"ntime_{self.n_time_features}_nlinear_" \
               f"{self.n_linear_features}_srcwindow_" \
               f"{self.src_window_len}_tgtwindow_" \
               f"{self.tgt_window_len}_dtimeembed_" \
               f"{self.d_time_embed}_dlinearembed_" \
               f"{self.d_linear_embed}_nout_{self.n_out_features}_nencoder_" \
               f"{self.n_encoder_layers}_ndecoder_{self.n_decoder_layers}"


class TrainingHyperparameters(object):
    def __init__(self, data: dict):
        self.batch_size = data["batch_size"]
        self.n_epochs = data["n_epochs"]
        self.learning_rate = data["learning_rate"]

    def create_meta_string(self):
        r"""The string is used for tensorboard and generally model saving."""
        return f"batchsize_{self.batch_size}_epochs_{self.n_epochs}_lr_" \
               f"{self.learning_rate}"
