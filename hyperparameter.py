class HyperParameter:

    def __init__(self):

        self.batch_size=25 #require above zero       1,16,256
        #CNNN hyperparameter
        self.epoch = 512
        self.n_embed = 0
        self.embed_dim = 300
        self.n_label = 0
        self.learn_rate = 0.001
        self.optim = "adam"
        self.weight_decay = 1e-4
        self.embed_learn_rate = 0.1
        self.kernel_num = 200
        self.kernel_size = [1,2]
        self.dropout = 0.5
        self.interval = 20
        self.clip_max_norm = None

        self.padding_index = 0
        self.word_embed = []
        self.pretrain = True
        # self.pretrain_file = "data/w2v103100-en"
        # self.pretrain_file = "data/converted_word_Subj.txt"
        # self.pretrain_file = "data/converted_word_CR.txt"
        self.pretrain_file = "data/glove.840B.300d.txt"
        # self.pretrain_file = "data/word2vec.test.bin"
        self.embed_pickle = ""
        self.embed_save_pickle = "sst_embed/embed.pkl"
        # self.embed_save_pickle = "sst_embed/sst_embed.pth"
        self.model_dict = {}

        #LSTM hyperparameter

        self.hidden_dim = 200
        self.num_layers = 1
        self.init_weight = True
        self.gain = 0.4

        self.cuda = False
        self.vocab = {}
        self.packet_nums = 10 #CV 包个数

        #Stanford Parser
        self.train_dataset = "dataset/stsa.binary.train"
        self.dev_dataset = "dataset/stsa.binary.dev"
        self.test_dataset = "dataset/stsa.binary.test"
        self.model_path = "F:\jcode\englishFactored.ser.gz"
        self.sst_train_tree_save_path = "tree_save/sst_train_tree.txt"
        self.sst_dev_tree_save_path = "tree_save/sst_dev_tree.txt"
        self.sst_test_tree_save_path = "tree_save/sst_test_tree.txt"

        self.fine_tune = False




