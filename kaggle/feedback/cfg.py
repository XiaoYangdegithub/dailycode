class CFG:
    input_path = './'
    model_path = 'microsoft/deberta-v3-base'  # nghuyong/ernie-2.0-large-en studio-ousia/luke-large
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_labels = 6
    max_position_embeddings = 1024 #文本长度
    loss_type = 'smooth-l1'
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0
    max_input_length = 1024 #最大输入
    epochs = 5  # 5 训练指示数/迭代次数
    encoder_lr = 10e-6 #学习率
    decoder_lr = 10e-4
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    num_fold = 5
    training_fold = 0
    batch_size = 4 #38
    seed = 1006
    OUTPUT_DIR = './'
    num_workers = 2
    device = 'cpu'
    print_freq = 10

    #以上是超参数的调整
