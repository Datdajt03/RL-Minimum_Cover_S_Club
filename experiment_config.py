class ExperimentConfig:
    def __init__(self):
        # Theo paper bảng 1
        self.learning_rate    = 0.0003   # Adam lr
        self.batch_size       = 64       # PPO batch size
        self.max_epochs       = 500     # số thế hệ (generations)
        self.hidden_dim       = 256      # chiều rộng ẩn MLP
        self.s_club_threshold = 2        # ngưỡng đường kính s-club
        self.pop_size         = 50       # kích thước quần thể
        self.tournament_size  = 2        # kích thước đấu giá
        self.num_seeds        = 20       # số seeds
        self.discount         = 0.99     # γ
        self.clip_eps         = 0.2      # PPO clip ε
        self.entropy_coef     = 0.01     # β entropy
        self.grad_clip        = 1.0      # gradient clipping
        self.penalty_lambda   = 10.0     # hệ số hình phạt vi phạm đường kính
        self.max_generations  = 1000     # số thế hệ tối đa trong 1 episode (T)
        self.stagnation_limit = 200      # số thế hệ không cải thiện tối đa để dừng sớm (L)