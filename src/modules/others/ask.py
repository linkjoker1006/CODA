import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridRequestNetwork(nn.Module):
    """
    一个混合模式的请求决策网络。

    该网络并联地输出两个决策：
    1. 闸门决策 (Gating Decision): 一个logit值，决定“是否”要发起请求。
    2. 选择决策 (Selection Decision): 一组注意力权重，作为“向谁”请求的概率分布。
    """
    def __init__(self, self_obs_dim, teacher_obs_dim, n_teachers, embedding_dim=128, novelty_dim=1):
        """
        初始化网络。

        参数:
            self_obs_dim (int): 智能体自身的私有观测维度。
            teacher_obs_dim (int): 能观测到的其他智能体的公共信息维度。
            embedding_dim (int): 内部特征嵌入的维度。
            novelty_dim (int): 新奇度分数的维度 (通常为2)。
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 编码器模块
        self.self_encoder = nn.Sequential(
            nn.Linear(self_obs_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.teacher_encoder = nn.Sequential(
            nn.Linear(teacher_obs_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 注意力机制的线性投影层
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # 闸门决策头 (Gating Head)
        # 输入维度 = 私有嵌入 + 上下文向量 + 新奇度分数
        gating_input_dim = embedding_dim + embedding_dim + novelty_dim
        self.gating_head = nn.Sequential(
            nn.Linear(gating_input_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, n_teachers + 1) # 输出请求与不请求的logit
        )

    def forward(self, self_obs, teacher_infos, novelty_score):
        """
        网络的前向传播。

        参数:
            self_obs (torch.Tensor): 请求者自身的私有观测。(B, 1, self_obs_dim)
            teacher_infos (torch.Tensor): 视野内其他智能体的公共信息。(B, 1, N - 1, teacher_obs_dim)
                                        B=batch_size, N=num_other_agents
            novelty_score (torch.Tensor): 请求者自身的新奇度分数。(B, 1, novelty_dim)
        
        返回:
            tuple: (gate_logit, attention_weights)
                - gate_logit (torch.Tensor): “不请求, 像0请求, 像1请求, ...”的logit值。(B, N)
                - attention_weights (torch.Tensor): 注意力矩阵。(B, 1, N - 1)
        """
        teacher_infos = teacher_infos.squeeze(1)
        # --- 步骤 1: 编码 ---
        # (B, 1, D_emb)
        self_embedding = F.relu(self.self_encoder(self_obs))
        # (B, N - 1, D_emb)
        teacher_embeddings = F.relu(self.teacher_encoder(teacher_infos))
        # 生成 Q, K, V
        # Q: (B, 1, D_emb)
        q = self.q_proj(self_embedding)
        # K, V: (B, N - 1, D_emb)
        k = self.k_proj(teacher_embeddings)
        v = self.v_proj(teacher_embeddings)
        # --- 步骤 2: 注意力机制计算上下文 ---
        # 计算注意力权重 (B, 1, N - 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        attention_weights = F.softmax(scores, dim=-1)
        # 生成上下文向量 (B, 1, D_emb)
        context_vector = torch.matmul(attention_weights, v)
        # --- 步骤 3: 闸门决策 ---
        # 融合所有信息 (B, D_emb + D_emb + D_novelty)
        gating_input = torch.cat([self_embedding, context_vector, novelty_score], dim=-1)
        # 通过闸门头得到最终的logit
        gate_logit = self.gating_head(gating_input)
        
        return gate_logit, attention_weights
