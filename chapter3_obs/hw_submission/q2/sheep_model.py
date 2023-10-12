import torch
import torch.nn as nn
from ding.torch_utils import Transformer, MLP, unsqueeze, to_tensor


class ItemEncoder(nn.Module):
    encoder_type = ['TF', 'MLP', 'two_stage_MLP']

    def __init__(self, item_obs_size=60, item_num=30, item_encoder_type='TF', hidden_size=64, activation=nn.ReLU()):
        super(ItemEncoder, self).__init__()
        assert item_encoder_type in self.encoder_type, "not support item encoder type: {}/{}".format(item_encoder_type, self.encoder_type)
        self.item_encoder_type = item_encoder_type
        self.item_num = item_num
        self.hidden_size = hidden_size

        if self.item_encoder_type == 'TF':
            self.encoder = Transformer(
                item_obs_size,
                hidden_dim=2 * hidden_size,
                output_dim=hidden_size,
                activation=activation
            )
        elif self.item_encoder_type == 'MLP':
            self.encoder = MLP(
                item_obs_size,
                hidden_size,
                hidden_size,
                layer_num=3,
                activation=activation
            )
        elif self.item_encoder_type == 'two_stage_MLP':
            self.trans_len = 16
            self.encoder_1 = MLP(
                item_obs_size,
                hidden_size,
                self.trans_len,
                layer_num=3,
                activation=activation
            )
            self.encoder_2 = MLP(
                self.trans_len*self.item_num,
                hidden_size,
                self.item_num*hidden_size,
                layer_num=2,
                activation=activation
            )

    def forward(self, item_obs):
        if self.item_encoder_type == 'two_stage_MLP':
            item_embedding_1 = self.encoder_1(item_obs)   # (B, M, L)
            item_embedding_2 = torch.reshape(item_embedding_1, [-1, self.trans_len*self.item_num])
            item_embedding = self.encoder_2(item_embedding_2)
            item_embedding = torch.reshape(item_embedding, [-1, self.item_num, self.hidden_size])
        else:
            item_embedding = self.encoder(item_obs)
        return item_embedding


class SheepModel(nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(self, item_obs_size=60, item_num=30, item_encoder_type='TF', bucket_obs_size=30, global_obs_size=17, hidden_size=64, activation=nn.ReLU()):
        super(SheepModel, self).__init__()
        self.item_encoder = ItemEncoder(item_obs_size, item_num, item_encoder_type, hidden_size, activation=activation)
        self.bucket_encoder = MLP(bucket_obs_size, hidden_size, hidden_size, layer_num=3, activation=activation)
        self.global_encoder = MLP(global_obs_size, hidden_size, hidden_size, layer_num=2, activation=activation)
        self.value_head = nn.Sequential(
            MLP(hidden_size, hidden_size, hidden_size, layer_num=2, activation=activation), nn.Linear(hidden_size, 1)
        )

    def compute_actor(self, x):
        item_embedding = self.item_encoder(x['item_obs'])
        bucket_embedding = self.bucket_encoder(x['bucket_obs'])
        global_embedding = self.global_encoder(x['global_obs'])

        key = item_embedding
        query = bucket_embedding + global_embedding
        query = query.unsqueeze(1)
        logit = (key * query).sum(2)
        logit.masked_fill_(~x['action_mask'].bool(), value=-1e9)
        return {'logit': logit}

    def compute_critic(self, x):
        item_embedding = self.item_encoder(x['item_obs'])
        bucket_embedding = self.bucket_encoder(x['bucket_obs'])
        global_embedding = self.global_encoder(x['global_obs'])

        embedding = item_embedding.mean(1) + bucket_embedding + global_embedding
        value = self.value_head(embedding)
        return {'value': value.squeeze(1)}

    def compute_actor_critic(self, x):
        item_embedding = self.item_encoder(x['item_obs'])
        bucket_embedding = self.bucket_encoder(x['bucket_obs'])
        global_embedding = self.global_encoder(x['global_obs'])

        key = item_embedding
        query = bucket_embedding + global_embedding
        query = query.unsqueeze(1)
        logit = (key * query).sum(2)
        logit.masked_fill_(~x['action_mask'].bool(), value=-1e9)

        embedding = item_embedding.mean(1) + bucket_embedding + global_embedding
        value = self.value_head(embedding)
        return {'logit': logit, 'value': value.squeeze(1)}

    def forward(self, x, mode):
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)

    def compute_action(self, x):
        x = unsqueeze(to_tensor(x))
        with torch.no_grad():
            logit = self.compute_actor(x)['logit']
            return logit.argmax(dim=-1)[0].item()
