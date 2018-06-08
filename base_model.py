import torch
import torch.nn as nn
from attention import Attention, NewAttention, TwoWayAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, SimpleRegressor
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    regressor = SimpleRegressor(
         num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    # pass classifier or regressor based on what you want to use as loss
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, regressor)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    regressor = SimpleRegressor(
         num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    # pass classifier or regressor based on what you want to use as loss
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, regressor)

   
class TwoWayModel(nn.Module):
    def __init__(self, num_hid, v_dim, feat_size, ntokens, num_ans_candidates):
        super(TwoWayModel, self).__init__()
        self.w_emb = WordEmbedding(ntokens, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.bi_att = TwoWayAttention(v_dim)
        self.w_emb.init_embedding('data/glove6b_init_300d.npy')
        self.v_att = NewAttention(v_dim, num_hid, num_hid)
        self.q_net = FCNet([num_hid, num_hid])
        self.v_net = FCNet([v_dim, num_hid])
        self.final = nn.Linear(feat_size,1)
        self.classifier = SimpleRegressor(
                num_hid, 2 * num_hid, num_ans_candidates, 0.5)
        
    def forward(self, v, b, q, a, d_iou, threshold):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        att = (att * v)
        v_att = self.bi_att(att.squeeze())
        v_att = v_att * d_iou        
#         a_tilde = v_att 
        a_tilde = self.final(v_att)
#         logits = self.final(v_att)
#         logits = nn.Linear(36,1).cuda()(a_tilde.squeeze())
        a_tilde = nn.Sigmoid()(a_tilde.squeeze())
#         logits = nn.Linear(36,1).cuda()(a_tilde)
        mask = (a_tilde>threshold).float()
        predictions = (mask * a_tilde).sum(1)
#         w_emb = self.w_emb(q)
#         q_emb = self.q_emb(w_emb) # [batch, q_dim]

#         att = self.v_att(v, q_emb)
#         v_emb = (v_att * v).sum(1) # [batch, v_dim]

#         q_repr = self.q_net(q_emb)
#         v_repr = self.v_net(v_emb)
#         joint_repr = q_repr * v_repr
#         logits = self.classifier(joint_repr)
        return predictions
