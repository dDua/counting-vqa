import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def instance_mse1(logits, labels, label2ans):
    true_answers = torch.FloatTensor([])
    predicted_answers = torch.FloatTensor([])
    cnt = 0
    for lbl in labels.data.cpu():
        val, indices = torch.max(lbl,0)
        indices = float(label2ans[indices.numpy()[0]])
#         answers = [float(label2ans[ans]) for ans in answers.numpy()[:,0]]
        true_answers = torch.cat([true_answers, torch.FloatTensor([indices])])
        prediction = torch.ones(1)*logits[cnt].data.cpu()
        predicted_answers = torch.cat([predicted_answers, prediction])
        cnt += 1
    predicted_answers = Variable(predicted_answers.cuda())
    true_answers = Variable(true_answers.cuda())
    loss = nn.MSELoss()(predicted_answers, true_answers)
    return loss

def instance_mse(predictions, labels):   
    loss = nn.MSELoss().cuda()(predictions, labels)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots = one_hots.cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits

def train(model, train_dset, eval_dset, num_epochs, output, batch_size, loss_fn):
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    
    print("Starting training...")
    threshold = 0.5
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print(len(train_loader))
        for i, (v, b, q, a, a_val, dist) in enumerate(train_loader):
            v = Variable(v)
            b = Variable(b)
            q = Variable(q)
            a = Variable(a)
            a_val = Variable(a_val)
            dist = Variable(dist)

            v = v.cuda()
            b = b.cuda()
            q = q.cuda()
            a = a.cuda()
            a_val = a_val.cuda()
            dist = dist.cuda()

            pred = model(v, b, q, a, dist, threshold)
            if loss_fn=="mse":
                loss = instance_mse(pred, a_val)
            else:
                loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()
            if loss_fn=="mse":
                batch_score= compute_score_with_logits(pred, a.data)
            else:
                batch_score = instance_mse(pred, a.data)
            batch_score = batch_score.sum()
            total_loss += loss.data[0] * v.size(0)
#             train_score += batch_score
            if i % 1000 == 0:
                print("epoch:{0} cnt:{1} total_loss:{2}".format(epoch, i, total_loss/(i+1)))
            
        total_loss /= len(train_loader.dataset)
        model.eval()
        if loss_fn=="mse":
            acc = evaluate_regression(model, eval_dset, batch_size, threshold)
        else:
            acc = evaluate_classification(model, eval_dset, batch_size, threshold)
        model.train()

        if epoch%10==0:
            threshold=0.04
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f, acc : %.2f ' % (total_loss, train_score, acc))
        if acc >= best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = acc
    
def evaluate_regression(model, eval_dset, batch_size, threshold=None):
    dataloader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    score = 0
    cnt = 0
    for v, b, q, a, a_val,dist in iter(dataloader):
        v = Variable(v, volatile=True)
        b = Variable(b, volatile=True)
        q = Variable(q, volatile=True)
        a_val = Variable(a_val, volatile=True)
        dist = Variable(dist, volatile=True)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        a_val = a_val.cuda()
        dist = dist.cuda()

        if threshold is None:
            predictions = model(v, b, q, None)
        else:
            predictions = model(v, b, q, None, dist, threshold)
        batch_score = nn.MSELoss()(predictions, a_val).data[0]
        score += batch_score
        cnt+=1

    score = score/float(cnt)
    return score
    
def evaluate_classification(model, eval_dset, batch_size, threshold=None):
    dataloader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    score = 0
    cnt = 0
    for v, b, q, a, a_val,dist in iter(dataloader):
        v = Variable(v, volatile=True)
        b = Variable(b, volatile=True)
        q = Variable(q, volatile=True)
        a_val = Variable(a_val, volatile=True)
        dist = Variable(dist, volatile=True)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        a_val = a_val.cuda()
        dist = dist.cuda()

        if threshold is None:
            predictions = model(v, b, q, None)
        else:
            predictions = model(v, b, q, None, dist, threshold)
        _, indices = predictions.max(1)
        pred_temp = [eval_dset.label2ans[ind] for ind in indices.data]
        preds = Variable(torch.zeros(predictions.size(0)).fill_(0)).cuda()
        for i in range(0, predictions.size(0)):
            if pred_temp[i].isdigit():
                preds[i].data.fill_(float(pred_temp[i]))
        batch_score, indices = compute_score_with_logits(preds, a)
        batch_score = batch_score.sum()
        
        score += batch_score
        cnt+=1

    score = score/float(cnt)
    return score
