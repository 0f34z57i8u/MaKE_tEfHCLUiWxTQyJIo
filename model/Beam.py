import torch
import numpy as np
from queue import Queue
import torch.nn.functional as F
from .Constant import Constants

class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.log_prob = log_prob
        self.length = length

class MMINode(object):
    def __init__(self, hidden, lm_hidden,previous_node, decoder_input, attn, mmi_prob,length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.mmi_prob = mmi_prob
        self.length = length
        self.lm_hidden = lm_hidden

Constants = Constants()

def beam_search(model, input_nodes, adj_matrix, node_lens, device, max_tgt_len, beam_width=3, output_num=3):
    model.eval()
    with torch.no_grad():
        nodes_resp = model.embedding(input_nodes)
        encoder_outputs, encoder_hidden = model.encoder(nodes_resp, adj_matrix, node_lens) # bs*seq*h, bs*h
        
        # decode
        decoder_input = torch.LongTensor([Constants.BOS])
        node = Node(encoder_hidden, None, decoder_input, None, 0, 1)
        q = Queue()
        q.put(node)
        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                prev_y = model.embedding(decoder_input)
                hidden = node.hidden
                
                if decoder_input.item() == Constants.EOS or node.length >= max_tgt_len:
                    end_nodes.append(node)
                    continue
                log_prob, hidden, attn = model.decoder(prev_y, hidden, encoder_outputs, None)
                log_prob = F.log_softmax(log_prob.squeeze(), dim=-1)
                log_prob, indices = log_prob.topk(beam_width)
                
                for k in range(beam_width):
                    index = indices[k].unsqueeze(0)
                    log_p = log_prob[k].item()
                    child = Node(hidden.squeeze(0), node, index, attn, node.log_prob + log_p, node.length+1)
                    candidates.append((node.log_prob + log_p, child))
            candidates = sorted(candidates, key = lambda x: x[0], reverse = True)
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i][1])
        candidates = []
        for node in end_nodes:
            value = node.log_prob
            candidates.append((value, node))
        candidates = sorted(candidates, key = lambda x: x[0], reverse=True)
        node = [x[1] for x in candidates[:output_num]]
        res = []
        attns = []
        for one_node in node:
            one_res = []
            one_attns = []
            while one_node.previous_node != None:
                one_res.append(one_node.decoder_input.item())
                one_attns.append(one_node.attn.squeeze(0).cpu().numpy().tolist())
                one_node = one_node.previous_node
            res.append(one_res[::-1])
            attns.append(attns[::-1])
    return attns, res


def beam_search_graph_vae(model, input_nodes, adj_matrix, node_lens, device, max_tgt_len, beam_width=3, output_num=3):
    model.eval()
    with torch.no_grad():
        nodes_resp = model.embedding(input_nodes)
        encoder_outputs, encoder_hidden = model.encoder(nodes_resp, adj_matrix, node_lens) # bs*seq*h, bs*h
        
        batch_size = nodes_resp.shape[0]
        z = model.sample_z_prior(batch_size,device)
        
        # decode
        decoder_input = torch.LongTensor([Constants.BOS])
        node = Node(torch.cat([encoder_hidden,z], dim=1), None, decoder_input, None, 0, 1)
        q = Queue()
        q.put(node)
        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                prev_y = model.embedding(decoder_input)
                hidden = node.hidden
                
                if decoder_input.item() == Constants.EOS or node.length >= max_tgt_len:
                    end_nodes.append(node)
                    continue
                log_prob, hidden, attn = model.decoder(prev_y, hidden, encoder_outputs, z, None)
                log_prob = F.log_softmax(log_prob.squeeze(), dim=-1)
                log_prob, indices = log_prob.topk(beam_width)
                
                for k in range(beam_width):
                    index = indices[k].unsqueeze(0)
                    log_p = log_prob[k].item()
                    child = Node(hidden.squeeze(0), node, index, attn, node.log_prob + log_p, node.length+1)
                    candidates.append((node.log_prob + log_p, child))
            candidates = sorted(candidates, key = lambda x: x[0], reverse = True)
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i][1])
        candidates = []
        for node in end_nodes:
            value = node.log_prob
            candidates.append((value, node))
        candidates = sorted(candidates, key = lambda x: x[0], reverse=True)
        node = [x[1] for x in candidates[:output_num]]
        res = []
        attns = []
        for one_node in node:
            one_res = []
            one_attns = []
            while one_node.previous_node != None:
                one_res.append(one_node.decoder_input.item())
                one_attns.append(one_node.attn.squeeze(0).cpu().numpy().tolist())
                one_node = one_node.previous_node
            res.append(one_res[::-1])
            attns.append(attns[::-1])
    return attns, res

class GateNode(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, log_prob, prev_d_dec, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.log_prob = log_prob
        self.prev_d_dec = prev_d_dec
        self.length = length

def beam_search_graph_gate_vae(model, input_nodes, adj_matrix, node_lens, scene, device, max_tgt_len, beam_width=3, output_num=3):
    model.eval()
    with torch.no_grad():
        nodes_resp = model.embedding(input_nodes)
        d_initial = model.embedding(scene)
        d_initial = torch.mean(d_initial, 1)

        encoder_outputs, encoder_hidden = model.encoder(nodes_resp, adj_matrix, node_lens) # bs*seq*h, bs*h
        
        batch_size = nodes_resp.shape[0]
        z = model.sample_z_prior(batch_size,device)
        
        # decode
        decoder_input = torch.LongTensor([Constants.BOS])
        node = GateNode(torch.cat([encoder_hidden,z], dim=1), None, decoder_input, None, 0, d_initial,1)
        q = Queue()
        q.put(node)
        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                prev_y = model.embedding(decoder_input)
                hidden = node.hidden
                d_dec = node.prev_d_dec
                
                if decoder_input.item() == Constants.EOS or node.length >= max_tgt_len:
                    end_nodes.append(node)
                    continue
#                 print("prev_y shape",prev_y.shape)
#                 print('hidden shape', hidden.shape)
#                 print("d_dec shape", d_dec.shape)
                
                log_prob, hidden, d_dec, attn = model.decoder(prev_y, hidden, encoder_outputs, z, d_dec, None)
                log_prob = F.log_softmax(log_prob.squeeze(), dim=-1)
                log_prob, indices = log_prob.topk(beam_width)
                
                for k in range(beam_width):
                    index = indices[k].unsqueeze(0)
                    log_p = log_prob[k].item()
                    child = GateNode(hidden, node, index, attn, node.log_prob + log_p, d_dec,node.length+1)
                    candidates.append((node.log_prob + log_p, child))
            candidates = sorted(candidates, key = lambda x: x[0], reverse = True)
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i][1])
        candidates = []
        for node in end_nodes:
            value = node.log_prob
            candidates.append((value, node))
        candidates = sorted(candidates, key = lambda x: x[0], reverse=True)
        node = [x[1] for x in candidates[:output_num]]
        res = []
        attns = []
        for one_node in node:
            one_res = []
            one_attns = []
            while one_node.previous_node != None:
                one_res.append(one_node.decoder_input.item())
                one_attns.append(one_node.attn.squeeze(0).cpu().numpy().tolist())
                one_node = one_node.previous_node
            res.append(one_res[::-1])
            attns.append(attns[::-1])
    return attns, res

def get_non_pad_mask(seq):
    assert seq.dim() ==2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def beam_search_graph_trans_vae(model, input_nodes, adj_matrix, node_lens, device, max_tgt_len, beam_width=3, output_num=3):
    model.eval()
    with torch.no_grad():
        nodes_resp = model.embedding(input_nodes)
        non_pad_mask = get_non_pad_mask(input_nodes)
        encoder_outputs, encoder_hidden, *_ = model.encoder(nodes_resp, adj_matrix, node_lens,non_pad_mask) # bs*seq*h, bs*h
        
        batch_size = nodes_resp.shape[0]
        z = model.sample_z_prior(batch_size,device)
        
        # decode
        decoder_input = torch.LongTensor([Constants.BOS])
        node = Node(torch.cat([encoder_hidden,z], dim=1), None, decoder_input, None, 0, 1)
        q = Queue()
        q.put(node)
        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                prev_y = model.embedding(decoder_input)
                hidden = node.hidden
                
                if decoder_input.item() == Constants.EOS or node.length >= max_tgt_len:
                    end_nodes.append(node)
                    continue
                log_prob, hidden, attn = model.decoder(prev_y, hidden, encoder_outputs, z, None)
                log_prob = F.log_softmax(log_prob.squeeze(), dim=-1)
                log_prob, indices = log_prob.topk(beam_width)
                
                for k in range(beam_width):
                    index = indices[k].unsqueeze(0)
                    log_p = log_prob[k].item()
                    child = Node(hidden.squeeze(0), node, index, attn, node.log_prob + log_p, node.length+1)
                    candidates.append((node.log_prob + log_p, child))
            candidates = sorted(candidates, key = lambda x: x[0], reverse = True)
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i][1])
        candidates = []
        for node in end_nodes:
            value = node.log_prob
            candidates.append((value, node))
        candidates = sorted(candidates, key = lambda x: x[0], reverse=True)
        node = [x[1] for x in candidates[:output_num]]
        res = []
        attns = []
        for one_node in node:
            one_res = []
            one_attns = []
            while one_node.previous_node != None:
                one_res.append(one_node.decoder_input.item())
                one_attns.append(one_node.attn.squeeze(0).cpu().numpy().tolist())
                one_node = one_node.previous_node
            res.append(one_res[::-1])
            attns.append(attns[::-1])
    return attns, res