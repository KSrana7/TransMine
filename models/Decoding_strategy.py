import torch

import logging
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------Added--------------------------------->
def dynamic_decode(model, batch, batch_x, batch_x_mark, dec_inp, batch_y_mark, beam_width, c_out, eos_token,device):
    batch_size = batch_x.size(0)
    seq_len = batch_x.size(1) 
    vocab_size = c_out 
    
    eos_token = eos_token.to(device)  

    seq = dec_inp
    
    for _ in range(seq_len):
        # Get the model's output probabilities
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, seq, batch_y_mark)[0]

        # -------------Greedy decoding-----------------> 
        next_words = torch.softmax(outputs[:,-1,:], dim=1)
        next_word = torch.max(next_words, dim=1)[1]
        seq = torch.cat([seq, next_word.unsqueeze(1).unsqueeze(2)], dim=1)

        # seq = torch.cat([seq, outputs[:,-1:,:]], dim=1)
   
        if (seq == eos_token).any():
            logger.info("EOS token reached for batch :{batch}")
            break
    return seq



def beam_search(model, batch, batch_x, batch_x_mark, dec_inp, batch_y_mark, beam_width, c_out, eos_token, device):
    batch_size = batch_x.size(0)
    seq_len = batch_x.size(1)
    vocab_size = c_out  # Assuming c_out is the vocabulary size

    eos_token = eos_token.to(device)

    # Initialize the beam with the start token
    beams = [(dec_inp, 0)]  # (sequence, score)

    for _ in range(seq_len):
        new_beams = []
        for seq, score in beams:
            # Get the model's output probabilities
            with torch.no_grad():
                outputs = model(batch_x, batch_x_mark, seq, batch_y_mark)[0]
                logits = outputs[:, -1, :]  # Get the last time step's logits
                probs = torch.softmax(logits, dim=1)

            # Get the top beam_width probabilities and their indices
            top_probs, top_indices = probs.topk(beam_width, dim=1)

            for i in range(beam_width):
                new_seq = torch.cat([seq, top_indices[:, i].unsqueeze(1).unsqueeze(2)], dim=1)
                new_score = score + torch.log(top_probs[:, i])
                new_beams.append((new_seq, new_score))

        if (seq == eos_token).any():
            logger.info("EOS token reached for batch :{batch}")
            break

        # Keep the top beam_width sequences
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    # Return the sequence with the highest score
    best_seq, best_score = beams[0]
    return best_seq

