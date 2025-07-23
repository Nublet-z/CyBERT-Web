import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from tqdm.auto import tqdm
import numpy as np
from transformers import BertTokenizer, EncoderDecoderModel, BertModel, BartTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from evaluate import load
import csv
from safetensors.torch import load_model
from wordfreq import zipf_frequency
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")
remove_postags = ['VERB', 'NOUN', 'AUX', 'ADJ', 'PUNCT']
style_postags = ['VERB', 'NOUN', 'AUX', 'ADJ']

splits = {'validation': 'simplification/validation-00000-of-00001.parquet', 'test': 'simplification/test-00000-of-00001.parquet'}
df_val = pd.read_parquet("hf://datasets/facebook/asset/" + splits["test"])

def capitalize_ent(text):
    title_text = text.title()
    doc=nlp(title_text)
    words=[]
    for x in doc:
        if nlp(x.text).ents:
            words.append(x.text)
    for word in words:
        text = text.replace(word.lower(),word)
    # text = text[0].upper() + text[1:]
    return text

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
def get_context_v2(sentences, isComplex=True):
    # Tokenize sentence
    contexts = []
    for sentence in sentences:
        s_doc = nlp(sentence.lower())

        # Compute Zipf frequency for each word
        word_freqs = {token.text: zipf_frequency(token.text, 'en') for token in s_doc} 

        # Compute average Zipf frequency
        avg_zipf_freq = sum(word_freqs.values()) / len(word_freqs) if word_freqs else 0.5

        # Filter words based on Zipf frequency and POS tags
        if isComplex:
            filtered_words = [
                # token.text if token.pos_ not in style_postags and word_freqs[token.text] >= avg_zipf_freq
                # else "[MASK]"
                # for token in s_doc

                token.text for token in s_doc if token.pos_ not in style_postags and word_freqs[token.text] >= avg_zipf_freq
            ]
        else:
            filtered_words = [
                # token.text if token.pos_ not in style_postags and word_freqs[token.text] <= avg_zipf_freq
                # else "[MASK]"
                # for token in s_doc

                token.text for token in s_doc if token.pos_ not in style_postags and word_freqs[token.text] <= avg_zipf_freq
            ]

        # Reconstruct the filtered sentence
        filtered_sentence = " ".join(filtered_words)
        contexts.append(filtered_sentence)

    return contexts

def get_style(sentences):
    styles = []
    for sentence in sentences:
        sentence = sentence.lower()
        s_doc = nlp(sentence)

        # Remove stopwords and token with pos as listed in remove_postags
        filtered = [token.lemma_ for token in s_doc if str(token.pos_) in style_postags and token.text not in STOP_WORDS]
        style = " ".join(filtered)
        styles.append(style)
    return styles

class SimplificationDataset(torch.utils.data.Dataset):
    def __init__(self, input, context, style, label=None):
        n = int(len(input['input_ids'])//2)
        # Complex
        self.source_ids = input['input_ids'][:n]
        self.source_masks = input['attention_mask'][:n]
        self.source_context_ids = context['input_ids'][:n]
        self.source_context_masks = context['attention_mask'][:n]
        self.source_style_ids = style['input_ids'][:n]
        self.source_style_masks = style['attention_mask'][:n]
        # self.source_class = label[:n]
        # Simplified
        self.target_ids = input['input_ids'][n:]
        self.target_masks = input['attention_mask'][n:]
        self.target_context_ids = context['input_ids'][n:]
        self.target_context_masks = context['attention_mask'][n:]
        self.target_style_ids = style['input_ids'][n:]
        self.target_style_masks = style['attention_mask'][n:]
        # self.target_class = label[:n]

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        data = {
            'input_ids': self.source_ids[idx],
            'attention_mask': self.source_masks[idx],
            'src_context_ids': self.source_context_ids[idx],
            'src_context_mask': self.source_context_masks[idx],
            'src_style_ids': self.source_style_ids[idx],
            'src_style_mask': self.source_style_masks[idx],
            # 'src_class': self.source_class[idx],
            'labels': self.target_ids[idx],
            'labels_attention_mask': self.target_masks[idx],
            'lbl_context_ids': self.target_context_ids[idx],
            'lbl_context_mask': self.target_context_masks[idx],
            'lbl_style_ids': self.target_style_ids[idx],
            'lbl_style_mask': self.target_style_masks[idx],
            # 'lbl_class': self.target_class[idx],
            'idx': idx
        }

        return data
        
def create_data_loader_mask(df, batch_size, tokenizer, shuffle=False, is_val=False, is_inference=False):
    texts = df['text'].tolist()
    # For CEFR
    label = torch.Tensor(df['label'].to_numpy())
    # texts = get_context_v2(texts)

    #Tokenize the entire dataset at once
    encoded_all = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=80)
    encoded_input = {
                        'input_ids': encoded_all['input_ids'],
                        'attention_mask': encoded_all['attention_mask'],
                    } # Full context for label from texts
    
    if not is_inference:
        dataset = TensorDataset(# A data
                                encoded_input['input_ids'], encoded_input['attention_mask'],
                                )

    else:
        dataset = TensorDataset(
                                encoded_input['input_ids'], encoded_input['attention_mask'], label
                                )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class BARTLSTMGenerator2(nn.Module):
    def __init__(self, config, bert_model_name='lucadiliello/bart-small', start_token_id=0, end_token_id=2, max_seq_length=80):
        super(BARTLSTMGenerator2, self).__init__()
        self.device = config.device
        # self.bart = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(bert_model_name)
        self.hidden_dim = config.hidden_size
        self.max_seq_length = max_seq_length
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers

        # Freeze BART if needed
        # for param in self.bart.parameters():
        #     param.requires_grad = False

        # Project BERT's [CLS] hidden state to LSTM hidden
        # self.linear = nn.Linear(config.content_dim, hidden_dim)

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(self.num_layers):
            # in_dim = config.content_dim if i == 0 else self.hidden_dim  # Bidirectional LSTM doubles output dim
            self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))

        # Decoder
        # Normmalization
        self.dropout = nn.Dropout(config.drop_prob)
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, decoder_hidden_states, encoder_hidden_state, hiddens=None):
        # BERT Encoding
        # bert_outputs = self.bert(input_ids=input_ids)
        # cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        N, h = encoder_hidden_state.shape
        # hidden = self.activation(self.linear(encoder_hidden_state)).unsqueeze(0)  # (1, batch, hidden)
        if hiddens==None:
            hiddens = (
                torch.zeros_like(encoder_hidden_state).unsqueeze(0),
                torch.zeros_like(encoder_hidden_state).unsqueeze(0))
        # elif hiddens==None and encoder_hidden_state!=None:
        #     hiddens = (
        #         torch.zeros_like(encoder_hidden_state).unsqueeze(0),
        #         encoder_hidden_state.unsqueeze(0))
        # else:
        #     hiddens = (
        #         hiddens[0],
        #         encoder_hidden_state.unsqueeze(0)
        #     )            

        x = decoder_hidden_states

        for i in range(self.num_layers):
            residual = x
            # print("--x", x.shape)
            # print("--h", hiddens)
            x, hiddens = self.lstm_layers[i](x, (hiddens[0], hiddens[1]))
            # print("--x[:,-1,:]", x[:,-1,:].shape)
            # print("--x[:,-1,:]", x[:,-1,:].shape)
            x = x + residual
            # print("--residual", residual.shape)
            # print("--x", x.shape)
            x = self.layer_norms[i](x)  # Apply LayerNorm
        
        # # Decoder Embedding
        # embeddings = self.embedding(decoder_input_ids)
        # output, _ = self.lstm(embeddings, (hidden, cell))

        # # Residual
        # output = output + embeddings
        # output = self.normalization(output)
        x = self.dropout(x)
        logits = self.out(x)  # (batch, seq_len, vocab_size)
        # sys.exit()
        return logits, hiddens

    def generate(self, decoder_hidden_states, encoder_hidden_state, hiddens=None):
        # BERT Encoding
        # bert_outputs = self.bert(input_ids=input_ids)
        # cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        N, h = encoder_hidden_state.shape
        # hidden = self.activation(self.linear(encoder_hidden_state)).unsqueeze(0)  # (1, batch, hidden)
        if hiddens==None:
            hiddens = (
                torch.zeros_like(encoder_hidden_state).unsqueeze(0),
                torch.zeros_like(encoder_hidden_state).unsqueeze(0))          

        x = decoder_hidden_states

        for i in range(self.num_layers):
            residual = x
            x, hiddens = self.lstm_layers[i](x, (hiddens[0], hiddens[1]))
            x = x + residual
            x = self.layer_norms[i](x)  # Apply LayerNorm
        
        # # Decoder Embedding
        x = self.dropout(x)
        logits = self.out(x)  # (batch, seq_len, vocab_size)
        return logits

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=1, bias=True, dropout=0.1, vocab_size=50265):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Feedforward network (for post-attention processing)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        decoder_hidden = self.embedding(decoder_hidden)
        if len(encoder_outputs.shape) < 3:
            encoder_outputs = encoder_outputs.unsqueeze(1)
        key = self.k_proj(encoder_outputs).permute(1, 0, 2)
        value = self.v_proj(encoder_outputs).permute(1, 0, 2)
        query = self.q_proj(decoder_hidden).permute(1, 0, 2)

        # Apply multi-head attention
        attn_output, attn_weights = self.attention(query, key, value, attn_mask=mask)

        # Add residual connection and normalize
        attn_output = self.norm1(query.permute(1, 0, 2) + self.dropout(attn_output.permute(1, 0, 2)))
        
        return attn_output, attn_weights

class Options():
    def __init__(self):
        self.name = "Cycle-Training"
        self.num_epochs = 95 # You can adjust this
        self.pre_train_epc = 0
        self.batch_size = 1
        self.lr = 2e-4
        self.word_embedding_dimension = 10
        self.sentence_max_size = 80
        self.label_num = 2
        self.out_channel = 1
        self.num_layers = 1
        self.hidden_size = 768
        self.vocab_size = 30522
        self.eos_token_id = 102
        self.bos_token_id = 101
        self.pad_token_id = 0
        self.style_ratio = 0.05
        self.style_dim = int(self.hidden_size*self.style_ratio)
        self.content_dim = self.hidden_size-self.style_dim
        self.num_classes = 2
        self.eval_step = 1
        self.vis_step = 5
        self.drop_prob = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.w_rec = 1.0
        self.beam_width = 3
        self.temperature = 0.5
        self.emb_train = True
        self.cls_train = False
        self.drec_train = False
        self.emb_w_train = True
        self.gen_train = True
        self.tf_train = True
        self.continueTrain = True

        # Dataset settings
        self.dataset_path = '/home/wimaya/small-experiments/simple-text-classification/dataset'
        self.dataset_name = 'asset'

class simplified():
    def __init__(self):
        self.config = Options()

        checkpoint_dir = './13-5_cycleGAN-wiki1.6M/13-5_cycleGAN-wiki1.6M'
        # checkpoint_dir = f'/home/wimaya/demo/03-asset-saricescoreLoss-bertFulltrain'
        # log_dir = f'/home/wimaya/evaluate/transfer/generated-text/C8-data-source/cycleGAN-trans-ss'

        # ================ Initialize Model ================

        # Initialize BERT tokenizer and model
        bert_model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load the pretrained BART model
        self.embedding_model = BertModel.from_pretrained(bert_model_name)

        # Print model architecture (optional)

        self.device = self.config.device

        # Define the decoder for text generation
        self.decoder_G_AB = BARTLSTMGenerator2(self.config)

        self.decoder_G_BA = BARTLSTMGenerator2(self.config)

        self.decoder_tf_AB = nn.LSTM(input_size=self.config.style_dim, hidden_size=self.config.style_dim,
                            num_layers=1, batch_first=True)
        self.decoder_tf_BA = nn.LSTM(input_size=self.config.style_dim, hidden_size=self.config.style_dim,
                            num_layers=1, batch_first=True)

        self.attention = Attention(self.config.hidden_size).to(self.config.device)

        if self.config.continueTrain:
            print("Load decoder transfer weight..")
            load_model(self.embedding_model, f'{checkpoint_dir}/model.safetensors')
            # load_model(classifier, f'{checkpoint_dir}/classifier.safetensors')
            # load_model(decoder, f'{checkpoint_dir}/decoderR.safetensors')
            # load_model(decoder_s_AB, f'{checkpoint_dir}/main-LSTM-decoderRs-AB.safetensors')
            # load_model(decoder_s_BA, f'{checkpoint_dir}/main-LSTM-decoderRs-BA.safetensors')
            load_model(self.decoder_G_AB, f'{checkpoint_dir}/LSTM-decoderGAB.safetensors')
            load_model(self.decoder_G_BA, f'{checkpoint_dir}/LSTM-decoderGBA.safetensors')
            # load_model(style_w, f'{log_dir}/style_w.safetensors')
            # load_model(content_w, f'{log_dir}/content_w.safetensors')
            load_model(self.decoder_tf_AB, f'{checkpoint_dir}/LSTM-decodertfAB.safetensors')
            load_model(self.decoder_tf_BA, f'{checkpoint_dir}/LSTM-decodertfBA.safetensors')
            load_model(self.attention, f'{checkpoint_dir}/attention.safetensors')
            if not self.config.tf_train:
                for param in self.decoder_tf_AB.parameters():
                    param.requires_grad = False
                    param.grad = None
                    param.to(self.config.device)
                for param in self.decoder_tf_BA.parameters():
                    param.requires_grad = False
                    param.grad = None
                    param.to(self.config.device)


        # classifier.to(device)
        self.embedding_model.to(self.device)
        self.decoder_G_AB.to(self.device)
        self.decoder_G_BA.to(self.device)
        self.decoder_tf_AB.to(self.device)
        self.decoder_tf_BA.to(self.device)
        # decoder.to(device)

        bertscore = load("bertscore")
        bleu = load("bleu")
        rouge = load("rouge")
        sari = load("sari")

        # classifier.eval()
        # decoder.eval()
        self.decoder_tf_AB.eval()
        self.decoder_tf_BA.eval()
        self.decoder_G_AB.eval()
        self.decoder_G_BA.eval()
        self.embedding_model.eval()
        self.attention.eval()
        # ori_model.eval()

    def load_data(self, text):# ASSET Dataset
        val_com = pd.DataFrame()

        print("Text:", text)

        val_com['text'] = [text.lower()]
        val_com['label'] = 0

        # Load the dataset
        val_loader = create_data_loader_mask(pd.concat([val_com], ignore_index=True), self.config.batch_size, self.tokenizer, is_val=True)
        for batch in val_loader:
            # ================ Set Input ================
            self.input_ids_AB, self.attention_mask_AB = batch
            self.input_ids_AB = self.input_ids_AB.to(self.device)
            self.attention_mask_AB = self.attention_mask_AB.to(self.input_ids_AB.device)

        self.batch_size, self.seq_len = self.input_ids_AB.shape
        return self.seq_len

    def generate(self):
        # Read the text files into pandas DataFrames
        predictions = []
        actual_labels = []

        gen_text_As = []
        gen_text_Bs = []
        rA_texts = []
        rB_texts = []
        lbl_As = []
        lbl_Bs = []

        with torch.no_grad():
                
            # ================ Cycle A: Complex > Simple > Complex ================
            # --- First Half
            batch_size, seq_len = self.input_ids_AB.shape

            outputs = self.embedding_model(self.input_ids_AB[:,1:], self.attention_mask_AB[:,1:])
            embeddings = outputs.last_hidden_state

            content_word_emb = embeddings[:,:,:-self.config.style_dim].to(self.device)
            style_emb = embeddings[:,:,-self.config.style_dim:].to(self.device)

            # Input style to RNN network to map to target style
            style_tf_emb,_ = self.decoder_tf_AB(style_emb)

            # content_word_emb, adapted_content = adain(content_word_emb, style_tf_emb, return_content=True)
            
            # Generate the text
            # embedding_tf = torch.cat([content_word_emb, style_tf_emb], dim=-1)
            full_mask = torch.triu(torch.full((seq_len-1, seq_len-1), float('-inf')), diagonal=1).to(self.config.device)
            embedding_tf = torch.cat([content_word_emb, style_tf_emb], dim=-1)
            embedding_pooled = mean_pooling(embedding_tf, self.attention_mask_AB[:,1:])
            hidden = None

            logits = []
            input = self.input_ids_AB.clone()

            ## BEAM SEARCH ##
            beams = [(torch.tensor([[input[:, 0]]], device=self.device), 0.0)]

            completed_sequences = []
            for t in range(1, seq_len):
                new_beams = []
                for seq, score in beams:
                    mask = full_mask[:t, :seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                    attention_emb, _ = self.attention(seq, embedding_tf, mask=mask)
                    logits, hidden = self.decoder_G_AB(attention_emb, embedding_pooled, hidden)
                    next_token_logits = logits[:, -1, :]/self.config.temperature  # (batch, vocab)
                    probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # log probs

                    topk_probs, topk_indices = torch.topk(probs, self.config.beam_width, dim=-1)  # (batch, beam)

                    for k in range(self.config.beam_width):
                        next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)  # shape (1, 1)
                        next_score = score + topk_probs[0, k].item()
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_beams.append((new_seq, next_score))

                # Keep top-k beams only
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.config.beam_width]

                completed_sequences = beams  # In case all sequences didn't reach end_token
                best_seq = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[0][0]
                fake_B = best_seq[0][t:]
                # fake_B = input
                fake_B = fake_B.long()
                eot_idx = (fake_B == self.config.eos_token_id).nonzero()

                
                # if not (fake_B == self.config.eos_token_id).any():  # Check if there's no '2' in the row
                #     fake_B[-1] = self.config.eos_token_id
                # if len(eot_idx) > 0:
                #     fake_B[eot_idx[0]:] = self.config.pad_token_id

                # fake_B_mask = torch.ones_like(fake_B)
                # fake_B_mask[fake_B==self.config.pad_token_id] = 0
                # fake_B_mask.view(fake_B.shape)

                gen_tokens_B = fake_B
                gen_text_B = [self.tokenizer.decode(gen_tokens_B, skip_special_tokens=True)]
                
                gen_text_Bs.extend([gen_text_B[0]])

                print("Text:", gen_text_B[0].replace('#',''))

                yield gen_text_B[0]

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# print(f"Save generated text to {log_dir}")

# csv_file = f'{log_dir}/(BtA)_{config.name}_metric.csv'

# for src, gen, tgt in zip(rA_texts, gen_text_Bs, rB_texts):
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         # check if the file is empty to write header
#         if file.tell() == 0:
#             writer.writerow(["source", "rec_text", "target"])
#         writer.writerow([src, gen, tgt])