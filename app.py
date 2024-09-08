import streamlit as st
import torch
from tokenizers import Tokenizer
from dataset import greedy_decode
from transformer import build_transformer

@st.cache_resource
def load_model_and_tokenizers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file("tokenizer_mr.json")

    config = {
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "mr",
    }

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), 
                              config["seq_len"], config["seq_len"], d_model=config["d_model"]).to(device)

    model_weights = "tmodel_49.pt"
    state = torch.load(model_weights, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    return model, tokenizer_src, tokenizer_tgt, config, device

def translate(sentence: str, model, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()
    
    encoder_input = tokenizer_src.encode(sentence).ids
    encoder_input = torch.tensor(encoder_input).unsqueeze(0).to(device)
    
    encoder_mask = (encoder_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    
    translation = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
    
    translated_sentence = tokenizer_tgt.decode(translation.detach().cpu().numpy())
    
    return translated_sentence

def main():
    st.title("English to Marathi Translator")
    
    model, tokenizer_src, tokenizer_tgt, config, device = load_model_and_tokenizers()

    english_sentence = st.text_input("Enter an English sentence:")
    if english_sentence:
        with st.spinner("Translating..."):
            marathi_translation = translate(english_sentence, model, tokenizer_src, tokenizer_tgt, config["seq_len"], device)
        st.success("Translation Complete!")
        st.write(f"**English:** {english_sentence}")
        st.write(f"**Marathi:** {marathi_translation}")

if __name__ == "__main__":
    main()
