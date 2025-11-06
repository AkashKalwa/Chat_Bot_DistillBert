#!/usr/bin/env python
# coding: utf-8

import os
import gradio as gr
from transformers import (
    DistilBertTokenizer, DistilBertForQuestionAnswering,
    BertTokenizer, BertForQuestionAnswering,
    RobertaTokenizer, RobertaForQuestionAnswering,
    GPT2Tokenizer, GPT2LMHeadModel
)
from datasets import load_dataset
import torch

# Load a small subset of SQuAD
dataset = load_dataset("squad", split="validation[:5%]")

# Prepare models/tokenizers
model_name_distilbert = "distilbert-base-cased-distilled-squad"
tokenizer_distilbert = DistilBertTokenizer.from_pretrained(model_name_distilbert)
model_distilbert = DistilBertForQuestionAnswering.from_pretrained(model_name_distilbert)

model_name_bert = "bert-base-uncased"
tokenizer_bert = BertTokenizer.from_pretrained(model_name_bert)
model_bert = BertForQuestionAnswering.from_pretrained(model_name_bert)

model_name_roberta = "roberta-base"
tokenizer_roberta = RobertaTokenizer.from_pretrained(model_name_roberta)
model_roberta = RobertaForQuestionAnswering.from_pretrained(model_name_roberta)

model_name_gpt = "gpt2"
tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_name_gpt)
model_gpt = GPT2LMHeadModel.from_pretrained(model_name_gpt)

def answer_question_with_model(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_index:end_index]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

def chatbot_response(user_question, context_selection):
    idx = int(context_selection.split(":")[0])
    context = dataset[idx]["context"]

    low = user_question.strip().lower()
    if low in {"thank you", "thanks", "bye", "ok"}:
        return ["Have a nice day! 😊"] * 4
    if low in {"hello", "hi", "greetings", "hey"}:
        return ["Hello! I am here to answer your questions. Ask me anything!"] * 4

    ans_distilbert = answer_question_with_model(user_question, context, model_distilbert, tokenizer_distilbert)
    ans_bert = answer_question_with_model(user_question, context, model_bert, tokenizer_bert)
    ans_roberta = answer_question_with_model(user_question, context, model_roberta, tokenizer_roberta)

    gpt_input = f"Question: {user_question}\nContext: {context}\nAnswer:"
    inputs_gpt = tokenizer_gpt.encode(gpt_input, return_tensors="pt", truncation=True, max_length=512)
    outputs_gpt = model_gpt.generate(inputs_gpt, max_length=150, do_sample=False)
    ans_gpt = tokenizer_gpt.decode(outputs_gpt[0], skip_special_tokens=True).split("Answer:")[-1].strip()

    return [ans_distilbert, ans_bert, ans_roberta, ans_gpt]

context_options = [f"{i}: {dataset[i]['context'][:100]}..." for i in range(len(dataset))]

interface = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your question here..."),
        gr.Dropdown(choices=context_options, label="Choose context", value=context_options[0]),
    ],
    outputs=[
        gr.Textbox(label="DistilBERT Answer"),
        gr.Textbox(label="BERT Answer"),
        gr.Textbox(label="RoBERTa Answer"),
        gr.Textbox(label="GPT-2 (Generative) Answer")
    ],
    title="Customer Service Bot — Model Comparison",
    description="Compare QA answers from DistilBERT, BERT, RoBERTa, and a GPT-2 generation.",
)

interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
