#!/usr/bin/env python
# coding: utf-8

import os
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from datasets import load_dataset

# Load model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Load a small subset of SQuAD
dataset = load_dataset("squad", split="train[:5%]")

def sample_questions():
    examples = []
    n = min(5, len(dataset))
    for i in range(n):
        question = dataset[i]["question"]
        context_intro = dataset[i]["context"][:150] + "..."
        examples.append(f"Q: {question}\nContext: {context_intro}")
    return "\n\n".join(examples)

def find_context(question):
    # Simple baseline: use the first context
    return dataset[0]["context"]

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_index:end_index]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

def chatbot_response(user_question):
    low = user_question.strip().lower()
    if low in {"thank you", "thanks", "bye", "ok"}:
        return "Have a nice day! 😊", sample_questions()
    if low in {"hello", "hi", "greetings", "hey"}:
        return "Hello! I am here to answer your questions. Ask me anything!", sample_questions()
    context = find_context(user_question)
    answer = answer_question(user_question, context)
    return answer, sample_questions()

interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs=[
        gr.Textbox(label="Answer", placeholder="The answer will appear here."),
        gr.Textbox(label="Sample Questions", value=sample_questions(), interactive=False)
    ],
    title="Customer Service Bot",
    description="Ask questions based on topics from the SQuAD dataset.",
)

interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
