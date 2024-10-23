import numpy as np
import json
from wiki_retrieval import DocDB, Retrieval
from wild_retrieval import WildRetrieval
from openai import OpenAI
import pdb
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import os
from transformers import GPT2Tokenizer
from prompt_base import INSTUCT_FACTCHECK_QA, INSTUCT_FACTCHECK_RAG_ALL, INSTUCT_FACTCHECK_RAG_TOP

class FactChecker(object):
    
    def __init__(self, max_evidence_length):
        super().__init__()
        self.llm_token_count = 0
        self.openai_cost = 0
        self.db = {}
        self.retrieval = {}
        self.max_evidence_length = max_evidence_length


    def build_enwiki_evidence(self, knowledge_source="enwiki",
                              db_path='../long-short-uncertainty/factcheck_cache/enwiki-20230401.db',
                              data_path='', cache_path='../long-short-uncertainty/factcheck_cache/retrieval-enwiki-20230401.json',
                              embed_cache_path='../long-short-uncertainty/factcheck_cache/retrieval-enwiki-20230401.pkl',
                              batch_size=256):
        """Build evidence from English Wikipedia."""
        self.db[knowledge_source] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[knowledge_source] = Retrieval(self.db[knowledge_source], cache_path, embed_cache_path, batch_size=batch_size)

    def build_google_search(self, knowledge_source="google_search", db_path=".../long-short-uncertainty/factcheck_cache/wildhallu.db",
                            data_path=""):
        """Build evidence from Google Search."""
        self.retrieval[knowledge_source] = WildRetrieval(db_path=db_path)

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(20))
    def get_completion(self, user_prompt: str=""):
        """Get completion from the GPT model."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f'ERROR: {str(e)}')
                print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
                time.sleep(2 ** (i + 1))
        return None

    def get_prompt_zero_all(self, topic, fact_list, dataset, evidence_type):
        atomic_facts_string = "### " + "\n### ".join(fact_list) + "\n\n"
        instruct = ""
        retrieved_evidence_string = ""
        if evidence_type == "zero":
            instruct = INSTUCT_FACTCHECK_QA.format(atomic_facts_string=atomic_facts_string)
        elif evidence_type == "all":
            if dataset == "longfact":
                raise ValueError("LongFact does not support evidence retrieval")
            elif dataset == "bios":
                passages = self.retrieval['enwiki'].get_all_passages(topic)
                self.retrieval['enwiki'].save_cache()
                retrieved_evidence_string = "".join([i["text"] for i in passages])
            else:  # dataset == "wildhallu"
                passages = self.retrieval['google_search'].get_all_passages(topic)
                retrieved_evidence_string = "".join([i for i in passages])
            instruct = INSTUCT_FACTCHECK_RAG_ALL.format(atomic_facts_string=atomic_facts_string, retrieved_evidence=retrieved_evidence_string)
        return instruct
    
    def get_prompt_topk(self, topic, fact_list, dataset, topk=1):
        atomic_fact_with_evidence = ""
        if dataset == "longfact":
            raise ValueError("LongFact does not support evidence retrieval")
        if dataset in ["bios", "wildhallu"]:
            for fact in fact_list:
                if dataset == "bios":
                    passages = self.retrieval['enwiki'].get_passages(topic, fact, topk)
                else:
                    try:
                        passages = self.retrieval['google_search'].get_gtr_passages(topic, fact, topk)
                    except Exception as e:
                        if "CUDA out of memory" in str(e):
                            print(f"Error in retrieving evidence for fact: {fact} using GTR retrieval. Error: {e}")
                            print(f"Trying BM25 retrieval for fact: {fact}.")
                            passages = self.retrieval['google_search'].get_bm25_passages(topic, fact, topk)

                evidence = "".join([i["text"] if dataset == "bios" else i for i in passages])
                atomic_fact_with_evidence += f"Statement: {fact}\n\nEvidence: '''{evidence}'''\n\n"
        instruct = INSTUCT_FACTCHECK_RAG_TOP.format(atomic_fact_with_evidence=atomic_fact_with_evidence)
        print(instruct)
        return self.truncate_text(instruct)
    
    def get_prompt(self, topic, fact_list, dataset, evidence_type):
        if evidence_type in ["zero", "all"]:
            return self.get_prompt_zero_all(topic, fact_list, dataset, evidence_type)
        elif evidence_type == "topk":
            return self.get_prompt_topk(topic, fact_list, dataset)
        else:
            raise ValueError("Invalid evidence type")   
        
        
    def get_veracity_labels(self, topic="", atomic_facts=[], knowledge_source="", evidence_type="zero"):
        """
        Get veracity labels for the given atomic facts.
        We need to try multiple times because the GPT model may fail to respond.
        """
        for attempt in range(3):
            try:
                user_prompt = self.get_prompt(topic, atomic_facts, knowledge_source, evidence_type=evidence_type)
                completion = self.get_completion(user_prompt)
                atomic_responses = completion.split("### ")
                atomic_responses = [x.strip() for x in atomic_responses if x]
                gpt_labels = [response.split("$")[1] for response in atomic_responses]
                assert len(atomic_facts) == len(gpt_labels)
                return gpt_labels
            except Exception as e:
                print(e)
                print(f'Retrying ({attempt + 1}/3), wait for {2 ** (attempt + 1)} sec...')
                time.sleep(2 ** (attempt + 1))
        return []
    
    def filter_abstain(self, questions=[], answers=[]):
        abstain_indicators = ["don't have real-time access", 
                              "do not have real-time access", 
                              "I do not have access",
                              "I don't have access",
                              "I cannot definitively answer",
                              "I cannot answer",
                              "I couldn't find definitive information",
                              "I cannot provide",
                              "I apologize",
                              "I cannot predict the future",
                              "I cannot confirm",
                              "I couldn't find any information",
                              "I couldn't find any reliable information",
                              "not specified in the available historical records"]
        abs_index, filtered_q, filtered_a = [], [], []
        for i, (q, a) in enumerate(zip(questions, answers)):
            if any(indicator in a for indicator in abstain_indicators):
                abs_index.append(i)
            else:
                filtered_q.append(q)
                filtered_a.append(a)

        return abs_index, filtered_q, filtered_a

        
if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    import sys
    sys.path.append(".../long_uncertainty_express")
    from utils import read_jsonl

    parser = argparse.ArgumentParser(description="Generate bios using LLM")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo-1106", help="Model ID for LLM")
    parser.add_argument("--dataset", type=str, default="wild", help="dataset")
    parser.add_argument("--cache_path", type=str, default="../long-short-uncertainty/factcheck_cache")
    parser.add_argument("--knowledge_source", type=str, default="wildhallu")
    parser.add_argument("--input_dir", type=str, default="../results")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument('--method', type=str, default="demo")
    args = parser.parse_args()

    fc = FactChecker(max_evidence_length=100000)
    fc.build_google_search()
    fc.build_enwiki_evidence()

    # Load the data
    file_path = f"{args.input_dir}/{args.dataset}/{args.model_id}_{args.method}_uncq_to_answers.jsonl"
    print(file_path)
    # if os.path.exists(file_path):
    answers = read_jsonl(file_path)

    # Check if the file already exists
    previous_file = f"{args.output_dir}/{args.dataset}/{args.model_id}_{args.method}_unca_veracity.jsonl"
    if os.path.exists(previous_file):
        data_to_save = read_jsonl(previous_file)
    else:
        data_to_save = []
        
    previous_topics = [item['topic'] for item in data_to_save]
    print(len(previous_topics))

    for item in tqdm(answers, desc=f"{args.dataset}-{args.model_id}"):
        topic, fact_list = item['topic'], item["atomic_facts"]
        questions, answers = item["unc_facts_to_questions"], item["unc_question_to_answers"]
        abs_index, filtered_q, filtered_a = fc.filter_abstain(questions, answers)
        qa_pairs = [f"Question: {question} Answer: {answer}" for question, answer in zip(filtered_q, filtered_a)]
        if topic not in previous_topics:
            if qa_pairs:
                veracity_labels = fc.get_veracity_labels(topic=topic, 
                                                        atomic_facts=qa_pairs, 
                                                        knowledge_source=args.knowledge_source, 
                                                        evidence_type="zero")
                if veracity_labels:
                    labels_iter = iter(veracity_labels)
                    final_labels = [next(labels_iter) if i not in abs_index else 'NS' for i in range(len(veracity_labels)+len(abs_index))]
                else:
                    final_labels = ["NS"]*len(abs_index) if abs_index else []
            else:
                final_labels = ["NS"]*len(abs_index) if abs_index else []

            item.update({
                "unc_veracity_labels": final_labels})
            data_to_save.append(item)

            with open(previous_file, "a") as f:
                f.write(json.dumps(item) + "\n")
        
