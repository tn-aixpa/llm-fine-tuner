from transformers import AutoTokenizer
from huggingface_hub import login
import random
from typing import Dict, List, Any
import os
import json
import yaml
import copy
from pprint import pprint
import argparse

def load_flags():
    parser = argparse.ArgumentParser(description="Parameters for the creation of finetuning dataset")

    parser.add_argument('--in_dir', type=str, default="../data/annotation", help='path that contains dialogues and documents')
    parser.add_argument('--out_dir', type=str, default="../data/annotation/llm_template_splits", help='path for saving the data for finetuning')
    parser.add_argument('--seed', type=int, default=1337, help='randomness seed')

    return parser.parse_args()


def make_llm_splits(input_list: List[Any], seed: int = 1337) -> Dict[str, List[Any]]:
    """
    Divides the input data into train, test, and validation splits.
    Data shuffling is applied.
    The division of the split is 90% for train, 10% for test, and 10% for validation.

    Arguments:
        input_list (List[Any]): the elements that will be randomly put in one of the three splits
        seed (int): seed for random shuffling (default 1337)

    Returns:
        (Dict[str, List[Any]]) A mapping that for each split gives the elements belonging to it
    """
    random.seed(seed)
    random.shuffle(input_list)
    n = len(input_list)
    split1 = int(n * 0.8)
    split2 = int(n * 0.9)
    return {
        "train": input_list[:split1],
        "validation": input_list[split1:split2],
        "test": input_list[split2:]}


def reshape_dialogue(dialogue: Dict, document: str) -> Dict:
    """
    Restructure the inner structure of the dialogue

    Arguments:
        dialogue (Dict): the dialogue to be reshaped
        document (str): a document to be added to the dialogue new structure

    Returns:
        (Dict) the reshaped dialogue with the document in it
    """

    ### Adjust single turn formatting
    adj_turns = list()
    for turn in dialogue["edited_turns"]:
        role = str()
        if any(turn['speaker'] in el for el in ["Cittadino", "Citizen", "Operatore PA", "PA Operator"]):
            role = "user"
        elif any(turn['speaker'] in el for el in ["Agente"]):
            role = "assistant"
        else:
            raise ValueError(f"speaker {turn['speaker']} is not expected!")
        
        ### make the new turn
        try:
            adj_turn = {
                "role":    role,
                "content": turn["message"] if "message" in turn else turn["manual_text"]
            }
        except KeyError as ke:
            raise(f"{ke}\n\n{turn}")
        
        adj_turn["content"] = adj_turn["content"].strip()

        # skip empty turns
        if adj_turn["content"] == "":
            continue

        adj_turns.append(adj_turn)
        
    ### Adjust user type
    user_type = str()
    if   dialogue["user"] in ["Cittadino", "Citizen"]:
        user_type = "Citizen"
    elif dialogue["user"] in ["Operatore PA", "PA Operator"]:
        user_type = "Public Operator"
    elif dialogue["user"] in ["Agente", "Agent"]:
        user_type = "Chatbot"
    else:
        raise ValueError(f"Unexpected user type \"{dialogue['user']}\"")

    ### Adjust interaction style
    int_style = str()
    if   dialogue["dial_style"] == "Tu":
        int_style = "informal"
    elif dialogue["dial_style"] == "Lei":
        int_style = "formal"
    else:
        raise ValueError(f"Unexpected interaction style \"{dialogue['dial_style']}\"")

    ### Prepare return
    adj_data = {
        "messages": adj_turns,
        "doc":      document.strip(),
        "metadata": {
            "user_type":         user_type,
            "interaction_style": int_style,
            "proactive":         dialogue["proactive"] if "proactive" in dialogue else None 
        }
    }
    return adj_data


if __name__ == "__main__":
    args = load_flags()
    
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    login(token = config["llama_token"])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    ########################################
    ### Read dialogue and document files ###
    ########################################
    main_dial_dir = os.path.join(args.in_dir, "annotation")
    main_doc_dir  = os.path.join(args.in_dir, "documents")

    dialogues_list  = list()
    doc_name_to_txt = dict()
    for data_source in ["Comune", "AmiciFamiglia"]:

        ### read dialogue files
        dial_dir =  os.path.join(main_dial_dir, data_source)
        for filename in os.listdir(dial_dir):
            with open(os.path.join(dial_dir, filename), "r") as file:
                dial = json.load(file)
                dial["data_src"] = data_source
                dialogues_list.append(dial)
        
        ### read document files
        doc_dir =  os.path.join(main_doc_dir, data_source)
        for filename in os.listdir(doc_dir):
            with open(os.path.join(doc_dir, filename), "r") as file:
                doc_name_to_txt[filename] = file.read() 

    #########################
    ### Reshape dialogues ###
    #########################
    reshaped_dials = list()
    for dial in dialogues_list:
        reshaped_dial = reshape_dialogue(dial, doc_name_to_txt[dial["doc_src"].replace("'", "_")])
        reshaped_dial["id"] = dial["data_src"] + "_" + dial["task_id"]
        reshaped_dials.append(reshaped_dial)

    ##########################
    ### Make LLM templates ###
    ##########################
    model_data_splits = make_llm_splits(reshaped_dials, seed = args.seed) 

    prompt_template = """You are an helpful assistant from the public administration, use a {{TONE}} tone.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.
    Answer in italian.
    The user is a {{USER_TYPE}}.

    <document>{{DOCUMENT}}</document>"""

    full_data_splits = dict()

    ### iterate each of the splits
    for split_type, model_data in model_data_splits.items():
        llm_templates_finetuning = list()

        ### Make template of each dialogue in the split
        for dial in model_data:
            try:
                prompt = prompt_template\
                    .replace("{{TONE}}", dial["metadata"]["interaction_style"])\
                    .replace("{{USER_TYPE}}", dial["metadata"]["user_type"])\
                    .replace("{{DOCUMENT}}", dial["doc"])

            except Exception as e:
                pprint(dial)
                raise Exception(e)

            ### Dictionary format expected by the template conversion
            conversation = copy.deepcopy(dial["messages"])
            conversation.insert(0, {"role": "system", "content": prompt})

            # add all segments of the conversation in which the agent has to reply
            conversation_segment = list()
            for message in conversation:
                conversation_segment.append(message)

                ### Make an entry if the current turn is made by the assistant
                if message["role"] == "assistant":
                    entry_txt  = tokenizer.apply_chat_template(copy.deepcopy(conversation_segment), tokenize=False)
                    entry_txt  = entry_txt\
                        .replace("Today Date: 26 Jul 2024", "Today Date: 1 Apr 2024")\
                        .replace("Cutting Knowledge Date: December 2023", "Cutting Knowledge Date: September 2024")
                    llm_templates_finetuning.append({"text": entry_txt, "id": dial["id"], "conv_len": len(conversation_segment)-1})


        ### shuffle the data and save it in the matching split
        random.shuffle(llm_templates_finetuning)
        print(f"# entries in {split_type}: {len(llm_templates_finetuning)}")
        full_data_splits[split_type] = llm_templates_finetuning

    ### save splits data on disk

    for split_type, split_data in full_data_splits.items():
        os.makedirs(split_path, exist_ok=True)
        with open(os.path.join(args.out_dir, f"{split_type}.json"), 'w') as file:
            json.dump([{"text": item["text"]} for item in split_data], file, indent=4)

        with open(os.path.join(args.out_dir, f"{split_type}_ids.txt"), 'w') as file:
            file.write("\n".join([item["id"] + "_len_" + str(item["conv_len"]) for item in split_data]))