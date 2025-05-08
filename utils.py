from datasets import Dataset

def should_exclude(d, exclude, training):
    """Helper function to determine whether to exclude a record based on guidelines."""
    if exclude is not None:
        if training and exclude in d["guidelines"]:
            return True
        if not training and exclude not in d["guidelines"]:
            return True
    return False

def format_user_string(events_list, previous_messages= False, guidelines=None, guideline_map=None):
    """Helper function to create the user prompt based on the events list and optional guidelines."""
    user_string = "Create a warning message informing on the current happening, providing a suggestion, for the last line in the following chain of events (be short, max 300 characters). No other output other than the message."

    # Add guidelines if provided
    if guidelines is not None and guideline_map is not None:
        user_string = "Based on the provided guidelines, " + user_string
        user_string += "\n\nGuidelines:"
        for g in guidelines:
            user_string += f"\n\n{guideline_map[g]}"

    # Add previous messages to the prompt
    if previous_messages:
        user_string += "\n\nPrevious messages:"
        for e in events_list[:-1]:
            user_string += f"\n-{e['message']}"

    # Add events to the prompt
    user_string += "\n\nChain of events:"
    for e in events_list:
        user_string += f"\n-{e['event_text']}"

    return user_string

def prepare_dataset(data, tokenizer, from_base=False, previous_messages= False, guidelines=None):
    """Prepare dataset with optional guidelines based on training mode and exclusion criteria."""
    formatted_dataset = []

    for d in data:

        # If guidelines are provided, pass them to the formatting function, else pass None
        user_string = format_user_string(d["events_list"], previous_messages, d.get("guidelines") if guidelines else None, guidelines)

        if not from_base:
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_string},
                {"role": "assistant", "content": d["events_list"][-1]['message']},
            ]

            formatted_text = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            formatted_text = "<|begin_of_text|>" + user_string + "\n\nWarning Message:\n\n" + d["events_list"][-1]['message'] + "<|end_of_text|>"
        formatted_dataset.append({"text": formatted_text})

    return Dataset.from_list(formatted_dataset)


def prepare_dpo_dataset(data, tokenizer, rejected_type):

    formatted_dataset = []

    for d in data:

        user_string = "Create a warning message informing on the current happening, providing a suggestion, for the last line in the following chain of events (be short, max 300 characters). No other output other than the message:\n"

        for e in d["events_list"]:
            user_string += f"\n-{e['event_text']}"

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_string},
        ]

        chosen = [{"role": "assistant", "content": d["events_list"][-1]['message']}]
        rejected = [{"role": "assistant", "content": d["events_list"][-1][rejected_type]}]

        formatted_dataset.append(
            {
                "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
                "chosen": tokenizer.apply_chat_template(chosen, tokenize=False).split("<|eot_id|>")[1]+"<|eot_id|>",
                "rejected": tokenizer.apply_chat_template(rejected, tokenize=False).split("<|eot_id|>")[1]+"<|eot_id|>",
            }
        )

    return Dataset.from_list(formatted_dataset)


def generate_text(model, tokenizer, input_text):
    """
    Generates text from a model based on the provided input, while separating and returning the gold message 
    and the generated message.

    Args:
        model (`PreTrainedModel`):
            The model used for text generation. This model should be compatible with the `generate` method 
            and capable of accepting input_ids and attention masks.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used to encode and decode the input and output texts. It must handle 
            tokenization of the input prompt and decoding of the generated output.
        input_text (`str`):
            The input text containing the user prompt and an original message.

    Returns:
        (`str`, `str`):
            A tuple containing:
            - The gold message.
            - The generated message.
    """

    original_message = input_text.split("<|start_header_id|>assistant<|end_header_id|>", 1)
    prompt = original_message[0]
    original_message = original_message[1]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    output_ids = model.generate(
        input_ids, attention_mask=attention_mask, temperature=1, top_p=1, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id
    )
    output_text = tokenizer.decode(output_ids[0])

    message = output_text.split("assistant<|end_header_id|>", 1)

    if len(message) > 1:
        message = message[1].strip()
    else:
        message = output_text

    return original_message[:-10], message[:-10]

