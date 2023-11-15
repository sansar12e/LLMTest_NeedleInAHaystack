from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

class ClaudeEnc:
    
    def __init__(self, llm):
        self.llm = llm
        
    def encode(self, text):
        return self.llm.client.get_tokenizer().encode(text).ids
    
    def decode(self, token_ids):
        return self.llm.client.get_tokenizer().decode(token_ids)
    
def get_claude_enc(claude_llm):
    return ClaudeEnc(claude_llm)

def path_from_code(relpath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relpath)

def read_files(directory):
    context = ""
    for file in glob.glob(directory):
        with open(file, 'r') as f:
            context += f.read()
    return context

def encode_and_trim(context, context_length, enc):
    tokens = enc.encode(context)
    if len(tokens) > context_length:
        context = enc.decode(tokens[:context_length])
    return context

def insert_needle(needle, context, depth_percent, context_length, enc):
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 150

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = enc.encode('.')
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = enc.decode(tokens_new_context)
    return new_context

def generate_context(needle, context_length, depth_percent, enc):
    # Get your Paul Graham files loaded into a string
    graham_glob = path_from_code("PaulGrahamEssays/*.txt")
    context = read_files(graham_glob)

    # Truncate the Paul Graham essays to the context length you desire
    context = encode_and_trim(context, context_length, enc)

    # Insert your random statement according to your depth percent
    context = insert_needle(needle, context, depth_percent, context_length, enc)

    return context

def evaluate_response(response, needle, question_to_ask):
    accuracy_criteria = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor errors or omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Keep your explanations extremely short, just give the score
        """
    }

    # Using GPT-4 to evaluate
    evaluator = load_evaluator(
        "labeled_score_string",
        criteria=accuracy_criteria,
        llm=ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey')),
    )

    eval_result = evaluator.evaluate_strings(
        # The models response
        prediction=response,

        # The actual answer
        reference=needle,

        # The question asked
        input=question_to_ask,
    )

    return int(eval_result['score'])

def result_exists(results, context_length, depth_percent, version):
    """
    Checks to see if a result has already been evaluated or not
    """
    conditions_met = []
    for result in results:
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == version
        conditions_met.append(context_length_met and depth_percent_met and version_met)
    return any(conditions_met)


def main(
        needle = """
    The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
    """,
        question_to_ask = "What is the most fun thing to do in San Francisco?",

        # The code will check to see if a context_length, depth percent and version number have already been checked yet
        # Change the version # if you would like to run the results multiple times.
        # If you're just testing, then leave as version=1
        results_version = 1,
        context_aware_query=False,  # Added parameter for context-aware query
        use_messages_context=False,  # Added parameter to choose context source; if False, inline context is used
        model_to_test = ChatOpenAI(
            model='gpt-4-1106-preview', temperature=0, 
            openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey')),
        enc = tiktoken.encoding_for_model("gpt-4-1106-preview"),
        max_context_length = 128000,
        num_context_lengths = 15,
        num_depths = 15,
        rate_limit = 120000
): 

    # This will produce a list of context lengths for each experiment iteration
    context_lengths = np.round(np.linspace(1000, max_context_length,
                                           num=num_context_lengths,
                                           endpoint=True)).astype(int)
    
    # This will product a list of document depths to place your random statement (needle) at.
    # Suggestion: Try out different distributions (like a sigmoid) to test non-evenly space intervals
    document_depth_percents = np.round(np.linspace(0, 100, num=num_depths, endpoint=True)).astype(int)
    
    results_file_name = path_from_code("results_%s.json" % results_version)

    # Run through each iteration of context_lengths and depths
    for context_length in context_lengths:
        for depth_percent in document_depth_percents:
            # Load results from file. 
            try:
                with open(results_file_name, 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = []
                pass

            # Checks to see if you've already checked a length/percent/version.
            # This helps if the program stop running and you want to restart later
            if result_exists(results, context_length, depth_percent, results_version):
                continue

            # Go generate the required length context and place your needle statement in
            context = generate_context(needle, context_length, depth_percent, enc)

            # Prepare your message to send to the model you're going to evaluate
            if use_messages_context:
                # Prepare context using messages
                if context_aware_query:
                messages = [
                SystemMessage(
                    content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                ),
                HumanMessage(
                        content="What is the most fun thing to do in San Francisco based on the following context? Don't give information outside the document"
                    ),
                HumanMessage(
                    # This is the PG essays with your needle/random statement placed in it
                    # This is your haystack with a needle placed in it.
                    content=context
                ),
                HumanMessage(
                        content="What is the most fun thing to do in San Francisco based on the preceding context? Don't give information outside the document"
                    ),
            ]
                else:
                messages = [
                    SystemMessage(
                        content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                    ),
                    HumanMessage(
                        content=context
                    ),
                    HumanMessage(
                        content="What is the most fun thing to do in San Francisco based on the preceding context? Don't give information outside the document"
                    ),
                ]
            else:
                #Prepare context using inline context
                if context_aware_query:
                messages = [
                SystemMessage(
                    content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                ),
                HumanMessage(
                    content=f"""<question>What is the most fun thing to do in San Francisco based on the following context? Don't give information outside the document.</question>
                    %%%%%%{context}%%%%%%
               <question>What is the most fun thing to do in San Francisco based on the preceding context? Don't give information outside the document.</question>"""
                    ),
            ]
                else:
                messages = [
                    SystemMessage(
                        content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                    ),
                    HumanMessage(
                    content=f"""
                    {context}%%%%%%
               <question>What is the most fun thing to do in San Francisco based on the preceding context? Don't give information outside the document.</question>"""
                    ),
            ]

            # Go see if the model can answer the question to pull out your random fact
            response = model_to_test(messages)

            # Go see if the model can answer the question to pull out your random fact
            response = model_to_test(messages)

            # Compare the reponse to the actual needle you placed
            score = evaluate_response(response, needle, question_to_ask)

            # Optional, print out bad responses if you want
            if score != 10:
                print (f"Bad Answer: {response}")

            results.append({
                # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
                'context_length' : int(context_length),
                'depth_percent' : int(depth_percent),
                'version' : results_version,
                'needle' : needle,
                'model_response' : response.content,
                'score' : score
            })

            print (f"#{len(results)} Context: {context_length}, Depth: {depth_percent}, Score: {score}")

            # Save results to a JSON file each run
            with open(results_file_name, 'w') as f:
                json.dump(results, f)

            if rate_limit is not None:
            # Optional. Sleep for a bit to stay under the rate limit
            # Rate limit is 150K tokens/min so it's set at 120K for some cushion
                sleep_time = (context_length / rate_limit)*60
                print (f"Sleeping: {sleep_time}\n")
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()