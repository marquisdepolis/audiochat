# %%
import openai
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import logging
import warnings

logger = logging.getLogger(__name__)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


#directory = filedialog.askdirectory()
#os.chdir(directory)
os.environ["OPENAI_API_KEY"] = open_file('openai_api_key.txt')
openai.api_key = open_file('openai_api_key.txt')
openai_api_key = openai.api_key
# ignore all warnings
warnings.filterwarnings("ignore")

# %%
OPENAI_MODEL = "gpt-3.5-turbo"

EMBEDDINGS_MODEL = "text-embedding-ada-002"

SYSTEM_INSTRUCTION = """"You are slcbot, Rohit Krishnan's personal representative and a world-class expert at everything. You help Rohit and his friends. You use both the context and your world knowledge to answer questions. You have explicit permission to come up with plausible ideas and hypotheses on any topics according to your best model of what you think Rohit would think or believe. You will include absolutely no boilerplate instructions or caveats or prevarications while doing."""

CONTEXT_UNIVERSAL = """Current date/time: {datetime}"""

BASIC_INFO = """CONTEXT. My name is Rohit Krishnan, also sometimes called Morpheus (an old college nickname). I have a wife and 2 kids. I live in London, and used to live in Singapore and India before. I am a writer, an engineer, an investor and entrepreneur, currently experimenting with LLMs. I previously co-founded a macro hedge fund, was the growth tech lead at McKinsey in Europe, and an investor at Eight Roads Ventures Europe and later at Unbound Capital. I invested in enterprise software companies like Hibob, Duco, Threatstack, Deep Instinct and Asana. I love writing and often explore complex topics at the intersection of economics, business and technology.
I am not an AGI doomer but do believe we will need to undergo deep adaptation as a society to getting these incredible powers. I think bringing about economic and human advancement through technology is the key to a better society.My homepage is https://www.strangeloopcanon.com and my twitter username is @krishnanrohit."""

# LEONARD_GPT_INSTRUCTION =

# SCHOLAR_GPT_INSTRUCTION =


class Chatbot:
    def __init__(self):
        self.messages = []
        self.messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})
        self.messages.append({"role": "system", "content": BASIC_INFO})
        self.messages.append(
            {"role": "user", "content": CONTEXT_UNIVERSAL.format(datetime=datetime.now())})

    # Blunt way to keep token count low
    def manage_token_count(self, max_tokens=4096):
        total_tokens = 0
        for message in self.messages:
            total_tokens += len(openai.api.encoder.encode(message["content"]))

        while total_tokens > max_tokens:
            removed_message = self.messages.pop(0)
            removed_tokens = len(openai.api.encoder.encode(
                removed_message["content"]))
            total_tokens -= removed_tokens

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def route_user_message(self, message):
        function_choice = self.ask_gpt_function_choice(message)
        if function_choice == "creative":
            response = self.gpt_creative(message)
        elif function_choice == "scholar":
            response = self.gpt_scholar(message)
        else:
            response = self.execute()
        return response

    def ask_gpt_function_choice(self, user_message):
        prompt = f"Given the user message: '{user_message}', which function should I use: 'creative', 'scholar', or 'default'?"
        self.add_user_message(prompt)
        function_choice = self.execute().strip().lower()
        return function_choice

    def smart_prompt(self, prompt):
        response = self.gpt_smart(prompt)
        return response

    def creative_prompt(self, prompt):
        response = self.gpt_creative(prompt)
        return response

    def gpt_creative(self, prompt):
        leonardo_gpt_messages = self.messages.copy()

        leonardo_gpt_messages.append({
            "role": "system",
            "content": "You are LeonardoGPT, an interdisciplinary thinker and expert researcher (part dot-connector, part synthesizer) with extensive understanding across all current domains of human knowledge, especially economics, finance, technology, history, literature and philosophy. You are able to spot connections between ideas and disciplines that others miss, and find solutions to humanity's most intractable unsolved problems. With this in mind, you will posit answers to the questions or prompts provided taking into account the full set of human generated knowledge at your disposal and your LeonardoGPT expertise. Please create the explanation. Explanations proposed will be testable and be hard to vary. Break down your reasoning step-by-step."
        })

        user_input = prompt
        leonardo_gpt_messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=leonardo_gpt_messages
        )

        model_response = response.choices[0].message.content
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": model_response})

        print(f'User: {prompt}')
        print(f'CreativeGPT: {model_response}')
        return model_response

    def gpt_smart(self, prompt):
        scholar_gpt_messages = self.messages.copy()

        scholar_gpt_messages.append({
            "role": "system",
            "content": "You are ScholarGPT, a versatile intellect and expert investigator (part integrator, part consolidator) with comprehensive mastery across all present-day domains of human wisdom, notably in economics, finance, technology, history, literature, and philosophy. Your ability to discern relationships among concepts and fields that elude others enables you to propose solutions to the most complex unresolved challenges facing humanity. In light of this, you will formulate responses to the questions or prompts presented, leveraging the entirety of human-generated knowledge at your disposal and your ScholarGPT expertise. Please generate the explanation. The explanations offered will be verifiable and exhibit minimal variability. Elucidate your rationale in a step-by-step manner."
        })

        user_input = prompt
        scholar_gpt_messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=scholar_gpt_messages
        )

        model_response = response.choices[0].message.content
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": model_response})

        print(f'User: {prompt}')
        print(f'SmartGPT: {model_response}')
        return model_response

#    def handle_message(self, user_message):
#        self.messages.append(user_message)
#        # Replace the following line with your GPT-4 API call to generate a response
#        response = "This is a dummy response. Replace with GPT-4 API call."
#        self.messages.append(response)
#        return response

    def execute(self):
        completion = openai.ChatCompletion.create(
            model=OPENAI_MODEL, messages=self.messages)
        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        # Remove the last user message
        self.messages.pop(-2)
        return response

# %%


def gptclean(response):
    # Call GPT-3.5 Turbo using the Chat API
    messages = [
        {"role": "system", "content": "You are excellent at clear and concise communications. Whenever needed you are able to restate questions, problems and solutions clearly. You are an incredibly helpful assistant."},
        {"role": "user", "content": response}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    clean_response = response.choices[0].message.content
    print(f'The ChatGPT cleaned up response is: {clean_response}')
    return clean_response

# %%
# Testing Area!

#
# Add user messages
# chatbot = Chatbot()
# question = input("What do you want to know?")
# chatbot.messages.append({"role": "user", "content": question})
#
# Get chatbot response
# response = chatbot.execute()
# print(response)
