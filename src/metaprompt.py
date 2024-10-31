import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")


class MetaPrompt:
    def __init__(self):
        # Get the directory where the current script is located
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the file
        prompt_guide_path = os.path.join(current_script_path, "metaprompt.txt")

        # Open the file using the full path
        with open(prompt_guide_path, "r", encoding="utf-8") as f:
            self.metaprompt = f.read()

        # region_name = os.getenv("REGION_NAME")
        # session = boto3.Session()
        # retry_config = Config(
        #     region_name=region_name,
        #     retries={
        #         "max_attempts": 5,
        #         "mode": "standard",
        #     },
        # )
        # service_name = "bedrock-runtime"
        # self.bedrock_client = session.client(
        #     service_name=service_name, config=retry_config
        # )
        self.openai_client = OpenAI(
            base_url=openai_base_url,
            api_key=openai_api_key,
        )

    def __call__(self, task, variables):
        variables = variables.split("\n")
        variables = [variable for variable in variables if len(variable)]

        variable_string = ""
        for variable in variables:
            variable_string += "\n{$" + variable.upper() + "}"
        prompt = self.metaprompt.replace("{{TASK}}", task)
        assistant_partial = "<Inputs>"
        if variable_string:
            assistant_partial += (
                variable_string + "\n</Inputs>\n<Instructions Structure>"
            )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_partial},
        ]
        # body = json.dumps(
        #     {
        #         "messages": messages,
        #         "max_tokens": 4096,
        #         "temperature": 0.0,
        #         "anthropic_version": "bedrock-2023-05-31",
        #     }
        # )
        # modelId = "anthropic.claude-3-haiku-20240307-v1:0"  # anthropic.claude-3-sonnet-20240229-v1:0 "anthropic.claude-3-haiku-20240307-v1:0"
        # accept = "application/json"
        # contentType = "application/json"
        #
        # response = self.bedrock_client.invoke_model(
        #     body=body, modelId=modelId, accept=accept, contentType=contentType
        # )
        # response_body = json.loads(response.get("body").read())
        # message = response_body["content"][0]["text"]
        message = self.generate_openai_response(messages=messages, model_id="gpt-4o")

        def pretty_print(message):
            print(
                "\n\n".join(
                    "\n".join(
                        line.strip()
                        for line in re.findall(
                            r".{1,100}(?:\s+|$)", paragraph.strip("\n")
                        )
                    )
                    for paragraph in re.split(r"\n\n+", message)
                )
            )

        extracted_prompt_template = self.extract_prompt(message)
        variables = self.extract_variables(message)

        return extracted_prompt_template.strip(), "\n".join(variables)

    def generate_openai_response(self, messages, model_id):
        completion = self.openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content

    def extract_between_tags(
        self, tag: str, string: str, strip: bool = False
    ) -> list[str]:
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list

    def remove_empty_tags(self, text):
        return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)

    def extract_prompt(self, metaprompt_response):
        between_tags = self.extract_between_tags("Instructions", metaprompt_response)[0]
        return (
            between_tags[:1000]
            + self.remove_empty_tags(
                self.remove_empty_tags(between_tags[1000:]).strip()
            ).strip()
        )

    def extract_variables(self, prompt):
        pattern = r"{([^}]+)}"
        variables = re.findall(pattern, prompt)
        return set(variables)


# test = MetaPrompt()
# TASK = "Draft an email responding to a customer complaint" # Replace with your task!
# # Optional: specify the input variables you want Claude to use. If you want Claude to choose, you can set `variables` to an empty list!

# VARIABLES = ["CUSTOMER_COMPLAINT", "COMPANY_NAME"]
# test(TASK, VARIABLES)
