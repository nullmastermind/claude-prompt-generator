import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

default_system = "You are a helpful and knowledgeable assistant who is able to provide detailed and accurate information on a wide range of topics. You are also able to provide clear and concise answers to questions and are always willing to go the extra mile to help others."
model_1_default_system = default_system
model_2_default_system = default_system

evaluate_response_prompt_template = """
You are an expert in linguistics and able to observe subtle differences in content between two paragraphs. Your task is to analyze responses from OpenAI and Claude and provide detailed feedback.

Here are the OpenAI response: 
<response>
{_OpenAI}
</response>

Here are the Claude response:
<response>
{_Bedrock}
</response>

Please follow these steps:
1. Carefully analyze both responses in terms of content accuracy, logical organization, and expression style.
2. Summarize the differences between the Claude response and the OpenAI response.
3. Provide recommendations on how the Claude response could be refactored to better align with the OpenAI response.
4. Encapsulate your analysis, including the differences, within <auto_feedback></auto_feedback> tags using bullet points.
5. Encapsulate recommendations, within <recommendation></recommendation> tags using bullet points.
""".strip()

generate_revised_prompt_template = """
You are an expert in prompt engineering for both OpenAI and Claude model and able to follow the human feedback to adjust the prompt to attain the optimal effect, you will be given the original Claude prompt, responses from OpenAI, responses from Claude and human feedback to revise the Claude prompt.

Here are the original Claude prompt: 
<prompt>
{_prompt}
</prompt>

Here are the OpenAI response:
<response>
{_OpenAI}
</response>

Here are the Claude response:
<response>
{_Bedrock}
</response>

Here are the human feedback:
<evaluation_summary>
{_feedback}
</evaluation_summary>

Please analyze whether Claude's response strictly aligns with OpenAI's response based on the human feedback. Then, consider how the original Claude prompt can be improved accordingly. Your revised prompt should only involve slight adjustments and must not drastically change the original prompt. Use the human feedback to guide your revision.

Finally, provide the revised prompt within the following XML tags:

<revised_prompt>
[Your revised prompt]
</revised_prompt>
""".strip()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")


class Alignment:
    def __init__(self):
        self.openai_client = OpenAI(
            base_url=openai_base_url,
            # This is the default and can be omitted
            api_key=openai_api_key,
        )

    def generate_bedrock_response(self, prompt, model_id):
        return self.generate_openai_response(prompt=prompt, model_id=model_id)

    def generate_openai_response(self, prompt, model_id):
        completion = self.openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": model_1_default_system},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def stream_bedrock_response(self, prompt, model_id, output_component):
        self.stream_openai_response(
            prompt=prompt,
            model_id=model_id,
            output_component=output_component,
        )

    def stream_openai_response(self, prompt, model_id, output_component):
        stream = self.openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                output_component.update(chunk.choices[0].delta.content, append=True)

    def invoke_prompt(
        self,
        original_prompt_replace,
        revised_prompt_replace,
        original_prompt,
        revised_prompt,
        openai_model_id,
        aws_model_id,
    ):
        if len(original_prompt_replace) == 0:
            original_prompt_replace = original_prompt
        if len(revised_prompt_replace) == 0:
            revised_prompt_replace = revised_prompt
        if self.openai_client is None:
            openai_result = "OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            aws_result = "OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            return openai_result, aws_result
        openai_result = self.generate_openai_response(
            original_prompt_replace, openai_model_id
        )
        aws_result = self.generate_bedrock_response(
            revised_prompt_replace, aws_model_id
        )
        return openai_result, aws_result

    def evaluate_response(
        self,
        openai_output,
        aws_output,
        eval_model_id,
        openai_model_id,
        aws_model_id,
    ):
        revised_prompt = evaluate_response_prompt_template.format(
            _OpenAI=openai_output,
            _Bedrock=aws_output,
            _Model_1=openai_model_id,
            _Model_2=aws_model_id,
        )
        aws_result = self.generate_bedrock_response(revised_prompt, eval_model_id)
        pattern = r"<auto_feedback>(.*?)</auto_feedback>"
        feedback = re.findall(pattern, aws_result, re.DOTALL)[0]

        pattern = r"<recommendation>(.*?)</recommendation>"
        recommendation = re.findall(pattern, aws_result, re.DOTALL)[0]

        # remove all the \n and []
        # matches = matches[0]#.replace("\n", "").replace("[", "").replace("]", "")
        return feedback + f"\n<recommendation>{recommendation}</recommendation>"

    def insert_kv(self, user_prompt, kv_string):
        # Split the key-value string by ';' to get individual pairs
        kv_pairs = kv_string.split(";")
        for pair in kv_pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)  # Only split on the first ':'
                user_prompt = user_prompt.replace(f"{{{key}}}", value)
        return user_prompt

    def generate_revised_prompt(
        self,
        feedback,
        prompt,
        openai_response,
        aws_response,
        eval_model_id,
        openai_model_id,
        aws_model_id,
    ):
        pattern = r"<recommendation>(.*?)</recommendation>"
        matches = re.findall(pattern, feedback, re.DOTALL)
        if len(matches):
            feedback = matches[0]
        revised_prompt = generate_revised_prompt_template.format(
            _feedback=feedback,
            _prompt=prompt,
            _OpenAI=openai_response,
            _Bedrock=aws_response,
            _Model_1=openai_model_id,
            _Model_2=aws_model_id,
        )
        aws_result = self.generate_bedrock_response(revised_prompt, eval_model_id)
        pattern = r"<revised_prompt>(.*?)</revised_prompt>"
        matches = re.findall(pattern, aws_result, re.DOTALL)
        # remove all the \n and []
        matches = matches[0]  # .replace("\n", "").replace("[", "").replace("]", "")
        return matches.strip()
