import json
import os
import re

import boto3
import gradio as gr
import threading
from botocore.config import Config
from dotenv import load_dotenv
from openai import OpenAI

from ape import APE
from translate import GuideBased

ape = APE()
rewrite = GuideBased()

load_dotenv()

# Initialize the LLM client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime", region_name=os.getenv("REGION_NAME")
)
client = OpenAI()

default_system = "you have profound knowledge and hands on experience in field of software engineering and artificial intelligence, you are also an experienced solution architect in Amazon Web Service and have expertise to impelment model application development with AWS in consdieration of well architect and industry best practice."
bedrock_default_system = default_system
openai_default_system = default_system

evaluate_response_prompt_template = """
You are an expert in linguistics and able to observe the most subtle content difference between two paragraph, you will be given responses from OpenAI, Bedrock to observe.

Here are the OpenAI response: 
<response>
{_OpenAI}
</response>

Here are the Bedrock response:
<response>
{_Bedrock}
</response>


Your goal is to summarize the difference in detail of Bedrock response compare to OpenAI response, first analyze both response carefully in content accuracy, logical organize method and expression style, then give your recommendation on how the Bedrock reponse could be refactored to align with the OpenAI response, finally give your answer including the difference and recommendation with bullet points <auto_feedback></auto_feedback> tags.
"""

generate_revised_prompt_template = """
You are an expert in prompt engineering for both OpenAI and Claude model and able to follow the human feedback to adjust the prompt to attain the optimal effect, you will be given original Claude prompt, responses from OpenAI, Claude and human feedback to revise the Claude prompt.

Here are the user original prompt: 
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

Please first analyze whether Claude responsded to original prompts in alignment OpenAI response according to the human feedback, then consider how the user original prompt can be improved accordingly, finally provide the revised prompt in <revised_prompt></revised_prompt> tags.
"""


def generate_prompt(original_prompt, level):
    if level == "一次生成":
        result = rewrite(original_prompt)  # , cost
        return [
            gr.Textbox(
                label="我们为您生成的prompt",
                value=result,
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [gr.Textbox(visible=False)] * 2

    elif level == "多次生成":
        candidates = []
        for i in range(3):
            result = rewrite(original_prompt)
            candidates.append(result)
        judge_result = rewrite.judge(candidates)
        textboxes = []
        for i in range(3):
            is_best = "Y" if judge_result == i else "N"
            textboxes.append(
                gr.Textbox(
                    label=f"我们为您生成的prompt #{i+1} {is_best}",
                    value=candidates[i],
                    lines=3,
                    show_copy_button=True,
                    visible=True,
                    interactive=False,
                )
            )
        return textboxes


def ape_prompt(original_prompt, user_data):
    result = ape(initial_prompt, 1, json.loads(user_data))
    return [
        gr.Textbox(
            label="我们为您生成的prompt",
            value=result["prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
    ] + [gr.Textbox(visible=False)] * 2


def generate_bedrock_response(prompt, model_id):
    """
    This function generates a test dataset by invoking a model with a given prompt.

    Parameters:
    prompt (str): The user input prompt.

    Returns:
    matches (list): A list of questions generated by the model, each wrapped in <case></case> XML tags.
    """
    message = {
        "role": "user",
        "content": [
            # {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image}},
            {"type": "text", "text": prompt}
        ],
    }
    messages = [message]
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": messages,
            "system": bedrock_default_system,
        }
    )
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get("body").read())
    return response_body["content"][0]["text"]


def generate_openai_response(prompt, model_id):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": openai_default_system},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def stream_bedrock_response(prompt, model_id, output_component):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }
    messages = [message]
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": messages,
            "system": bedrock_default_system,
        }
    )
    response = bedrock_runtime.invoke_model_with_response_stream(modelId=model_id, body=body)

    stream = response.get('body')
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                output = json.loads(chunk.get('bytes').decode())
                output_component.update(output)


def stream_openai_response(prompt, model_id, output_component):
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            output_component.update(chunk.choices[0].delta.content, append=True)


def invoke_prompt_stream(original_prompt, revised_prompt, openai_model_id, aws_model_id, openai_output_component, aws_output_component):
    # Start streaming in separate threads to allow concurrent streaming
    openai_thread = threading.Thread(target=stream_openai_response, args=(original_prompt, openai_model_id, openai_output_component))
    bedrock_thread = threading.Thread(target=stream_bedrock_response, args=(revised_prompt, aws_model_id, aws_output_component))

    openai_thread.start()
    bedrock_thread.start()


def invoke_prompt(original_prompt, revised_prompt, openai_model_id, aws_model_id):
    openai_result = generate_openai_response(original_prompt, openai_model_id)
    aws_result = generate_bedrock_response(revised_prompt, aws_model_id)
    return openai_result, aws_result


def evaluate_response(openai_output, aws_output, eval_model_id):
    revised_prompt = evaluate_response_prompt_template.format(_OpenAI=openai_output, _Bedrock=aws_output)
    aws_result = generate_bedrock_response(revised_prompt, eval_model_id)
    pattern = r'<auto_feedback>(.*?)</auto_feedback>'
    matches = re.findall(pattern, aws_result, re.DOTALL)
    # remove all the \n and []
    matches = matches[0].replace("\n", "").replace("[", "").replace("]", "")
    return matches

def insert_kv(user_prompt, kv_string):
    # Split the key-value string by ';' to get individual pairs
    kv_pairs = kv_string.split(";")
    for pair in kv_pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)  # Only split on the first ':'
            user_prompt = user_prompt.replace(f"{{{key}}}", value).replace(
                f"<{key}>", value
            )
    return user_prompt


def generate_revised_prompt(feedback, prompt, openai_response, aws_response, eval_model_id):
    revised_prompt = generate_revised_prompt_template.format(_feedback=feedback, _prompt=prompt, _OpenAI=openai_response, _Bedrock=aws_output)
    aws_result = generate_bedrock_response(revised_prompt, eval_model_id)
    pattern = r'<revised_prompt>(.*?)</revised_prompt>'
    matches = re.findall(pattern, aws_result, re.DOTALL)
    # remove all the \n and []
    matches = matches[0].replace("\n", "").replace("[", "").replace("]", "")
    return matches

with gr.Blocks(
    title="Automatic Prompt Engineering",
    theme="soft",
    css="#textbox_id textarea {color: red}",
) as demo:
    with gr.Tab("Prompt 生成"):
        gr.Markdown("# Automatic Prompt Engineering")
        original_prompt = gr.Textbox(label="请输入您的原始prompt", lines=3)
        gr.Markdown("其中用户自定义变量使用{\{xxx\}}表示，例如{\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(
                    ["一次生成", "多次生成"], label="优化等级", value="一次生成"
                )
                b1 = gr.Button("优化prompt")
            with gr.Column(scale=2):
                user_data = gr.Textbox(label="测试数据JSON", lines=2)
                b2 = gr.Button("APE优化prompt")
        textboxes = []
        for i in range(3):
            t = gr.Textbox(
                label="我们为您生成的prompt",
                elem_id="textbox_id",
                lines=3,
                show_copy_button=True,
                interactive=False,
                visible=False if i > 0 else True,
            )
            textboxes.append(t)
        log = gr.Markdown("")
        b1.click(generate_prompt, inputs=[original_prompt, level], outputs=textboxes)
        b2.click(ape_prompt, inputs=[original_prompt, user_data], outputs=textboxes)
    with gr.Tab("Prompt 评估"):
        with gr.Row():
            user_prompt_original = gr.Textbox(label="请输入您的原始prompt", lines=3)
            kv_input = gr.Textbox(
                label="[可选]输入需要替换的模版参数",
                placeholder="参考格式: key1:value1;key2:value2",
                lines=2,
            )
            user_prompt_eval = gr.Textbox(label="请输入您要评估的prompt", lines=3)
            kv_input = gr.Textbox(
                label="[可选]输入需要替换的模版参数",
                placeholder="参考格式: key1:value1;key2:value2",
                lines=2,
            )
        with gr.Row():
            insert_button_original = gr.Button("替换原始模版参数")
            insert_button_original.click(
                insert_kv,
                inputs=[user_prompt_original, kv_input],
                outputs=user_prompt_original,
            )
            insert_button_revise = gr.Button("替换评估模版参数")
            insert_button_revise.click(
                insert_kv, inputs=[user_prompt_eval, kv_input], outputs=user_prompt_eval
            )
        with gr.Row():
            # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
            openai_model_dropdown = gr.Dropdown(
                label="选择 OpenAI 模型",
                choices=[
                    "gpt-3.5-turbo",
                    "gpt-4-32k",
                    "gpt-4-1106-preview",
                    "gpt-4-turbo-preview",
                ],
                value="gpt-3.5-turbo",
            )
            # aws bedrock list-foundation-models --region us-east-1 --output json | jq -r '.modelSummaries[].modelId'
            aws_model_dropdown = gr.Dropdown(
                label="选择 AWS 模型",
                choices=[
                    "anthropic.claude-instant-v1:2:100k",
                    "anthropic.claude-instant-v1",
                    "anthropic.claude-v2:0:18k",
                    "anthropic.claude-v2:0:100k",
                    "anthropic.claude-v2:1:18k",
                    "anthropic.claude-v2:1:200k",
                    "anthropic.claude-v2:1",
                    "anthropic.claude-v2",
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                ],
                value="anthropic.claude-3-sonnet-20240229-v1:0",
            )
        invoke_button = gr.Button("调用prompt")
        with gr.Row():
            openai_output = gr.Textbox(
                label="OpenAI 输出", lines=3, interactive=False, show_copy_button=True
            )
            aws_output = gr.Textbox(
                label="AWS Bedrock 输出", lines=3, interactive=False, show_copy_button=True
            )
        invoke_button.click(
            invoke_prompt,
            inputs=[
                user_prompt_original,
                user_prompt_eval,
                openai_model_dropdown,
                aws_model_dropdown,
            ],
            outputs=[openai_output, aws_output],
        )
        # invoke_button.click(
        #     invoke_prompt_stream,
        #     inputs=[
        #         user_prompt_original,
        #         user_prompt_eval,
        #         openai_model_dropdown,
        #         aws_model_dropdown,
        #         openai_output,
        #         aws_output,
        #     ],
        #     outputs=[]
        # )


        with gr.Row():
            feedback_input = gr.Textbox(
                label="评估prompt效果", placeholder="手动填入反馈或自动评估", lines=3, show_copy_button=True
            )
            with gr.Column():
                eval_model_dropdown = gr.Dropdown(
                    label="选择评价模型",
                    # Use Bedrock to evaluate the prompt, sonnet or opus are recommended
                    choices=[
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        # opus placeholder
                    ],
                    value="anthropic.claude-3-sonnet-20240229-v1:0",
                )
                evaluate_button = gr.Button("自动评估prompt效果")
                evaluate_button.click(
                    evaluate_response,
                    inputs=[
                        openai_output,
                        aws_output,
                        eval_model_dropdown,
                    ],
                    outputs=[feedback_input],
                )
        revise_button = gr.Button("修正Prompt")
        revised_prompt_output = gr.Textbox(
            label="修正后的Prompt", lines=3, interactive=False
        )
        revise_button.click(
            generate_revised_prompt,
            inputs=[feedback_input, user_prompt_eval, openai_output, aws_output, eval_model_dropdown],
            outputs=revised_prompt_output,
        )

demo.launch()
