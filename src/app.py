import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from ape import APE
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased

# Initialize components
ape = APE()
rewrite = GuideBased()
alignment = Alignment()
metaprompt = MetaPrompt()
calibration = CalibrationPrompt()
# Load environment variables
load_dotenv(override=True)

language = os.getenv("LANGUAGE", "en")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

# Load translations from JSON file
with open("translations.json", "r", encoding="utf-8") as f:
    lang_store = json.load(f)


def generate_prompt(original_prompt, level):
    if level == "One-time Generation":
        result = rewrite(original_prompt)
        return [
            gr.Textbox(
                label=lang_store[language]["Prompt Template Generated"],
                value=result,
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [gr.Textbox(visible=False)] * 2
    elif level == "Multiple-time Generation":
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
                    label=f"{lang_store[language]['Prompt Template Generated']} #{i+1} {is_best}",
                    value=candidates[i],
                    lines=3,
                    show_copy_button=True,
                    visible=True,
                    interactive=False,
                )
            )
        return textboxes


def ape_prompt(original_prompt, user_data):
    result = ape(original_prompt, 1, json.loads(user_data))
    return [
        gr.Textbox(
            label="Prompt Generated",
            value=result["prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
    ] + [gr.Textbox(visible=False)] * 2


def get_models():
    openai_client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_api_key,
    )
    models = openai_client.models.list()
    model_ids = [model.id for model in models.data]

    return model_ids


with gr.Blocks(
    title=lang_store[language]["Automatic Prompt Engineering"], theme="soft"
) as demo:
    model_ids = get_models()

    gr.Markdown(f"# {lang_store[language]['Automatic Prompt Engineering']}")

    with gr.Tab(lang_store[language]["Meta Prompt"]):
        original_task = gr.Textbox(
            label=lang_store[language]["Task"],
            lines=3,
            info=lang_store[language]["Please input your task"],
            placeholder=lang_store[language][
                "Draft an email responding to a customer complaint"
            ],
        )
        variables = gr.Textbox(
            label=lang_store[language]["Variables"],
            info=lang_store[language][
                "Please input your variables, one variable per line"
            ],
            lines=5,
            placeholder=lang_store[language]["CUSTOMER_COMPLAINT\nCOMPANY_NAME"],
        )
        metaprompt_button = gr.Button(lang_store[language]["Generate Prompt"])
        prompt_result = gr.Textbox(
            label=lang_store[language]["Prompt Template Generated"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        variables_result = gr.Textbox(
            label=lang_store[language]["Variables Generated"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        metaprompt_button.click(
            metaprompt,
            inputs=[original_task, variables],
            outputs=[prompt_result, variables_result],
        )

    with gr.Tab(lang_store[language]["Prompt Translation"]):
        original_prompt = gr.Textbox(
            label=lang_store[language]["Please input your original prompt"],
            lines=3,
            placeholder=lang_store[language][
                'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
            ],
        )
        gr.Markdown("Use {XXX} to express custom variable, e.g. {DOCUMENT}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(
                    ["One-time Generation", "Multiple-time Generation"],
                    label=lang_store[language]["Optimize Level"],
                    value="One-time Generation",
                )
                b1 = gr.Button(lang_store[language]["Generate Prompt"])
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        label=lang_store[language]["Prompt Template Generated"],
                        elem_id="textbox_id",
                        lines=3,
                        show_copy_button=True,
                        interactive=False,
                        visible=False if i > 0 else True,
                    )
                    textboxes.append(t)
                b1.click(
                    generate_prompt, inputs=[original_prompt, level], outputs=textboxes
                )

    with gr.Tab(lang_store[language]["Prompt Evaluation"]):
        with gr.Row():
            user_prompt_original = gr.Textbox(
                label=lang_store[language]["Please input your original prompt"], lines=3
            )
            kv_input_original = gr.Textbox(
                label=lang_store[language][
                    "[Optional]Input the template variable need to be replaced"
                ],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_original_replaced = gr.Textbox(
                label=lang_store[language]["Replace Result"], lines=3, interactive=False
            )
            user_prompt_eval = gr.Textbox(
                label=lang_store[language][
                    "Please input the prompt need to be evaluate"
                ],
                lines=3,
            )
            kv_input_eval = gr.Textbox(
                label=lang_store[language][
                    "[Optional]Input the template variable need to be replaced"
                ],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_eval_replaced = gr.Textbox(
                label=lang_store[language]["Replace Result"], lines=3, interactive=False
            )

        with gr.Row():
            insert_button_original = gr.Button(
                lang_store[language]["Replace Variables in Original Prompt"]
            )
            insert_button_original.click(
                alignment.insert_kv,
                inputs=[user_prompt_original, kv_input_original],
                outputs=user_prompt_original_replaced,
            )

            insert_button_revise = gr.Button(
                lang_store[language]["Replace Variables in Revised Prompt"]
            )
            insert_button_revise.click(
                alignment.insert_kv,
                inputs=[user_prompt_eval, kv_input_eval],
                outputs=user_prompt_eval_replaced,
            )

        with gr.Row():
            model_1_dropdown = gr.Dropdown(
                label=lang_store[language]["Choose Model 1"],
                choices=model_ids,
                value="gpt-4o-mini",
            )
            model_2_dropdown = gr.Dropdown(
                label=lang_store[language]["Choose Model 2"],
                choices=model_ids,
                value="gpt-4o",
            )

            invoke_button = gr.Button(lang_store[language]["Execute prompt"])

        with gr.Row():
            openai_output = gr.Textbox(
                label=lang_store[language]["Model 1 Output"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )
            aws_output = gr.Textbox(
                label=lang_store[language]["Model 2 Output"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )

            invoke_button.click(
                alignment.invoke_prompt,
                inputs=[
                    user_prompt_original_replaced,
                    user_prompt_eval_replaced,
                    user_prompt_original,
                    user_prompt_eval,
                    model_1_dropdown,
                    model_2_dropdown,
                ],
                outputs=[openai_output, aws_output],
            )

        with gr.Row():
            feedback_input = gr.Textbox(
                label=lang_store[language]["Evaluate the Prompt Effect"],
                placeholder=lang_store[language][
                    "Input your feedback manually or by model"
                ],
                lines=3,
                show_copy_button=True,
            )
            eval_model_dropdown = gr.Dropdown(
                label=lang_store[language]["Choose the Evaluation Model"],
                choices=model_ids,
                value="o1-mini",
            )
            evaluate_button = gr.Button(
                lang_store[language]["Auto-evaluate the Prompt Effect"]
            )
            evaluate_button.click(
                alignment.evaluate_response,
                inputs=[
                    openai_output,
                    aws_output,
                    eval_model_dropdown,
                    model_1_dropdown,
                    model_2_dropdown,
                ],
                outputs=[feedback_input],
            )

            revise_button = gr.Button(lang_store[language]["Iterate the Prompt"])
            revised_prompt_output = gr.Textbox(
                label=lang_store[language]["Revised Prompt"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )
            revise_button.click(
                alignment.generate_revised_prompt,
                inputs=[
                    feedback_input,
                    user_prompt_eval,
                    openai_output,
                    aws_output,
                    eval_model_dropdown,
                    model_1_dropdown,
                    model_2_dropdown,
                ],
                outputs=revised_prompt_output,
            )

    with gr.Tab(lang_store[language]["Prompt Calibration"]):
        default_code = """
def postprocess(llm_output):
    return llm_output
""".strip()
        with gr.Row():
            with gr.Column(scale=2):
                calibration_task = gr.Textbox(
                    label=lang_store[language]["Please input your task"], lines=3
                )
                calibration_prompt_original = gr.Textbox(
                    label=lang_store[language]["Please input your original prompt"],
                    lines=5,
                    placeholder=lang_store[language][
                        'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
                    ],
                )
            with gr.Column(scale=2):
                postprocess_code = gr.Textbox(
                    label=lang_store[language]["Please input your postprocess code"],
                    lines=3,
                    value=default_code,
                )
                dataset_file = gr.File(file_types=["csv"], type="binary")
        with gr.Row():
            with gr.Column(scale=2):
                calibration_task = gr.Radio(
                    ["classification"],
                    value="classification",
                    label=lang_store[language]["Task type"],
                )
            with gr.Column(scale=2):
                steps_num = gr.Slider(
                    1, 5, value=1, step=1, label=lang_store[language]["Epoch"]
                )
            calibration_optimization = gr.Button(
                lang_store[language]["Optimization based on prediction"]
            )
            calibration_prompt = gr.Textbox(
                label=lang_store[language]["Revised Prompt"],
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
            calibration_optimization.click(
                calibration.optimize,
                inputs=[
                    calibration_task,
                    calibration_prompt_original,
                    dataset_file,
                    postprocess_code,
                    steps_num,
                ],
                outputs=calibration_prompt,
            )

demo.launch()
