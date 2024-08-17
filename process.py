import json
import click
import logging
from litellm import completion

def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_system_prompt(prompt_file):
    with open(prompt_file, "r") as f:
        return f.read().strip()


@click.group()
@click.option(
    "-m",
    "--model",
    default="ollama/llama3.1:8b",
    help="Model name to use for processing. Default: ollama/llama3.1:8b",
)
@click.option(
    "--num_ctx",
    default=2048,
    help="Context length for inference. Ollama only. Default: 2048",
)
@click.option(
    "--temperature", default=0.2, help="Temperature for processing. Default: 0.2"
)
@click.option('--log-level', default='INFO', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
              help='Set the logging level')
@click.pass_context
def main(ctx, model, num_ctx, temperature, log_level):
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["num_ctx"] = num_ctx
    ctx.obj["temperature"] = temperature
    ctx.obj["log_level"] = log_level

    setup_logging(log_level)


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "--prompt_file",
    default="assets/summarize_system_prompt.txt",
    help="File containing the system prompt for summarization. Default: assets/summarize_system_prompt.txt",
)
@click.pass_context
def summarize(ctx, filepath, prompt_file):
    model = ctx.obj["model"]
    num_ctx = ctx.obj["num_ctx"]
    temperature = ctx.obj["temperature"]

    # Load the transcript
    with open(filepath, "r") as f:
        transcript = json.load(f)

    # Join the text from each segment
    text = "\n".join(x["text"] for x in transcript)

    # Load the system prompt
    system_prompt = load_system_prompt(prompt_file)

    # Set extra options for Ollama model
    extra_options = {"num_ctx": num_ctx} if model.startswith("ollama") else {}

    # Generate the summary
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Summarize this transcript into the format similar as the sample:: \n\n{text}",
            },
        ],
        temperature=temperature,
        api_base="http://localhost:11434" if model.startswith("ollama/") else None,
        stream=True,
        **extra_options,
    )

    # Stream the response
    summary = ""
    for chunk in response:
        text_chunk = chunk["choices"][0]["delta"]["content"]
        if text_chunk:
            print(text_chunk, end="")
            summary += text_chunk

    return summary


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "--prompt_file",
    default="assets/dialog_system_prompt.txt",
    help="File containing the system prompt for dialog conversion. Default: assets/dialog_system_prompt.txt",
)
@click.pass_context
def convert_to_dialog(ctx, filepath, prompt_file):    
    model = ctx.obj["model"]
    num_ctx = ctx.obj["num_ctx"]
    temperature = ctx.obj["temperature"]

    allowed_models = ["gemini", "gpt-4", "anthropic"]
    logging.info(model.split("/")[-1])
    if not any(model.split("/")[-1].startswith(x) for x in allowed_models):
        raise ValueError(f"Invalid model: {model}. Must be one of the classes: {', '.join(allowed_models)}")


    # Load the transcript
    with open(filepath, "r") as f:
        transcript = json.load(f)

    # Join the text from each segment
    text = "\n".join(x["text"] for x in transcript)

    # Load the system prompt
    system_prompt = load_system_prompt(prompt_file)

    # Set extra options for Ollama model
    extra_options = {"num_ctx": num_ctx} if model.startswith("ollama") else {}

    # Generate the dialog
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Convert this transcript into a dialog format. Try infering the speakers name: \n\n{text}",
            },
        ],
        temperature=temperature,
        api_base="http://localhost:11434" if model.startswith("ollama/") else None,
        stream=True,
        **extra_options,
    )

    # Stream the response
    dialog = ""
    for chunk in response:
        text_chunk = chunk["choices"][0]["delta"]["content"]
        if text_chunk:
            print(text_chunk, end="")
            dialog += text_chunk

    # save the dialog
    with open(filepath[:filepath.rindex(".")] + "_dialog.md", "w") as f:
        f.write(dialog)

    return


if __name__ == "__main__":
    main()
