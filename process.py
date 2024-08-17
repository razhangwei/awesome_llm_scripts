import json
import click
import logging
from litellm import completion, encode


def load_system_prompt(prompt_file):
    with open(prompt_file, "r") as f:
        return f.read().strip()


@click.group()
def main():
    pass


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model",
    default="ollama/llama3.1:8b",
    help="Model name to use for summarization. Default: ollama/llama3.1:8b",
)
@click.option(
    "--num_ctx",
    default=2048,
    help="Context length for inference. Ollama only. Default: 2048",
)
@click.option(
    "--temperature", default=0.2, help="Temperature for summarization. Default: 0.2"
)
@click.option(
    "--prompt_file",
    default="assets/summarize_system_prompt.txt",
    help="File containing the system prompt. Default: assets/summarize_system_prompt.txt",
)
def summarize(filepath, model, num_ctx, temperature, prompt_file):
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


if __name__ == "__main__":
    main()
