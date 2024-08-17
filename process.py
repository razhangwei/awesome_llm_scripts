from litellm import completion, encode
import json 
import click
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@click.group()
def main():
    pass


@main.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('-m', '--model', default='ollama/llama3.1:8b', help='Model name to use for summarization. Default is ollama/llama3.1:8b')
@click.option('--num_ctx', default=2048, help='Context length for inference. Ollama only. Default is 2048')
@click.option('--temperature', default=0.2, help='Temperature for summarization. Default is 0.2')
def summarize(filepath, model, num_ctx, temperature):
    """
    Summarize the transcript in the given file.

    Args:
        filepath (str): Path to the JSON file containing the transcript.
        model (str): Model name to use for summarization. Default is 'ollama/llama3.1:8b'.
        num_ctx (int): Context length for inference. Ollama only. Default is 2048.
        temperature (float): Temperature for summarization. Default is 0.2.

    Returns:
        str: The generated summary.
    """
    # Load the transcript from the JSON file
    with open(filepath, "r") as f:
        transcript = json.load(f)

    # Join the text from each segment in the transcript into a single string
    text = "\n".join(x['text'] for x in transcript)
    logging.debug(f"Input text: \n {text}")

    # Check if the input text length is greater than the number of context tokens
    token_count = len(encode(model=model, text=text))
    logging.info(f"Input text token count: {token_count}. num_ctx = {num_ctx}.")

    # Set extra options for Ollama model
    extra_options = {}
    if model.startswith("ollama"): 
        extra_options['num_ctx'] = num_ctx

    # Generate the summary using the specified model
    response = completion(
        model=model, 
        messages=[
            # Define the system message
            {"role": "system", "content": """
The preferred style: 
- The summary must mimic the structure, format, and tone of the following sample. 
- Sometimes presenting the answers in bullet point format for quick and easy reading and added variety of the style. 

# Summary

Brett Adcock, founder and CEO of Figure AI, discusses his company's mission to develop humanoid robots for general-purpose use. He explains the rationale behind choosing a humanoid form, the technological advancements enabling this development, and the challenges faced in creating such complex systems. Adcock also touches on the company's vertical integration, product development process, and future goals for deploying robots in industrial and home settings

# Topics Breakdown

## 1. Brett Adcock's Background and Path to Humanoid Robots

**Q: Can you talk about your journey from farming to software to vertical take-off and landing to humanoid robots?**

A: I grew up on a farm in Illinois and got into coding at an early age. I've spent about 20 years building companies - over 10 in software and under 10 in hardware. I started a software company, sold it, then started Archer Aviation for electric vertical take-off and landing aircraft. About 21 months ago, I started Figure.

**Key Quote:** 
> "The thesis here is that if you assume the technologies are possible, to build a humanoid robot... It's going to be the biggest business in the world, by probably order of magnitude."

## 2. Why Humanoid Robots?

**Q: Why does being humanoid matter? Why not other forms of robots?**

A: The whole world was built for humans and around humans. If we want to automate work, we want to build a general interface to that world. The human form is that interface - you can do everything a human can. Building one hardware system that can do everything is more efficient than creating specialized robots for every task.

**Key Quote:** 
> "We can either rebuild like thousands or millions of special types of robots that do special use cases all over the world, or we could build a humanoid robot."

## 3. Technological Advancements Enabling Humanoid Robots

**Q: Why is now the right time for humanoid robots?**

A: Several factors make it possible now:
1. Improved power systems (batteries and motors)
2. Advanced locomotion controllers
3. AI systems and computation power
4. Speech interfaces

**Key Quote:** 
> "I don't think this was really possible 5 years ago."
            """ },
            # Define the user message
            {"role": "user", "content": f"Summarize this transcript into the format similar as the sample:: \n\n{text}" }
        ],
        temperature=temperature,
        api_base="http://localhost:11434" if model.startswith("ollama/") else None,
        stream=True, 
        **extra_options, 
    )

    # Generate the summary by streaming the response
    summary = ""
    for chunk in response:
        text_chunk = chunk['choices'][0]['delta']['content']
        if text_chunk: 
            print(text_chunk, end="")
            summary += text_chunk 

    return summary

if __name__ == '__main__':
    main()
