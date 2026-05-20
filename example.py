from nanovllm import LLM


def main():
    llm = LLM()

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    outputs = llm.generate(prompts)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
