from nanovllm import LLM, RequestConfig


def main():
    llm = LLM()

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    outputs = llm.generate(
        prompts,
        request_configs = [
            RequestConfig(max_tokens=32),
            RequestConfig(max_tokens=128),
        ]
    )

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("\n")
    llm.print_traces()

if __name__ == "__main__":
    main()
