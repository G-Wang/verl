import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def format_as_deepseek_prompt(question_raw: str) -> str:
    """Create a prompt similar to the DeepSeek format, as described in Table 1 of the DeepSeek paper."""
    full_prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {question_raw}. Assistant:"""
    return full_prompt

def load_model(checkpoint_path: str):
    """Load a fine-tuned model, tokenizer, and generation config from a given checkpoint path."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model onto GPU
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map={"": 0},  # Force model to a single GPU
        trust_remote_code=True
    ).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load generation configuration, fallback if missing
    try:
        gen_config = GenerationConfig.from_pretrained(checkpoint_path)
        print("Loaded generation config from checkpoint.")
    except:
        gen_config = GenerationConfig()  # Default config
        print("No generation_config.json found. Using default settings.")

    return model, tokenizer, gen_config, device

def generate_response(model, tokenizer, gen_config, device, question: str):
    """Generates a response to a given question using the loaded model."""
    # Format question as a DeepSeek-style prompt
    prompt = format_as_deepseek_prompt(question)

    # Tokenize and move input to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=gen_config, max_length=2048)

    # Decode and return generated text
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--question", type=str, default=None, help="Question to ask the model (optional)")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, gen_config, device = load_model(args.checkpoint)

    # Default question if none provided
    default_question = 'If $x = 2$ and $y = 5$, then what is the value of $\frac{x^4+2y^2}{6}$ ?'
    question = args.question if args.question else default_question

    # Generate output
    generated_text = generate_response(model, tokenizer, gen_config, device, question)

    print("\nGenerated Output:\n", generated_text)
