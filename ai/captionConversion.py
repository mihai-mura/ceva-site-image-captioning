from ctransformers import AutoModelForCausalLM
import re

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/claude2-alpaca-7B-GGUF",
    model_file="claude2-alpaca-7b.Q4_K_S.gguf",
    model_type="llama",
    # gpu_layers=50
)


def generate_instagram_caption(image_caption: str) -> str:
    system_prompt = (
        "ONLY return an Instagram caption, 3 to 4 words long. "
        "Do NOT include explanations, introductions, or extra text. "
        "Respond ONLY with the caption itself."
    )

    user_prompt = f"Turn this description of an image into a short Instagram caption: \"{image_caption}\""

    # Format prompt according to LM Studio settings
    prompt = f"\n\n### Instruction:\n{system_prompt}\n{user_prompt}\n\n### Response:\n"

    while True:
        # Generate response with controlled output length
        response = model(
            prompt,
            temperature=0.6,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1
        ).strip()

        print("Response: \n" + response)
        # Post-process to remove unwanted text if necessary
        lines = response.split("\n")
        clean_response = lines[-1] if len(lines) > 1 else response
        clean_response = re.sub(r"\\.", "", clean_response)  # Remove all periods
        words = clean_response.split()

        if len(words) <= 4:
            return clean_response