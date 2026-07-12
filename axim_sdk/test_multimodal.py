from axim_sdk.multimodal import AximMultimodalEngine

def main():
    print("Initializing Axim Multimodal Engine natively...")
    
    # Testing Gemma
    try:
        # Changed to gemma4 because that's the tag available in local ollama
        gemma_engine = AximMultimodalEngine(model_name="gemma4")
        print("\nTesting basic prompt with local Gemma...")
        gemma_response = gemma_engine.ask("Say 'Axim Custom Engine integration successful!' and nothing else.")
        print("Gemma Response:\n" + gemma_response)
    except Exception as e:
        print("Gemma failed:", e)

    # Testing Mistral
    try:
        mistral_engine = AximMultimodalEngine(model_name="mistral")
        print("\nTesting basic prompt with local Mistral...")
        mistral_response = mistral_engine.ask("Say 'Axim Custom Engine integration successful!' and nothing else.")
        print("Mistral Response:\n" + mistral_response)
    except Exception as e:
        print("Mistral failed:", e)

    # Testing native MLX VLM
    try:
        # Use a lightweight 4-bit model to save memory
        mlx_model_name = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        mlx_engine = AximMultimodalEngine(model_name=mlx_model_name)
        print(f"\nTesting basic prompt with native Apple Silicon MLX ({mlx_model_name})...")
        mlx_response = mlx_engine.ask("Say 'Axim MLX native integration successful!' and nothing else.")
        print("MLX Response:\n" + mlx_response)
    except Exception as e:
        print("MLX VLM failed:", e)

if __name__ == "__main__":
    main()
