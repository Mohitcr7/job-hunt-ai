# main.py

from config import validate_config, get_llm, JOB_PREFERENCES, MATCH_THRESHOLD

def check_setup():
    print("=== Job Hunt AI — Setup Check ===\n")

    ok = validate_config()
    if not ok:
        print("Fix the errors above before continuing.")
        return

    print(f"Match threshold : {MATCH_THRESHOLD}%")
    print(f"Target roles    : {JOB_PREFERENCES['roles']}")
    print(f"Target locations: {JOB_PREFERENCES['locations']}")

    # Quick live test — sends one real message to Gemini
    # to confirm your API key actually works.
    print("\nTesting LLM connection...")
    try:
        llm = get_llm(temperature=0.0)

        # .invoke() sends a message to the model and returns its reply.
        # This is the most basic LangChain operation you'll use everywhere.
        response = llm.invoke("Reply with exactly three words: setup is working")

        # response.content is where LangChain puts the model's text reply
        print(f"LLM replied: {response.content}")
        print("\nAll good — ready to build agents!")

    except Exception as e:
        # Exception handling: if anything goes wrong inside the try block,
        # Python jumps here instead of crashing the whole program.
        print(f"LLM test failed: {e}")
        print("Double-check your API key in .env")


if __name__ == "__main__":
    check_setup()