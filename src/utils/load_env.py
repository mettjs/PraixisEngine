from dotenv import find_dotenv, load_dotenv

def load_env():
    env_file = find_dotenv()

    if not env_file:
        # No .env found — env vars must be injected directly (Docker, k8s, CI).
        # Don't crash; the app will use its defaults and whatever is in the environment.
        return

    if not load_dotenv(env_file):
        raise RuntimeError(f"Found .env at '{env_file}' but failed to load it.")