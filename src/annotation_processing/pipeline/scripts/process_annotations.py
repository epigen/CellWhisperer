from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path

import os
from pathlib import Path
import subprocess
from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManager

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import openai

print(openai.__version__)

batch_chat = None
stream_chat = None


def load_openai_api(api_key: Optional[str] = None, model="gpt-4"):
    """Make sure OpenAI API key is available. If not, then load it

    Args:
        api_key: The OpenAI API key to use. If None, then try to load it from the environment variable OPENAI_API_KEY or from the password store.
        model: The model to use. Defaults to "gpt-4". alternatively use gpt-3.5-turbo for faster and cheaper exploration
    """
    global batch_chat, stream_chat
    try:
        os.environ["OPENAI_API_KEY"]
    except KeyError:
        if api_key is None:

            def get_password_from_pass_store(key: str) -> str:
                # Run the pass command to get the password for the specified key
                password = subprocess.run(["pass", key], capture_output=True, text=True)

                # Check if the command execution was successful
                if password.returncode != 0:
                    raise RuntimeError(f"Failed to get password for key: {key}")

                # Remove trailing newline character and return the password
                return password.stdout.rstrip("\n")

            key = "openai.com/meduni_matthias_api_key"  # Moritz' setup
            os.environ["OPENAI_API_KEY"] = get_password_from_pass_store(key)
            # print(f"Password for {key}: {openai_api_key}")
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    batch_chat = ChatOpenAI(
        temperature=0,
        model_name=model,
        request_timeout=600,
        max_retries=1,
    )
    stream_chat = ChatOpenAI(
        streaming=True,
        temperature=0,
        model_name=model,
        request_timeout=600,
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
        # verbose=True,
    )
    return batch_chat, stream_chat


batch_chat, stream_chat = load_openai_api()

request = [
    SystemMessage(content=Path(snakemake.input.system_message).read_text()),
    HumanMessage(content=Path(snakemake.input.human_message).read_text()),
]
result = stream_chat(request)

out_path = Path(snakemake.output[0])
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(result.content)
