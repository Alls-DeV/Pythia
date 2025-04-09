import os
from time import sleep

from openai import OpenAI, RateLimitError

from common import PNUMBER1


class DeepSeekPlayer:
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def get_LLM_action(
        self,
        system_prompt,
        user_prompt,
        model="deepseek-chat",
        temperature=0.3,
        stop=[],
        max_tokens=200,
        log_metadata=None,
    ) -> str:
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=temperature,
                stop=stop,
                max_tokens=max_tokens,
            )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)
            print("rate limit error")
            return self.get_LLM_action(
                system_prompt,
                user_prompt,
                model,
                temperature,
                stop,
                max_tokens,
                log_metadata,
            )
        outputs = response.choices[0].message.content
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        # Logging
        if log_metadata is not None:
            with open(f"./alessio/log_{PNUMBER1}", "a") as f:
                f.write(log_metadata + "\n")
                f.write("System Prompt:\n" + system_prompt + "\n\n")
                f.write("User Prompt:\n" + user_prompt + "\n\n")
                f.write("Output:\n" + outputs + "\n\n")
                f.write("--" * 50 + "\n")
        return outputs
