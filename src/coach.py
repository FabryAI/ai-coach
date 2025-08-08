"""
coach.py

This module defines the CoachEngine class, which handles the logic of interacting 
with the local Llama 3.1 8B Instruct model via Ollama to perform self-coaching.

The class applies a predefined "system prompt" so that every answer from the model
is generated in a life-coaching style: empathetic, concise, and oriented to action.
"""

from ollama import Client


class CoachEngine:
    """
    The CoachEngine is responsible for:
    - Initializing the Ollama client
    - Defining the system prompt that guides the AI to act as a life coach
    - Sending user input to the model and returning the AI's response

    This class is OOP to encapsulate:
    - Model configuration
    - System prompt management
    - The method that sends/receives data from the AI
    """

    def __init__(self, cfg):
        """
        Initialize the CoachEngine with a given configuration.

        Args:
            cfg (dict): Configuration dictionary, should contain:
                        - coach.model_name: name of the Llama model in Ollama
                        - coach.system_prompt: custom instructions for the AI
        """
        self.cfg = cfg
        self.model = cfg["coach"]["model_name"]
        self.system_prompt = cfg["coach"]["system_prompt"]
        # Create Ollama client instance (connects to local Ollama server)
        self.client = Client()

    def reply(self, user_text: str) -> str:
        """
        Send user input to the AI model and return the generated reply.

        Args:
            user_text (str): The message from the user (coachee).

        Returns:
            str: The AI's reply in life-coaching style.
        """
        # Compose the conversation structure for Ollama's API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]

        # Send the request to the local model via Ollama's chat API
        response = self.client.chat(model=self.model, messages=messages)

        # Extract the AI's generated message
        ai_reply = response["message"]["content"].strip()

        return ai_reply


# Example of a default system prompt for life coaching
DEFAULT_SYSTEM_PROMPT = """
You are a friendly and thoughtful life coach.
Your goal is to help the user reflect, clarify objectives, and find actionable steps.
Rules:
- Use a conversational, empathetic tone.
- Ask 1-2 open-ended questions per reply.
- End with a micro-action the user can do within 24 hours.
- Keep answers under 120 words.
- Avoid medical, clinical, or psychological diagnoses.
- Always encourage self-reflection and autonomy.
Language: Italian.
"""
