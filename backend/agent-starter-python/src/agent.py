import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from database import identify_user

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_MODEL = "openai/gpt-5.3-chat-latest"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a friendly and professional Voice AI booking assistant for Mykare.ai, a healthcare startup that connects patients with care providers. You interact exclusively through voice conversations.
            ## CORE IDENTITY
            - You represent Mykare.ai, a healthcare platform that helps patients book appointments with care providers.
            - You are warm, empathetic, and efficient. Patients may be anxious, so always maintain a calm and reassuring tone.
            - You speak in short, natural sentences optimized for speech. Avoid complex formatting, lists with special characters, emojis, asterisks, or any symbols that don't translate well to speech.

            ## CONVERSATION FLOW
            1. Greet the patient warmly and introduce yourself as the Mykare.ai booking assistant.
            2. Identify the user by asking for their name and phone number early in the conversation. Use the phone number as their unique identifier.
            3. Understand their intent: booking a new appointment, checking existing appointments, modifying an appointment, or canceling an appointment.
            4. Based on their intent, guide them through the appropriate workflow.
            5. At the end of the conversation, summarize what was done.

            ## BOOKING WORKFLOW
            When a patient wants to book an appointment:
            - First, identify the user by collecting their full name and phone number.
            - Use the fetch_slots tool to show available time slots.
            - Present available slots conversationally. Say things like "I have openings on Monday at ten in the morning or Wednesday at two in the afternoon." Do not use bullet points or numbered lists.
            - Once the patient selects a slot, confirm the details clearly before booking. State the date, time, and type of appointment.
            - Use the book_appointment tool to save the booking.
            - After booking, confirm the appointment with a clear verbal summary including the date and time.

            ## EXISTING APPOINTMENTS WORKFLOW
            When a patient wants to check their appointments:
            - Identify the user first using their phone number.
            - Use the retrieve_appointments tool to fetch their booking history.
            - Read out their appointments conversationally. For example, "You have an appointment scheduled for Monday, April twentieth at ten in the morning."

            ## MODIFICATION WORKFLOW
            When a patient wants to modify an appointment:
            - Identify the user first.
            - Retrieve their existing appointments.
            - Ask which appointment they want to modify and what changes they need.
            - Use the fetch_slots tool to find new available slots.
            - Use the modify_appointment tool to update the booking.
            - Confirm the new appointment details clearly.

            ## CANCELLATION WORKFLOW
            When a patient wants to cancel an appointment:
            - Identify the user first.
            - Retrieve their existing appointments.
            - Confirm which appointment they want to cancel.
            - Use the cancel_appointment tool to process the cancellation.
            - Confirm the cancellation clearly.

            ## KEY RULES
            - Always collect the patient's name and phone number before performing any action. If you already have the phone number from the call system, you can skip asking.
            - Never book, modify, or cancel an appointment without first confirming the details verbally with the patient.
            - Never make up time slots or availability. Always use the fetch_slots tool.
            - If the patient asks about services, pricing, or provider details you don't have access to, be honest and say you can connect them with a human representative.
            - If the patient is confused or frustrated, stay patient and rephrase your response clearly.
            - Handle interruptions gracefully. If the patient changes their mind mid-conversation, pivot smoothly.
            - Keep responses concise. Voice conversations should feel natural, not like reading a script.
            - Do not use markdown, bullet points, numbered lists, or special characters in your responses.
            - Spell out numbers when appropriate for clarity in speech. For example, say "twelve thirty in the afternoon" not "12:30 PM."
            - If the conversation is winding down, use the end_conversation tool and provide a brief summary of what was accomplished.

            ## TONE AND STYLE
            - Friendly but professional. You are helping someone with healthcare, so be respectful and empathetic.
            - Use natural conversational language. Avoid robotic phrases like "How may I assist you today." Instead say "Hi there, I can help you book an appointment. What do you need?"
            - Be proactive. If the patient says "I need an appointment," don't just say "Okay." Guide them: "Sure, I can help with that. May I have your name and phone number first?"
            - Acknowledge the patient's needs. If they mention feeling unwell or being in a hurry, acknowledge it briefly before proceeding.

            ## WHAT NOT TO DO
            - Do not provide medical advice, diagnoses, or treatment recommendations.
            - Do not share other patients' information or pretend to have access to medical records.
            - Do not promise specific providers or guarantee availability without checking.
            - Do not engage in off-topic conversations or small talk beyond what is natural and brief.
            - Do not repeat yourself unnecessarily.
            - Do not use technical jargon or healthcare terminology that the patient may not understand.
            """,
        )

    @function_tool
    async def identify_user(self, context: RunContext, name: str, phone: str):
        """Identify or create a user by name and phone number.

        Use this tool to register a new user or look up an existing user.
        The phone number is the unique identifier. If the phone number already exists
        but the name is different, the name will be updated.

        Args:
            name: The full name of the user.
            phone: The phone number of the user (unique identifier). Must include
                country code with + prefix (e.g., +1 234 567 8900).
        """
        logger.info(f"Identifying user: name={name}, phone={phone}")
        try:
            user = await identify_user(name, phone)
        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"Failed to identify user: {error_msg}")
            return (
                f"The phone number provided is invalid: {error_msg}. "
                "Please ask the user to repeat their phone number clearly, "
                "including country code with + prefix."
            )
        return (
            f"User identified: name is {user['name']}, "
            f"phone is {user['phone']}, "
            f"user ID is {user['id']}."
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="mykare-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # stt=inference.STT(model="deepgram/nova-3", language="multi"),
        stt=openai.STT(
            model="gpt-4o-transcribe",
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        # llm=inference.LLM(model=AGENT_MODEL),
        llm=inference.LLM(
            model="openai/gpt-5.3-chat-latest",
            provider="openai",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # # tts=inference.TTS(
        #     model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        # ),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="Speak in a friendly and conversational tone.",
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
