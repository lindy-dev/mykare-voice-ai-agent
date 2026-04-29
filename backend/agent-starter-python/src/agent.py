import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

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
    get_job_context,
    metrics,
    MetricsCollectedEvent,
    room_io,
    ChatContext,
)
from livekit.plugins import ai_coustics, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import tavus
import os
from database import (
    book_slot,
    find_slot_by_date_time,
    get_available_slots,
    get_user_appointments,
    identify_user,
)
from database import (
    cancel_appointment as cancel_appointment_db,
)
from database import (
    modify_appointment as modify_appointment_db,
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")
tavus_api_key = os.getenv("TAVUS_API_KEY")
# logger.debug(f"Loaded Tavus API key: {'set' if tavus_api_key else 'not set'}")
AGENT_MODEL = "openai/gpt-5.3-chat-latest"

_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _get_ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _format_time(time_str: str) -> str:
    parts = time_str.split(":")
    h = int(parts[0])
    m = int(parts[1])
    period = (
        "in the morning"
        if h < 12
        else "in the afternoon"
        if h < 18
        else "in the evening"
    )
    if h == 0:
        display_h = 12
    elif h > 12:
        display_h = h - 12
    else:
        display_h = h

    if m == 0:
        return f"{display_h} {period}"
    return f"{display_h} thirty {period}"


def _format_date(date_str: str) -> str:
    parts = date_str.split("-")
    day = int(parts[2])
    month = int(parts[1])
    day_name = _DAY_NAMES[datetime.strptime(date_str, "%Y-%m-%d").weekday()]
    ordinal = _get_ordinal(day)
    month_name = _MONTH_NAMES[month - 1]
    return f"{day_name}, {month_name} {ordinal}"


async def _send_tool_status(
    room,
    tool: str,
    status: str,
    message: str,
) -> None:
    """Send a tool-call status update to the first remote (frontend) participant."""
    
    frontend = next(iter(room.remote_participants.values()), None)
    if frontend is None:
        return
    try:
        await room.local_participant.perform_rpc(
            destination_identity=frontend.identity,
            method="tool_status",
            payload=json.dumps({"tool": tool, "status": status, "message": message}),
        )
    except Exception as e:
        logger.warning(f"tool_status RPC failed ({tool}/{status}): {e}")


@dataclass
class MySessionInfo:
    user_name: str | None = None
    user_phone: str | None = None
    user_id: str | None = None


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
            stt=openai.STT(
                model="gpt-4o-transcribe",
            ),
            llm=inference.LLM(
                model="openai/gpt-5.3-chat-latest",
                provider="openai",
                extra_kwargs={"reasoning_effort": "low"},
            ),
            tts=openai.TTS(
                model="gpt-4o-mini-tts",
                voice="ash",
                instructions="Speak in a friendly and conversational tone.",
            ),
        )

    @function_tool
    async def identify_user(
        self, context: RunContext[MySessionInfo], name: str, phone: str
    ):
        """Identify or create a user by name and phone number.

        Use this tool to register a new user or look up an existing user.
        The phone number is the unique identifier. If the phone number already exists
        but the name is different, the name will be updated.

        Args:
            name: The full name of the user.
            phone: The phone number of the user (unique identifier). Must include
                country code with + prefix (e.g., +1 234 567 8900).
        """
        userdata = context.userdata
        logger.info(f"Identifying user: name={name}, phone={phone}")
        room = get_job_context().room
        await _send_tool_status(room, "identify_user", "started", "Looking up your account…")
        try:
            user = await identify_user(name, phone)
        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"Failed to identify user: {error_msg}")
            await _send_tool_status(room, "identify_user", "error", "Unable to look up your account.")
            return (
                f"The phone number provided is invalid: {error_msg}. "
                "Please ask the user to repeat their phone number clearly, "
                "including country code with + prefix."
            )
        # Store user data under the database user_id as the key,
        # so other tools can look it up with userdata.read_object(user_id)
        # userdata.data_objects[user["id"]] = {
        #     "id": user["id"],
        #     "type": "user",
        #     "created_at": datetime.now(timezone.utc).isoformat(),
        #     "data": user,
        # }
        context.userdata.user_id = user['id']
        context.userdata.user_name = user['phone']
        context.userdata.user_phone = user['phone']
        logger.info(
            f"Added to user context: user_id={user['id']}, "
            f"user_name={user['name']}, user_phone={user['phone']}"
        )
        await _send_tool_status(room, "identify_user", "completed", "Account found.")
        # current_user_data = {"name":user['name'], "phone": user['phone'], "user_id":user['id']}
        # current_user_data_id = userdata.create_object("current_user_data", current_user_data)
        return (
            f"User identified: name is {user['phone']}, "
            f"phone is {user['phone']}, "
            f"user ID is {user['id']}."
            # f"current user data id is {current_user_data_id}."
        )

    @function_tool
    async def fetch_slots(
        self,
        context: RunContext[MySessionInfo],
        date_from: str = "",
        date_to: str = "",
    ):
        """Fetch available appointment time slots.

        Use this tool to find available time slots for booking an appointment.
        The agent should present the slots conversationally to the patient.

        Args:
            date_from: Optional start date filter in ISO format (YYYY-MM-DD).
                If not provided, returns slots starting from tomorrow.
            date_to: Optional end date filter in ISO format (YYYY-MM-DD).
                If not provided, returns slots for the next 5 business days.
        """
        logger.info(
            f"Fetching slots: date_from={date_from or 'none'}, date_to={date_to or 'none'}"
        )
        room = get_job_context().room
        await _send_tool_status(room, "fetch_slots", "started", "Checking available appointment slots…")

        slots = await get_available_slots(
            date_from=date_from if date_from else None,
            date_to=date_to if date_to else None,
        )

        if not slots:
            await _send_tool_status(room, "fetch_slots", "completed", "No slots found for the requested dates.")
            return (
                "I don't have any available slots for the requested dates. "
                "Would you like me to check a different date range?"
            )

        grouped: dict[str, list[str]] = {}
        for slot in slots:
            date_key = slot["date"]
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(slot["start_time"])

        parts = []
        for date_key, times in grouped.items():
            date_label = _format_date(date_key)
            time_labels = [_format_time(t) for t in times]
            if len(time_labels) == 1:
                times_str = time_labels[0]
            elif len(time_labels) == 2:
                times_str = f"{time_labels[0]} and {time_labels[1]}"
            else:
                times_str = f"{', '.join(time_labels[:-1])} and {time_labels[-1]}"
            parts.append(f"I have openings on {date_label} at {times_str}")

        logger.debug(f"Available slots (parts): {' '.join(parts) + '.'}")
        await _send_tool_status(room, "fetch_slots", "completed", "Available slots retrieved.")
        return ". ".join(parts) + "."

    @function_tool
    async def book_appointment(
        self,
        context: RunContext[MySessionInfo],
        date: str,
        start_time: str,
        user_id: str,
    ):
        """Book an appointment for the given user and slot.

        Use this tool to confirm and save a booking after the patient has
        selected a time slot. The slot_id and user_id must have been obtained
        from previous tool calls.

        Args:
            date: date filter in ISO format (YYYY-MM-DD).
            start_time: time of the appointment
            user_id: The ID of the user to book for.
        """
        user_id = context.userdata.user_id
        logger.debug(f"Extracted user id from user context: {user_id}")
        logger.info(
            f"Booking appointment: user_id={user_id}, date={date}, start_time={start_time}"
        )
        room = get_job_context().room
        await _send_tool_status(room, "book_appointment", "started", "Booking your appointment…")
        slot = await find_slot_by_date_time(date, start_time)
        if slot is None:
            await _send_tool_status(room, "book_appointment", "error", "No available slot found for the requested time.")
            return (
                f"No available slot found for {date} at {start_time}. "
                "Please fetch available slots again and ask the user to choose a different time."
            )
        try:
            result = await book_slot(slot["id"], user_id)
        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"Failed to book appointment: {error_msg}")
            await _send_tool_status(room, "book_appointment", "error", "Unable to book the appointment.")
            return (
                f"Unable to book the appointment: {error_msg}. "
                "Please fetch available slots again and ask the user to choose a different time."
            )

        date_label = _format_date(result["date"])
        time_label = _format_time(result["start_time"])

        await _send_tool_status(room, "book_appointment", "completed", "Appointment booked successfully.")
        return (
            f"Appointment booked successfully. "
            f"The appointment is confirmed for {date_label} at {time_label}. "
            f"The appointment ID is {result['appointment_id']}."
        )

    @function_tool
    async def retrieve_appointments(
        self, context: RunContext[MySessionInfo], user_id: str
    ):
        """Retrieve existing appointments for a user.

        Use this tool to look up a user's appointment history.
        The agent should present the results conversationally to the patient.

        Args:
            user_id: The ID of the user to look up appointments for.
        """
        room = get_job_context().room
        logger.info(f"Retrieving appointments for user: user_id={user_id}")
        await _send_tool_status(room, "retrieve_appointments", "started", "Retrieving your appointments…")
        appointments = await get_user_appointments(user_id)

        if not appointments:
            await _send_tool_status(room, "retrieve_appointments", "completed", "No appointments found.")
            return (
                "I don't see any appointments for this user. "
                "Would you like to book one?"
            )

        parts = []
        for appt in appointments:
            date_label = _format_date(appt["date"])
            time_label = _format_time(appt["start_time"])
            parts.append(
                f"You have an appointment on {date_label} at {time_label}. "
                f"The appointment ID is {appt['id']}."
            )

        await _send_tool_status(room, "retrieve_appointments", "completed", "Appointments retrieved.")
        return " ".join(parts)

    @function_tool
    async def cancel_appointment_by_appointment_id(
        self, context: RunContext[MySessionInfo], appointment_id: str, user_id: str
    ):
        """Cancel an existing appointment.

        Use this tool to cancel an appointment after confirming with the patient.
        The slot will be freed and made available for others.

        Args:
            appointment_id: The ID of the appointment to cancel.
            user_id: The ID of the user who owns the appointment.
        """
        room = get_job_context().room
        logger.info(
            f"Cancelling appointment: appointment_id={appointment_id}, user_id={user_id}"
        )
        await _send_tool_status(room, "cancel_appointment_by_appointment_id", "started", "Cancelling your appointment…")
        try:
            result = await cancel_appointment_db(appointment_id, user_id)
        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"Failed to cancel appointment: {error_msg}")
            await _send_tool_status(room, "cancel_appointment_by_appointment_id", "error", "Unable to cancel the appointment.")
            return (
                f"Unable to cancel the appointment: {error_msg}. "
                "Please verify the appointment ID and try again."
            )

        date_label = _format_date(result["date"])
        time_label = _format_time(result["start_time"])

        await _send_tool_status(room, "cancel_appointment_by_appointment_id", "completed", "Appointment cancelled successfully.")
        return (
            f"Appointment cancelled successfully. "
            f"The appointment on {date_label} at {time_label} has been cancelled. "
            f"The appointment ID was {result['id']}."
        )

    @function_tool
    async def modify_appointment(
        self,
        context: RunContext[MySessionInfo],
        user_id: str,
        to_be_cancelled_date: str,
        to_be_cancelled_time: str,
        to_be_booked_date: str,
        to_be_booked_time: str,
    ):
        """Modify an existing appointment by cancelling one and booking a new slot.

        Use this tool to change an appointment to a different time slot after
        confirming with the patient. The old appointment will be cancelled,
        the old slot freed, and the new slot booked.

        Args:
            user_id: The ID of the user who owns the appointment.
            to_be_cancelled_date: The date of the appointment to cancel (YYYY-MM-DD).
            to_be_cancelled_time: The start time of the appointment to cancel (HH:MM).
            to_be_booked_date: The date of the new appointment (YYYY-MM-DD).
            to_be_booked_time: The start time of the new appointment (HH:MM).
        """
        room = get_job_context().room
        logger.info(
            f"Modifying appointment: user_id={user_id}, "
            f"cancel {to_be_cancelled_date} {to_be_cancelled_time}, "
            f"book {to_be_booked_date} {to_be_booked_time}"
        )
        await _send_tool_status(room, "modify_appointment", "started", "Updating your appointment…")
        try:
            result = await modify_appointment_db(
                user_id,
                to_be_cancelled_date,
                to_be_cancelled_time,
                to_be_booked_date,
                to_be_booked_time,
            )
        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"Failed to modify appointment: {error_msg}")
            await _send_tool_status(room, "modify_appointment", "error", "Unable to update the appointment.")
            return (
                f"Unable to modify the appointment: {error_msg}. "
                "Please fetch available slots again and ask the user to choose a different time."
            )

        date_label = _format_date(result["date"])
        time_label = _format_time(result["start_time"])

        await _send_tool_status(room, "modify_appointment", "completed", "Appointment updated successfully.")
        return (
            f"Appointment modified successfully. "
            f"The new appointment is confirmed for {date_label} at {time_label}. "
            f"The new appointment ID is {result['appointment_id']}."
        )
    
    @function_tool()
    async def end_conversation(self, context: RunContext[MySessionInfo]) -> str:
        """Generate a concise summary of the conversation and send it to the frontend."""
        room = get_job_context().room
        await _send_tool_status(room, "end_conversation", "started", "Generating conversation summary…")
        context.disallow_interruptions()
        chat_ctx = self.chat_ctx
        summary_ctx = ChatContext()
        summary_ctx.add_message(
            role="system",
            content="""
            Generate a concise summary of the entire conversation so far. Include the following in your summary: 
                - Summary of conversation
                - List of appointments made/modified/cancelled
                - User preferences (if any)
                - Timestamp
                Respond with only the summary text
            """,
        )

        n_summarized = 0
        for item in chat_ctx.items:
            if item.type != "message":
                continue
            if item.role not in ("user", "assistant"):
                continue
            if item.extra.get("is_summary") is True:  # avoid summarizing previous summaries
                continue
            text = (item.text_content or "").strip()
            if text:
                summary_ctx.add_message(role="user", content=f"{item.role}: {text}")
                n_summarized += 1

        if n_summarized == 0:
            return None
        summarizer = self.llm 
        response = await summarizer.chat(chat_ctx=summary_ctx).collect()
        summary= response.text.strip() if response.text else None
        logger.debug(f"Generated conversation summary: {summary}")

        await _send_tool_status(room, "end_conversation", "completed", summary)

        # 2. Return summary so the LLM presents it naturally
        return summary
    
    async def on_enter(self) -> None:
        
        await self.session.generate_reply(
            instructions=f"Greet user and ask the user for their full name and phone number to begin the chat"
        )
    


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
    userdata = MySessionInfo()
    session = AgentSession[MySessionInfo](
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
        preemptive_generation=True,
        turn_detection=MultilingualModel(),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)
    # "default_replica_id": "r72f7f7f7c8b"
    # {"persona_id":"p885c08bf44c","persona_name":"Mykare Persona","created_at":"2026-04-29T16:45:39.988597Z"}
    avatar = tavus.AvatarSession(
        replica_id="r72f7f7f7c8b",  # ID of the Tavus replica to use
        persona_id="p885c08bf44c",  # ID of the Tavus persona to use (see preceding section for configuration details)
        api_key=tavus_api_key,  # Your Tavus API key (set in .env.local and loaded with load_dotenv
    )
    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)
    # Start the session, which initializes the voice pipeline and warms up the models
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

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
