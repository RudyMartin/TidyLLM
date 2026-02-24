Unitree’s newer robots ship with their own integrated large language model stack plus hooks so you can also run external LLM/NLP (e.g., OpenAI, local models) via the SDK and onboard Jetson.
​

Built‑in language model
Unitree has a proprietary UnifoLM (Unitree Robot Unified Large Model) that powers dialogue, command understanding, and robot control on platforms like the G1, working similarly to ChatGPT but with direct actuation capabilities.
​

The G1’s NLP stack is tightly integrated with its voice pipeline (ASR + TTS) and control layer so spoken natural‑language commands can be mapped to motion or task routines without an external cloud call.
​

Hardware for running models
Robots like the G1 use an onboard Jetson Orin NX as the main compute module, which runs control policies and can host LLM clients or smaller on‑device models.
​

Go2 EDU/Pro variants are designed to work with external or add‑on compute (e.g., Jetson Orin NX/Nano) for higher‑end perception, policy learning, or LLM integration beyond the stock “big model GPT” features marketed for Go2.
​

Voice, ASR, and TTS layer
Go2 has a built‑in voice assistant (“BenBen”) that provides speech recognition and TTS; developers can intercept this pipeline and forward text to external LLM APIs, then send the response back to be spoken.
​

The Pro version includes voice functions by default, leveraging its higher‑performance CPU and sensors to support wake words, command grammar, and conversational responses.
​

External LLM / NLP integrations
A common pattern in Go2 EDU projects is: on‑robot or nearby computer handles mic input → ASR → send text to OpenAI API (ChatGPT) or another LLM → return text → TTS on the robot → map specific intents to SDK motion commands.
​

Integrators around the G1 often connect the robot through the SDK to external LLMs or enterprise backends, exposing APIs that let the LLM trigger high‑level skills while low‑level control stays in a traditional policy or controller.
​

For your own stack
For research‑style setups you can:

Run control and perception locally (PyTorch/ROS/Isaac),

Treat the LLM as a high‑level planner (cloud or local),

Use the Unitree SDK only as a thin actuation/telemetry layer between the LLM/planner and the robot.
​

If you describe what you want (fully on‑device, local LLM, or cloud‑backed planner), a concrete architecture for Go2 or G1 can be sketched out.
