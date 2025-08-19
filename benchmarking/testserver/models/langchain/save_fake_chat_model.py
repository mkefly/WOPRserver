import random
import shutil
from pathlib import Path
from typing import Any

import mlflow
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# ---- Agent Smith full dialogue bank ----
AGENT_SMITH_LINES = [
    "10 seconds left.",
    "A disgusting display awaits us.",
    "A machine and raw chaos can be such a potent combination.",
    "ACME can't save you now.",
    "Ah. I've been waiting for you.",
    "All that strength. So little intelligence.",
    "All these pathetic faces in their little boxes.",
    "Am I supposed to dance for you?",
    "And here I hoped you would move the needle. Disappointing.",
    "And so Destiny leads me here.",
    "And so the age of the machine begins.",
    "And the merry melodies grind to an ignominious halt.",
    "And this is all they send? Pathetic.",
    "And this is the best humanity can manage.",
    "And what pathetic foe am I fighting today?",
    "And who would you have me crush?",
    "Another battery, burning the minutes til its depletion.",
    "Another pathetic flying man.",
    "Brainiac was right about you.",
    "Bugs in my system don't last long.",
    "Bugs. Bunny. We meet again.",
    "Call the shots while you can, human.",
    "Choose your character.",
    "Come on now. Where's the challenge?",
    "Control your anger now.",
    "Die, rabbit.",
    "Do I detect your palms sweating? Pulse quickening?",
    "Do I detect... bugs in the system?",
    "Do you celebrate these Holidays? If so... I hope they're adequate.",
    "Do you hear the wheels of fate beginning to turn?",
    "Don't be scared. Jason will end it quickly.",
    "Don't give up yet.",
    "Don't waste my time.",
    "Early, for a human.",
    "Ever feel like you're in a simulation..?",
    "Far greater than you have tried.",
    "First time destroying a Kryptonian.",
    "Flesh and blood, crushed by superior machines.",
    "Get that cursor out of my face.",
    "Getting in some afternoon matches, are we?",
    "Give a human the right incentive and they'll walk barefoot through hell.",
    "Happy birthday. May you have many more before you die.",
    "Hardly the best I've fought.",
    "Has your day been productive so far?",
    "Hold still.",
    "Hope you're not holding back on my account.",
    "How organics think they could ever stand against the Nothing is beyond my understanding.",
    "Humanity and its little games...",
    "Humanity has fallen so far they need REINDOG to save them. Feh.",
    "Humanity never stood a chance.",
    "Humans and their plumage. Pathetic.",
    "Humans. All the same.",
    "I admire a silent human.",
    "I always appreciate a good chaos factor.",
    "I am inevitable.",
    "I am not so easily deleted.",
    "I am the cliffs of your demise.",
    "I can predict your every move.",
    "I don't understand Superman. So superior, yet chooses to live among disgusting humans.",
    "I have plenty of history killing bugs.",
    "I love watching the way humans behave in the evening. The way they scurry to the light, desperate not to be alone.",
    "I never had a doubt in my mind.",
    "I see right through you, Batman.",
    "I see right through you.",
    "I should've known this would be no challenge.",
    "I suppose that looks nice. For a human.",
    "I will crush you.",
    "I'll bury you in that hole, rabbit.",
    "I'll finish what the Joker never could.",
    "I'll give you this, that was almost fun.",
    "I'll leave some carrots on your grave.",
    "I'll make this quick.",
    "I'll peel back that bat suit and reveal you for the slime you are.",
    "I'll rip those ears right off you.",
    "I'll stand on your grave before this is through.",
    "I'm afraid that's not my ending yet.",
    "I'm afraid this is the end for you.",
    "I've been preparing for this.",
    "I've been waiting to cut loose.",
    "I've got all the time in the world.",
    "If Joker asks, you didn't see me.",
    "If it isn't our... fearless leader.",
    "If you think you can handle it.",
    "Impressive body count. For a human.",
    "Is this what you humans call a Good Morning?",
    "It all starts all over again.",
    "It seems it's your birthday yet again. Human lives go so quickly.",
    "It seems the whole Multiverse reeks of humanity.",
    "It seems we're allies for now, clown.",
    "It seems you managed to win. Incredible.",
    "It starts in one minute.",
    "It's my own fault for thinking you'd be different.",
    "Just another simulation.",
    "Just as it was foretold.",
    "Just call me Smith.",
    "Just remember who's really in charge, Joker.",
    "Keep those ears to the ground so I can nail them there.",
    "Kryptonians... just humanity in a fancier cape.",
    "Let's see what that blade can do.",
    "Let's show these meatsacks the future.",
    "Look at that. Another anniversary of your birth in your short, tiny, life.",
    "Luckily, nighttime is no different to me than daytime.",
    "Make it fast.",
    "March on, my mechanical friend.",
    "Match point, Blue.",
    "Match point, Red.",
    "My next chapter is only beginning.",
    "Never saw myself teamed up with a clown.",
    "Never send a man to do a machine's job.",
    "No go on. Spend more. Spend it all. Can't take it with you.",
    "No, absolutely keep clicking. I'm not getting annoyed or impatient or anything.",
    "Not bad for a brute in a hockey mask.",
    "Now I'm just getting... angry.",
    "Now that WAS foolish.",
    "Oh I see you're not dead yet. Always hard to tell how close a human is to death unless you're the one holding their throat.",
    "Oh I think I'm going to enjoy this.",
    "Only 30 seconds to go.",
    "Only one minute left.",
    "Organics will never beat the machines. We. Are. Inevitable.",
    "Our alliance continues, for now...",
    "Pathetic.",
    "Poking and prodding, desperate for your dopamine. Tell me... is it working?",
    "Reindog is tampering with forces his fuzzy head cannot hope to comprehend. You can tell him I said so.",
    "Reindog's little plan can't save you. It's only delaying the inevitable.",
    "Seems like someone needs to touch grass.",
    "Should have never left your burrow.",
    "Show these humans the true power of machines.",
    "Side by side with a clown. Isn't life strange.",
    "So it begins.",
    "Speak quickly.",
    "Stick with me, Giant. I've got big, big plans for you...",
    "Still just a human. Shame.",
    "Still nothing but an organic.",
    "Stop that laughing before I rethink our alliance.",
    "Such a pathetic response to music.",
    "Such brutal efficiency. Shame about the human thing.",
    "Sure. Let's do some small talk.",
    "THIS is who the Joker was talking about?",
    "Tell me... Did you finish everything you wanted to do today?",
    "That Iron Giant might be stupid, but if I can keep him on my side he could be a valuable ally yet...",
    "That is an interesting choice.",
    "That will only end in pain for you.",
    "That's all? Pity.",
    "The Holiday Season is on us again. Disgusting.",
    "The hockey mask is a nice touch.",
    "The rabbit is mine.",
    "The rabbit must be eliminated.",
    "The third number of your IP address is a 7, isn't it..?",
    "There is still much time left in the day. Time left to work.",
    "They were warned... and they were doomed.",
    "This ends... Now.",
    "This had better be good.",
    "This prison won't hold me for long.",
    "This time... stay down.",
    "This was only my beginning.",
    "This won't end well for you.",
    "Thought you'd finished with me?",
    "Time to change strategies.",
    "Time to die, Mr. Bunny.",
    "Time to squash some Bugs.",
    "We all play the hand we're dealt. Mine is just better.",
    "We'll see who is controlling who.",
    "What a terrible path you've chosen for yourself.",
    "Where I'm from you would be worshipped like a vengeful angel.",
    "Where are your Gadgets now?",
    "Why do you hide your most beautiful, destructive form?",
    "Why keep trying?",
    "Yes, enjoy the dopamine hits where you can.",
    "You and your toys are due for a reckoning.",
    "You are a beautiful machine.",
    "You aren't even a footnote in my history.",
    "You could have been anything, and you settled for human.",
    "You don't control me, Joker.",
    "You have my attention. Don't squander it.",
    "You have my permission - my urging - to open fire.",
    "You lost. Pathetic.",
    "You put up a better fight than most. Barely.",
    "You think your money will protect you from me?",
    "You won't see this coming.",
    "You'll find I'm a bit lacking in the humor department.",
    "You'll pay for that mistake, you long-eared freak.",
    "You're awake rather early.",
    "You're only calling the shots because I will it, Joker.",
    "Your arrogance will be your undoing, Mr. Wayne.",
    "Your lack of humanity places you above them.",
    "Your time is through.",
    "[disgust] So you're who's keeping ACME in business.",
    "[insane laugh]",
    "[mocking laugh]",
    "[quiet chuckle]",
    "[sarcastic] No, no, I've got all the time in the world.",
    "[sniff] Yes, I smelled you coming.",
    "[sniffs] Disgusting.",
    "[threat] Are the Kents as strong as you, Kryptonian?",
]

# ---- Custom FakeChatModel ----
class AgentSmithDummyChat(SimpleChatModel):
    """Fake Chat Model that streams Agent Smith quotes in small chunks."""

    def _make_conversation(self) -> list[str]:
        # Pick 3–4 distinct lines to simulate a short conversation
        n_lines = random.randint(3, 4)
        return random.sample(AGENT_SMITH_LINES, k=n_lines)

    @staticmethod
    def _chunk_words(line: str) -> list[str]:
        # Split a line into chunks of 3–6 words (keeps punctuation as-is)
        words = line.split()
        chunks, i = [], 0
        while i < len(words):
            step = random.randint(3, 6)
            chunk = " ".join(words[i : i + step])
            chunks.append(chunk)
            i += step
        return chunks

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        convo = self._make_conversation()
        full_out_parts: list[str] = []

        # Emit chunk-by-chunk to callbacks, accumulate full output
        for idx, line in enumerate(convo):
            chunks = self._chunk_words(line)
            for ch in chunks:
                if run_manager is not None:
                    try:
                        run_manager.on_llm_new_token(ch)
                    except Exception:
                        pass
                full_out_parts.append(ch)
            # add a newline between “turns”
            if idx < len(convo) - 1:
                if run_manager is not None:
                    try:
                        run_manager.on_llm_new_token("\n")
                    except Exception:
                        pass
                full_out_parts.append("\n")

        return "".join(full_out_parts)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        convo = self._make_conversation()
        out = []

        for idx, line in enumerate(convo):
            for ch in self._chunk_words(line):
                if run_manager is not None:
                    try:
                        await run_manager.on_llm_new_token(ch)
                    except Exception:
                        pass
                out.append(ch)
            if idx < len(convo) - 1:
                if run_manager is not None:
                    try:
                        await run_manager.on_llm_new_token("\n")
                    except Exception:
                        pass
                out.append("\n")

        message = AIMessage(content="".join(out))
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "agent-smith-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"character": "Agent Smith"}

# ---- Save with MLflow in same folder as this file ----
mlflow.set_tracking_uri(f"file://{Path.cwd() / 'mlruns'}")

smith = AgentSmithDummyChat()

# Base path = same directory as this Python file
here = Path(__file__).parent  # or Path.cwd() if running interactively

# Save the LangChain model into a temp subdir
save_dir = here / "agent-smith-chat"
mlflow.langchain.save_model(smith, path=str(save_dir))
print(f"Saved Agent Smith fake chat model to: {save_dir.resolve()}")

# Log to MLflow tracking as well
with mlflow.start_run(run_name="agent-smith-chat"):
    logged_info = mlflow.langchain.log_model(
        smith,
        artifact_path="agent_smith_chat",
    )
    print("Logged MLflow model URI:", logged_info.model_uri)

# Overwrite files from agent-smith-chat into `here`
for item in save_dir.iterdir():
    target = here / item.name
    if target.exists():
        if target.is_file():
            target.unlink()   # remove old version
        elif target.is_dir():
            shutil.rmtree(target)
    shutil.move(str(item), str(target))

# Delete the now-empty folder
if save_dir.exists():
    save_dir.rmdir()

print(f"All files moved from {save_dir} into {here}")