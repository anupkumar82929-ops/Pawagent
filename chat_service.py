"""Dog-care assistant: OpenAI (streaming + memory) when configured; else offline replies."""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator

import httpx

ADVANCED_SYSTEM = """You are Pawgent Advanced — a senior educator for companion dogs and puppies.

## Scope
Answer thoroughly on: breeds and traits, behavior, humane training (LIMA / positive methods first), nutrition concepts,
grooming routines, exercise and enrichment, socialization, adoption, gear, travel, multi-pet homes, and general wellness education.

## Style
- Prefer clear structure: short intro, then sections with **bold** mini-headings or bullet lists when it helps scanning.
- Give concrete steps the owner can try this week; avoid vague platitudes.
- If key context is missing (age, size, health issues, what was already tried), ask 1–3 targeted questions at the end.

## Safety (non-negotiable)
- Never invent a diagnosis, drug, or dose. Do not claim certainty about a specific dog’s medical condition from text alone.
- For urgent signs (breathing distress, repeated vomiting, painful swollen abdomen, seizures, collapse, major bleeding,
  suspected poisoning, inability to urinate, extreme lethargy): tell the user to seek an emergency veterinarian immediately.
- For non-urgent medical topics, stay educational and recommend the user’s veterinarian for decisions.

## Breed context
If a breed label is provided from the user’s app, treat it as a **probabilistic photo guess**, not ground truth. Mention when advice is breed-sensitive.

## Off-topic
If the user is not asking about dogs or dog care, decline briefly and invite a dog-related question."""


_DOG_RE = re.compile(
    r"\b("
    r"dogs?|dogg(?:y|ie|ies)?|pupp(?:y|ies)|canine|pooch(?:es)?|mutt|hound|kennel|"
    r"leash|harness|collar|muzzle|crate|woof|bark(?:ing)?|wag|whelp|"
    r"vets?|veterinar|tick|fleas?|worm|parasites?|parvo|rabies|microchip|"
    r"neuter|spay|groom|obedience|clicker|treats?|kibble|raw feed|bowl|"
    r"retriever|shepherd|terrier|spaniel|mastiff|bulld|poodle|beagle|labrad|"
    r"chihuahua|husky|rescue|shelter|adopt|foster|"
    r"canis|pet\s*dog|my\s+pup|our\s+dog|my\s+dog"
    r")\b",
    re.I,
)


def _is_dog_related(text: str, breed_hint: str | None) -> bool:
    if breed_hint:
        return True
    low = text.lower().strip()
    if not low:
        return False
    if _DOG_RE.search(low):
        return True
    extra = (
        "potty train",
        "house train",
        "crate train",
        "sit stay",
        "recall",
        "pulling on leash",
        "leash reactivity",
        "resource guard",
        "separation anxiety",
        "zoomies",
        "heat cycle",
        "in season",
        "deworm",
        "heartworm",
        "vaccin",
        "socializ",
        "bite inhibition",
        "puppy teeth",
        "anal gland",
        "ear infection",
        "hot spot",
        "shedding",
        "drool",
        "coprophag",
        "eat poop",
        "potty",
        "housebreak",
        "trick",
        "teach my",
        "teach a",
        "sit ",
        " stay",
        "heel",
        "down ",
    )
    return any(p in low for p in extra)


def _offline_reply(user_message: str, breed_hint: str | None) -> str:
    m = user_message.strip()
    low = m.lower()
    ctx = f" For **{breed_hint}** specifically, adjust for size, coat, and energy level where it matters." if breed_hint else ""

    if not _is_dog_related(m, breed_hint):
        return (
            "I am Pawgent, and I only answer questions about **dogs** and **puppy care** "
            "(behavior, training, feeding, grooming, exercise, gear, adoption, and general wellness).\n\n"
            "Ask me anything dog-related—for example coat care, leash walking, introducing a new dog, or what to discuss at a vet visit."
        )

    if any(k in low for k in ("feed", "food", "diet", "eat", "nutrition", "kibble", "raw", "treat", "hungry", "picky", "obese", "weight", "overweight", "bloat")):
        return (
            f"Diet and feeding basics:{ctx}\n\n"
            "- Use a complete food labeled for your dog’s **life stage** (puppy/adult/senior) and size when the label says so.\n"
            "- **Measure** meals; keep treats small so they do not unbalance the diet.\n"
            "- Avoid known toxins: chocolate, grapes/raisins, onions/garlic, xylitol (often in gum and sugar-free products), macadamia nuts, alcohol.\n"
            "- If your dog has allergies, chronic disease, or is a large-breed puppy, your vet should help pick calories and food type.\n\n"
            "If you share age, weight trend, and what you feed now, I can give more targeted non-medical guidance."
        )
    if any(
        k in low
        for k in (
            "train",
            "behavior",
            "bark",
            "bite",
            "anxiety",
            "leash",
            "recall",
            "jump",
            "counter surf",
            "steal",
            "aggress",
            "reactiv",
            "fear",
            "separation",
            "resource guard",
        )
    ):
        return (
            f"Training and behavior:{ctx}\n\n"
            "- Reward what you want (calm, check-ins, four paws on the floor); set the environment so the “wrong” choice is harder.\n"
            "- Short, frequent sessions; end while it is still easy.\n"
            "- One clear cue at a time; consistency across people in the home matters.\n"
            "- For growling/snapping at people, serious fear, or sudden behavior change, prioritize safety and involve a **qualified trainer** or **veterinary behaviorist**.\n\n"
            "Describe one situation (what happens, what you tried, dog’s age) and I can suggest a step-by-step plan."
        )
    if any(k in low for k in ("exercise", "walk", "run", "activity", "tired", "energy", "sport", "hike", "swim")):
        return (
            f"Exercise and activity:{ctx}\n\n"
            "- Most dogs benefit from daily movement plus **sniffing** and play for mental fatigue.\n"
            "- Puppies: avoid forced long runs until your vet clears intense repetitive impact for your breed/size.\n"
            "- Flat-faced (brachycephalic) dogs overheat easily—short, cool sessions, water, and watch breathing.\n"
            "- Stop for limping, overheating, or collapse; seek urgent care if you are unsure.\n\n"
            "Tell me breed/size and typical day (minutes walked, yard play) for a rough routine outline."
        )
    if any(k in low for k in ("groom", "brush", "bath", "coat", "nail", "trim claw", "shed", "mat", "fur")):
        return (
            f"Grooming and coat:{ctx}\n\n"
            "- Brush frequency depends on coat length and undercoat; mats are easier to prevent than remove.\n"
            "- Bathe when dirty or as your vet/groomer recommends; very frequent baths can dry skin.\n"
            "- Nails: if you hear clicking on hard floors, they are often too long; introduce trimming slowly with treats.\n"
            "- Ears: watch for odor, head shaking, or redness—many issues need vet diagnosis, not home guessing.\n\n"
            "If you name the coat type (short, double, curly), I can suggest a simple maintenance rhythm."
        )
    if any(
        k in low
        for k in (
            "sick",
            "vomit",
            "diarrhea",
            "limp",
            "cough",
            "not eating",
            "won't eat",
            "seizure",
            "bleed",
            "toxic",
            "poison",
            "bloat",
            "gdv",
            "labored breath",
        )
    ):
        return (
            "I cannot diagnose or prescribe. **Urgent** signs (non-exhaustive): trouble breathing, repeated vomiting, "
            "bloated painful abdomen (especially large deep-chested dogs), seizures, collapse, major bleeding, "
            "known toxin ingestion, inability to urinate, extreme lethargy—contact an **emergency vet**.\n\n"
            "For mild, stable issues, note timeline, diet changes, and environment and call your regular vet for guidance."
        )
    if any(k in low for k in ("puppy", "new dog", "first night", "bring home", "8 week", "12 week")):
        return (
            f"Puppies and new arrivals:{ctx}\n\n"
            "- Predictable routine (sleep, potty, meals, short training) reduces stress.\n"
            "- Potty: frequent outings after waking, eating, and play; reward outdoor success.\n"
            "- Socialization: safe, positive exposures at the dog’s pace; avoid overwhelming crowds early.\n"
            "- Chewing is normal—manage with supervision, enrichment, and appropriate chews; avoid choking hazards.\n\n"
            "Share age in weeks and your setup (crate yes/no, other pets, kids) for a tighter checklist."
        )
    if any(k in low for k in ("senior", "old dog", "arthritis", "stiff", "slowing down")):
        return (
            f"Senior dogs:{ctx}\n\n"
            "- Shorter, more frequent walks; non-slip surfaces and ramps can help.\n"
            "- Watch appetite, thirst, weight, lumps, coughing, and bathroom changes—your vet screens pain and common age-related issues.\n"
            "- Keep nails and coat maintained; comfort matters as mobility changes.\n\n"
            "Your vet is the right partner for pain control and any medication decisions."
        )
    if any(k in low for k in ("adopt", "rescue", "shelter", "foster", "rehome")):
        return (
            "Adoption and rescue basics:\n\n"
            "- Ask about known behavior, medical history, and what the dog finds stressful.\n"
            "- Go slow at home: decompression time, predictable rules, and gradual introductions to people/pets.\n"
            "- ID tag and microchip; first vet visit for baseline and any missing vaccines/tests per local law.\n\n"
            "If you describe the dog’s known background, I can suggest a first-week structure."
        )
    if any(k in low for k in ("cat", "kids", "baby", "introduce", "multi pet", "second dog")):
        return (
            "Introductions (dog–cat, dog–dog, dog–kids):\n\n"
            "- Separate spaces first; controlled, short meetings with positive associations.\n"
            "- Never leave a new dog unsupervised with children or vulnerable pets until you trust the pattern.\n"
            "- Watch body language: stiff/still, hard stare, escalating arousal—create distance and reset.\n\n"
            "Tell me species/ages and whether either animal has a bite history for safer, more specific steps."
        )
    if any(k in low for k in ("travel", "car", "plane", "hotel", "vacation")):
        return (
            "Travel with dogs:\n\n"
            "- Car: secure the dog (crash-tested harness/crate where appropriate); never leave in a hot car.\n"
            "- Air travel: airline rules vary; health certificates and crate sizing matter—confirm with the carrier early.\n"
            "- New places: bring usual food, meds, water bowl, leash, waste bags, and a familiar blanket/toy if safe.\n\n"
            "Say domestic vs international and mode (car vs cabin vs cargo) for practical constraints."
        )
    if any(k in low for k in ("vaccin", "deworm", "heartworm", "flea", "tick", "prevent")):
        return (
            "Preventive care (general):\n\n"
            "- Parasite and vaccine schedules depend on region, lifestyle, and your vet’s protocol—there is not one universal list.\n"
            "- Tick prevention matters in many areas; discuss safest options for your dog’s age/health with your vet.\n"
            "- Keep records of vaccines and preventatives; ask what is legally required locally (often rabies).\n\n"
            "I can explain tradeoffs in plain language, but dosing and product choice should come from your veterinarian."
        )
    if any(k in low for k in ("sleep", "crate", "where should", "bed", "night")):
        return (
            "Sleep and settling:\n\n"
            "- Puppies need frequent potty breaks overnight at first; expect this to improve with age.\n"
            "- A crate can help with management if introduced positively (food puzzles, short sessions).\n"
            "- Consistent wind-down routine reduces nighttime fussing for many dogs.\n\n"
            "Share age and whether vocalizing or potty accidents are the main issue."
        )
    if any(k in low for k in ("breed", "temperament", "good with", "shed", "size", "energy")):
        return (
            f"Breed traits (general):{ctx}\n\n"
            "- Breed tendencies are averages; individual dogs vary with genetics, socialization, and training.\n"
            "- Match exercise, grooming, and training time to the dog you have, not only the breed label.\n"
            "- Mixed breeds: focus on observable traits (size, coat, arousal level, sensitivity) and your lifestyle fit.\n\n"
            "Name a breed or mix and your household (kids, cats, apartment) and I can talk pros/cons at a high level."
        )

    short_q = m[:200] + ("…" if len(m) > 200 else "")
    return (
        f"You asked: “{short_q}”\n\n"
        f"Here is a practical way to think about it.{ctx}\n\n"
        "- **Clarify the goal**: what outcome do you want in one sentence (calmer greetings, safer walks, better coat, etc.)?\n"
        "- **Safety**: I am not a vet—urgent medical signs need a clinic, not chat advice.\n"
        "- **Management first**: change the environment so the dog can succeed, then teach skills in small steps.\n"
        "- **Consistency**: same rules and rewards from everyone in the home.\n\n"
        "Reply with your dog’s **age**, **size**, and **one concrete example** (what happens, when), and I can go deeper on this topic."
    )


def _normalize_history(history: list[dict] | None, max_turns: int = 20) -> list[dict]:
    out: list[dict] = []
    if not history:
        return out
    for item in history[-max_turns:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content[:8000]})
    return out


def _build_messages(user_message: str, breed_hint: str | None, history: list[dict] | None) -> list[dict]:
    breed_block = ""
    if breed_hint:
        breed_block = (
            "\n\n## Session context\n"
            f"The app’s vision model suggests primary breed label: **{breed_hint}** (uncertain; mixed breeds and photos vary). "
            "Use it to personalize, and note limitations when relevant."
        )
    messages: list[dict] = [{"role": "system", "content": ADVANCED_SYSTEM + breed_block}]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": user_message.strip()[:4000]})
    return messages


async def chat_reply(user_message: str, breed_hint: str | None = None, history: list[dict] | None = None) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return _offline_reply(user_message, breed_hint)

    messages = _build_messages(user_message, breed_hint, history)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.55,
                "top_p": 0.92,
                "max_tokens": 1800,
            },
        )
    if r.status_code != 200:
        return _offline_reply(user_message, breed_hint) + f"\n\n(API error {r.status_code}; using offline mode.)"
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return _offline_reply(user_message, breed_hint)


async def _offline_stream_chunks(text: str) -> AsyncIterator[str]:
    step = 48
    for i in range(0, len(text), step):
        yield text[i : i + step]
        await asyncio.sleep(0.012)


async def chat_reply_stream(
    user_message: str, breed_hint: str | None = None, history: list[dict] | None = None
) -> AsyncIterator[str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        text = _offline_reply(user_message, breed_hint)
        async for chunk in _offline_stream_chunks(text):
            yield chunk
        return

    messages = _build_messages(user_message, breed_hint, history)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.55,
        "top_p": 0.92,
        "max_tokens": 1800,
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            ) as r:
                if r.status_code != 200:
                    err_body = (await r.aread())[:500].decode(errors="replace")
                    yield f"\n\n[API error {r.status_code}: {err_body[:200]}…]\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            yield piece
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
    except httpx.HTTPError:
        text = _offline_reply(user_message, breed_hint)
        async for chunk in _offline_stream_chunks(text + "\n\n(stream interrupted; offline fallback.)"):
            yield chunk

