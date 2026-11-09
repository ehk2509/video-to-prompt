"""Prompt-writing rules for each family of video generators."""

from __future__ import annotations

from dataclasses import dataclass

_COMMON_NEGATIVE = (
    "blurry, low quality, jitter, flicker, warped anatomy, extra fingers, "
    "distorted faces, watermark, subtitles, logo"
)


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    rules: str
    negative_prompt: str
    # Added verbatim after the prompt, so a model can't paraphrase what is said. Empty = no audio.
    speech_line: str = ""
    max_words: int = 200


TARGETS: dict[str, Target] = {
    "ltx": Target(
        key="ltx",
        label="LTX-2.x (image + son)",
        rules=(
            "Write one single flowing paragraph of at most 200 words, in the present tense, "
            "as a chronological description from the first moment to the last. Open directly "
            "on the shot and the subject. Use concrete action verbs. Name the camera move "
            "explicitly (static, slow dolly-in, pan left, handheld tracking...). Weave in "
            "lighting and color. LTX generates sound with the picture: describe any ambient "
            "sound and effects from the notes inline."
        ),
        negative_prompt=_COMMON_NEGATIVE,
        speech_line='A voice says: "{speech}"',
    ),
    "wan": Target(
        key="wan",
        max_words=120,
        label="Wan 2.2 (image seule)",
        rules=(
            "Write one paragraph of about 80 to 120 words following this order: subject and "
            "its appearance, scene, motion of the subject and of the environment, camera "
            "language, then aesthetic control (lighting, shot size, color tone) and "
            "stylization. Use the present tense and concrete motion. Wan does not generate "
            "audio here: never mention sound, music or dialogue."
        ),
        negative_prompt=(
            _COMMON_NEGATIVE + ", overexposed, oversaturated"
        ),
    ),
    "generic": Target(
        key="generic",
        max_words=150,
        label="Générique (Sora, Veo, Kling, Runway...)",
        rules=(
            "Write one natural-language paragraph of 60 to 150 words in the present tense: "
            "shot type and camera move first, then subject, action in chronological order, "
            "setting, lighting, color and style. End with one sentence describing any ambient "
            "sound from the notes."
        ),
        negative_prompt=_COMMON_NEGATIVE,
        speech_line='A voice says: "{speech}"',
    ),
}
