import unittest

from video2prompt.media import (
    Shot,
    build_shots,
    frame_times,
    parse_cut_times,
    select_shots,
    speech_for_shot,
)
from video2prompt.pipeline import add_speech, clean_output, slugify, writer_notes, writer_style
from video2prompt.targets import TARGETS


class BuildShots(unittest.TestCase):
    def test_no_cuts_gives_one_shot(self):
        shots = build_shots([], 5.0)
        self.assertEqual([(s.start, s.end) for s in shots], [(0.0, 5.0)])

    def test_cuts_become_boundaries(self):
        shots = build_shots([2.0, 4.0], 6.0)
        self.assertEqual([(s.start, s.end) for s in shots], [(0, 2), (2, 4), (4, 6)])
        self.assertEqual([s.index for s in shots], [1, 2, 3])

    def test_flash_cuts_are_ignored(self):
        shots = build_shots([2.0, 2.2, 5.9], 6.0)
        self.assertEqual([(s.start, s.end) for s in shots], [(0, 2), (2, 6)])

    def test_long_take_is_split_and_flagged(self):
        shots = build_shots([], 25.0, max_segment=10.0)
        self.assertEqual(len(shots), 3)
        self.assertEqual([s.continues_previous for s in shots], [False, True, True])
        self.assertAlmostEqual(shots[-1].end, 25.0)


class SelectShots(unittest.TestCase):
    def test_keeps_longest_in_original_order(self):
        shots = [Shot(1, 0, 5), Shot(2, 5, 6), Shot(3, 6, 9)]
        self.assertEqual([s.index for s in select_shots(shots, 2)], [1, 3])

    def test_returns_all_under_limit(self):
        shots = [Shot(1, 0, 5)]
        self.assertEqual(select_shots(shots, 4), shots)


class FrameTimes(unittest.TestCase):
    def test_times_stay_strictly_inside_shot(self):
        times = frame_times(Shot(1, 10.0, 14.0), 4)
        self.assertEqual(len(times), 4)
        self.assertTrue(all(10.0 < t < 14.0 for t in times))
        self.assertEqual(times, sorted(times))

    def test_very_short_shot_uses_two_frames(self):
        self.assertEqual(len(frame_times(Shot(1, 0.0, 0.7), 4)), 2)


class Parsing(unittest.TestCase):
    def test_parse_cut_times(self):
        log = "[Parsed_showinfo_2] n:0 pts:90 pts_time:3.75 duration\nx pts_time:12 y"
        self.assertEqual(parse_cut_times(log), [3.75, 12.0])

    def test_speech_goes_to_shot_holding_its_midpoint(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "un"},
            {"start": 1.5, "end": 3.5, "text": "deux"},
        ]
        self.assertEqual(speech_for_shot(segments, Shot(1, 0, 2)), "un")
        self.assertEqual(speech_for_shot(segments, Shot(2, 2, 4)), "deux")

    def test_slugify(self):
        self.assertEqual(slugify("Été à Paris (final)!"), "ete-a-paris-final")
        self.assertEqual(slugify("???"), "video")


class WriterInput(unittest.TestCase):
    analysis = {"subject": "a cat", "on_screen_text": "SALE", "implied_sound": "wind"}

    def test_burned_in_text_never_reaches_the_writer(self):
        self.assertNotIn("on_screen_text", writer_notes(self.analysis, TARGETS["ltx"]))
        self.assertNotIn("on_screen_text", writer_notes(self.analysis))

    def test_silent_target_gets_no_sound_notes(self):
        self.assertNotIn("implied_sound", writer_notes(self.analysis, TARGETS["wan"]))
        self.assertIn("implied_sound", writer_notes(self.analysis, TARGETS["ltx"]))

    def test_speech_is_appended_verbatim_for_audio_targets(self):
        self.assertEqual(
            add_speech("A cat.", TARGETS["ltx"], "Où {est} il ?"),
            'A cat. A voice says: "Où {est} il ?"',
        )

    def test_overlay_sentences_are_removed_but_textured_stays(self):
        self.assertEqual(
            clean_output("A textured wall. White text appears; A cat runs.", TARGETS["ltx"]),
            "A textured wall. A cat runs.",
        )

    def test_sound_sentences_are_removed_for_silent_targets_only(self):
        prompt = "A cat runs. Soft music plays."
        self.assertEqual(clean_output(prompt, TARGETS["wan"]), "A cat runs.")
        self.assertEqual(clean_output(prompt, TARGETS["ltx"]), prompt)

    def test_prompt_is_kept_when_cleaning_would_empty_it(self):
        self.assertEqual(clean_output("White text appears.", TARGETS["ltx"]), "White text appears.")

    def test_style_loses_overlay_mentions(self):
        style = {"pacing": "Slow. Text overlays animate.", "recurring_subjects": ["A cat", "Text overlays"]}
        self.assertEqual(writer_style(style), {"pacing": "Slow.", "recurring_subjects": ["A cat"]})

    def test_speech_is_skipped_when_absent_or_silent_target(self):
        self.assertEqual(add_speech("A cat.", TARGETS["ltx"], ""), "A cat.")
        self.assertEqual(add_speech("A cat.", TARGETS["wan"], "Bonjour"), "A cat.")


if __name__ == "__main__":
    unittest.main()
