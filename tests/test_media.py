import unittest

from video2prompt.media import (
    Shot,
    build_shots,
    frame_times,
    parse_cut_times,
    select_shots,
    speech_for_shot,
)
from video2prompt.pipeline import (
    add_speech,
    build_timeline,
    clean_output,
    finish_prompt,
    word_count,
    slugify,
    writer_notes,
    writer_style,
)
from video2prompt.backends import BackendError, parse_claude_code_output
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


class AdaptiveFrameTimes(unittest.TestCase):
    shot = Shot(1, 0.0, 10.0)

    def curve(self, busy_from, busy_to):
        return [(i / 6, 20.0 if busy_from <= i / 6 < busy_to else 1.0) for i in range(1, 60)]

    def test_frames_gather_where_the_picture_changes(self):
        times = frame_times(self.shot, 8, self.curve(7.0, 8.0))
        even = frame_times(self.shot, 8)
        self.assertGreater(sum(7.0 <= t < 8.0 for t in times), sum(7.0 <= t < 8.0 for t in even))
        self.assertEqual(times, sorted(times))

    def test_quiet_parts_keep_some_frames(self):
        times = frame_times(self.shot, 8, self.curve(7.0, 8.0))
        self.assertTrue(any(t < 3.0 for t in times))

    def test_times_stay_inside_shot_and_are_distinct(self):
        times = frame_times(self.shot, 8, self.curve(4.0, 4.4))
        self.assertTrue(all(0.0 < t < 10.0 for t in times))
        self.assertEqual(len(times), len(set(times)))

    def test_flat_or_missing_curve_is_evenly_spread(self):
        even = frame_times(self.shot, 4)
        self.assertEqual(even, [1.25, 3.75, 6.25, 8.75])
        self.assertEqual(frame_times(self.shot, 4, []), even)
        flat = frame_times(self.shot, 4, [(i / 6, 0.0) for i in range(1, 60)])
        self.assertEqual(len(flat), 4)
        self.assertLess(max(flat) - min(flat), 8.0)
        self.assertGreater(max(flat) - min(flat), 6.0)


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


class FakeBackend:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, system, text, images, schema):
        self.calls += 1
        return {"prompt": self.replies.pop(0)}


class FinishPrompt(unittest.TestCase):
    def test_timestamps_and_frame_mentions_are_removed(self):
        text = "A cat sits. At 1.8 seconds, it jumps. By 2.5 s, it lands, its head turning in later frames."
        self.assertEqual(
            clean_output(text, TARGETS["ltx"]),
            "A cat sits. It jumps. It lands, its head turning.",
        )

    def test_a_display_reading_a_word_is_dropped(self):
        self.assertEqual(clean_output("A sign reading 'Paris' glows. A cat sits.", TARGETS["ltx"]), "A cat sits.")

    def test_short_prompt_needs_no_model_call(self):
        backend = FakeBackend([])
        self.assertEqual(finish_prompt(backend, "A cat sits.", TARGETS["wan"]), "A cat sits.")
        self.assertEqual(backend.calls, 0)

    def test_long_prompt_is_shortened(self):
        backend = FakeBackend(["A short cat."])
        self.assertEqual(finish_prompt(backend, "word " * 130, TARGETS["wan"]), "A short cat.")
        self.assertEqual(backend.calls, 1)

    def test_shortening_stops_after_three_attempts(self):
        backend = FakeBackend(["word " * 135, "word " * 130, "word " * 125])
        self.assertEqual(word_count(finish_prompt(backend, "word " * 140, TARGETS["wan"])), 125)
        self.assertEqual(backend.calls, 3)


class ClaudeCodeOutput(unittest.TestCase):
    def test_structured_answer_is_returned(self):
        stream = '{"type":"system"}\nnot json\n{"type":"result","subtype":"success","structured_output":{"prompt":"A cat."}}\n'
        self.assertEqual(parse_claude_code_output(stream), {"prompt": "A cat."})

    def test_usage_limit_error_is_reported(self):
        stream = '{"type":"result","subtype":"success","is_error":true,"result":"Usage limit reached"}\n'
        with self.assertRaisesRegex(BackendError, "Usage limit reached"):
            parse_claude_code_output(stream)

    def test_missing_result_is_an_error(self):
        with self.assertRaises(BackendError):
            parse_claude_code_output('{"type":"system"}\n')

    def test_missing_structured_answer_is_an_error(self):
        with self.assertRaises(BackendError):
            parse_claude_code_output('{"type":"result","subtype":"success","result":"plain text"}\n')


class Timeline(unittest.TestCase):
    def test_beats_are_merged_in_time_order_without_overlays(self):
        shots = [
            {"analysis": {"beats": [{"time": 7.25, "event": "A fist hits the glass."}]}},
            {"analysis": {"beats": [
                {"time": 2.0, "event": "People run."},
                {"time": 3.0, "event": "White text appears."},
                {"time": 4.0, "event": "  "},
            ]}},
        ]
        self.assertEqual(
            build_timeline(shots),
            [{"time": 2.0, "event": "People run."}, {"time": 7.2, "event": "A fist hits the glass."}],
        )

    def test_writer_notes_filter_beats_too(self):
        notes = writer_notes({"beats": [{"time": 1, "event": "A caption fades in."}], "subject": "a cat"})
        self.assertEqual(notes, {"beats": [], "subject": "a cat"})


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
