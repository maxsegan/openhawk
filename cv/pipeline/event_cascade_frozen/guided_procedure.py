"""Step-by-step procedure for local VLM labellers (2026-09-23).

Written from Qwen3.8-27B traces on the fair-evidence grid: the model described all 25 frames in
near-identical prose ("ball visible, green dot"), misread the tracker-centred crop (the ball barely
moves inside it, so it chased court lines "jumping" past the ball), re-examined the same frames
with 29-61 hedges per answer, and spent a median 10.5k tokens to end on abstain or no_event in 36
of 52 replies. The procedure below replaces open-ended looking with a fixed, bounded pass.
"""

GUIDED_PROCEDURE = """
HOW TO WORK - follow these steps in order, once. Be brief: short lines, no essays.

1. READ THE CROP CORRECTLY. Every crop follows the tracker, so the ball stays near the centre while
   it is in flight and the BACKGROUND slides past it. Court lines, the net or a player "jumping"
   from one side of the ball to the other is camera-following, not an event. Judge the ball's
   motion by how it moves RELATIVE TO the court, net and players.

2. FIND THE BALL. Use the FALLIBLE TRACKER HINT image to see where the tracker thought the ball
   was, then confirm the ball in the CLEAN image at those places. The ball is a small bright
   blob of a few native pixels (a few 8x8 blocks), often motion-blurred into a short streak.
   Write ONE line per frame, e.g. "104: centre" or "104: left of centre, near racket" or
   "104: not seen". Do not describe anything else yet.

3. EARLY EXITS.
   - If you cannot point to the ball in at least 8 of the clean frames: answer abstain now.
   - If the ball is seen throughout and at no frame touches a racket/player, the court surface
     (or its own shadow), or the net: answer no_event now.

4. FIND THE ONE CHANGE. Look only at the frames where the ball is next to a racket/player, the
   court, or the net, and decide:
   - contact: the ball meets the racket or player and its path relative to the court reverses
     or sharply changes. Frame = where ball and racket strings coincide.
   - bounce: the ball comes down to the court surface (meets its shadow) and leaves upward.
     Frame = where ball and shadow/court meet.
   - net_hit: the ball meets the net tape or mesh and stops, drops or deflects.
   The nomination is only a hint; it is often wrong about the type and a few frames off.

5. STOP. Do not re-examine frames you already wrote a line for, and do not re-derive the
   geometry. If after steps 2-4 you are still torn between two answers, pick abstain.
   Keep your whole reasoning under about 800 words.
"""

# Rejection-first variant (2026-09-23). guided_nothink emits an event on almost every
# nomination (3 no_event in 150 on the 11-clip read). This pass asks whether any
# physical impact is visible before it is allowed to name one. Same crops as the
# zero-shot guided pass; only the procedure text changes.
REJECT_PROCEDURE = """
HOW TO WORK - follow these steps in order, once. Be brief: short lines, no essays.

Many nominations are not events. The earlier automatic pass often marks ordinary
flight, a preparation pose, or a tracker error. Rejecting those is the task.
Naming contact, bounce or net_hit when you do not see the impact is an error.

1. IS THERE A PHYSICAL IMPACT IN THIS WINDOW AT ALL?
   Answer this before you name a type. Judge the ball against the court, the
   racket or player, and the net — not against the crop edges.
   no_event — the window supports no physical impact. Choose it when you see any of:
   - uninterrupted flight: the ball never meets a racket, the court, or the net;
   - the ball passes near a player or racket but its path relative to the court
     does not reverse or kink at that meeting;
   - a raised racket, toss, trophy pose, or jump, with no strike inside the window;
   - a ball already resting, or a replay of an impact that already happened;
   - court lines, the net, or a player sliding across the frame. The crop follows
     the tracker, so the background moves and the ball stays near the centre.
     That slide is the camera, not an impact.
   abstain — only when you cannot point to the ball in at least 8 of the clean
   frames, or the crop never shows the ball against the surface that would have
   to be involved. Do not abstain to avoid a clear no_event. Do not name an
   event type to avoid a clear no_event. An unreadable ball is abstain, not
   no_event.

2. READ THE CROP. Confirm the ball in the CLEAN image. The FALLIBLE TRACKER HINT
   dots are not the ball. Write one line per frame, e.g. "104: centre" or
   "104: not seen". Do not describe anything else yet.

3. ONLY IF STEP 1 WAS YES, name the one impact:
   - contact: the ball meets the racket or player AND its path relative to the
     court reverses or sharply changes. Frame = where ball and racket coincide.
     Proximity or a pose is not enough.
   - bounce: the ball descends to the court (meets its shadow) and leaves upward.
     A stationary speck or a post-bounce apex is not a new bounce.
   - net_hit: the ball meets the tape or mesh and stops, drops, or deflects.
     Crossing in front of the net, or lying near it afterwards, is not a net_hit.
   The nomination is only a hint. It is often the wrong type, a few frames off,
   or not an event.

4. STOP. Do not re-examine a frame you already wrote a line for. If step 1 is
   not a clear yes, return no_event or abstain as step 1 says. Keep the whole
   reply under about 800 words.
"""


def procedure_for(profile: str) -> str:
    """Procedure text for a profile, or empty when the profile is not guided."""
    if "reject" in profile:
        return REJECT_PROCEDURE
    if profile.startswith("guided"):
        return GUIDED_PROCEDURE
    return ""
