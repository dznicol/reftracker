# RefTracker — Roadmap

## Decision Classification

### Double-pass architecture
Use referee velocity data from the tracking pass to identify decision windows, then send only short clips to Gemini for classification.

**Problem:** Sending full match video to Gemini produces hallucinated decisions during open play. The referee is running and watching — not signalling — for 90%+ of the match.

**Approach:**
1. Compute velocity from `tracked.json` position deltas
2. Find "ref stopped" windows (velocity below threshold for N consecutive frames)
3. Extract short clips around each window (~3s before to ~5s after the stop)
4. Send only those clips to Gemini with signal reference images
5. Map Gemini's responses back to full video timestamps via `segment_offset_seconds`

**Impact:** For an 80-minute match, expect ~30-40 decision windows of ~8s each = ~5 minutes of video to Gemini instead of 80 minutes. Cheaper, faster, and less hallucination.

### Player clustering for team assignment
Use rugby's forward-pass rule to determine which team occupies which side of the pitch.

**Problem:** Gemini can see which direction the referee points but doesn't know which team is on which side, so team assignment is random.

**Approach:** In rugby, players must stay behind the ball (forward passing is illegal). The majority of same-coloured players clustered on one side of the ball reveals which team occupies that half. Add this as a general prompt rule: "The ref points toward the team the decision favours."

**Impact:** Correct team assignment without video-specific configuration.

### Whistle detection via audio
Detect referee whistle blows from the audio track as an alternative or complement to velocity-based decision windows.

**Approach:** Rugby whistles have a distinctive high-frequency acoustic signature. Use librosa or scipy to detect sharp peaks in the 2-4kHz range. Cross-reference with velocity data for higher confidence.

## Tracking

### Floodlight colour adaptation
Under artificial floodlights, the referee's green shirt appears darker and more saturated than in daylight, sometimes falling inside the grass exclusion mask.

**Approach:** Adapt the grass mask dynamically based on detected lighting conditions, or use a separate colour profile for the referee's shirt learned during calibration rather than fixed HSV ranges.

### Breakdown cluster handling
When the referee enters a dense cluster of players (ruck/maul), multiple detections overlap and the tracker loses the ref's identity.

**Approach:**
- Velocity-predicted search: predict where the ref should emerge based on their pre-cluster trajectory
- "No lock is better than wrong lock" — prefer dropping tracking temporarily over locking onto the wrong person
- Post-hoc interpolation: fill tracking gaps by interpolating position between last-known and re-acquired positions

### Linesman exclusion
The linesman wears the same kit as the referee and can trigger false re-identification, especially near the touchline.

**Approach:** Exclude detections near the frame edges (within ~5% of frame width) from re-ID candidates, since the linesman operates along the touchline.

## Video & Output

### Match-length scalability
Current pipeline processes a single 62s clip. Full matches are 80 minutes (2 x 40-minute halves).

**Approach:**
- Split video into segments with ffmpeg (already supported in classify_decisions.py)
- Process segments in parallel where possible
- Stitch outputs maintaining correct timestamps across segments

### Broadcast footage support
Current approach requires wide-angle footage (Veo-style fixed elevated camera). Broadcast footage follows the ball, with the referee off-screen most of the time.

**Approach:** Detect referee appearances in broadcast footage and track only during visible segments. Lower coverage but still useful for key decision moments which broadcasters typically show.
