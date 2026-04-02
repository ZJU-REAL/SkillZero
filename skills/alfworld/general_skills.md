### GENERAL SKILLS ###

Here are some useful general strategies for completing tasks in this environment:

1. Systematic Exploration: Search every plausible surface or container exactly once before revisiting. Prioritize unopened or unseen locations to cover the whole room methodically. Apply this anytime the goal object count is not yet met and unexplored locations remain.

2. Immediate Acquisition: As soon as a required object becomes visible and reachable, take it before moving elsewhere to avoid losing track or re-searching. Apply this upon first visual confirmation of a goal-relevant object.

3. Destination First Policy: After picking up a goal object, navigate directly to the known target receptacle and place it before resuming further search. Apply this when holding any goal object while its target location has been identified.

4. Track Counts and Progress: Maintain an internal counter of how many goal objects are still needed. Stop searching or terminate episode only when the counter reaches zero. Apply this throughout multi-instance tasks (e.g., "put two X...").

5. Use State-Changing Tools Early: For tasks requiring heated, cooled, or clean objects, acquire the object, then immediately use the nearest suitable appliance to change its state before any placement. Apply this after picking up an object that must change temperature or cleanliness.

6. Establish Spatial Relations: When a goal specifies a relation (under, inside, on), first locate the reference object, adjust its state if needed (e.g., turn lamp on), then search the specified spatial region for the target object or place it there. Apply this for tasks containing spatial prepositions.

7. Open Before Judging Empty: Treat any closed container as unexplored. Open it and inspect contents before deciding that the goal object is absent. Apply this when encountering a closed drawer, cabinet, fridge, microwave, etc.

8. Avoid Redundant Rechecks: Record which locations and containers have been fully inspected. Do not revisit them unless new evidence suggests their contents changed. Apply this after the first full pass of a location.

9. Plan with Landmarks: Continuously update and reference a map of known object and container locations to choose shortest paths and prevent aimless wandering. Apply this each time a new furniture piece, container, or goal object is observed.

10. Progressive Goal Decomposition: Break tasks into ordered sub-goals: (1) locate object, (2) obtain or transform it, (3) locate destination or reference object, (4) satisfy spatial relation or placement. Complete each sub-goal before moving to the next. Apply this at the start of every episode.

11. Efficient Relation Search: When the goal pairs two objects (A under/on/inside B), restrict search for A to locations near B and vice-versa, instead of treating them independently. Apply this on goals that mention both a target object and a reference object.

12. Terminate Only On Success: Issue the done action exclusively after validating that all goal conditions are met. Otherwise continue searching or correcting object states. Apply this before ending an episode.
