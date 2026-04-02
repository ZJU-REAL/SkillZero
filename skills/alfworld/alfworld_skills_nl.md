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

### TASK: pick_and_place ###

For this pick-and-place task, apply these specific strategies:

1. Systematic First-Pass Search: Maintain a checklist of all visible and closed containers and surfaces. Open or inspect each unseen candidate exactly once before revisiting any location. Apply this after reading the goal and before acquiring every required object.

2. Grab When Seen: Whenever a needed object is visible and reachable, immediately take it before moving elsewhere. Do not leave targets uncollected. Apply this upon first sight of an unheld object that matches the goal specification.

3. Transform Before Transport: If the goal specifies a state change (e.g., heated, cooled, cleaned), perform the transformation at the nearest appropriate appliance before heading to the final destination. Apply this right after acquiring an object that must change state.

4. Place Before More Search: When holding any goal object and the target location is known and reachable, navigate there and place it immediately, then resume searching if more items are needed. Apply this while carrying a required object and the destination has been identified.

5. Track Counts for Multiples: Maintain a tally of how many target objects are required, held, and already placed. Stop searching only after the placed count meets the goal. Apply this throughout tasks demanding two or more instances of the same object.

### TASK: look_at_obj_in_light ###

For this examine-object-under-light task, apply these specific strategies:

1. Seek Lamp Surfaces First: Head straight to furniture that commonly hosts a desklamp (desk, sidetable, nightstand) because the target must end up under that light. Apply this right after parsing the goal.

2. Switch Lamp On: Issue the use desklamp command as soon as you reach it so the light condition is satisfied before or immediately after handling the target object. Apply this upon arriving at a desklamp that is currently off.

3. Acquire Or Fetch Target: If the target is on the lit surface, take it. If found elsewhere, pick it up and carry it back to the lit lamp area. Apply this whenever the target object becomes visible, whether under the lamp or not.

4. Pair Objects Early: When both the target and desklamp are seen on the same surface, immediately pick up the target there and use the desklamp without moving elsewhere. Apply this upon first observation that the target object and desklamp share the current location.

5. Grab Target First: If you see the target but not the desklamp yet, take the target immediately so you can carry it to wherever the desklamp is found. Apply this when the target object is visible and not yet held, while the desklamp location is unknown.

6. Shortcut To Tool: Once holding the target, navigate straight to the nearest known desklamp and issue a single use command with no intervening searches or detours. Apply this when the target object is in inventory and at least one desklamp location is known.

7. Single Toggle Rule: Use the desklamp only after the target is in hand or co-located. Avoid repeated or premature toggles that waste steps. Apply this when about to interact with a desklamp.

8. Systematic Surface Search: On each candidate lamp surface, look, open nearby containers and drawers once, then move on, ensuring all plausible lamp locations are covered. Apply this after the lamp at the current surface is on and the target has not been found.

9. Explore New Surfaces: If either the target or the desklamp has not been found, systematically check unvisited surfaces and containers instead of re-inspecting places already confirmed empty. Apply this when the location of a required object remains unknown after current area is exhausted.

10. Lock In Observed Items: After spotting either required object, do not leave without acting. Pick it up if it is the target, or note its location if it is the desklamp, to prevent costly backtracking loops. Apply this just after observing a required object during scanning.

### TASK: clean ###

For this clean-and-place task, apply these specific strategies:

1. Phase-Ordered Plan: Always execute clean tasks in the fixed sequence: (1) locate and acquire target object, (2) bring it to an available water source or sink to clean, (3) navigate to final location, (4) place object. Apply this as soon as the goal specifies the object must be clean before placement.

2. Pick Before You Wander: If the target object is visible or discovered, take it immediately. Never leave it behind to explore other places, as possession enables direct cleaning and prevents redundant searches. Apply this after visually confirming the presence of the target object on any surface or inside an opened container.

3. Sink First for Cleaning: Upon holding the target object, go straight to the nearest sink, basin, or faucet and issue the clean or use command before any placement attempts. Apply this once the target object is in hand and its required state is clean.

4. Systematic Container Sweep: When the object is not found on open surfaces, iterate through all unopened or unexamined containers in the room before revisiting already-checked spots to avoid search loops. Apply this after initial obvious surfaces are empty and the target object remains unfound.

5. State Verification Before Drop: Always inspect or infer the object state after cleaning. If still not confirmed clean, re-clean before final placement to satisfy goal conditions. Apply this immediately after a cleaning action and before placing the object at its target location.

6. Use Location Priors: Begin search at the most probable category-specific surfaces (e.g., kitchenware on countertop or stove, food on dining table) to minimize exploration steps and avoid aimless roaming. Apply this at the very start of the task to choose the first search destination for the target object.

### TASK: heat ###

For this heat-and-place task, apply these specific strategies:

1. Secure Exact Target First: Always identify and pick up the exact object type named in the goal before interacting with the microwave or destination. Ignore look-alikes (e.g., do not substitute a mug for a cup). Apply this after spotting any candidate object or during initial search phase, before opening or using appliances.

2. Systematic One-Pass Search: Search each plausible surface or closed container once by opening it and inspecting contents. Mark searched spots mentally to avoid redundant revisits. Apply this while locating the target object and you have not yet found it.

3. Open Then Heat: Upon reaching the microwave with the target in hand, always open the door, place the object inside, and execute the heat action before leaving. Apply this immediately after navigating to the microwave with the target object held.

4. No Appliance Before Object: Do not move to or interact with the microwave or final placement location until the target object is picked up, preventing wasted navigation steps. Apply this whenever you are tempted to head to microwave or destination without holding the required object.

5. Direct Post-Heat Placement: After heating, navigate straight to the specified destination and place the object once, avoiding extra exploration or detours. Apply this right after the heating action completes and the object is in hand.

### TASK: cool ###

For this cool-and-place task, apply these specific strategies:

1. Systematic Object Hunt: Sweep every unvisited surface and open each closed container until the exact required object is found. Never loop over already-checked empty spots. Apply this immediately after receiving the task and before any object is held.

2. Confirm Object Match: After picking up an item, verify it matches the requested type. If not, drop it and resume the search. Apply this whenever you acquire an object but the goal object is still unmet.

3. Prep Cooling Appliance: Locate the fridge first and open it so it is ready for use before or immediately after grabbing the target object. Apply this as soon as the fridge comes into view or right after acquiring the target object.

4. Enforce Cooling Before Placement: Do not place the target object in its final location until you have successfully executed a cooling action with the fridge. Apply this when holding the correct object and before any placement action is attempted.

5. Direct Post-Cooling Delivery: Once the object is cooled, navigate straight to the destination and place it without extraneous detours or re-searching. Apply this right after the cooling action succeeds.
