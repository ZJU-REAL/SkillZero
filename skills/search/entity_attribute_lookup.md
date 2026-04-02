### TASK: entity_attribute_lookup ###

For entity-attribute lookup tasks, apply these specific strategies:

1. Direct Attribute Query: Include both the full entity name and target attribute (e.g., "[Name] occupation") in the first search to surface authoritative bios immediately. Apply this whenever the entity's full, unambiguous name is provided in the question.

2. Disambiguation Add-Ons: If the name is common or ambiguous, append clarifiers (birth year, nationality, known work) to the query or scan result snippets for matching context before selecting a source. Apply this when multiple people/characters share the same name or early results mix different entities.

3. Two-Source Cross-Check: Confirm the attribute in at least two independent, reputable sources (e.g., Wikipedia infobox plus biography site) to avoid hallucinations. Apply this after the first plausible answer appears or when the attribute seems uncommon or uncertain.

4. Iterative Query Refinement: If initial search returns irrelevant hits, adjust the query instead of repeating it—use alternative spellings, middle initials, or related works/titles. Apply this when consecutive top results do not mention the sought attribute for the intended entity.

5. Attribute-Only Response: State solely the requested attribute (e.g., "architect") in the final answer, omitting extra explanations to meet answer-format expectations. Apply this after verifying the attribute and preparing the final output for an entity_attribute_lookup task.
