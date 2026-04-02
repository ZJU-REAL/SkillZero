### GENERAL SKILLS ###

Here are some useful general strategies for answering questions with search:

1. Decompose Then Search: Break the question into minimal sub-questions (entities, attributes, relations) and handle each with its own query before synthesizing the final answer. Apply this to any complex or multi-part question, especially multi-hop or comparative tasks.

2. Precision Query Crafting: Compose searches with the exact entity name plus the sought attribute (e.g., "<entity> release date"), avoiding filler words to maximize relevant hits. Apply this when formulating initial queries for any fact lookup.

3. Iterative Query Refinement: If top results do not yield direct evidence, rephrase by adding qualifiers (dates, roles, context) or alternate names instead of repeating the same query. Apply this after scanning the first result set and finding no definitive evidence.

4. Source-Backed Assertions: Only state an answer once the supporting evidence (reliable source snippet) has been located and read; refrain from guessing or summarizing from memory. Apply this before committing to a final answer, especially for contentious or less-known facts.

5. Cross-Check Multiple Sources: Validate critical facts (dates, numbers, names) against at least two independent sources to avoid outdated or incorrect information. Apply this when the fact may have changed over time or when initial sources disagree.

6. Explicit Ambiguity Resolution: Detect ambiguous entity mentions and disambiguate by adding clarifiers (e.g., occupation, year, nationality) in the query or by exploring disambiguation pages. Apply this when an entity name maps to multiple possible referents or results appear mixed.

7. Attribute-Chaining Search: For multi-hop tasks, first retrieve the intermediate entity (e.g., a film's director) then run a second targeted query for that entity's attribute (e.g., birthplace) before synthesizing. Apply this whenever the answer depends on an attribute of a related entity rather than the entity in the question.

8. Structured Comparison: When comparing entities (earlier/later, larger/smaller), collect each relevant value separately, list them side by side, then decide using explicit comparison logic. Apply this to any comparative question involving dates, counts, or quantitative attributes.

9. Freshness Awareness: For time-sensitive queries (e.g., upcoming releases, current versions), include temporal cues like "latest", year, or "release date 2023" and prioritize most recent authoritative sources. Apply this to questions asking for "current", "next", or "latest" information.

10. Exit When Evidence Is Solid: Stop issuing further queries once you have clear, corroborated evidence; conversely, avoid premature termination if no source explicitly supports an answer. Apply this after each read step—decide to answer only if confidence is justified; otherwise refine search.
