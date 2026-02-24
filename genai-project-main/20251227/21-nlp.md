 Here's the new command-level NLP architecture for Unitree integration:

  New Module: yrsn.core.nlp

  Voice → ASR → IntentParser → CommandDecomposer → SkillMatcher → SDK
                ↓                    ↓
           ParsedIntent         R/S/N Quality Gate
           (action, object,     (α > 0.6 → execute)
            color, location)    (α < 0.6 → clarify)

  Components

  | Component          | Purpose                                                         |
  |--------------------|-----------------------------------------------------------------|
  | IntentParser       | "pick up the red cup" → {action: pick, object: cup, color: red} |
  | CommandDecomposer  | R/S/N at command level, not word level                          |
  | SkillMatcher       | Match intent to robot skill catalog                             |
  | GroundingValidator | "the red cup" → cup_id_123                                      |

  R/S/N Semantics (Fixed)

  | Signal | Meaning     | High Score =                                      |
  |--------|-------------|---------------------------------------------------|
  | R      | Relevance   | Clear intent + grounded target + capability match |
  | S      | Superfluous | Politeness, filler, verbose phrasing              |
  | N      | Noise       | Ambiguity + ungrounded refs + safety risk         |

  Usage for Unitree

  from yrsn.core.nlp import IntentParser, CommandDecomposer

  parser = IntentParser()
  decomposer = CommandDecomposer(capabilities=robot.get_skills())

  intent = parser.parse(voice_command)
  quality = decomposer.decompose(intent, scene_objects=perception.objects)

  if quality.can_execute:
      robot.execute(intent.action_type, **quality.grounded_objects)
  else:
      robot.say(quality.clarification_prompt)

  This replaces the static word lists with semantic command understanding that integrates with UnifoLM or external LLMs
