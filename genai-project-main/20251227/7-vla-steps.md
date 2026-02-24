This paper is gold for our VLAQualityNode! Key insights:

  OmniVIC → YRSN Mapping

  | OmniVIC Concept            | YRSN Equivalent          | How to Use                                         |
  |----------------------------|--------------------------|----------------------------------------------------|
  | RAG Memory Bank            | SkillMemory              | Store successful (instruction, K, D, phase) tuples |
  | ICL Examples               | CurriculumManager.replay | Retrieve similar experiences for context           |
  | Phase Labels               | Quality phases           | Free_motion→HIGH, Contact→LOW quality              |
  | Force Threshold            | Collapse detection       | F > Fmax = CollapseType.SATURATION                 |
  | Self-Improving Bank        | commit_if_improved()     | Only store if α improved                           |
  | Diversity-aware Collection | ReplayConsolidator       | Prevent near-duplicates                            |

  Critical Design Updates for VLAQualityNode

  1. Add Phase Detection (like OmniVIC's 4 phases):
  class VLAPhase(Enum):
      FREE_MOTION = "free_motion"    # High α expected
      APPROACHING = "approaching"     # Medium α
      CONTACT = "contact"            # Low α acceptable
      RETREAT = "retreat"            # Recovery phase

  2. Store RAG-style records (not just skills):
  @dataclass
  class VLAExperience:
      instruction: str           # Ttext
      instruction_embedding: np.ndarray  # Temb (for retrieval)
      phase: VLAPhase
      twist: np.ndarray         # 6D velocity
      wrench: np.ndarray        # 6D force/torque
      impedance_K: np.ndarray   # Stiffness (what worked)
      impedance_D: np.ndarray   # Damping (what worked)
      alpha: float              # Quality at this moment
      success: bool             # Outcome

  3. RAG Retrieval for ICL:
  def _retrieve_similar_experiences(self, instruction: str, phase: VLAPhase) -> List[VLAExperience]:
      """OmniVIC-style 4-step retrieval:
      1. Instruction embedding similarity
      2. Phase filtering
      3. Twist/wrench similarity
      4. Top-5 for ICL
      """
