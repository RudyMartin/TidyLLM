  User Call: tidyllm.chat("message")
      ↓
  Context (determines compliance rules)
      ↓
  Solution (selects appropriate workflow)
      ↓
  Processing (optimizes prompt for context)
      ↓
  ReviewExecution (executes with proper model)
      ↓
  Response back to user (simple string/object)

“Context → Solution → Processing → ReviewExecution- Response”

sequenceDiagram
  participant U as User
  participant T as tidyllm.chat()
  participant X as Context
  participant S as Solution
  participant P as Processing
  participant R as ReviewExecution
  participant O as Output

  U->>T: "message"
  T->>X: infer role, rules, doc types
  X-->>T: context object
  T->>S: select workflow (mvr_scope)
  S-->>T: workflow plan
  T->>P: build prompt + inputs
  P-->>T: prompt package
  T->>R: execute with model + audit
  R-->>T: results (scores, evidence, commentary)
  T-->>O: stringify or JSON
  O-->>U: response
