### Multi Agent System using Langgraph###

#### Tools ###
Database Access: Menu search 
Database Access: order search

#### Nodes ####
Each tool encapsulated using toolnode (langgraph)
- Menu Agent
- Order Agent
- Orchestrator
  - Uses structured LLM output to identify the right agent to answer quers
- Synthesizer
  - When parallel call are made (SEND), this node will synthesize the output using LLM
