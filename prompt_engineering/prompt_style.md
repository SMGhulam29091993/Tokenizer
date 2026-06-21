# Prompt style

## Alpaca Prompting

### INSTRUCTIONS : <SYSTEM_PROMPT>\n

### INPUT : <USER_QUERY>\n

### RESPONSE : \n

## ChatML Prompting

### Mainly used by GPT, Gemini, Claude

{
"role" : "System" | "User" | "Assisstant",
"content" : "string"
}

## INST Prompting

### Mainly used in LAMA LLM

[INST] What is the time ? [INST]
