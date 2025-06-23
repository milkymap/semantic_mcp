await mcp_router.recommend_tools(
                problem="recherche moi les dernières informations sur l'IA, je veux une analyse en profondeur et comparative des boites OpenAI, Anthropic, Google. Au moment de l'analyse, je veux que tu me parle a chaque instant avec ton system vocal. comme l'experience jarvis ok. Quand tu auras fini, crée un repo github dans lequel tu mettras tous tes resultats au format markdown. N'oublie pas à la fin de m'appeler sur mon numero de tel pour m'expliquer tout ce que tu as trouve.",
                thinking_budget=10_000
            )



{
  "mcpServers": {
    "Time/Timezone": {
      "command": "docker",
       "args": ["run", "-i", "--rm", "mcp/time"]
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/milkymap/Workspace"
        ]
    },
    "modal-sandbox": {
      "command": "docker",
      "args": [
        "run", 
        "--rm", 
        "-i", 
        "-e", 
        "MODAL_TOKEN_ID", 
        "-e", 
        "MODAL_TOKEN_SECRET", 
        "-v", 
        "/Users/milkymap/Workspace:/home/solver/workspace",
        "modal:mcp", 
        "--transport", 
        "stdio"
      ],
      "env": {
        "MODAL_TOKEN_ID": "ak-Uoplcn7Y68fjQFhQlj3Yfc",
        "MODAL_TOKEN_SECRET": "as-ACRhdotDhcM35PX8An68t7"
      }
    }
  }
}
