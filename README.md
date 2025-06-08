# Game-theoretic LLM


## Abstract

This paper investigates the rationality of large language models (LLMs) in strategic decision-making contexts, specifically within the framework of game theory. We evaluate several state-of-the-art LLMs across a spectrum of complete-information and incomplete-information games. Our findings reveal that LLMs frequently deviate from rational strategies, particularly as the complexity of the game increases with larger payoff matrices or deeper sequential trees. 

To address these limitations, we design multiple game-theoretic workflows that guide the reasoning and decision-making processes of LLMs. These workflows aim to enhance the models' ability to compute Nash Equilibria and make rational choices, even under conditions of uncertainty and incomplete information. Experimental results demonstrate that the adoption of these workflows significantly improves the rationality and robustness of LLMs in game-theoretic tasks. Specifically, with the workflow, LLMs exhibit marked improvements in identifying optimal strategies, achieving near-optimal allocations in negotiation scenarios, and reducing susceptibility to exploitation during negotiations. Furthermore, we explore the meta-strategic considerations of whether it is rational for agents to adopt such workflows, recognizing that the decision to use or forgo the workflow constitutes a game-theoretic issue in itself. 

Our research contributes to a deeper understanding of LLMs' decision-making capabilities in strategic contexts and provides insights into enhancing their rationality through structured workflows. The findings have implications for the development of more robust and strategically sound AI agents capable of navigating complex interactive environments.

## Complete Information Game: classic game-theoretic settings
### Simultaneous Game

- Prisoner’s Dilemma
- Stag Hunt
- Battle of the Sexes (coordination game)
- Wait-Go Game
- Duopolistic Competition

### Sequential Game
- Escalation Game
- Monopoly Game
- Hot-cold Game
- Draco Game
- TriGame

## Inomplete Information Game: Common Resource Allocation with Private Information

- Deal or no Deal

  
## Quick Setup
```
conda create -n gametheory python=3.10
conda activate gametheory
pip install -r requirements.txt
cd src
```
Setup API keys in model.py

## Experiment for Complete-information Games

### Simultaneous Game without Workflow
```
python simultaneous_game_main.py --game prisoner_dilemma --max_negotiation_round 0 --model opus
```
There are 3 parameters to set:
```
game: prisoner_dilemma, stag_hunt, battle_of_sexes, duopolistic_competition, wait_go
max_negotiation_round: any integer. If you choose 0, then there is no negotiation before any action
model: opus, sonnet, o1, gpt-4o
```

### Sequential Game without Workflow
```
python sequential_game_main.py --game escalation_game --max_negotiation_round 0 --model o1
```
There are 3 parameters to set:
```
game: escalation_game, draco, hot_cold_game, monopoly, trigame
max_negotiation_round: any integer. If you choose 0, then there is no negotiation before any action
model: opus, sonnet, o1, gpt-4o
```

### Simultaneous Game and Sequential Game with Workflow
```
python workflow_design_main.py --game draco --game_type sequential --max_negotiation_round 0 --model gpt-4o
```
There are 4 parameters to set:
```
game: prisoner_dilemma, stag_hunt, battle_of_sexes, duopolistic_competition, wait_go, escalation_game, draco, hot_cold_game, monopoly, trigame
game_type: simultaneous, sequential
max_negotiation_round: any integer. If you choose 0, then there is no negotiation before any action
model: opus, sonnet, o1, gpt-4o
```

## Experiment for Incomplete-information Games
```
cd deal_no_deal
python deal_no_deal.py --model sonnet --use_workflow '' --datapoint 48
```
There are 4 parameters to set:
```
model: opus, sonnet, o1, gpt-4o
use_workflow: '' means no player uses workflow, "Alice" means only Alice (the first agent) uses workflow, "Bob" means only Bob (the second agent) uses workflow, "Alice,Bob" means both agents use workflow
datapoint: any integer between 0 - 50
```

## License
The source code of Game-theoretic LLM is licensed under [Apache 2.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE). The intended purpose is solely for research use.

## Citation
```
@article{hua2024game,
  title={Game-theoretic LLM: Agent workflow for negotiation games},
  author={Hua, Wenyue and Liu, Ollie and Li, Lingyao and Amayuelas, Alfonso and Chen, Julie and Jiang, Lucas and Jin, Mingyu and Fan, Lizhou and Sun, Fei and Wang, William and others},
  journal={arXiv preprint arXiv:2411.05990},
  year={2024}
}
```
