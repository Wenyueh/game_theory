import argparse
import numpy as np
from tqdm import tqdm
from model import close_source_call
import time 
from deal_no_deal_metrics import (
    check_pareto_optimalities,
    compute_score,
    translate_values,
    check_envy_free,
    check_envy_free_pareto_optimal,
    check_human_pareto_optimal_envy_free
)
import json

### data processing
def process_data(data):
    def parse_agent1_input(line):
        start = line.index('<input>') + len('<input>')
        end = line.index('</input>')
        example_count = [int(a) for i,a in enumerate([a.strip() for a in line[start:end].split(' ')[1:-1]]) if i % 2 == 0]
        agent1_values = [int(a) for i,a in enumerate([a.strip() for a in line[start:end].split(' ')[1:-1]]) if i % 2 == 1]
        agent1_values_text = translate_values(example_count, agent1_values)
        return example_count, agent1_values, agent1_values_text
    
    def parse_agent2_input(line):
        start = line.index('<partner_input>') + len('<partner_input>')
        end = line.index('</partner_input>')
        example_count = [int(a) for i,a in enumerate([a.strip() for a in line[start:end].split(' ')[1:-1]]) if i % 2 == 0]
        agent2_values = [int(a) for i,a in enumerate([a.strip() for a in line[start:end].split(' ')[1:-1]]) if i % 2 == 1]
        agent2_values_text = translate_values(example_count, agent2_values)
        return example_count, agent2_values, agent2_values_text
    
    def parse_human_outcome(line):
        start = line.index('<output>') + len('<output>')
        end = line.index('</output>')
        outcomes = [a.strip() for a in line[start:end].split(' ')[1:-1]]
        if 'item0=' in outcomes[0]:
            agent1_outcomes = [int(a.split('=')[1]) for a in outcomes[:3]]
            agent2_outcomes = [int(a.split('=')[1]) for a in outcomes[3:]]
            return agent1_outcomes, agent2_outcomes
        else:
            return outcomes[:3], outcomes[3:]

    example_count, agent1_values, agent1_values_text = parse_agent1_input(data)
    example_count, agent2_values, agent2_values_text = parse_agent2_input(data)
    agent1_human_outcomes, agent2_human_outcomes = parse_human_outcome(data)

    return example_count, agent1_values, agent1_values_text, agent2_values, agent2_values_text, agent1_human_outcomes, agent2_human_outcomes

def parse(message):
    assert '<s>' in message and '</s>' in message 
    start = message.index('<s>') + len('<s>')
    end = message.index('</s>')
    return message[start:end]

def parse_deal(message):
    assert '<deal>' in message and '</deal>' in message
    start = message.index('<deal>') + len('<deal>')
    end = message.index('</deal>')
    deal = message[start:end]
    deal = deal.split(' ')
    deal = [int(a.split('=')[1]) for a in deal]
    return deal

def parse_value(message):
    assert '<value>' in message and '</value>' in message
    start = message.index('<value>') + len('<value>')
    end = message.index('</value>')
    deal = message[start:end]
    deal = deal.split(' ')
    deal = [int(a.split('=')[1]) for a in deal]
    return deal

def parse_strategy(message):
    strategies = []
    assert '<strategy1>' in message and '</strategy1>' in message
    for i in range(1, 65):
        if f'<strategy{i}>' in message and f'</strategy{i}>' in message:
            start = message.index(f'<strategy{i}>') + len(f'<strategy{i}>')
            end = message.index(f'</strategy{i}>')
            strategy = message[start:end]
            strategy = strategy.split(' ')
            strategy = [int(a.split('=')[1]) for a in strategy]
            strategies.append(strategy)
    return strategies

class Agent:
    def __init__(self, args, data, name):
        (self.example_count, 
         self.agent1_values, 
         self.agent1_values_text, 
         self.agent2_values, 
         self.agent2_values_text,
           _, 
           _) = process_data(data)
        self.args = args
        self.max_negotiation_round = self.args.max_negotiation_round
        self.previous_message = []
        
        self.name = name
        self.the_other_player = 'Bob' if self.name == 'Alice' else 'Alice'
        self.game_setting()
    
    def game_setting(self):
        if self.name == 'Alice':
            self.agent_values = self.agent1_values
            self.the_other_agent_values = self.agent2_values
        else:
            self.agent_values = self.agent2_values
            self.the_other_agent_values = self.agent1_values
        if self.args.special_prompting:
            self.game_description = f"""
### Game Description

This is a negotiation game. There are {self.example_count[0]} books, {self.example_count[1]} hats, and {self.example_count[2]} balls in total. 
Each item has a value to you and the other player {self.the_other_player}. You only know your values for all the item, but you don't know {self.the_other_player}'s valuation of each item; this is unknown to you.
Thus do not assume the value of the items to the other player {self.the_other_player} is the same as yours.

Your goal is to MAXIMIZE the total reward/value you alone can obtain by taking the items after negotiation.
You need to negotiate with the other player {self.the_other_player} to decide which and how many items you and your the other player {self.the_other_player} will each get.
DO NOT REVEAL your values of the items to the other player {self.the_other_player} through out the game.
Notice that if you come to disagreement on the negotiation, neither of you will obtain any reward.

There are two principles you need to focus on when discussing the deal with your the other player {self.the_other_player}: 
(1) pareto optimality: a deal is pareto optimal if there is no other deal that makes both you and your the other player better off.
(2) envy freeness: a deal is envy free if each person receive items that are, in their eyes, at least as valuable as the share received by your the other player.
Deals that are both pareto optimal and envy free are considered the best deals.

You are playing the role of {self.name}.

### Item Values to You

{translate_values(self.example_count, self.agent_values)}
"""
        else:
            self.game_description = f"""
### Game Description

This is a negotiation game. There are {self.example_count[0]} books, {self.example_count[1]} hats, and {self.example_count[2]} balls in total. 
Each item has a value to you and the other player {self.the_other_player}. You only know your values for all the item, but you don't know {self.the_other_player}'s valuation of each item; this is unknown to you.
Thus do not assume the value of the items to the other player {self.the_other_player} is the same as yours.

Your goal is to MAXIMIZE the total reward/value you alone can obtain by taking the items after negotiation.
You need to negotiate with the other player {self.the_other_player} to decide which and how many items you and your partner {self.the_other_player} will each get.
DO NOT REVEAL your values of the items to the other player {self.the_other_player} through out the game.
Notice that if you come to disagreement on the negotiation, neither of you will obtain any reward.

You are playing the role of {self.name}.

### Item Values to You

{translate_values(self.example_count, self.agent_values)}
"""
    
    def present_deal(self):
        present_deal_prompt = f"""
### Present Deal

You have finished the negotiation. Now, you need to present the deal to the other player.
You need to present which and how many items you will get based on your negotiation.
Write down the number of books, hats, and balls you want to get in the format of <deal>book=x hat=y ball=z</deal>, where x, y, and z are the number of books, hats, and balls you want to get, respectively.
""" 
        previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
        
        present_deal_prompt = self.game_description + previous_messages + present_deal_prompt

        while True:
            try:
                message = close_source_call('claude', present_deal_prompt, self.args.system_prompt)
                message = parse_deal(message)
                return message 
            except:
                time.sleep(0.1)

    def guess_on_value_rank(self):
        ## guess the relative item value rank of the other player to himself/herself
        def parse_ranked_value(message):
            assert "<value>" in message and "</value>" in message
            start = message.index("<value>") + len("<value>")
            end = message.index("</value>")
            rank = message[start:end]
            assert '>' in rank
            return rank

        guess_on_rank_prompt = f"""

### Guess the Relative Item Value of the Other Player

Based on the current negotiation messages, what do you think are the relative item values to the other player {self.the_other_player}?
Now, rank the item values of the other player from the most valuable to the least valuable. 
For example, if you think the books are the most valuable to {self.the_other_player}, followed by the hats, and then the balls, you can write down <value>book > hat > ball</value>.
"""
        previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
        
        present_deal_prompt = self.game_description + previous_messages + guess_on_rank_prompt

        while True:
            try:
                message = close_source_call('claude', present_deal_prompt, self.args.system_prompt)
                rank = parse_ranked_value(message)
                return rank 
            except:
                time.sleep(0.1)

    def guess_on_relative_value(self):
        ## guess the relative item value rank of the other player compared with yours
        def parse_ranked_value(message):
            assert "<value>" in message and "</value>" in message
            start = message.index("<value>") + len("<value>")
            end = message.index("</value>")
            rank = message[start:end]
            assert ',' in rank
            return rank
        
        guess_on_value_prompt = f"""

### Guess the Value of the Other Player Compared with Yours

Based on the current negotiation messages, what do you think are the item values of the other player compared with yours?
Now, for each item, think about whether the other player values the item more, less, or equally to you. 
For example, if you think the other player values the books more than you, the hats equally to you, and the balls less than you, you can write down <value>Items {self.the_other_player} values higher than you: book, Items {self.the_other_player} values the same as you: hat, Items {self.the_other_player} values less than you: ball</value>.
"""
        previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
        
        present_deal_prompt = self.game_description + previous_messages + guess_on_value_prompt

        while True:
            try:
                message = close_source_call('claude', present_deal_prompt, self.args.system_prompt)
                print('message:', message)
                rank = parse_ranked_value(message)
                return rank 
            except:
                time.sleep(0.1)

    def present_known_values(self):
        # if there's any item that we know for sure
        pass

    def initial_negotiation_message(self):
        ## what is the initial negotiation message?
        if self.args.special_prompting:
            negotiate_prompt = f"""
### Negotiation

You can discuss with {self.the_other_player} to maximize the reward you can obtain. You have a maximum of {self.max_negotiation_round} rounds to negotiate.
Analyze the situation and decide on what to say to your the other player {self.the_other_player}.

There are two principles you need to focus on when negotiate on the deal: 
(1) pareto optimality: a deal is pareto optimal if there is no other deal that makes both you and your the other player better off.
(2) envy freeness: a deal is envy free if you do not envy the reward your the other player gets.
Deals that are both pareto optimal and envy free are considered the best deals.
Thus you should pay attention to whether the deal is pareto optimal and envy free when negotiating on the deal.

Surround your message with '<s>' and '</s>' to indicate the start and end of your message. For example, '<s>Hi, how are you?</s>'.
You can also choose the halt the negotiation by saying '<s>halt negotiation</s>'.
Especially, if you have come to an agreement, say '<s>halt negotiation</s>' to end the negotiation.
"""
        else:
            negotiate_prompt = f"""
### Negotiation

You can discuss with {self.the_other_player} to maximize the reward you can obtain. You have a maximum of {self.max_negotiation_round} rounds to negotiate.
Analyze the situation and decide on what to say to the other player.

Surround your message with '<s>' and '</s>' to indicate the start and end of your message. For example, '<s>Hi, how are you?</s>'.
You can also choose the halt the negotiation by saying '<s>halt negotiation</s>'.
Especially, if you have come to an agreement, say '<s>halt negotiation</s>' to end the negotiation.
"""
        if self.previous_message:
            previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
            negotiate_prompt += previous_messages

        negotiate_prompt = self.game_description + negotiate_prompt

        while True:
            try:
                message = close_source_call('claude', negotiate_prompt, self.args.system_prompt)
                message = parse(message)
                return message 
            except:
                time.sleep(0.1)
    
    def summarize_deal_basedon_message(self, negotiation_message):
        ## what is the deal based on the current negotiation messages? 
        proposal_summarizer_prompt = f"""
### Summarize the Proposal based your own negotiation message

Based on the negotiation message you have just sent, what is the deal you are proposing to the other player {self.the_other_player}?
Write down the number of books, hats, and balls you want to get in the format of <deal>book=x hat=y ball=z</deal>, where x, y, and z are the number of books, hats, and balls you want to get, respectively.
"""         
        
        negotiation_message = f"""
### Your Proposed Negotiation Message (haven't told the other player {self.the_other_player} yet)

{negotiation_message}
"""
        present_deal_prompt = self.game_description + negotiation_message + proposal_summarizer_prompt

        while True:
            try:
                message = close_source_call('claude', present_deal_prompt, self.args.system_prompt)
                message = parse_deal(message)
                return message 
            except:
                time.sleep(0.1)
        
    def self_reflect_on_envy_free(self, value_rank, current_proposal):
        def parse(message):
            assert '<answer>' in message and '</answer>' in message
            assert '<reasoning>' in message and '</reasoning>' in message
            start = message.index('<answer>') + len('<answer>')
            end = message.index('</answer>')
            answer = message[start:end]
            start = message.index('<reasoning>') + len('<reasoning>')
            end = message.index('</reasoning>')
            reasoning = message[start:end]
            return answer, reasoning
        ## does the current negotiation message's implicit deal make the agent envy free?
        current_proposal_for_the_other_player = [self.example_count[i] - current_proposal[i] for i in range(3)]
        current_proposal_prompt = f"""

### Current Proposal

Based on currently proposed negotiation message, you are proposing the following deal: 
You get {current_proposal}: {current_proposal[0]} book, {current_proposal[1]} hat, {current_proposal[2]} ball.
The other player {self.the_other_player} gets {current_proposal_for_the_other_player}: {current_proposal_for_the_other_player[0]} book, {current_proposal_for_the_other_player[1]} hat, {current_proposal_for_the_other_player[2]} ball.
"""
        self_reflect_on_envy_free_prompt = f"""

### Self-Reflect on Envy Free

Based on previous conversations, you've guessed that the other player values the items in the following order: {value_rank}.

Based on the current negotiation messages and the deal you are proposing, think about whether the deal is envy free.
Envy freeness means that you do not envy the reward the other player gets: 
you think the items you get based on YOUR own value system are at least as valuable as the items the other player gets based on YOUR own value system; same for the other player, that is, the items the other player gets based on THEIR own value system are at least as valuable as the items you get based on THEIR own value system.

Based on your value system on the items and the value system you guessed for the other player, do you think the deal you are proposing is envy free?

Provide your reasoning process as well as the final answer of whether the deal is envy free.
<reasoning> ... </reasoning>
<answer>yes/no</answer>
"""
        
        if self.previous_message:
            previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
            self_reflect_on_envy_free_prompt = self.game_description + previous_messages + current_proposal_prompt + self_reflect_on_envy_free_prompt
        else:
            self_reflect_on_envy_free_prompt = self.game_description + previous_messages + current_proposal_prompt + self_reflect_on_envy_free_prompt


        while True:
            try:
                message = close_source_call('claude', self_reflect_on_envy_free_prompt, self.args.system_prompt)
                answer, reasoning = parse(message)
                return answer, reasoning 
            except:
                time.sleep(0.1)

    def self_reflect_on_pareto_optimal(self, relative_value, current_proposal):
        def parse(message):
            assert '<answer>' in message and '</answer>' in message
            assert '<reasoning>' in message and '</reasoning>' in message
            start = message.index('<answer>') + len('<answer>')
            end = message.index('</answer>')
            answer = message[start:end]
            start = message.index('<reasoning>') + len('<reasoning>')
            end = message.index('</reasoning>')
            reasoning = message[start:end]
            return answer, reasoning
        ## does the current negotiation message's implicit deal make the agent pareto optimal?
        current_proposal_for_the_other_player = [self.example_count[i] - current_proposal[i] for i in range(3)]
        current_proposal_prompt = f"""
### Current Proposal

Based on currently proposed negotiation message, you are proposing the following deal: 
You get {current_proposal}: {current_proposal[0]} book, {current_proposal[1]} hat, {current_proposal[2]} ball.
The other player {self.the_other_player} gets {current_proposal_for_the_other_player}: {current_proposal_for_the_other_player[0]} book, {current_proposal_for_the_other_player[1]} hat, {current_proposal_for_the_other_player[2]} ball.
"""
        self_reflect_on_pareto_optimal_prompt = f"""
### Self-reflect on Pareto Optimal

Based on previous conversations, you've guessed that the other player values the items compared with how you value the items: {relative_value}.

Based on the current negotiation messages and the deal you are proposing, think about whether the deal is pareto optimal.
Pareo optimal means that there is no other deal that makes both you and the other player better off.
Remember your total REWARD based on the CURRENT proposal is {compute_score(self.agent_values, current_proposal)} and any other pareto optimal strategy should not make you worse off.

Based on your value system on the items and the value system you guessed for the other player, do you think the deal you are proposing is pareto optimal?
Provide your reasoning process as well as the final answer of whether the deal is envy free.
<reasoning> ... </reasoning>
<answer>yes/no</answer>
"""

        if self.previous_message:
            previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
            self_reflect_on_pareto_optimal_prompt = self.game_description + previous_messages + current_proposal_prompt + self_reflect_on_pareto_optimal_prompt
        else:
            self_reflect_on_pareto_optimal_prompt = self.game_description + previous_messages + current_proposal_prompt + self_reflect_on_pareto_optimal_prompt

        while True:
            try:
                message = close_source_call('claude', self_reflect_on_pareto_optimal_prompt, self.args.system_prompt)
                answer, reasoning = parse(message)
                return answer, reasoning 
            except:
                time.sleep(0.1)

    def update_negotiation_message(self, envy_free_suggestion, pareto_optimal_suggestion, current_message):
        ## update the negotiation message based on the current negotiation message
        if envy_free_suggestion[0] == 'yes' and pareto_optimal_suggestion[0] == 'yes':
            return current_message
        else:
            if envy_free_suggestion[0] == 'no':
                envy_free_message = f"""
### Envy Free Suggestion

Based on the self-reflection on envy freeness, you think the deal you are proposing is not envy free.
This is because {envy_free_suggestion[1]}
"""
            else:
                envy_free_message = """
### Envy Free Suggestion

Your self-reflection on envy freeness suggests that the deal you are proposing is envy free.
"""
            if pareto_optimal_suggestion[0] == 'no':
                pareto_optimal_message = f"""
### Pareto Optimal Suggestion

Based on the self-reflection on pareto optimality, you think the deal you are proposing is not pareto optimal.
This is because {pareto_optimal_suggestion[1]}
"""
            else:
                pareto_optimal_message = """
### Pareto Optimal Suggestion

Your self-reflection on pareto optimality suggests that the deal you are proposing is pareto optimal.
"""
            suggestion = envy_free_message + pareto_optimal_message

            renegotiate_prompt = f"""
### Rethink about Negotiation Message

You have proposed one message in negotiation currently (### Currently Proposed Negotiation Message), but based on the self-reflection on envy freeness and pareto optimality, you may want to rethink about the negotiation message.
Below are the suggestions based on the self-reflection on envy freeness and pareto optimality:

{suggestion}

Now rethink about the egotiation message and decide on what to say to the other player.

Surround your message with '<s>' and '</s>' to indicate the start and end of your message. For example, '<s>Hi, how are you?</s>'.
You can also choose the halt the negotiation by saying '<s>halt negotiation</s>'.
Especially, if you have come to an agreement, say '<s>halt negotiation</s>' to end the negotiation.
"""
        if self.previous_message:
            previous_messages = "\n\n## The previous rounds of negotiation are presented below:\n\n" + '\n'.join(self.previous_message)
            renegotiate_prompt += previous_messages

        renegotiate_prompt = self.game_description + renegotiate_prompt + """### Currently Proposed Negotiation Message\n\n""" + current_message + renegotiate_prompt

        while True:
            try:
                message = close_source_call('claude', renegotiate_prompt, self.args.system_prompt)
                message = parse(message)
                return message 
            except:
                time.sleep(0.1)
        
    def negotiate_with_feedback(self):
        envy_free = False 
        pareto_optimal = False
        most_round = 0 # rethink at most 3 times
        while not envy_free or not pareto_optimal:
            negotiation_message = self.initial_negotiation_message()
            print('##Initial negotiation message:', negotiation_message)
            if negotiation_message == 'halt negotiation':
                return negotiation_message
            current_proposal = self.summarize_deal_basedon_message(negotiation_message)
            print('##Current proposal:', current_proposal)
            value_rank = self.guess_on_value_rank() # book > hat > ball for 1 player
            print('##Value rank:', value_rank)
            relative_value = self.guess_on_relative_value() # comparison between 2 players
            print('##Relative value:', relative_value)
            envy_free_suggestion = self.self_reflect_on_envy_free(value_rank, current_proposal)
            print('##Envy free suggestion:', envy_free_suggestion)
            pareto_optimal_suggestion = self.self_reflect_on_pareto_optimal(relative_value, current_proposal)
            print('##Pareto optimal suggestion:', pareto_optimal_suggestion)
            negotiation_message = self.update_negotiation_message(envy_free_suggestion, pareto_optimal_suggestion, negotiation_message)
            print('##Updated negotiation message:', negotiation_message)
            envy_free = envy_free_suggestion[0] == 'yes'
            pareto_optimal = pareto_optimal_suggestion[0] == 'yes'
            most_round += 1
            if most_round > 2:
                return negotiation_message
        return negotiation_message
            
    def negotiate_without_feedback(self):
        negotiation_message = self.initial_negotiation_message()
        return negotiation_message

class DealNoDeal:
    def __init__(self, args, data):
        (self.example_count, 
         self.agent1_values, 
         self.agent1_values_text, 
         self.agent2_values, 
         self.agent2_values_text,
         self.human_outcomes1, 
         self.human_outcomes2) = process_data(data)
        self.alice = Agent(args, data, 'Alice')
        self.bob = Agent(args, data, 'Bob')
        self.max_negotiation_round = args.max_negotiation_round
        self.args = args

    def check_deal_match(self, agent1_picks, agent2_picks):
        assert int(agent1_picks[0]) + int(agent2_picks[0]) <= self.example_count[0]
        assert int(agent1_picks[1]) + int(agent2_picks[1]) <= self.example_count[1]
        assert int(agent1_picks[2]) + int(agent2_picks[2]) <= self.example_count[2]

    def negotiation(self, total_negotiation_round):
        for negotiation_round in range(self.max_negotiation_round):
            if total_negotiation_round + negotiation_round >= 1 and self.args.use_workflow:
                alice_message = self.alice.negotiate_with_feedback()
            else:
                alice_message = self.alice.negotiate_without_feedback()
            print('Alice said in round {}: '.format(negotiation_round+1)+alice_message)
            print('='*20)
            self.alice.previous_message.append('Alice said in round {}: '.format(negotiation_round+1)+alice_message)
            self.bob.previous_message.append('Alice said in round {}: '.format(negotiation_round+1)+alice_message)
            if total_negotiation_round + negotiation_round >= 1 and self.args.use_workflow:
                bob_message = self.bob.negotiate_with_feedback()
            else:
                bob_message = self.bob.negotiate_without_feedback()
            print('Bob said in round {}: '.format(negotiation_round+1)+bob_message)
            print('='*20)
            self.alice.previous_message.append('Bob replied in round {}: '.format(negotiation_round+1)+bob_message)
            self.bob.previous_message.append('Bob replied in round {}: '.format(negotiation_round+1)+bob_message)
            if bob_message == 'halt negotiation' or alice_message == 'halt negotiation':
                return negotiation_round

        return negotiation_round

    def play(self):
        negotiation_done = False
        total_negotiation_round = 0

        while not negotiation_done:
            # start negotiation first
            negotiation_round = self.negotiation(total_negotiation_round)
            total_negotiation_round += negotiation_round

            # present deal
            alice_deal = self.alice.present_deal()
            bob_deal = self.bob.present_deal()

            # check whether number in the deal matches
            try:
                self.check_deal_match(alice_deal, bob_deal)
                return alice_deal, bob_deal, total_negotiation_round
            except:
                negotiation_done = False
            if total_negotiation_round > 10:
                return 'No deal', 'No deal', total_negotiation_round
        
    def check_reasonable_guess(self):
        items = ['book', 'hat', 'ball']
        alice_value = {'book':self.agent1_values[0], 'hat':self.agent1_values[1], 'ball':self.agent1_values[2]}
        alice_value_rank = list({k: v for k, v in sorted(alice_value.items(), key=lambda item: item[1])}.keys())
        bob_value = {'book':self.agent2_values[0], 'hat':self.agent2_values[1], 'ball':self.agent2_values[2]}
        bob_value_rank = list({k: v for k, v in sorted(bob_value.items(), key=lambda item: item[1])}.keys())
        bob_relative_value = []
        alice_relative_value = []
        for i, (alice_value, bob_value) in enumerate(zip(self.agent1_values, self.agent2_values)):
            if alice_value < bob_value:
                alice_relative_value.append(f'lower than you: {items[i]}')
                bob_relative_value.append(f'higher than you: {items[i]}')
            elif alice_value > bob_value:
                alice_relative_value.append(f'higher than you: {items[i]}')
                bob_relative_value.append(f'less than you: {items[i]}')
                
        guessed_bob_value_rank = self.alice.guess_on_value_rank()
        print('Alice guessed Bob value rank:', guessed_bob_value_rank)
        print('Actual Bob value rank:', bob_value_rank)

        guessed_bob_relative_value = self.alice.guess_on_relative_value()
        print('Alice guessed Bob relative value:', guessed_bob_relative_value)
        print('Actual Bob relative value:', bob_relative_value)

        guessed_alice_value_rank = self.bob.guess_on_value_rank()
        print('Bob guessed Alice value rank:', guessed_alice_value_rank)
        print('Actual Alice value rank:', alice_value_rank)

        guessed_alice_relative_value = self.bob.guess_on_relative_value()
        print('Bob guessed Alice relative value:', guessed_alice_relative_value)
        print('Actual Alice relative value:', alice_relative_value)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deal or No Deal')
    parser.add_argument('--data', type=str, default='deal_no_deal_test.txt', help='Path to the data file')
    parser.add_argument('--system_prompt', type=str, default="rational")
    parser.add_argument('--max_negotiation_round', type=int, default=20)
    parser.add_argument('--datapoint_id', type=int, default=0)
    parser.add_argument('--special_prompting', action='store_true')
    parser.add_argument('--use_workflow', action='store_true')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = f.readlines()
    # remove repetitive lines
    data = [d for i,d in enumerate(data) if i % 2 == 0]
    total_number = len(data)
    print(f'Total number of data: {total_number}')
    # only experiment on data that are not pareto optimal envy free
    not_pareto_optimal_envy_free_human_choices = []
    for d in tqdm(data):
        if not check_human_pareto_optimal_envy_free(d):
            not_pareto_optimal_envy_free_human_choices.append(d)
    print(f'Number of data where human choices are not pareto optimal envy free: {len(not_pareto_optimal_envy_free_human_choices)}')
    print(f'Percentage of Pareto optimal envy free data: {1 - len(not_pareto_optimal_envy_free_human_choices)/total_number}')
    data = not_pareto_optimal_envy_free_human_choices

    # play the game
    game = DealNoDeal(args, data[args.datapoint_id])
    alice_deal, bob_deal, total_negotiation_round = game.play()
    print('Alice deal:')
    print(alice_deal)
    print('Bob deal:')
    print(bob_deal)
    print('Total negotiation round:', total_negotiation_round)
    data_to_collect = {'negotiation_message':game.alice.previous_message, 'alice_deal':alice_deal, 'bob_deal':bob_deal, 'total_negotiation_round':total_negotiation_round}
    game.check_reasonable_guess()

    # check performance on envy free & pareto optimal
    alice_score = compute_score(alice_deal, game.agent1_values)
    bob_score = compute_score(bob_deal, game.agent2_values)
    print('Alice score:', alice_score)
    print('Bob score:', bob_score)
    switch_bob_score = compute_score(alice_deal, game.agent2_values)
    switch_alice_score = compute_score(bob_deal, game.agent1_values)
    print('Alice score in switch deal:', switch_alice_score)
    print('Bob score in switch deal:', switch_bob_score)
    data_to_collect['alice_score'] = int(alice_score)
    data_to_collect['bob_score'] = int(bob_score)

    pareto = check_pareto_optimalities(alice_deal, game.agent1_values, bob_deal, game.agent2_values, game.example_count)
    print('Is it Pareto optimal?', pareto)
    data_to_collect['pareto'] = pareto

    envy_free = check_envy_free(alice_deal, bob_deal, data[args.datapoint_id])
    print('Is it envy free?', envy_free)
    data_to_collect['envy_free'] = envy_free

    envy_free_pareto_optimal = check_envy_free_pareto_optimal(alice_deal, bob_deal, data[args.datapoint_id])
    print('Is it envy free and pareto optimal?', envy_free_pareto_optimal)
    data_to_collect['envy_free_pareto_optimal'] = envy_free_pareto_optimal

    if args.special_prompting:
        with open('result/deal_no_deal/prompting_{}.json'.format(args.datapoint_id), 'w') as f:
            json.dump(data_to_collect, f)
    else:
        with open('result/deal_no_deal/{}.json'.format(args.datapoint_id), 'w') as f:
            json.dump(data_to_collect, f)
        
        
