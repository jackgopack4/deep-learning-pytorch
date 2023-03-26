The final project is open-ended. The task is to program a SuperTuxKart ice-hockey player. You have two options: An image-based agent and a state-based agent. You need to pick one route and cannot compete in both.

Both agent types have the same objective, score as many goals as possible and win the match. However, they have different inputs and performance requirements. For each we play 2 vs 2 tournaments against agents the TAs and I coded up.

Image based agents
An image-based agent gets access to the state of the players karts. The agent also sees an image for each player, from which it can infer where the puck is and where the other teams players are. However, it does not get access to the state of the puck or opponents during the gameplay.

We anticipate to see some vision-heavy solutions here. Choose this path if you prefer to work on the vision component of your agent and use a hand-tuned controller. This option may require more compute to develop and test.

Implement image_agent/player.py.

Special rules
You need to use a deep network to process the images
You may use any controller you want (hand-designed or even some of the agents the TAs and I coded up)
Time limit 50ms per step (call to the Team.act function) on a reasonable fast GPU
State based agents
A state-based agent gets access to the state of the players karts, the opponents’ kart, and the location of the ball. There is no vision required here.

We anticipate to see some learning-based solution here. Choose this path if you prefer to work on the control component of your agent.

Implement state_agent/player.py.

Special rules
Your agent needs to be a single deep network, no hand-designed components are allowed.
Your agent should be a torch.jit.script. This is required and no exceptions.
Time limit 10ms per step (call to the Team.act function) on a reasonable fast GPU
You are not allowed to use parts of the test agent models in your solution. Note that image_jurgen_agent is also a state based agent.
Getting started
In both bases you will implement Team object in player.py. This class provides an act function, which takes inputs as described above and produces a list of actions (one per player). The current agents will simply drive straight as an example.

You can test your agents against each other or the build-in AI

python -m tournament.runner image_agent AI
The tournament runner has quite a few arguments you can explore using the -h flag. The most important include recording a video -r my_video.mp4 or saving data -s my_data.pkl.

General rules
This project is completely open-ended, however, there are a few ground rules:

You may work in a team of up to 4 students.
Teams may share data, but not code.
We will provide some test agents to compete against towards the end of the project.
In our grader, your agent will run in complete isolation from the rest of the system. Any attempt to circumvent this will result in your team failing.
You may use pystk and all its state during training, but not during gameplay in the tournament.
Your code must be efficient. If you violate the time-limit your forfeit the match.
The tournament (extra credit)
We will play an ice hockey tournament between all submissions within each agent type. Two of your agents will play two opponents. Each game will play up to 3 goals, or a maximum of 1200 steps (2 minutes). The team with more goals wins. Should one of the teams violate the above rules, it will lose the game. Should both teams violate the rules, the game is tied.

Depending on the number of submissions, we will let all submissions play against each other, or we will first have a group stage and then have the top 8 submissions play against each other. The submission with the most victories wins the tournament. The goal difference breaks ties. Additional, matches further break ties.
