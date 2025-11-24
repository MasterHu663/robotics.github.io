# RL Env

## 1. OpenAI Gymnasium

### 1.1 Agent-Environment Loop

In reinforcement learning:

1. **Agent** observes the current situation (like looking at a game screen)
2. **Agent chooses an action** based on what it sees (like pressing a button)
3. **Environment responds** with a new situation and a reward (game state changes, score updates)
4. **Repeat** until the episodes ends

<img src="./image/AE_loop.png" alt="AE_loop" style="zoom:15%;" />

```python
class Env():
  
  # Set this in SOME subclasses
  metadata: dict[str, Any] = {"render_modes": []}
  # define render_mode if your environment supports rendering
  render_mode: str | None = None
  spec: EnvSpec | None = None

  # Set these in ALL subclasses
  action_space: spaces.Space[ActType]
  observation_space: spaces.Space[ObsType]

  # Created
  _np_random: np.random.Generator | None = None
  # will be set to the "invalid" value -1 if the seed of the currently set rng is unknown
  _np_random_seed: int | None = None
  
  def step(
    self, action: ActType
  ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """
    Args:
      action (ActType): an action provided by the agent to update the environment state.

    Returns:
    	observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
      reward (SupportsFloat): The reward as a result of taking the action.
      terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
      truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
      info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
    """
  	raise NotImplementedError
    
	def reset(
    self,
    *,
    seed: int | None = None,
    option: dict[str, Any] | None = None,
  ) -> tuple[ObsType, dict[str, Any]]:
    """
    Args:
      seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`) and the read-only attribute `np_random_seed`.
      
    Returns: 
    	observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
    	info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to the ``info`` returned by :meth:`step`.
    """
    # Initialize the RNG if the seed is manually passed
    if seed is not None:
        self._np_random, self._np_random_seed = seeding.np_random(seed)
 
  def render(self) -> RenderFrame | list[RenderFrame] | None:
    raise NotImplementedError
  
  def close(self):
    pass
  

  
```

